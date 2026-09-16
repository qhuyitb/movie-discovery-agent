"""Collaborative filtering, user profiling and the hybrid ranker.

The matrix is 610 x 5,135 so everything fits comfortably in dense float32 -
no sparse-matrix acrobatics needed, which keeps the code readable.

Three independent scorers, deliberately kept separable so the evaluation can
attribute wins and losses to a specific signal:

* `user_cf`    - "people whose taste matches yours liked this"
* `item_cf`    - "this resembles films *you* rated highly" (adjusted cosine)
* `content`    - "this reads like the films you love" (plot/LSA similarity)

`hybrid` fuses the normalised scores. Predictions are shrunk toward the user's
mean by how much evidence supports them, which matters a lot here: 51% of the
catalogue has fewer than 5 ratings, so an unshrunk neighbour average will
happily promise 5.0 on the strength of one stranger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import MovieData
from .index import PlotIndex


def _minmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale to [0,1] using only the candidate entries, so fusion weights mean something."""
    out = np.zeros_like(x)
    if not mask.any():
        return out
    vals = x[mask]
    lo, hi = float(vals.min()), float(vals.max())
    out[mask] = 0.5 if hi - lo < 1e-9 else (vals - lo) / (hi - lo)
    return out


@dataclass
class Recommender:
    data: MovieData
    index: PlotIndex
    ratings: pd.DataFrame
    user_ids: np.ndarray = field(repr=False, default=None)
    item_ids: np.ndarray = field(repr=False, default=None)

    def __post_init__(self):
        self.user_ids = np.sort(self.ratings["userId"].unique()).astype(int)
        self.item_ids = self.data.movies["movieId"].to_numpy(dtype=int)
        self.u_pos = {int(u): i for i, u in enumerate(self.user_ids)}
        self.i_pos = {int(m): i for i, m in enumerate(self.item_ids)}

        n_u, n_i = len(self.user_ids), len(self.item_ids)
        R = np.zeros((n_u, n_i), dtype=np.float32)
        rows = self.ratings["userId"].map(self.u_pos).to_numpy()
        cols = self.ratings["movieId"].map(self.i_pos)
        keep = cols.notna().to_numpy()
        R[rows[keep], cols[keep].to_numpy().astype(int)] = self.ratings["rating"].to_numpy(dtype=np.float32)[keep]

        self.R = R
        self.rated = R > 0
        counts = self.rated.sum(axis=1)
        self.user_mean = np.divide(R.sum(axis=1), np.maximum(counts, 1), dtype=np.float32)
        self.user_mean[counts == 0] = np.float32(self.data.global_mean)
        self.user_count = counts

        # Mean-centred ratings; unrated entries stay exactly 0 (= "no opinion").
        self.C = np.where(self.rated, R - self.user_mean[:, None], 0).astype(np.float32)

        # Item popularity/quality is derived from *this* Recommender's ratings frame,
        # not from the full dataset - otherwise an offline evaluation leaks the test set
        # into the candidate filter and the popularity prior.
        self.item_count = self.rated.sum(axis=0).astype(np.int32)
        item_sum = R.sum(axis=0)
        item_mean = np.divide(item_sum, np.maximum(self.item_count, 1))
        gmean = float(self.ratings["rating"].mean())
        m = config.BAYES_PRIOR_COUNT
        v = self.item_count.astype(np.float32)
        self.item_bayes = ((v / (v + m)) * np.where(self.item_count > 0, item_mean, gmean)
                           + (m / (v + m)) * gmean).astype(np.float32)

        self._user_sim = None
        self._item_norm = None
        self._hist_cache = None

    @classmethod
    def build(cls, data: MovieData, index: PlotIndex, ratings: pd.DataFrame | None = None):
        return cls(data=data, index=index, ratings=data.ratings if ratings is None else ratings)

    def user_history(self, user_id: int) -> pd.DataFrame:
        """The user's ratings *as this model can see them*, newest first.

        Always read history through here rather than through `MovieData`: during
        evaluation the Recommender is fitted on a training split, and going
        straight to the full dataframe would silently leak the test ratings into
        profiling and content scoring.
        """
        if self._hist_cache is None:
            meta = self.data.movies[["movieId", "title", "year", "genres", "genre_list"]].reset_index(drop=True)
            joined = self.ratings.merge(meta, on="movieId", how="left")
            self._hist_cache = {
                int(uid): df.sort_values("timestamp", ascending=False).reset_index(drop=True)
                for uid, df in joined.groupby("userId")
            }
        return self._hist_cache.get(
            int(user_id),
            pd.DataFrame(columns=["movieId", "rating", "timestamp", "title", "year", "genres", "genre_list"]),
        )

    # ------------------------------------------------------- user similarity
    @property
    def user_sim(self) -> np.ndarray:
        """Significance-weighted cosine between mean-centred user vectors."""
        if self._user_sim is None:
            norms = np.linalg.norm(self.C, axis=1, keepdims=True)
            U = self.C / np.maximum(norms, 1e-6)
            sim = (U @ U.T).astype(np.float32)
            co = (self.rated.astype(np.float32) @ self.rated.astype(np.float32).T)
            sim *= co / (co + config.SHRINKAGE)          # trust overlap, not luck
            sim[co < config.MIN_CO_RATED] = 0.0
            np.fill_diagonal(sim, 0.0)
            self._user_sim, self._co_counts = sim, co
        return self._user_sim

    def neighbours(self, user_id: int, k: int = config.NEIGHBOURS_K) -> list[dict]:
        u = self.u_pos.get(int(user_id))
        if u is None:
            return []
        sims = self.user_sim[u]
        k = min(k, int((sims > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            {
                "userId": int(self.user_ids[v]),
                "similarity": round(float(sims[v]), 3),
                "co_rated": int(self._co_counts[u, v]),
                "n_ratings": int(self.user_count[v]),
                "avg_rating": round(float(self.user_mean[v]), 2),
            }
            for v in top
        ]

    # ------------------------------------------------------- item similarity
    @property
    def item_norm(self) -> np.ndarray:
        if self._item_norm is None:
            M = self.C.T                                  # items x users
            self._item_norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-6)
        return self._item_norm

    def similar_items(self, movie_id: int, k: int = 10, min_support: int = 5) -> list[dict]:
        j = self.i_pos.get(int(movie_id))
        if j is None:
            return []
        sims = self.item_norm @ self.item_norm[j]
        sims[self.item_count < min_support] = -np.inf
        sims[j] = -np.inf
        top = np.argsort(-sims)[:k]
        return [
            {"movieId": int(self.item_ids[i]), "title": self.data.title_of(int(self.item_ids[i])),
             "similarity": round(float(sims[i]), 3)}
            for i in top if np.isfinite(sims[i])
        ]

    # -------------------------------------------------------------- scorers
    def user_cf_scores(self, user_id: int, k: int = config.NEIGHBOURS_K):
        """Returns (predicted rating per item, support per item)."""
        u = self.u_pos.get(int(user_id))
        n_i = len(self.item_ids)
        if u is None:
            return np.full(n_i, self.data.global_mean, np.float32), np.zeros(n_i, np.float32)

        sims = self.user_sim[u].copy()
        if (sims > 0).sum() > k:
            cut = np.partition(-sims, k)[k]
            sims[sims < -cut] = 0.0
        sims[sims < 0] = 0.0

        weight = self.rated.T @ sims                      # sum of |sim| of neighbours who rated i
        numer = self.C.T @ sims
        pred = self.user_mean[u] + numer / np.maximum(weight, 1e-6)
        # Shrink toward the user's own mean when few neighbours have an opinion.
        conf = weight / (weight + 0.5)
        pred = self.user_mean[u] + (pred - self.user_mean[u]) * conf
        return pred.astype(np.float32), weight.astype(np.float32)

    def item_cf_scores(self, user_id: int, k: int = config.ITEM_NEIGHBOURS_K):
        u = self.u_pos.get(int(user_id))
        n_i = len(self.item_ids)
        if u is None:
            return np.full(n_i, self.data.global_mean, np.float32), np.zeros(n_i, np.float32)

        rated_idx = np.flatnonzero(self.rated[u])
        if rated_idx.size == 0:
            return np.full(n_i, self.user_mean[u], np.float32), np.zeros(n_i, np.float32)

        S = self.item_norm @ self.item_norm[rated_idx].T   # all items x user's items
        S[S < 0] = 0.0
        if S.shape[1] > k:                                 # keep the k most similar seen items
            thresh = np.partition(S, -k, axis=1)[:, -k][:, None]
            S = np.where(S >= thresh, S, 0.0)
        dev = self.C[u, rated_idx]
        weight = S.sum(axis=1)
        pred = self.user_mean[u] + (S @ dev) / np.maximum(weight, 1e-6)
        conf = weight / (weight + 0.5)
        pred = self.user_mean[u] + (pred - self.user_mean[u]) * conf
        return pred.astype(np.float32), weight.astype(np.float32)

    def content_scores(self, user_id: int, seed_ids=None) -> np.ndarray:
        """Plot/LSA similarity to a taste centroid built from what the user liked."""
        if seed_ids is not None and len(seed_ids):
            return self.index.score_against_movies(list(seed_ids))
        hist = self.user_history(user_id)
        if hist.empty:
            return np.zeros(len(self.item_ids), dtype=np.float32)
        # Centroid of what the user actually liked. Using *all* ratings with signed
        # weights was tried first and was measurably mushier: a 190-film history
        # averages out to "generic movie".
        mean = float(hist["rating"].mean())
        liked = hist[hist["rating"] >= max(4.0, mean)]
        if liked.empty:
            liked = hist.nlargest(max(3, len(hist) // 5), "rating")
        w = (liked["rating"].to_numpy(dtype=float) - mean + 0.1)
        return self.index.score_against_movies(liked["movieId"].tolist(), weights=w)

    def popularity_scores(self) -> np.ndarray:
        return self.item_bayes

    # ----------------------------------------------------------- recommending
    def candidate_mask(
        self, user_id: int, min_ratings: int = config.MIN_RATINGS_FOR_CONFIDENCE,
        include_genres=None, exclude_genres=None, year_range=None,
        exclude_ids=None, exclude_seen: bool = True,
    ) -> np.ndarray:
        movies = self.data.movies
        mask = np.ones(len(self.item_ids), dtype=bool)
        mask &= self.item_count >= min_ratings

        if exclude_seen:
            u = self.u_pos.get(int(user_id))
            if u is not None:
                mask &= ~self.rated[u]
        if include_genres:
            want = {g.lower() for g in include_genres}
            mask &= movies["genre_list"].apply(lambda gl: bool(want & {g.lower() for g in gl})).to_numpy()
        if exclude_genres:
            skip = {g.lower() for g in exclude_genres}
            mask &= ~movies["genre_list"].apply(lambda gl: bool(skip & {g.lower() for g in gl})).to_numpy()
        if year_range:
            lo, hi = year_range
            years = movies["year"].to_numpy(dtype=float)
            mask &= np.nan_to_num(years, nan=-1) >= (lo or -np.inf)
            mask &= np.nan_to_num(years, nan=1e9) <= (hi or np.inf)
        if exclude_ids:
            for mid in exclude_ids:
                j = self.i_pos.get(int(mid))
                if j is not None:
                    mask[j] = False
        return mask

    def recommend(
        self, user_id: int, n: int = 10, method: str = "hybrid",
        seed_ids=None, **filters,
    ) -> list[dict]:
        mask = self.candidate_mask(user_id, **filters)
        if not mask.any():
            return []

        ucf, usup = self.user_cf_scores(user_id)
        icf, isup = self.item_cf_scores(user_id)
        content = self.content_scores(user_id, seed_ids=seed_ids)
        pop = self.popularity_scores()

        if method == "popularity":
            score = _minmax(pop, mask)
        elif method == "user_cf":
            score = _minmax(ucf, mask)
        elif method == "item_cf":
            score = _minmax(icf, mask)
        elif method == "content":
            score = _minmax(content, mask)
        elif method == "hybrid":
            w = config.HYBRID_WEIGHTS
            score = (
                w["user_cf"] * _minmax(ucf, mask)
                + w["item_cf"] * _minmax(icf, mask)
                + w["content"] * _minmax(content, mask)
            )
            # Small quality prior breaks ties among equally-plausible candidates.
            wp = config.POPULARITY_PRIOR
            score = (1 - wp) * score + wp * _minmax(pop, mask)
        else:
            raise ValueError(f"unknown method: {method}")

        score = np.where(mask, score, -np.inf)
        top = np.argsort(-score)[:n]
        out = []
        for j in top:
            if not np.isfinite(score[j]):
                continue
            mid = int(self.item_ids[j])
            rec = self.data.brief(mid)
            raw = float(0.9 * ucf[j] + 0.1 * icf[j])
            rec.update(
                score=round(float(score[j]), 4),
                predicted_rating=round(float(np.clip(raw, 0.5, 5.0)), 2),
                # Mean-centred CF over-shoots for users whose own average is already
                # near the ceiling; the raw value is kept so the UI can admit the cap.
                predicted_rating_capped=bool(raw > 5.02 or raw < 0.48),
                predicted_rating_raw=round(raw, 2),
                evidence={
                    "neighbour_support": round(float(usup[j]), 2),
                    "similar_item_support": round(float(isup[j]), 2),
                    "plot_similarity": round(float(content[j]), 3),
                },
            )
            out.append(rec)
        return out

    # ------------------------------------------------------------ explaining
    def peer_opinions(self, user_id: int, movie_id: int, k_users: int = 30) -> dict:
        """What the user's nearest neighbours actually rated this film."""
        j = self.i_pos.get(int(movie_id))
        if j is None:
            return {"error": "unknown movie"}
        rows = []
        for nb in self.neighbours(user_id, k=k_users):
            v = self.u_pos[nb["userId"]]
            if self.rated[v, j]:
                rows.append({**nb, "rating": float(self.R[v, j]),
                             "vs_their_average": round(float(self.R[v, j] - self.user_mean[v]), 2)})
        ratings = [r["rating"] for r in rows]
        everyone = self.R[:, j][self.rated[:, j]]
        return {
            "movie": self.data.brief(int(movie_id)),
            "n_similar_users_who_rated_it": len(rows),
            "similar_user_mean": round(float(np.mean(ratings)), 2) if ratings else None,
            "similar_user_mean_vs_own_average": (
                round(float(np.mean([r["vs_their_average"] for r in rows])), 2) if rows else None
            ),
            "everyone_mean": round(float(everyone.mean()), 2) if everyone.size else None,
            "everyone_n": int(everyone.size),
            "raters": sorted(rows, key=lambda r: -r["similarity"])[:10],
        }

    def why(self, user_id: int, movie_id: int, k: int = 5) -> dict:
        """Evidence bundle for 'why would I like that?' - all of it traceable to data."""
        j = self.i_pos.get(int(movie_id))
        if j is None:
            return {"error": "unknown movie"}
        u = self.u_pos.get(int(user_id))
        hist = self.user_history(user_id)

        # 1. Which films in the user's own history drive the item-CF prediction?
        drivers = []
        if u is not None:
            rated_idx = np.flatnonzero(self.rated[u])
            if rated_idx.size:
                sims = self.item_norm[rated_idx] @ self.item_norm[j]
                order = np.argsort(-sims)[:k]
                drivers = [
                    {"title": self.data.title_of(int(self.item_ids[rated_idx[o]])),
                     "your_rating": float(self.R[u, rated_idx[o]]),
                     "taste_overlap": round(float(sims[o]), 3)}
                    for o in order if sims[o] > 0.05
                ]

        # 2. Which films in the history read most like this one, textually?
        content_matches = []
        if not hist.empty:
            pos = [self.index.position[m] for m in hist["movieId"] if m in self.index.position]
            if pos:
                sims = self.index.latent[pos] @ self.index.latent[self.index.position[int(movie_id)]]
                order = np.argsort(-sims)[:k]
                content_matches = [
                    {"title": self.data.title_of(int(hist.iloc[int(o)]["movieId"])),
                     "your_rating": float(hist.iloc[int(o)]["rating"]),
                     "plot_similarity": round(float(sims[o]), 3)}
                    for o in order if sims[o] > 0.1
                ]

        genre_fit = []
        profile = self.genre_affinity(user_id)
        row = self.data.movie(int(movie_id))
        if row is not None:
            for g in row["genre_list"]:
                if g in profile:
                    genre_fit.append({"genre": g, **profile[g]})

        return {
            "movie": self.data.brief(int(movie_id)),
            "predicted_rating": round(float(np.clip(self.user_cf_scores(user_id)[0][j], 0.5, 5.0)), 2),
            "predicted_rating_raw": round(float(self.user_cf_scores(user_id)[0][j]), 2),
            "because_you_rated": drivers,
            "plot_resembles_your_favourites": content_matches,
            "genre_fit": genre_fit,
            "peers": self.peer_opinions(user_id, movie_id, k_users=30),
            "plot_keywords": self.index.top_terms(int(movie_id), 8),
        }

    # -------------------------------------------------------------- profiling
    def genre_affinity(self, user_id: int) -> dict[str, dict]:
        hist = self.user_history(user_id)
        if hist.empty:
            return {}
        mean = float(hist["rating"].mean())
        rows: dict[str, list[float]] = {}
        for _, r in hist.iterrows():
            for g in (r["genre_list"] if isinstance(r["genre_list"], list) else []):
                rows.setdefault(g, []).append(float(r["rating"]))
        return {
            g: {"n": len(v), "avg": round(float(np.mean(v)), 2),
                "vs_your_average": round(float(np.mean(v) - mean), 2)}
            for g, v in sorted(rows.items(), key=lambda kv: -len(kv[1]))
        }

    def profile(self, user_id: int) -> dict:
        hist = self.user_history(user_id)
        if hist.empty:
            return {"userId": int(user_id), "error": "no ratings for this user"}
        affinity = self.genre_affinity(user_id)
        liked = hist[hist["rating"] >= max(4.0, hist["rating"].mean())]
        years = hist["year"].dropna()
        return {
            "userId": int(user_id),
            "n_ratings": int(len(hist)),
            "avg_rating": round(float(hist["rating"].mean()), 2),
            "rating_spread_std": round(float(hist["rating"].std(ddof=0)), 2),
            "generosity_vs_all_users": round(float(hist["rating"].mean() - self.data.global_mean), 2),
            "top_rated": [
                {"title": self.data.title_of(int(r["movieId"])), "rating": float(r["rating"]),
                 "genres": r["genres"]}
                for _, r in hist.sort_values("rating", ascending=False).head(8).iterrows()
            ],
            "lowest_rated": [
                {"title": self.data.title_of(int(r["movieId"])), "rating": float(r["rating"])}
                for _, r in hist.sort_values("rating").head(4).iterrows()
            ],
            "favourite_genres": [
                {"genre": g, **v} for g, v in
                sorted(affinity.items(), key=lambda kv: (-kv[1]["vs_your_average"], -kv[1]["n"]))[:5]
                if v["n"] >= 3
            ],
            "most_watched_genres": [{"genre": g, **v} for g, v in list(affinity.items())[:5]],
            "era": {
                "median_year": int(years.median()) if len(years) else None,
                "share_pre_1990": round(float((years < 1990).mean()), 2) if len(years) else None,
            },
            "n_liked_4_plus": int((hist["rating"] >= 4).sum()),
            "sample_of_favourites": [self.data.title_of(int(m)) for m in liked["movieId"].head(6)],
        }

    def blind_spots(self, user_id: int) -> dict:
        """Genres the user has barely touched, ranked by how much the catalogue offers."""
        hist = self.user_history(user_id)
        affinity = self.genre_affinity(user_id)
        catalogue = {}
        # IMAX is a projection format, not a taste signal - it would otherwise top
        # every blind-spot list purely because nobody tags their ratings with it.
        for g in [g for g in self.data.all_genres if g != "IMAX"]:
            sub = self.data.movies[self.data.movies["genre_list"].apply(lambda gl: g in gl)]
            well_rated = sub[(sub["n_ratings"] >= 20)]
            catalogue[g] = {"catalogue_n": int(len(sub)), "well_known_n": int(len(well_rated))}

        gaps = []
        for g, cat in catalogue.items():
            seen = affinity.get(g, {"n": 0, "avg": None, "vs_your_average": None})
            share = seen["n"] / max(len(hist), 1)
            gaps.append({
                "genre": g,
                "you_have_rated": seen["n"],
                "share_of_your_ratings": round(share, 3),
                "your_avg_here": seen["avg"],
                "vs_your_average": seen["vs_your_average"],
                "catalogue_size": cat["catalogue_n"],
                "well_known_titles_available": cat["well_known_n"],
            })
        gaps.sort(key=lambda d: (d["you_have_rated"], -d["well_known_titles_available"]))
        return {
            "userId": int(user_id),
            "n_ratings": int(len(hist)),
            "unexplored_genres": [g for g in gaps if g["you_have_rated"] <= 2][:6],
            "under_explored_genres": [g for g in gaps if 2 < g["you_have_rated"] <= 8][:6],
            "dominant_genres": [
                {"genre": g, **v} for g, v in list(affinity.items())[:3]
            ],
        }
