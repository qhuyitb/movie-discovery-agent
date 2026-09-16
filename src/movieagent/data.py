"""Dataset loading plus the small lookups every other module needs.

One `MovieData` object owns the dataframes, the derived per-movie statistics and
title resolution. Everything downstream takes it as a dependency, so there is a
single place where "what does the data actually look like" is decided.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from . import config

_ARTICLE_SUFFIX = re.compile(r",\s*(the|a|an|la|le|les|il|el)$", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalise_title(title: str) -> str:
    """'Godfather, The' -> 'the godfather'; strips punctuation for fuzzy matching."""
    t = str(title).strip().lower()
    m = _ARTICLE_SUFFIX.search(t)
    if m:
        t = f"{m.group(1)} {t[: m.start()]}"
    t = _NON_ALNUM.sub(" ", t)
    return " ".join(t.split())


@dataclass
class MovieData:
    movies: pd.DataFrame            # indexed by movieId
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame
    _norm_titles: dict[str, int] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, data_dir=config.DATA_DIR) -> "MovieData":
        movies = pd.read_csv(data_dir / "movies_with_plots.csv")
        ratings = pd.read_csv(data_dir / "ratings.csv")
        tags = pd.read_csv(data_dir / "tags.csv")
        links = pd.read_csv(data_dir / "links.csv")

        movies["plot"] = movies["plot"].fillna("")
        movies["genres"] = movies["genres"].fillna("(no genres listed)")
        movies["genre_list"] = movies["genres"].str.split("|")
        movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

        # Tags are sparse but high signal where present - fold them into the movie row.
        tag_blob = (
            tags.assign(tag=tags["tag"].astype(str))
            .groupby("movieId")["tag"]
            .apply(lambda s: ", ".join(sorted({t.strip().lower() for t in s if t.strip()})))
        )
        movies["tags"] = movies["movieId"].map(tag_blob).fillna("")

        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        movies["n_ratings"] = movies["movieId"].map(stats["count"]).fillna(0).astype(int)
        movies["mean_rating"] = movies["movieId"].map(stats["mean"])

        # Bayesian damping: a 5.0 from two people should not outrank a 4.4 from 200.
        global_mean = ratings["rating"].mean()
        m = config.BAYES_PRIOR_COUNT
        v, r = movies["n_ratings"], movies["mean_rating"].fillna(global_mean)
        movies["bayes_score"] = (v / (v + m)) * r + (m / (v + m)) * global_mean

        movies = movies.set_index("movieId", drop=False).sort_index()

        norm = {}
        for mid, title in zip(movies["movieId"], movies["title"]):
            norm.setdefault(normalise_title(title), int(mid))

        return cls(movies=movies, ratings=ratings, tags=tags, links=links, _norm_titles=norm)

    # --------------------------------------------------------------- lookups
    @cached_property
    def global_mean(self) -> float:
        return float(self.ratings["rating"].mean())

    @cached_property
    def all_genres(self) -> list[str]:
        seen: set[str] = set()
        for gl in self.movies["genre_list"]:
            seen.update(gl)
        seen.discard("(no genres listed)")
        return sorted(seen)

    @cached_property
    def _user_groups(self) -> dict[int, pd.DataFrame]:
        return {int(uid): df for uid, df in self.ratings.groupby("userId")}

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_groups

    def user_ratings(self, user_id: int) -> pd.DataFrame:
        """That user's ratings joined to movie metadata, newest first."""
        df = self._user_groups.get(int(user_id))
        if df is None:
            return pd.DataFrame(columns=["movieId", "rating", "timestamp", "title", "year", "genres"])
        meta = self.movies[["movieId", "title", "year", "genres", "genre_list"]].reset_index(drop=True)
        out = df.merge(meta, on="movieId", how="left")
        return out.sort_values("timestamp", ascending=False).reset_index(drop=True)

    def movie(self, movie_id: int) -> pd.Series | None:
        try:
            return self.movies.loc[int(movie_id)]
        except KeyError:
            return None

    def title_of(self, movie_id: int) -> str:
        row = self.movie(movie_id)
        if row is None:
            return f"<unknown movie {movie_id}>"
        year = "" if pd.isna(row["year"]) else f" ({int(row['year'])})"
        return f"{row['title']}{year}"

    def resolve_movie(self, query: str | int) -> int | None:
        """Best-effort title -> movieId. Handles ids, exact, prefix and fuzzy matches."""
        if isinstance(query, (int, np.integer)) or (isinstance(query, str) and query.isdigit()):
            mid = int(query)
            return mid if mid in self.movies.index else None

        q = normalise_title(query)
        if not q:
            return None
        if q in self._norm_titles:
            return self._norm_titles[q]

        keys = list(self._norm_titles)
        # Prefer the most-rated movie whose title starts with / contains the query,
        # so "star wars" lands on the popular entry rather than an obscure one.
        prefixed = [k for k in keys if k.startswith(q)] or [k for k in keys if q in k]
        if prefixed:
            best = max(prefixed, key=lambda k: self.movies.loc[self._norm_titles[k], "n_ratings"])
            return self._norm_titles[best]

        close = difflib.get_close_matches(q, keys, n=1, cutoff=0.82)
        return self._norm_titles[close[0]] if close else None

    def search_titles(self, query: str, k: int = 5) -> list[int]:
        """Several plausible title matches, popular first - used for disambiguation."""
        q = normalise_title(query)
        hits = [mid for key, mid in self._norm_titles.items() if q and q in key]
        hits.sort(key=lambda mid: -self.movies.loc[mid, "n_ratings"])
        return hits[:k]

    def brief(self, movie_id: int) -> dict:
        """Compact, JSON-safe movie record - the shape tools hand back to the agent."""
        row = self.movie(movie_id)
        if row is None:
            return {"movieId": int(movie_id), "title": "<unknown>"}
        return {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": row["genres"],
            "n_ratings": int(row["n_ratings"]),
            "mean_rating": None if pd.isna(row["mean_rating"]) else round(float(row["mean_rating"]), 2),
            "tags": row["tags"][:120] or None,
        }
