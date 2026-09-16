"""Content retrieval over plot summaries.

Two views of the same text, blended at query time:

* **TF-IDF** - exact word/bigram overlap. Strong when the user names concrete
  plot elements ("heist", "amnesia", "hitman").
* **LSA** (truncated SVD of the TF-IDF matrix) - soft topical similarity, which
  is what rescues vibe queries like "slow burn character study" where no single
  word has to match.

No neural embedding model on purpose: it keeps the whole system CPU-only,
dependency-light and reproducible. See the report for the tradeoff.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config
from .data import MovieData


def _document(row) -> str:
    """Full text of a movie: title, genres, viewer tags and plot."""
    genres = str(row["genres"]).replace("|", " ")
    return " ".join([str(row["title"]), genres, str(row["tags"]), str(row["plot"])])


def _labels(row) -> str:
    """Genres + tags only. Added back as a weighted overlay (see `build`)."""
    return str(row["genres"]).replace("|", " ") + " " + str(row["tags"])


@dataclass
class PlotIndex:
    movie_ids: np.ndarray
    vectorizer: TfidfVectorizer
    tfidf: "np.ndarray"          # sparse csr
    svd: TruncatedSVD
    latent: np.ndarray           # L2-normalised dense doc vectors
    position: dict[int, int]

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, data: MovieData) -> "PlotIndex":
        docs = [_document(row) for _, row in data.movies.iterrows()]
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=config.TFIDF_MIN_DF,
            max_features=config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        )
        base = vec.fit_transform(docs)
        # Genres and tags are short and high-signal, so they need more weight than a
        # single mention inside a 3,200-character plot would give them. Re-projecting a
        # labels-only corpus through the *same* vocabulary does that without the
        # degenerate bigrams ("psychological psychological") that simply repeating the
        # text inline produced.
        labels = vec.transform([_labels(row) for _, row in data.movies.iterrows()])
        tfidf = normalize(base + config.LABEL_BOOST * labels)
        svd = TruncatedSVD(n_components=config.SVD_COMPONENTS, random_state=0)
        latent = normalize(svd.fit_transform(tfidf))
        ids = data.movies["movieId"].to_numpy(dtype=int)
        return cls(ids, vec, tfidf, svd, latent, {int(m): i for i, m in enumerate(ids)})

    @classmethod
    def load_or_build(cls, data: MovieData) -> "PlotIndex":
        config.CACHE_DIR.mkdir(exist_ok=True)
        key = hashlib.md5(
            f"{len(data.movies)}-{config.SVD_COMPONENTS}-{config.TFIDF_MIN_DF}-{config.LABEL_BOOST}-v2".encode()
        ).hexdigest()[:10]
        path = config.CACHE_DIR / f"plot_index_{key}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        index = cls.build(data)
        with path.open("wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return index

    # ----------------------------------------------------------------- query
    def score_query(self, query: str) -> np.ndarray:
        """Blended similarity of every movie to a free-text query."""
        q_tfidf = normalize(self.vectorizer.transform([query]))
        lexical = np.asarray(self.tfidf @ q_tfidf.T.toarray()).ravel()
        q_latent = normalize(self.svd.transform(q_tfidf))
        semantic = self.latent @ q_latent.ravel()
        w = config.LEXICAL_WEIGHT
        return w * lexical + (1.0 - w) * semantic

    def score_against_movies(self, movie_ids, weights=None) -> np.ndarray:
        """Similarity of every movie to a weighted set of seed movies (a taste profile)."""
        rows = [self.position[int(m)] for m in movie_ids if int(m) in self.position]
        if not rows:
            return np.zeros(len(self.movie_ids))
        w = np.ones(len(rows)) if weights is None else np.asarray(
            [weights[i] for i, m in enumerate(movie_ids) if int(m) in self.position], dtype=float
        )
        w = w / (np.abs(w).sum() + 1e-9)
        centroid = normalize((self.latent[rows] * w[:, None]).sum(axis=0).reshape(1, -1)).ravel()
        return self.latent @ centroid

    def similar_to(self, movie_id: int, k: int = 10) -> list[tuple[int, float]]:
        pos = self.position.get(int(movie_id))
        if pos is None:
            return []
        sims = self.latent @ self.latent[pos]
        sims[pos] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.movie_ids[i]), float(sims[i])) for i in top]

    def top_terms(self, movie_id: int, n: int = 8) -> list[str]:
        """Highest-TF-IDF terms for a movie - cheap, honest evidence for 'why this one'."""
        pos = self.position.get(int(movie_id))
        if pos is None:
            return []
        row = self.tfidf[pos].tocoo()
        names = self.vectorizer.get_feature_names_out()
        pairs = sorted(zip(row.col, row.data), key=lambda p: -p[1])
        terms, seen = [], set()
        for c, _ in pairs:
            t = names[c]
            parts = t.split()
            if len(parts) == 2 and parts[0] == parts[1]:
                continue                       # "psychological psychological" from tag boundaries
            if any(t in s or s in t for s in seen):
                continue                       # skip near-duplicates of a term already shown
            terms.append(t)
            seen.add(t)
            if len(terms) >= n:
                break
        return terms
