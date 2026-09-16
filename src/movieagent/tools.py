"""The investigation layer: everything the assistant is allowed to look up.

Design rule: **tools return evidence, never prose.** Each one hands back a plain
JSON-able dict of numbers and titles pulled from the dataset. The narration layer
(rule-based or LLM) may only rephrase what a tool returned. That is what keeps
"explains its reasoning using historical data" honest rather than a vibe the
model produced from its own memory of movies.

Every call is appended to `MovieTools.trace`, so any answer can be replayed as
"which questions did it ask the data, in what order".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cf import Recommender
from .data import MovieData
from .index import PlotIndex


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: Any
    ms: float

    def summary(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items() if v not in (None, [], ""))
        return f"{self.name}({args})"


@dataclass
class MovieTools:
    data: MovieData
    index: PlotIndex
    model: Recommender
    trace: list[ToolCall] = field(default_factory=list)

    @classmethod
    def bootstrap(cls) -> "MovieTools":
        data = MovieData.load()
        index = PlotIndex.load_or_build(data)
        return cls(data=data, index=index, model=Recommender.build(data, index))

    # ------------------------------------------------------------- dispatch
    def call(self, name: str, arguments: dict | None = None) -> Any:
        arguments = dict(arguments or {})
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name not in TOOL_NAMES:
            return {"error": f"unknown tool: {name}"}
        started = time.perf_counter()
        try:
            result = fn(**arguments)
        except TypeError as exc:
            result = {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:                                   # surfaced, not swallowed
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.trace.append(ToolCall(name, arguments, result, (time.perf_counter() - started) * 1000))
        return result

    def reset_trace(self) -> None:
        self.trace = []

    # ---------------------------------------------------------------- tools
    def get_user_profile(self, user_id: int) -> dict:
        """Taste summary for a user: volume, generosity, favourite genres, top films."""
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset",
                    "hint": "valid ids are 1-610"}
        return self.model.profile(int(user_id))

    def get_user_ratings(self, user_id: int, titles: list[str] | None = None,
                         limit: int = 20, min_rating: float | None = None) -> dict:
        """The user's own ratings - all of them, or just for specific titles."""
        hist = self.model.user_history(int(user_id))
        if hist.empty:
            return {"error": f"user {user_id} has no ratings"}
        if titles:
            wanted = {}
            for t in titles:
                mid = self.data.resolve_movie(t)
                if mid is None:
                    wanted[t] = None
                else:
                    row = hist[hist["movieId"] == mid]
                    wanted[self.data.title_of(mid)] = (
                        float(row.iloc[0]["rating"]) if len(row) else None
                    )
            return {"userId": int(user_id), "your_ratings": wanted,
                    "note": "null means the user has not rated that film"}
        if min_rating is not None:
            hist = hist[hist["rating"] >= float(min_rating)]
        hist = hist.sort_values("rating", ascending=False).head(int(limit))
        return {
            "userId": int(user_id),
            "n_shown": len(hist),
            "ratings": [{"title": self.data.title_of(int(r["movieId"])),
                         "rating": float(r["rating"]), "genres": r["genres"]}
                        for _, r in hist.iterrows()],
        }

    def search_movies(self, query: str, n: int = 8, include_genres: list[str] | None = None,
                      exclude_genres: list[str] | None = None, year_from: int | None = None,
                      year_to: int | None = None, min_ratings: int = 5,
                      min_avg_rating: float | None = None, for_user: int | None = None) -> dict:
        """Content search over plot summaries, genres and tags, with quality filters.

        `for_user` drops films that user has already rated and reports the
        personalised prediction alongside each hit.
        """
        scores = self.index.score_query(query)
        mask = self.model.candidate_mask(
            int(for_user) if for_user else -1,
            min_ratings=int(min_ratings), include_genres=include_genres,
            exclude_genres=exclude_genres,
            year_range=(year_from, year_to) if (year_from or year_to) else None,
            exclude_seen=for_user is not None,
        )
        if min_avg_rating is not None:
            mask &= np.nan_to_num(self.data.movies["mean_rating"].to_numpy(dtype=float), nan=0) >= float(min_avg_rating)
        if not mask.any():
            return {"query": query, "results": [], "note": "no movie passed the filters"}

        pred = None
        if for_user is not None and self.data.has_user(int(for_user)):
            pred, _ = self.model.user_cf_scores(int(for_user))

        ranked = np.argsort(-np.where(mask, scores, -np.inf))[: int(n)]
        results = []
        for j in ranked:
            mid = int(self.model.item_ids[j])
            row = self.data.brief(mid)
            row["plot_match"] = round(float(scores[j]), 3)
            row["why_it_matched"] = self.index.top_terms(mid, 6)
            if pred is not None:
                row["predicted_rating_for_user"] = round(float(pred[j]), 2)
            results.append(row)
        return {"query": query, "filters": {"include_genres": include_genres,
                                            "exclude_genres": exclude_genres,
                                            "year_from": year_from, "year_to": year_to,
                                            "min_ratings": min_ratings},
                "results": results}

    def recommend_for_user(self, user_id: int, n: int = 5, method: str = "hybrid",
                           include_genres: list[str] | None = None,
                           exclude_genres: list[str] | None = None,
                           similar_to: list[str] | None = None,
                           year_from: int | None = None, year_to: int | None = None,
                           min_ratings: int = 5) -> dict:
        """Personalised ranking. `similar_to` steers the content component toward given films."""
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset"}
        seeds = []
        unresolved = []
        for t in (similar_to or []):
            mid = self.data.resolve_movie(t)
            (seeds.append(mid) if mid else unresolved.append(t))
        recs = self.model.recommend(
            int(user_id), n=int(n), method=method, seed_ids=seeds or None,
            include_genres=include_genres, exclude_genres=exclude_genres,
            year_range=(year_from, year_to) if (year_from or year_to) else None,
            min_ratings=int(min_ratings),
        )
        return {
            "userId": int(user_id), "method": method,
            "seeded_on": [self.data.title_of(m) for m in seeds] or None,
            "unresolved_titles": unresolved or None,
            "constraints": {"include_genres": include_genres, "exclude_genres": exclude_genres},
            "recommendations": recs,
        }

    def find_similar_users(self, user_id: int, n: int = 5) -> dict:
        """Nearest neighbours by rating pattern, with the overlap that justifies each."""
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset"}
        nbrs = self.model.neighbours(int(user_id), k=int(n))
        for nb in nbrs:
            hist = self.model.user_history(nb["userId"])
            top = hist.nlargest(3, "rating")
            nb["their_favourites"] = [self.data.title_of(int(m)) for m in top["movieId"]]
        return {"userId": int(user_id), "similar_users": nbrs,
                "note": "similarity = cosine on mean-centred ratings, shrunk by co-rated count"}

    def how_similar_users_rated(self, user_id: int, movie: str, n_users: int = 30) -> dict:
        """The multi-hop one: find peers, then look up what *they* thought of a film."""
        mid = self.data.resolve_movie(movie)
        if mid is None:
            return {"error": f"could not find a movie called {movie!r}",
                    "did_you_mean": [self.data.title_of(m) for m in self.data.search_titles(movie, 5)]}
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset"}
        out = self.model.peer_opinions(int(user_id), mid, k_users=int(n_users))
        own = self.model.user_history(int(user_id))
        seen = own[own["movieId"] == mid]
        out["you_already_rated_it"] = float(seen.iloc[0]["rating"]) if len(seen) else None
        return out

    def explain_recommendation(self, user_id: int, movie: str) -> dict:
        """Full evidence bundle for 'why would I like that?'."""
        mid = self.data.resolve_movie(movie)
        if mid is None:
            return {"error": f"could not find a movie called {movie!r}",
                    "did_you_mean": [self.data.title_of(m) for m in self.data.search_titles(movie, 5)]}
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset"}
        return self.model.why(int(user_id), mid)

    def find_blind_spots(self, user_id: int) -> dict:
        """Genres the user has barely rated, weighted by what the catalogue offers."""
        if not self.data.has_user(int(user_id)):
            return {"error": f"user {user_id} has no ratings in this dataset"}
        out = self.model.blind_spots(int(user_id))
        # Give each gap a concrete entry point, otherwise the answer is just a scolding.
        for gap in out["unexplored_genres"] + out["under_explored_genres"]:
            picks = self.model.recommend(int(user_id), n=2, method="hybrid",
                                         include_genres=[gap["genre"]], min_ratings=20)
            gap["suggested_entry_points"] = [
                {"title": p["title"], "mean_rating": p["mean_rating"], "n_ratings": p["n_ratings"]}
                for p in picks
            ]
        return out

    def get_movie_details(self, movie: str, plot_chars: int = 600) -> dict:
        """Metadata, audience numbers, tags and a trimmed plot for one film."""
        mid = self.data.resolve_movie(movie)
        if mid is None:
            return {"error": f"could not find a movie called {movie!r}",
                    "did_you_mean": [self.data.title_of(m) for m in self.data.search_titles(movie, 5)]}
        row = self.data.movie(mid)
        out = self.data.brief(mid)
        out["plot"] = str(row["plot"])[: int(plot_chars)]
        out["rating_confidence"] = ("high" if row["n_ratings"] >= 50 else
                                    "medium" if row["n_ratings"] >= 10 else "low - few ratings")
        return out

    def find_similar_movies(self, movie: str, n: int = 6, basis: str = "both") -> dict:
        """Neighbours of a film by plot text, by audience overlap, or both."""
        mid = self.data.resolve_movie(movie)
        if mid is None:
            return {"error": f"could not find a movie called {movie!r}",
                    "did_you_mean": [self.data.title_of(m) for m in self.data.search_titles(movie, 5)]}
        out = {"movie": self.data.title_of(mid), "basis": basis}
        if basis in ("plot", "both"):
            out["similar_by_plot"] = [
                {**self.data.brief(m), "plot_similarity": round(s, 3)}
                for m, s in self.index.similar_to(mid, int(n))
            ]
        if basis in ("audience", "both"):
            out["similar_by_audience"] = [
                {**self.data.brief(int(d["movieId"])), "audience_similarity": d["similarity"]}
                for d in self.model.similar_items(mid, int(n))
            ]
        return out


# --------------------------------------------------------------------------
# JSON-Schema tool definitions. Shared verbatim by the OpenAI and Anthropic
# backends, and used by the rule-based planner as the list of legal actions.
# --------------------------------------------------------------------------
def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}


_USER = {"user_id": {"type": "integer", "description": "MovieLens user id (1-610)"}}
_GENRES = {"type": "array", "items": {"type": "string"},
           "description": "Genre names, e.g. ['Animation','Horror']"}

TOOL_SCHEMAS = [
    _tool("get_user_profile",
          "Summarise a user's taste: how much they rate, how generously, favourite and "
          "most-watched genres, their highest and lowest rated films, era preference.",
          {**_USER}, ["user_id"]),
    _tool("get_user_ratings",
          "Look up a user's own ratings. Pass `titles` to check whether they have seen "
          "specific films (null = not rated), or omit it for their highest-rated films.",
          {**_USER,
           "titles": {"type": "array", "items": {"type": "string"}},
           "limit": {"type": "integer", "default": 20},
           "min_rating": {"type": "number"}}, ["user_id"]),
    _tool("search_movies",
          "Content search over plot summaries, genres and tags. Use for mood/theme requests "
          "like 'dark psychological thriller with a twist'. Supports genre, year and quality "
          "filters; pass `for_user` to exclude what they have seen and get predicted ratings.",
          {"query": {"type": "string"},
           "n": {"type": "integer", "default": 8},
           "include_genres": _GENRES, "exclude_genres": _GENRES,
           "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
           "min_ratings": {"type": "integer", "default": 5,
                           "description": "Drop films with fewer ratings than this (quality filter)"},
           "min_avg_rating": {"type": "number"},
           "for_user": {"type": "integer"}}, ["query"]),
    _tool("recommend_for_user",
          "Rank unseen films for a user. method is one of hybrid (default), user_cf, item_cf, "
          "content, popularity. `similar_to` steers results toward given films; "
          "`exclude_genres` applies hard constraints like 'no animation'.",
          {**_USER,
           "n": {"type": "integer", "default": 5},
           "method": {"type": "string", "enum": ["hybrid", "user_cf", "item_cf", "content", "popularity"]},
           "include_genres": _GENRES, "exclude_genres": _GENRES,
           "similar_to": {"type": "array", "items": {"type": "string"}},
           "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
           "min_ratings": {"type": "integer", "default": 5}}, ["user_id"]),
    _tool("find_similar_users",
          "Find users with the most similar rating patterns, with their overlap and favourites.",
          {**_USER, "n": {"type": "integer", "default": 5}}, ["user_id"]),
    _tool("how_similar_users_rated",
          "What users with similar taste actually rated a specific film, versus the whole "
          "audience. Use for 'what do people like me think of X'.",
          {**_USER, "movie": {"type": "string"},
           "n_users": {"type": "integer", "default": 30}}, ["user_id", "movie"]),
    _tool("explain_recommendation",
          "Evidence for why a specific film suits a user: which of their own ratings drive it, "
          "which of their favourites it resembles textually, genre fit, and peer ratings.",
          {**_USER, "movie": {"type": "string"}}, ["user_id", "movie"]),
    _tool("find_blind_spots",
          "Genres the user has barely explored relative to what the catalogue offers, each "
          "with concrete entry-point films.",
          {**_USER}, ["user_id"]),
    _tool("get_movie_details",
          "Plot, genres, year, audience rating and rating-count confidence for one film.",
          {"movie": {"type": "string"}, "plot_chars": {"type": "integer", "default": 600}}, ["movie"]),
    _tool("find_similar_movies",
          "Films similar to a given one, by plot text ('plot'), by who rated them highly "
          "('audience'), or both.",
          {"movie": {"type": "string"}, "n": {"type": "integer", "default": 6},
           "basis": {"type": "string", "enum": ["plot", "audience", "both"]}}, ["movie"]),
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}


def openai_tool_schemas() -> list[dict]:
    """Same tools in OpenAI's function-calling envelope."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in TOOL_SCHEMAS]
