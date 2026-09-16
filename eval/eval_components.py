#!/usr/bin/env python3
"""Component-level evaluation: language understanding and content retrieval.

The end-to-end ranking metrics in `evaluate.py` say nothing about the two parts a
user actually feels first - whether the assistant understood the question, and
whether a mood query returns sensible films. Both are measured here against
small hand-written sets.

The intent set deliberately includes paraphrases the rule system was *not*
written against, so the number reflects generalisation rather than the author's
own vocabulary.

Run:  python eval/eval_components.py   -> outputs/component_eval.{json,md}
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movieagent import config                     # noqa: E402
from movieagent.data import MovieData             # noqa: E402
from movieagent.index import PlotIndex            # noqa: E402
from movieagent.planner import understand         # noqa: E402

# (utterance, gold intent, written-against?)  -- "seen" = a phrasing the rules target
INTENT_CASES = [
    # --- phrasings the rules were written against -------------------------
    ("I'm user 30", "identify", True),
    ("user 15", "identify", True),
    ("What should I watch tonight?", "recommend", True),
    ("Recommend me something", "recommend", True),
    ("What do people with similar taste to mine think about Pulp Fiction?", "peer_opinion", True),
    ("Why do you think I'd like that?", "explain", True),
    ("What's my blind spot? What genres am I missing?", "blind_spots", True),
    ("I want a dark psychological thriller with a twist", "content_search", True),
    ("I liked Toy Story but I'm tired of animated movies - what else?", "recommend", True),
    ("Tell me about Inception", "movie_info", True),
    ("movies like Fight Club", "similar_movies", True),
    ("What's my taste?", "profile", True),
    ("Who has taste like mine?", "similar_users", True),
    ("banana helicopter", "unknown", True),
    # --- paraphrases the rules were NOT written against -------------------
    ("my id is 42", "identify", False),
    ("put something on for me", "recommend", False),
    ("give me a film for tonight", "recommend", False),
    ("anything worth watching?", "recommend", False),
    ("surprise me", "recommend", False),
    ("how do folks with my kind of taste rate Fargo?", "peer_opinion", False),
    ("do people like me rate Goodfellas well", "peer_opinion", False),
    ("justify that pick", "explain", False),
    ("on what basis?", "explain", False),
    ("which corners of cinema have I ignored", "blind_spots", False),
    ("am I stuck in a rut genre wise", "blind_spots", False),
    ("find me a slow burn movie about grief", "content_search", False),
    ("something gritty set in new york in the 70s", "content_search", False),
    ("a heist film where the crew double crosses each other", "content_search", False),
    ("what is Fargo about", "movie_info", False),
    ("give me the plot of Alien", "movie_info", False),
    ("what else is like Memento", "similar_movies", False),
    ("who rates films the way I do", "similar_users", False),
    ("describe my rating history", "profile", False),
    ("asdf qwerty", "unknown", False),
    ("What do people like me think about The Matrix?", "unresolved_movie", False),
]

# (query, genres that ought to dominate the results)
SEARCH_CASES = [
    ("dark psychological thriller with a twist", {"Thriller", "Mystery"}),
    ("heist crew pulls off an impossible robbery", {"Crime", "Action", "Thriller"}),
    ("space opera with aliens and starships", {"Sci-Fi", "Adventure"}),
    ("feel good comedy about friendship", {"Comedy"}),
    ("haunted house horror where something is in the basement", {"Horror", "Thriller"}),
    ("courtroom drama about a wrongful conviction", {"Drama", "Crime"}),
    ("animated film for children about talking animals", {"Animation", "Children"}),
    ("war film about soldiers on the front line", {"War", "Drama", "Action"}),
    ("romantic story where two people meet in a foreign city", {"Romance", "Drama"}),
    ("documentary about music", {"Documentary"}),
]


def eval_intents(data: MovieData) -> dict:
    state = SimpleNamespace(user_id=1, last_movie="Fight Club", last_recommendations=[{"title": "Fight Club"}])
    rows, confusion = [], Counter()
    for text, gold, seen in INTENT_CASES:
        got = understand(text, data, state).intent
        rows.append({"utterance": text, "gold": gold, "predicted": got, "correct": got == gold, "seen_phrasing": seen})
        if got != gold:
            confusion[f"{gold} -> {got}"] += 1
    seen_rows = [r for r in rows if r["seen_phrasing"]]
    novel_rows = [r for r in rows if not r["seen_phrasing"]]
    return {
        "n": len(rows),
        "accuracy_overall": round(np.mean([r["correct"] for r in rows]), 3),
        "accuracy_on_written_against_phrasings": round(np.mean([r["correct"] for r in seen_rows]), 3),
        "accuracy_on_novel_paraphrases": round(np.mean([r["correct"] for r in novel_rows]), 3),
        "confusions": dict(confusion),
        "failures": [r for r in rows if not r["correct"]],
    }


def eval_search(data: MovieData, index: PlotIndex, k: int = 10) -> dict:
    counts = data.movies["n_ratings"].to_numpy()
    means = data.movies["mean_rating"].to_numpy(dtype=float)
    rows = []
    for query, want in SEARCH_CASES:
        scores = index.score_query(query)
        order = np.argsort(-scores)[:k]
        ids = [int(index.movie_ids[j]) for j in order]
        hits = [bool(set(data.movies.loc[m, "genre_list"]) & want) for m in ids]
        n_r = np.array([counts[j] for j in order], dtype=float)
        avg = np.array([means[j] for j in order], dtype=float)
        rows.append({
            "query": query,
            "expected_genres": sorted(want),
            f"genre_precision@{k}": round(float(np.mean(hits)), 3),
            "median_n_ratings_of_hits": float(np.median(n_r)),
            "mean_audience_rating_of_hits": round(float(np.nanmean(avg)), 3),
            "share_with_fewer_than_10_ratings": round(float(np.mean(n_r < 10)), 3),
            "top_3": [data.title_of(m) for m in ids[:3]],
        })
    return {
        f"mean_genre_precision@{k}": round(float(np.mean([r[f"genre_precision@{k}"] for r in rows])), 3),
        "mean_audience_rating_of_returned_films": round(
            float(np.mean([r["mean_audience_rating_of_hits"] for r in rows])), 3),
        "dataset_mean_rating": round(float(data.ratings["rating"].mean()), 3),
        "mean_share_with_fewer_than_10_ratings": round(
            float(np.mean([r["share_with_fewer_than_10_ratings"] for r in rows])), 3),
        "per_query": rows,
    }


def render(rep: dict) -> str:
    i, s = rep["intent_classification"], rep["content_search"]
    L = ["# Component evaluation\n", "## 1. Intent classification (offline planner)\n",
         f"{i['n']} hand-written utterances. **Overall accuracy {i['accuracy_overall']:.2f}** - "
         f"{i['accuracy_on_written_against_phrasings']:.2f} on phrasings the rules target, "
         f"**{i['accuracy_on_novel_paraphrases']:.2f} on paraphrases they do not**. "
         "The gap is the honest cost of a rule-based planner.\n"]
    if i["failures"]:
        L.append("| utterance | expected | got |")
        L.append("|---|---|---|")
        for f in i["failures"]:
            L.append(f"| {f['utterance']} | `{f['gold']}` | `{f['predicted']}` |")
    L.append("\n## 2. Content search\n")
    L.append(f"10 mood/theme queries with hand-labelled target genres. "
             f"**Mean genre precision@10 = {s['mean_genre_precision@10']:.2f}.**\n")
    L.append(f"Films returned average **{s['mean_audience_rating_of_returned_films']:.2f}** "
             f"against a dataset mean of {s['dataset_mean_rating']:.2f}, and "
             f"{s['mean_share_with_fewer_than_10_ratings']*100:.0f}% of them have fewer than 10 ratings - "
             "content search is quality-blind by construction.\n")
    L.append("| query | genre P@10 | avg audience rating | top 3 |")
    L.append("|---|---|---|---|")
    for r in s["per_query"]:
        L.append(f"| {r['query']} | {r['genre_precision@10']:.2f} | "
                 f"{r['mean_audience_rating_of_hits']:.2f} | {', '.join(r['top_3'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    data = MovieData.load()
    index = PlotIndex.load_or_build(data)
    report = {"intent_classification": eval_intents(data), "content_search": eval_search(data, index)}
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "component_eval.json").write_text(json.dumps(report, indent=2))
    (config.OUTPUT_DIR / "component_eval.md").write_text(render(report))
    i = report["intent_classification"]
    print(f"intent accuracy: overall {i['accuracy_overall']:.2f} | "
          f"written-against {i['accuracy_on_written_against_phrasings']:.2f} | "
          f"novel {i['accuracy_on_novel_paraphrases']:.2f}")
    for f in i["failures"]:
        print(f"   MISS  {f['utterance']!r}: expected {f['gold']}, got {f['predicted']}")
    print(f"search genre precision@10: {report['content_search']['mean_genre_precision@10']:.2f}")


if __name__ == "__main__":
    main()
