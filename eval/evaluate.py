#!/usr/bin/env python3
"""Offline evaluation of the recommender layer.

Protocol - **temporal leave-last-out per user**:

    for every user with >= 20 ratings, sort by timestamp and hold out the most
    recent 20% (capped at 10 films). Fit on the rest. Rank the whole unseen
    catalogue and check where the held-out films the user *liked* (>= 4.0) land.

A temporal split rather than a random one, because the product question is
"what should I watch next", not "can you reconstruct a rating I already gave".
Random splits let the model peek at a user's later taste and flatter it.

The candidate filter and popularity prior are rebuilt from the training split
only, so nothing about the held-out ratings leaks into ranking.

Run:  python eval/evaluate.py            (writes outputs/evaluation.json + .md)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movieagent import config                       # noqa: E402
from movieagent.cf import Recommender               # noqa: E402
from movieagent.data import MovieData               # noqa: E402
from movieagent.index import PlotIndex              # noqa: E402

K = 10
LIKE_THRESHOLD = 4.0
MIN_HISTORY = 20
HOLDOUT_FRACTION = 0.2
HOLDOUT_CAP = 10
METHODS = ["popularity", "user_cf", "item_cf", "content", "hybrid"]


# ----------------------------------------------------------------- metrics
def ndcg_at_k(ranked_ids, relevant, k=K) -> float:
    gains = [1.0 if m in relevant else 0.0 for m in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def average_precision(ranked_ids, relevant, k=K) -> float:
    hits, score = 0, 0.0
    for i, m in enumerate(ranked_ids[:k]):
        if m in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0


def split(ratings: pd.DataFrame):
    train_parts, test_parts = [], []
    for _, df in ratings.groupby("userId"):
        if len(df) < MIN_HISTORY:
            train_parts.append(df)
            continue
        df = df.sort_values("timestamp")
        n_test = min(HOLDOUT_CAP, max(1, int(round(len(df) * HOLDOUT_FRACTION))))
        train_parts.append(df.iloc[:-n_test])
        test_parts.append(df.iloc[-n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


def bucket(n: int) -> str:
    return "sparse (<50)" if n < 50 else ("medium (50-150)" if n <= 150 else "dense (>150)")


def main() -> None:
    data = MovieData.load()
    index = PlotIndex.load_or_build(data)
    train, test = split(data.ratings)

    print(f"users={data.ratings['userId'].nunique()}  train={len(train):,}  test={len(test):,}")
    model = Recommender.build(data, index, ratings=train)

    test_by_user = {int(u): df for u, df in test.groupby("userId")}
    per_method: dict[str, dict] = {}
    seen_lists: dict[str, list[list[int]]] = defaultdict(list)

    for method in METHODS:
        rows, by_bucket = [], defaultdict(list)
        for uid, tdf in test_by_user.items():
            relevant = set(tdf.loc[tdf["rating"] >= LIKE_THRESHOLD, "movieId"].astype(int))
            if not relevant:
                continue
            recs = model.recommend(uid, n=K, method=method)
            ranked = [r["movieId"] for r in recs]
            seen_lists[method].append(ranked)
            m = {
                "recall": len(relevant & set(ranked)) / len(relevant),
                "precision": len(relevant & set(ranked)) / max(len(ranked), 1),
                "ndcg": ndcg_at_k(ranked, relevant),
                "map": average_precision(ranked, relevant),
                "hit": 1.0 if relevant & set(ranked) else 0.0,
            }
            rows.append(m)
            by_bucket[bucket(int(model.user_count[model.u_pos[uid]]))].append(m)

        flat = [r for lst in seen_lists[method] for r in lst]
        counts = model.item_count
        long_tail = np.mean([counts[model.i_pos[m]] < 50 for m in flat]) if flat else 0.0
        per_method[method] = {
            "users_evaluated": len(rows),
            **{f"{k}@{K}": round(float(np.mean([r[k] for r in rows])), 4)
               for k in ("recall", "precision", "ndcg", "map", "hit")},
            "catalogue_coverage": round(len(set(flat)) / len(data.movies), 4),
            "long_tail_share(<50 train ratings)": round(float(long_tail), 4),
            "by_history_size": {
                b: {"users": len(v), **{f"{k}@{K}": round(float(np.mean([r[k] for r in v])), 4)
                                        for k in ("recall", "ndcg", "hit")}}
                for b, v in sorted(by_bucket.items())
            },
        }
        print(f"  {method:<11} recall@{K}={per_method[method][f'recall@{K}']:.4f} "
              f"ndcg@{K}={per_method[method][f'ndcg@{K}']:.4f} "
              f"cov={per_method[method]['catalogue_coverage']:.3f}")

    # ---------------------------------------------- rating-prediction sanity
    pred_rows = []
    for uid, tdf in test_by_user.items():
        if uid not in model.u_pos:
            continue
        ucf, _ = model.user_cf_scores(uid)
        icf, _ = model.item_cf_scores(uid)
        umean = float(model.user_mean[model.u_pos[uid]])
        for _, r in tdf.iterrows():
            j = model.i_pos.get(int(r["movieId"]))
            if j is None:
                continue
            pred_rows.append({
                "actual": float(r["rating"]),
                "user_cf": float(ucf[j]), "item_cf": float(icf[j]),
                "user_mean_baseline": umean,
                "global_mean_baseline": float(train["rating"].mean()),
            })
    pr = pd.DataFrame(pred_rows)
    rating_errors = {
        col: {"rmse": round(float(np.sqrt(((pr[col] - pr["actual"]) ** 2).mean())), 4),
              "mae": round(float((pr[col] - pr["actual"]).abs().mean()), 4)}
        for col in ("user_cf", "item_cf", "user_mean_baseline", "global_mean_baseline")
    }
    print("  rating prediction:", {k: v["rmse"] for k, v in rating_errors.items()})

    report = {
        "protocol": {
            "split": "temporal leave-last-out per user",
            "holdout": f"last {int(HOLDOUT_FRACTION*100)}% of each user's ratings, capped at {HOLDOUT_CAP}",
            "min_history": MIN_HISTORY,
            "relevance": f"held-out rating >= {LIKE_THRESHOLD}",
            "k": K,
            "candidate_pool": f"unseen movies with >= {config.MIN_RATINGS_FOR_CONFIDENCE} training ratings",
            "train_ratings": int(len(train)), "test_ratings": int(len(test)),
            "users_with_holdout": len(test_by_user),
        },
        "ranking": per_method,
        "rating_prediction": rating_errors,
    }
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    (config.OUTPUT_DIR / "evaluation.md").write_text(render_markdown(report))
    print(f"\nwrote {config.OUTPUT_DIR/'evaluation.json'} and evaluation.md")


def render_markdown(rep: dict) -> str:
    p, L = rep["protocol"], []
    L.append("# Offline evaluation\n")
    L.append(f"*{p['split']}; holdout = {p['holdout']}; relevant = {p['relevance']}; "
             f"candidates = {p['candidate_pool']}.*\n")
    L.append(f"Train {p['train_ratings']:,} ratings / test {p['test_ratings']:,} ratings "
             f"over {p['users_with_holdout']} users.\n")
    L.append(f"## Ranking quality (@{K})\n")
    L.append("| method | recall | precision | NDCG | MAP | hit-rate | catalogue coverage | long-tail share |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m, v in rep["ranking"].items():
        L.append(f"| `{m}` | {v[f'recall@{K}']:.3f} | {v[f'precision@{K}']:.3f} | {v[f'ndcg@{K}']:.3f} | "
                 f"{v[f'map@{K}']:.3f} | {v[f'hit@{K}']:.3f} | {v['catalogue_coverage']:.3f} | "
                 f"{v['long_tail_share(<50 train ratings)']:.3f} |")
    L.append(f"\n## Ranking quality by history size (recall@{K} / NDCG@{K})\n")
    buckets = list(next(iter(rep["ranking"].values()))["by_history_size"])
    L.append("| method | " + " | ".join(f"{b} (n={rep['ranking']['hybrid']['by_history_size'][b]['users']})"
                                        for b in buckets) + " |")
    L.append("|---" * (len(buckets) + 1) + "|")
    for m, v in rep["ranking"].items():
        cells = [f"{v['by_history_size'][b][f'recall@{K}']:.3f} / {v['by_history_size'][b][f'ndcg@{K}']:.3f}"
                 for b in buckets]
        L.append(f"| `{m}` | " + " | ".join(cells) + " |")
    L.append("\n## Rating prediction (secondary check)\n")
    L.append("| scorer | RMSE | MAE |")
    L.append("|---|---|---|")
    for m, v in rep["rating_prediction"].items():
        L.append(f"| `{m}` | {v['rmse']:.3f} | {v['mae']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
