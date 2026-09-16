#!/usr/bin/env python3
"""Small, explicit sweep over the hybrid fusion weights.

Not hyperparameter worship - the point is to know whether the chosen weights sit
on a plateau or on a lucky spike, and to record the alternatives I actually
tried. Scores per user are computed once and re-fused, so the sweep is cheap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movieagent.cf import Recommender, _minmax     # noqa: E402
from movieagent.data import MovieData              # noqa: E402
from movieagent.index import PlotIndex             # noqa: E402
from movieagent import config                      # noqa: E402

from evaluate import K, LIKE_THRESHOLD, ndcg_at_k, split   # noqa: E402

GRID = [
    # (user_cf, item_cf, content, popularity prior)
    (1.00, 0.00, 0.00, 0.0), (0.00, 1.00, 0.00, 0.0), (0.00, 0.00, 1.00, 0.0),
    (0.70, 0.30, 0.00, 0.0), (0.50, 0.50, 0.00, 0.0), (0.60, 0.20, 0.20, 0.0),
    (0.45, 0.35, 0.20, 0.0), (0.45, 0.35, 0.20, 0.1), (0.45, 0.35, 0.20, 0.2),
    (0.55, 0.25, 0.20, 0.1), (0.40, 0.30, 0.30, 0.1), (0.80, 0.10, 0.10, 0.1),
]


def main() -> None:
    data = MovieData.load()
    index = PlotIndex.load_or_build(data)
    train, test = split(data.ratings)
    model = Recommender.build(data, index, ratings=train)

    cached = []
    for uid, tdf in test.groupby("userId"):
        uid = int(uid)
        relevant = set(tdf.loc[tdf["rating"] >= LIKE_THRESHOLD, "movieId"].astype(int))
        if not relevant or uid not in model.u_pos:
            continue
        mask = model.candidate_mask(uid)
        if not mask.any():
            continue
        ucf, _ = model.user_cf_scores(uid)
        icf, _ = model.item_cf_scores(uid)
        cached.append((relevant, mask,
                       _minmax(ucf, mask), _minmax(icf, mask),
                       _minmax(model.content_scores(uid), mask),
                       _minmax(model.popularity_scores(), mask)))
    print(f"cached scores for {len(cached)} users")

    results = []
    for wu, wi, wc, wp in GRID:
        recalls, ndcgs, covered = [], [], set()
        for relevant, mask, u, i, c, p in cached:
            s = wu * u + wi * i + wc * c
            s = (1 - wp) * s + wp * p if wp else s
            s = np.where(mask, s, -np.inf)
            top = np.argsort(-s)[:K]
            ranked = [int(model.item_ids[j]) for j in top if np.isfinite(s[j])]
            covered.update(ranked)
            recalls.append(len(relevant & set(ranked)) / len(relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant))
        results.append({"user_cf": wu, "item_cf": wi, "content": wc, "pop_prior": wp,
                        f"recall@{K}": round(float(np.mean(recalls)), 4),
                        f"ndcg@{K}": round(float(np.mean(ndcgs)), 4),
                        "coverage": round(len(covered) / len(data.movies), 4)})
        r = results[-1]
        print(f"  {wu:.2f}/{wi:.2f}/{wc:.2f} pop={wp:.1f}  recall={r[f'recall@{K}']:.4f} "
              f"ndcg={r[f'ndcg@{K}']:.4f} cov={r['coverage']:.3f}")

    results.sort(key=lambda r: -r[f"ndcg@{K}"])
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "hybrid_sweep.json").write_text(json.dumps(results, indent=2))
    print("\nbest:", results[0])


if __name__ == "__main__":
    main()
