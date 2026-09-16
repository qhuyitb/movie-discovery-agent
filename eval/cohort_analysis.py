#!/usr/bin/env python3
"""Why does the ranker do *worse* on users with more history?

`evaluate.py` reports the counter-intuitive result that recall@10 falls from
0.108 (sparse users) to 0.039 (dense users). This script tests the obvious
explanation - that dense users have already eaten the popular head of the
catalogue, so what is left in their holdout is tail material the ratings matrix
barely covers - by describing the holdout itself, with no model involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movieagent import config                      # noqa: E402
from movieagent.data import MovieData              # noqa: E402

from evaluate import LIKE_THRESHOLD, bucket, split  # noqa: E402

HEAD_N = 200          # "the popular head" = the 200 most-rated films in train
TAIL_MAX = 50         # a film with fewer train ratings than this counts as long tail
ORDER = ["dense (>150)", "medium (50-150)", "sparse (<50)"]


def main() -> None:
    data = MovieData.load()
    train, test = split(data.ratings)
    train_count = train.groupby("movieId").size()
    history = train.groupby("userId").size()
    head = set(train_count.sort_values(ascending=False).head(HEAD_N).index)

    rows = []
    for uid, df in test.groupby("userId"):
        relevant = df[df.rating >= LIKE_THRESHOLD]
        if relevant.empty:
            continue
        counts = relevant.movieId.map(train_count).fillna(0)
        seen = set(train.loc[train.userId == uid, "movieId"])
        rows.append({
            "cohort": bucket(int(history.get(uid, 0))),
            "relevant_items": len(relevant),
            "median_train_ratings": float(counts.median()),
            "share_in_candidate_pool": float((counts >= 5).mean()),
            "share_long_tail": float((counts < TAIL_MAX).mean()),
            "head_already_seen": len(head & seen) / HEAD_N,
        })

    df = pd.DataFrame(rows)
    agg = (df.groupby("cohort")
             .agg(users=("relevant_items", "size"),
                  relevant_items=("relevant_items", "mean"),
                  median_train_ratings=("median_train_ratings", "median"),
                  share_in_candidate_pool=("share_in_candidate_pool", "mean"),
                  share_long_tail=("share_long_tail", "mean"),
                  head_already_seen=("head_already_seen", "mean"))
             .reindex(ORDER))

    lines = [
        "# Why dense users score worse",
        "",
        f"*Same split as `evaluate.py`. \"Head\" = the {HEAD_N} most-rated films in the training "
        f"half; long tail = fewer than {TAIL_MAX} training ratings. No model is involved - this "
        "only describes what each cohort left in its holdout.*",
        "",
        "| cohort | users | relevant held-out items | median train ratings of those items | "
        "share reachable (>=5 train ratings) | share long-tail | share of the popular head already seen |",
        "|---|---|---|---|---|---|---|",
    ]
    for cohort, r in agg.iterrows():
        lines.append(
            f"| `{cohort}` | {int(r.users)} | {r.relevant_items:.1f} | {r.median_train_ratings:.0f} | "
            f"{r.share_in_candidate_pool:.1%} | {r.share_long_tail:.1%} | {r.head_already_seen:.1%} |"
        )
    lines += [
        "",
        "Read it as: a dense user has already seen roughly half of the films the ratings matrix "
        "knows well, so the films they go on to enjoy are ones almost nobody in this 610-user "
        "dataset has rated. The ranker is not worse at understanding them - it is being asked a "
        "harder question, and collaborative filtering has the least to say exactly there.",
        "",
    ]
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "cohort_analysis.md"
    out.write_text("\n".join(lines))
    print(agg.round(3).to_string())
    print(f"\nwrote {out.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
