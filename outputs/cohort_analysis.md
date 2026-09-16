# Why dense users score worse

*Same split as `evaluate.py`. "Head" = the 200 most-rated films in the training half; long tail = fewer than 50 training ratings. No model is involved - this only describes what each cohort left in its holdout.*

| cohort | users | relevant held-out items | median train ratings of those items | share reachable (>=5 train ratings) | share long-tail | share of the popular head already seen |
|---|---|---|---|---|---|---|
| `dense (>150)` | 112 | 5.6 | 35 | 87.8% | 67.5% | 48.4% |
| `medium (50-150)` | 160 | 5.4 | 42 | 93.4% | 56.6% | 19.3% |
| `sparse (<50)` | 225 | 4.5 | 66 | 96.8% | 38.5% | 8.1% |

Read it as: a dense user has already seen roughly half of the films the ratings matrix knows well, so the films they go on to enjoy are ones almost nobody in this 610-user dataset has rated. The ranker is not worse at understanding them - it is being asked a harder question, and collaborative filtering has the least to say exactly there.
