# Offline evaluation

*temporal leave-last-out per user; holdout = last 20% of each user's ratings, capped at 10; relevant = held-out rating >= 4.0; candidates = unseen movies with >= 5 training ratings.*

Train 69,427 ratings / test 4,637 ratings over 533 users.

## Ranking quality (@10)

| method | recall | precision | NDCG | MAP | hit-rate | catalogue coverage | long-tail share |
|---|---|---|---|---|---|---|---|
| `popularity` | 0.051 | 0.025 | 0.045 | 0.023 | 0.189 | 0.012 | 0.089 |
| `user_cf` | 0.080 | 0.038 | 0.067 | 0.034 | 0.296 | 0.056 | 0.128 |
| `item_cf` | 0.024 | 0.014 | 0.021 | 0.008 | 0.117 | 0.289 | 0.655 |
| `content` | 0.012 | 0.007 | 0.012 | 0.005 | 0.062 | 0.183 | 0.843 |
| `hybrid` | 0.086 | 0.042 | 0.073 | 0.036 | 0.328 | 0.102 | 0.233 |

## Ranking quality by history size (recall@10 / NDCG@10)

| method | dense (>150) (n=112) | medium (50-150) (n=160) | sparse (<50) (n=225) |
|---|---|---|---|
| `popularity` | 0.035 / 0.032 | 0.049 / 0.041 | 0.061 / 0.055 |
| `user_cf` | 0.041 / 0.036 | 0.084 / 0.072 | 0.097 / 0.079 |
| `item_cf` | 0.024 / 0.024 | 0.029 / 0.026 | 0.021 / 0.015 |
| `content` | 0.013 / 0.011 | 0.009 / 0.008 | 0.013 / 0.015 |
| `hybrid` | 0.036 / 0.034 | 0.090 / 0.081 | 0.109 / 0.087 |

## Rating prediction (secondary check)

| scorer | RMSE | MAE |
|---|---|---|
| `user_cf` | 0.912 | 0.691 |
| `item_cf` | 0.898 | 0.678 |
| `user_mean_baseline` | 0.973 | 0.751 |
| `global_mean_baseline` | 1.066 | 0.862 |
