"""Paths and tunable constants, in one place so experiments are easy to reproduce."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "ml-latest-small-filtered"
CACHE_DIR = REPO_ROOT / ".cache"
OUTPUT_DIR = REPO_ROOT / "outputs"

# --- Retrieval -------------------------------------------------------------
SVD_COMPONENTS = 256          # latent dims for the LSA view of plot text
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 120_000
LEXICAL_WEIGHT = 0.5          # blend of raw TF-IDF vs LSA cosine in plot search
# Swept over 0.5/1.0/1.5/2.0 - the top-5 for the probe queries barely moves, so this
# stays at the neutral value rather than being tuned to look good on examples.
LABEL_BOOST = 1.0

# --- Collaborative filtering ----------------------------------------------
MIN_CO_RATED = 5              # ignore "neighbours" who overlap on fewer films
SHRINKAGE = 25.0              # significance weighting: sim * n_co / (n_co + SHRINKAGE)
NEIGHBOURS_K = 40             # users consulted per prediction
ITEM_NEIGHBOURS_K = 30

# --- Scoring ---------------------------------------------------------------
BAYES_PRIOR_COUNT = 10        # damping for the "is this actually good" prior
MIN_RATINGS_FOR_CONFIDENCE = 5

# Chosen from the sweep in eval/tune_hybrid.py (outputs/hybrid_sweep.json). Not the
# top NDCG@10 row - 0.80/0.10/0.10 + 0.1 scores .0741 - but it gives up .0011 NDCG to
# nearly double catalogue coverage (.055 -> .102), which is the right trade for a
# discovery assistant. The surface is a broad plateau (NDCG moves between .067 and
# .074 across every sensible mix), so read these as a deliberate point on a flat
# surface rather than a tuned optimum.
HYBRID_WEIGHTS = {"user_cf": 0.60, "item_cf": 0.20, "content": 0.20}
POPULARITY_PRIOR = 0.0        # tie-breaker weight blended in; the sweep preferred none
