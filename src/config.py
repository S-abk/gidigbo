"""Central configuration: paths, seeds and the chronological split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "scrape_ufc_stats"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RANDOM_SEED = 42

# Fights before this date are still used as *history* for later fights, but are
# not used as training examples (early no-rules / tournament era, very sparse stats).
MIN_TRAIN_DATE = "2001-01-01"

# Chronological split boundaries (inclusive lower bounds).
VALIDATION_START = "2022-01-01"
TEST_START = "2024-01-01"

# Walk-forward CV inside the training period: each fold tests one calendar year
# and trains on everything strictly before it.
WALK_FORWARD_YEARS = [2017, 2018, 2019, 2020, 2021]

# Fighters with fewer prior UFC fights than this trigger a "limited data" warning.
LIMITED_HISTORY_THRESHOLD = 3

# Model staleness policy (deep-research robustness recommendation: treat model age as
# an ongoing decision, not a one-time training run). These are cheap, concrete proxies
# for the "retain vs. recalibrate vs. refit" tradeoff, not a formal utility-optimal
# schedule -- just enough to surface a visible warning before weights silently go stale
# while fighter profiles (data/processed) keep advancing with every `python -m src.train`.
RECALIBRATE_AFTER_DAYS = 180   # ~6 months: re-fit just the probability calibrator
RETRAIN_AFTER_DAYS = 365       # ~12 months: rebuild features/models on fresh data

# Production refit: once evaluation has chosen a configuration on held-out data, the
# deployed model is that same configuration refit on EVERY fight through the latest
# event, so it learns from recent fights too. Evaluation metrics still come from the
# held-out run (models/evaluated_model.joblib is that exact artifact).
PRODUCTION_REFIT = True
