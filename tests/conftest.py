import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_dataset import build_all  # noqa: E402
from src.config import MODELS_DIR  # noqa: E402

STAT_COLUMNS = ["sig_landed", "sig_att", "opp_sig_landed", "opp_sig_att", "td_landed", "td_att",
                "opp_td_landed", "opp_td_att", "sub_att", "kd", "opp_kd", "ctrl_s"]


@pytest.fixture(scope="session")
def data():
    """Real canonical dataset built from the raw CSVs (not saved)."""
    return build_all(save=False)


@pytest.fixture(scope="session")
def predictor():
    if not (MODELS_DIR / "model.joblib").exists():
        pytest.skip("no trained model: run `python -m src.train` first")
    from src.predict import Predictor
    return Predictor()


def make_appearances(rows):
    """Tiny synthetic appearance table: rows = (fight_id, fighter_id, date, result, method, sig_landed)."""
    out = []
    for fight_id, fighter_id, date, result, method, sig in rows:
        r = {"fight_id": fight_id, "fighter_id": fighter_id, "event_date": pd.Timestamp(date),
             "result": result, "method": method, "duration_s": 900.0, "has_stats": True}
        r.update({c: 0.0 for c in STAT_COLUMNS})
        r.update({"sig_landed": float(sig), "sig_att": 2.0 * sig, "opp_sig_landed": 20.0, "opp_sig_att": 50.0})
        out.append(r)
    return pd.DataFrame(out)


def make_fighters(ids):
    return pd.DataFrame({"fighter_id": ids, "dob": pd.Timestamp("1990-01-01"), "height_in": 70.0,
                         "reach_in": 72.0, "stance": "Orthodox"})


@pytest.fixture
def rng():
    return np.random.default_rng(0)
