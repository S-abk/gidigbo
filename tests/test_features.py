import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_appearances, make_fighters

from src import features as F
from src.config import MODELS_DIR
from src.data_loader import (fight_duration_seconds, parse_height_inches, parse_inches, parse_landed_of,
                             parse_mmss_seconds, parse_round_lengths)


# --------------------------------------------------------------------------- parsers
def test_parsers():
    assert parse_height_inches(pd.Series(["5' 11\""])).iloc[0] == 71
    assert np.isnan(parse_height_inches(pd.Series([None])).iloc[0])
    assert parse_inches(pd.Series(['72"'])).iloc[0] == 72
    assert parse_mmss_seconds(pd.Series(["3:44"])).iloc[0] == 224
    landed, att = parse_landed_of(pd.Series(["29 of 73"]))
    assert (landed.iloc[0], att.iloc[0]) == (29, 73)
    assert parse_round_lengths("3 Rnd (5-5-5)") == [5, 5, 5]
    assert parse_round_lengths("1 Rnd + OT (12-3)") == [12, 3]
    assert parse_round_lengths("No Time Limit") is None
    assert fight_duration_seconds("3 Rnd (5-5-5)", 3, 300) == 900
    assert fight_duration_seconds("1 Rnd + OT (12-3)", 2, 60) == 780


# --------------------------------------------------------------------------- features
def test_debutant_gets_prior_values_not_nan():
    app = make_appearances([("f1", "x", "2020-01-01", "W", "DEC", 40), ("f1", "y", "2020-01-01", "L", "DEC", 20)])
    feats = F.compute_prefight_features(app, make_fighters(["x", "y"])).set_index("fighter_id")
    x = feats.loc["x"]
    assert x["n_fights"] == 0 and x["win_pct"] == 0.5
    assert x["slpm"] == pytest.approx(F.PRIOR_SLPM)
    assert np.isnan(x["days_since_last"])
    assert feats[F.PROFILE_FEATURES].drop(columns="days_since_last").notna().all().all()


def test_streak_and_windows():
    rows = []
    for i, res in enumerate(["W", "W", "L", "W", "W", "W"]):
        rows += [(f"f{i}", "x", f"2020-0{i + 1}-01", res, "DEC", 30),
                 (f"f{i}", f"o{i}", f"2020-0{i + 1}-01", "L" if res == "W" else "W", "DEC", 30)]
    rows += [("f9", "x", "2020-09-01", "W", "DEC", 30), ("f9", "o9", "2020-09-01", "L", "DEC", 30)]
    app = make_appearances(rows)
    feats = F.compute_prefight_features(app, make_fighters(["x"] + [f"o{i}" for i in (0, 1, 2, 3, 4, 5, 9)]))
    last = feats[(feats.fighter_id == "x") & (feats.fight_id == "f9")].iloc[0]
    assert last["n_fights"] == 6 and last["wins"] == 5
    assert last["streak"] == 3
    assert last["wins_last3"] == 3 and last["wins_last5"] == 4
    assert last["days_since_last"] == (pd.Timestamp("2020-09-01") - pd.Timestamp("2020-06-01")).days


def test_matchup_is_antisymmetric(data):
    prof = F.compute_prefight_features(data["appearances"], data["fighters"]).sample(200, random_state=0)
    a, b = prof.iloc[:100], prof.iloc[100:]
    ab, ba = F.matchup_features(a, b), F.matchup_features(b, a)
    np.testing.assert_allclose(ab[F.DIFF_FEATURES].values, -ba[F.DIFF_FEATURES].values)
    np.testing.assert_allclose(ab[F.CONTEXT_FEATURES].values, ba[F.CONTEXT_FEATURES].values)


def test_training_table_schema_and_balance(data):
    t = F.build_training_table(data["fights"], data["appearances"], data["fighters"])
    assert list(t[F.MODEL_FEATURES].columns) == F.MODEL_FEATURES
    assert not t[F.MODEL_FEATURES].isna().any().any()
    assert t["fight_id"].is_unique
    assert 0.45 < t["label"].mean() < 0.55  # raw data lists the winner first ~64% of the time
    # deterministic orientation
    t2 = F.build_training_table(data["fights"], data["appearances"], data["fighters"])
    assert (t["fighter_a_id"] == t2["fighter_a_id"]).all()


def test_mirror_negates_diffs_and_flips_labels():
    X = pd.DataFrame(np.ones((3, len(F.MODEL_FEATURES))), columns=F.MODEL_FEATURES)
    Xm, ym = F.mirror(X, np.array([1, 0, 1]))
    assert len(Xm) == 6 and list(ym) == [1, 0, 1, 0, 1, 0]
    assert (Xm.loc[3:, F.DIFF_FEATURES] == -1).all().all()
    assert (Xm.loc[3:, F.CONTEXT_FEATURES] == 1).all().all()


def test_saved_model_feature_schema_matches_code():
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        pytest.skip("no trained model")
    meta = json.loads(meta_path.read_text())
    assert meta["feature_names"] == F.MODEL_FEATURES
