"""Leakage tests: features for a fight must not depend on that fight or any later fight."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_appearances, make_fighters

from src.features import PROFILE_FEATURES, build_training_table, compute_prefight_features
from src.predict import fighter_profiles_as_of

CUTOFF = pd.Timestamp("2015-06-01")
CHECK_COLS = PROFILE_FEATURES + ["wins", "losses", "draws_nc"]


def _features(app, fighters):
    f = compute_prefight_features(app, fighters)
    return f.set_index(["fight_id", "fighter_id"]).sort_index()[CHECK_COLS]


def _scramble_from(app: pd.DataFrame, cutoff: pd.Timestamp, rng) -> pd.DataFrame:
    """Randomise outcomes, methods, durations and stats of every appearance on/after cutoff."""
    app = app.copy()
    m = app["event_date"] >= cutoff
    n = int(m.sum())
    app.loc[m, "result"] = rng.choice(["W", "L", "D", "NC"], n)
    app.loc[m, "method"] = rng.choice(["KO/TKO", "SUB", "DEC"], n)
    app.loc[m, "duration_s"] = rng.uniform(10, 1500, n)
    for col in ["sig_landed", "sig_att", "opp_sig_landed", "opp_sig_att", "td_landed", "td_att",
                "opp_td_landed", "opp_td_att", "sub_att", "kd", "opp_kd", "ctrl_s"]:
        app.loc[m, col] = rng.integers(0, 200, n).astype(float)
    return app


def test_perturbing_current_and_future_fights_does_not_change_features(data, rng):
    app, fighters = data["appearances"], data["fighters"]
    base = _features(app, fighters)
    scrambled = _features(_scramble_from(app, CUTOFF, rng), fighters)

    dates = app.set_index(["fight_id", "fighter_id"])["event_date"]
    on_or_before = dates[dates <= dates[dates >= CUTOFF].min()].index  # includes the first event on/after cutoff
    pd.testing.assert_frame_equal(base.loc[on_or_before].sort_index(), scrambled.loc[on_or_before].sort_index())
    # Sanity: the scramble really did change later features (otherwise the test proves nothing).
    later = dates[dates > CUTOFF + pd.Timedelta(days=60)].index
    assert not base.loc[later].equals(scrambled.loc[later])


def test_truncated_history_gives_identical_features(data):
    """Features computed with all future fights deleted equal those computed on the full data."""
    app, fighters = data["appearances"], data["fighters"]
    full = _features(app, fighters)
    truncated = _features(app[app["event_date"] <= CUTOFF], fighters)
    pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)


def test_prior_fight_counts_only_count_strictly_earlier_dates(data):
    app = data["appearances"]
    feats = compute_prefight_features(app, data["fighters"])
    merged = app[["fight_id", "fighter_id", "event_date"]].merge(feats[["fight_id", "fighter_id", "n_fights", "wins"]])
    a = app.sort_values("event_date")
    per_day = a.groupby(["fighter_id", "event_date"]).agg(n=("fight_id", "size"),
                                                         w=("result", lambda s: (s == "W").sum()))
    per_day[["exp_n", "exp_w"]] = per_day.groupby(level=0)[["n", "w"]].cumsum().values - per_day[["n", "w"]].values
    merged = merged.join(per_day[["exp_n", "exp_w"]], on=["fighter_id", "event_date"])
    assert (merged["n_fights"] == merged["exp_n"]).all()
    assert (merged["wins"] == merged["exp_w"]).all()


def test_same_night_tournament_bouts_do_not_see_each_other(data):
    app = data["appearances"]
    multi = app.groupby(["fighter_id", "event_date"]).size()
    multi = multi[multi > 1]
    assert len(multi) > 0, "expected early tournament nights with several bouts per fighter"
    feats = compute_prefight_features(app, data["fighters"]).set_index(["fighter_id", "event_date", "fight_id"]).sort_index()
    for (fid, date) in multi.index[:20]:
        rows = feats.loc[(fid, date)]
        assert (rows[CHECK_COLS].nunique(dropna=False) == 1).all(), "same-night bouts got different histories"


def test_changing_a_fights_own_result_does_not_change_its_features():
    app = make_appearances([
        ("f1", "x", "2020-01-01", "W", "KO/TKO", 30), ("f1", "y", "2020-01-01", "L", "KO/TKO", 10),
        ("f2", "x", "2020-06-01", "L", "DEC", 50), ("f2", "z", "2020-06-01", "W", "DEC", 60),
    ])
    fighters = make_fighters(["x", "y", "z"])
    before = compute_prefight_features(app, fighters).set_index(["fight_id", "fighter_id"])
    flipped = app.copy()
    flipped.loc[flipped["fight_id"] == "f2", "result"] = ["W", "L"]
    flipped.loc[flipped["fight_id"] == "f2", "sig_landed"] = [999.0, 0.0]
    after = compute_prefight_features(flipped, fighters).set_index(["fight_id", "fighter_id"])
    pd.testing.assert_series_equal(before.loc[("f2", "x"), CHECK_COLS], after.loc[("f2", "x"), CHECK_COLS])
    assert before.loc[("f2", "x"), "n_fights"] == 1 and before.loc[("f2", "x"), "wins"] == 1


def test_training_table_rows_use_only_prior_fights(data):
    table = build_training_table(data["fights"], data["appearances"], data["fighters"])
    app = data["appearances"]
    for side in ("a", "b"):
        m = table[["fight_id", "event_date", f"fighter_{side}_id", f"{side}_prior_fights"]]
        counts = m.merge(app[["fighter_id", "event_date"]], left_on=f"fighter_{side}_id", right_on="fighter_id",
                         suffixes=("", "_other"))
        earlier = counts[counts["event_date_other"] < counts["event_date"]].groupby("fight_id").size()
        expected = m.set_index("fight_id")[f"{side}_prior_fights"]
        assert (earlier.reindex(expected.index, fill_value=0) == expected).all()


def test_inference_profile_rejects_dates_not_after_history(data):
    app, fighters = data["appearances"], data["fighters"]
    fid = app["fighter_id"].iloc[-1]
    last = app.loc[app["fighter_id"] == fid, "event_date"].max()
    with pytest.raises(ValueError):
        fighter_profiles_as_of(app, fighters, [fid], last)


def test_inference_profile_equals_training_profile_for_next_fight(data):
    """Profile 'as of' a fighter's real fight date (history cut there) equals the training features."""
    app, fighters = data["appearances"], data["fighters"]
    feats = compute_prefight_features(app, fighters).set_index(["fight_id", "fighter_id"])
    sample = app[app["event_date"] > "2018-01-01"].sample(25, random_state=1)
    for r in sample.itertuples():
        history = app[app["event_date"] < r.event_date]
        prof = fighter_profiles_as_of(history, fighters, [r.fighter_id], r.event_date).iloc[0]
        expected = feats.loc[(r.fight_id, r.fighter_id)]
        np.testing.assert_allclose(prof[PROFILE_FEATURES].astype(float).values,
                                   expected[PROFILE_FEATURES].astype(float).values, equal_nan=True)
