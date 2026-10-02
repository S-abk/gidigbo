import numpy as np
import pytest

from src.features import MODEL_FEATURES, build_training_table


def _pairs(predictor, n=30, seed=0):
    rng = np.random.default_rng(seed)
    active = predictor.roster[predictor.roster["n_fights"] >= 1]["fighter_id"].to_numpy()
    return [tuple(rng.choice(active, 2, replace=False)) for _ in range(n)]


def test_probabilities_in_range_and_sum_to_one(predictor):
    for a, b in _pairs(predictor):
        r = predictor.predict_fight(a, b, explain=False)
        assert 0 < r["prob_a"] < 1 and 0 < r["prob_b"] < 1
        assert r["prob_a"] + r["prob_b"] == pytest.approx(1.0)


def test_reversal_consistency(predictor):
    for a, b in _pairs(predictor, seed=1):
        ab = predictor.predict_fight(a, b, explain=False)
        ba = predictor.predict_fight(b, a, explain=False)
        assert ab["prob_a"] == pytest.approx(1 - ba["prob_a"], abs=1e-9)  # symmetric rule: exact
        # The underlying (pre-symmetrisation) model should already be close to consistent.
        assert abs(ab["oriented_prob_a"] - (1 - ba["oriented_prob_a"])) < 0.05


def test_same_fighter_rejected(predictor):
    fid = predictor.roster["fighter_id"].iloc[0]
    with pytest.raises(ValueError):
        predictor.predict_fight(fid, fid)


def test_unknown_fighter_rejected(predictor):
    with pytest.raises(KeyError):
        predictor.predict_fight("Not A Real Fighter 123", predictor.roster["fighter_id"].iloc[0])


def test_inference_features_match_training_schema(predictor, data):
    a, b = _pairs(predictor, n=1)[0]
    X = predictor.matchup(a, b)
    assert list(X.columns) == MODEL_FEATURES
    t = build_training_table(data["fights"], data["appearances"], data["fighters"])
    assert list(t[MODEL_FEATURES].columns) == list(X.columns)
    assert not X.isna().any().any()


def test_limited_history_and_missing_physicals_do_not_crash(predictor):
    r = predictor.roster
    one_fight = r[r["n_fights"] == 1]["fighter_id"].iloc[0]
    synthetic = r[r["fighter_id"].str.startswith("unk_")]["fighter_id"].iloc[0]  # no tott data at all
    veteran = r.sort_values("n_fights")["fighter_id"].iloc[-1]
    for a, b in [(one_fight, veteran), (synthetic, veteran), (one_fight, synthetic)]:
        res = predictor.predict_fight(a, b)
        assert 0 < res["prob_a"] < 1
        assert res["warnings"], "expected a limited-history / missing-data warning"


def test_explanations_present_and_signed(predictor):
    a, b = _pairs(predictor, n=1, seed=3)[0]
    ab = predictor.predict_fight(a, b)
    ba = predictor.predict_fight(b, a)
    imp_ab = {f["group"]: f["impact"] for f in ab["factors"]}
    imp_ba = {f["group"]: f["impact"] for f in ba["factors"]}
    for g in imp_ab:
        assert imp_ab[g] == pytest.approx(-imp_ba[g], abs=1e-9)


def test_resolve_by_name(predictor):
    name = predictor.roster[~predictor.roster["name"].duplicated(keep=False)]["name"].iloc[0]
    assert predictor.resolve(name) == predictor.roster.set_index("name").loc[name, "fighter_id"]


def test_method_breakdown_is_a_consistent_distribution(predictor):
    if predictor.method_model is None:
        pytest.skip("no method model")
    for a, b in _pairs(predictor, n=10, seed=4):
        ab = predictor.predict_fight(a, b, explain=False)
        ba = predictor.predict_fight(b, a, explain=False)
        m, mr = ab["method"], ba["method"]
        assert sum(m["A"].values()) + sum(m["B"].values()) == pytest.approx(1.0)
        assert sum(m["A"].values()) == pytest.approx(ab["prob_a"])  # method splits the win probability
        assert sum(m["given_A_wins"].values()) == pytest.approx(1.0)
        for k in m["A"]:  # reversal mirrors the six-way outcome exactly
            assert m["A"][k] == pytest.approx(mr["B"][k]) and m["B"][k] == pytest.approx(mr["A"][k])


def test_method_context_changes_method_not_winner(predictor):
    if predictor.method_model is None:
        pytest.skip("no method model")
    a, b = _pairs(predictor, n=1, seed=5)[0]
    hw = predictor.predict_fight(a, b, explain=False, weight_class="Heavyweight", scheduled_rounds=5)
    ws = predictor.predict_fight(a, b, explain=False, weight_class="Women's Strawweight", scheduled_rounds=3)
    assert hw["prob_a"] == pytest.approx(ws["prob_a"])
    assert hw["method"]["overall"]["KO/TKO"] > ws["method"]["overall"]["KO/TKO"]
    assert hw["method"]["overall"]["DEC"] < ws["method"]["overall"]["DEC"]


def test_build_roster_names_are_always_unique():
    """Two different fighters sharing a name, division AND last-fight year must still
    get distinct display_names (the real cause of a past app crash: 'Bruno Silva')."""
    import pandas as pd
    from src.predict import build_roster

    appearances = pd.DataFrame([
        {"fighter_id": "x1", "event_date": pd.Timestamp("2026-03-01"), "weight_class": "Lightweight"},
        {"fighter_id": "x2", "event_date": pd.Timestamp("2026-07-01"), "weight_class": "Lightweight"},
    ])
    fighters = pd.DataFrame({"fighter_id": ["x1", "x2"], "name": ["Bruno Silva", "Bruno Silva"]})
    profiles = pd.DataFrame({"n_fights": [5, 5]}, index=["x1", "x2"])

    roster = build_roster(appearances, fighters, profiles)
    assert roster["display_name"].is_unique
    assert roster.set_index("fighter_id").loc["x1", "display_name"] != \
        roster.set_index("fighter_id").loc["x2", "display_name"]


def test_duplicate_named_fighters_dont_collide_in_comparison_tables(predictor):
    """Regression test for the real 'Bruno Silva' bug: selecting two different fighters
    who share a registered name must not produce duplicate pandas column labels."""
    dup_names = predictor.roster[predictor.roster["name"].duplicated(keep=False)]
    if dup_names.empty:
        pytest.skip("no same-named fighters in the current data to regression-test against")
    a_id, b_id = dup_names["fighter_id"].iloc[0], dup_names["fighter_id"].iloc[1]
    pa, pb = predictor.profile(a_id), predictor.profile(b_id)
    assert pa["name"] == pb["name"]  # the trap: same raw name
    assert pa["display_name"] != pb["display_name"]  # but distinct, safe table keys


def test_explanations_are_not_degenerate(predictor):
    """Occlusion explanations need a smooth probability mapping. A step-function
    calibrator (isotonic) once left ~half of all factor groups at exactly 0 impact.
    Any group whose matchup differences are non-zero should move the probability."""
    rng = np.random.default_rng(7)
    ids = predictor.roster[predictor.roster["n_fights"] >= 5]["fighter_id"].to_numpy()
    moved = total = 0
    for _ in range(15):
        a, b = rng.choice(ids, 2, replace=False)
        for f in predictor.predict_fight(a, b)["factors"]:
            if any(abs(v) > 1e-9 for v in f["features"].values()):
                total += 1
                moved += abs(f["impact"]) > 1e-9
    assert total > 0
    assert moved / total >= 0.9, f"only {moved}/{total} non-zero factor groups moved the probability"


def test_round_odds_form_a_distribution(predictor):
    if predictor.method_model is None or predictor.round_split is None:
        pytest.skip("no method model / round split")
    a, b = _pairs(predictor, n=1, seed=6)[0]
    for rounds in (3, 5):
        m = predictor.predict_fight(a, b, explain=False, scheduled_rounds=rounds)["method"]
        r = m["rounds"]
        assert len(r["by_round"]) == rounds
        assert r["distance"] == pytest.approx(m["overall"]["DEC"])  # distance = P(decision)
        assert sum(r["by_round"]) + r["distance"] == pytest.approx(1.0)
