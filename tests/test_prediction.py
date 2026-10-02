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
