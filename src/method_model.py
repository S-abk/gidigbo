"""Method-of-victory model: P(KO/TKO, SUB, DEC | winner W beat loser L).

Combined with the winner model:
    P(A wins by m) = P(A wins) * P(m | A beat B)
which gives a six-way outcome distribution (A/B x KO/SUB/DEC) that sums to 1 and is
exactly consistent when A and B are swapped.

Rows are built from the winner's point of view using the same leakage-safe pre-fight
profiles as the winner model (src.features.compute_prefight_features), plus bout
context (weight class, women's bout, five rounds, title). DQ / overturned / other
outcomes (< 1%) are too rare to model and are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, TEST_START, VALIDATION_START
from src.features import compute_prefight_features

METHODS = ["KO/TKO", "SUB", "DEC"]
# Method mix shifted toward decisions during the 2000s (37% -> ~49%); it has been
# stable since about 2010, so older fights are history only for this model.
METHOD_MIN_TRAIN_DATE = "2010-01-01"
DEFAULT_WEIGHT_LBS = 170.0  # catch / open weight bouts (no listed limit)

# The same finishing / durability / style profile for BOTH fighters, plus a few differences
# (incl. the Glicko opponent-strength rating). Chosen over a smaller, asymmetric set by a
# pre-registered validation test (C tuned inside training for both): -0.0030 log loss,
# 95% CI [-0.0059, -0.0003]. See README "Method of victory".
PROFILE_COLS = ["ko_win_share", "sub_win_share", "finish_rate", "kd15", "sub15", "slpm", "td15", "ctrl_pct",
                "sig_acc", "fin_loss_rate", "sapm", "sig_def", "td_def", "avg_fight_min", "n_fights", "age"]
WINNER_COLS = PROFILE_COLS
LOSER_COLS = PROFILE_COLS
DIFF_COLS = ["glicko_rating", "win_pct", "sig_diff_pm", "reach_in"]
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)
C_HOLDOUT_YEARS = 2  # C is chosen on the last 2 training years, fit on the years before them
CONTEXT_COLS = ["weight_lbs", "is_female", "five_rounds", "is_title"]
METHOD_FEATURES = ([f"w_{c}" for c in WINNER_COLS] + [f"l_{c}" for c in LOSER_COLS]
                   + [f"wl_{c}_diff" for c in DIFF_COLS] + CONTEXT_COLS)


def method_features(prof_w: pd.DataFrame, prof_l: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Model input from aligned winner / loser profiles and bout context (shared train/inference)."""
    w, l, c = (d.reset_index(drop=True) for d in (prof_w, prof_l, context))
    X = pd.DataFrame(index=w.index)
    for col in WINNER_COLS:
        X[f"w_{col}"] = w[col].astype(float)
    for col in LOSER_COLS:
        X[f"l_{col}"] = l[col].astype(float)
    for col in DIFF_COLS:
        X[f"wl_{col}_diff"] = w[col].astype(float) - l[col].astype(float)
    X["weight_lbs"] = c["weight_lbs"].astype(float).fillna(DEFAULT_WEIGHT_LBS)
    X["is_female"] = c["is_female"].astype(float)
    X["five_rounds"] = (c["scheduled_rounds"] == 5).astype(float)
    X["is_title"] = c["is_title"].astype(float)
    return X[METHOD_FEATURES].fillna(0.0)


def build_method_table(fights: pd.DataFrame, appearances: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    prof = compute_prefight_features(appearances, fighters).set_index(["fight_id", "fighter_id"])
    f = fights[fights["winner_side"].notna() & fights["method"].isin(METHODS)].reset_index(drop=True)
    w_id = np.where(f["winner_side"] == 1, f["fighter_1_id"], f["fighter_2_id"])
    l_id = np.where(f["winner_side"] == 1, f["fighter_2_id"], f["fighter_1_id"])
    pw = prof.loc[list(zip(f["fight_id"], w_id))].reset_index()
    pl = prof.loc[list(zip(f["fight_id"], l_id))].reset_index()
    X = method_features(pw, pl, f)
    return pd.concat([f[["fight_id", "event_date", "method"]], X], axis=1)


def _fit(train: pd.DataFrame, C: float):
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=C, max_iter=3000, random_state=RANDOM_SEED))])
    return model.fit(train[METHOD_FEATURES], train["method"])


def predict_method_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Columns ordered as METHODS."""
    p = model.predict_proba(X[METHOD_FEATURES])
    order = [list(model.classes_).index(m) for m in METHODS]
    return p[:, order]


def _scores(y: pd.Series, p: np.ndarray) -> dict:
    """Multiclass log loss computed explicitly against METHODS column order
    (sklearn's log_loss assumes alphabetically sorted columns)."""
    idx = pd.Series(y).map({m: i for i, m in enumerate(METHODS)}).to_numpy()
    p_true = np.clip(p[np.arange(len(idx)), idx], 1e-12, 1)
    return {"n": int(len(idx)), "log_loss": float(-np.log(p_true).mean()),
            "accuracy": float(accuracy_score(y, np.array(METHODS)[p.argmax(1)]))}


def train_method_model(fights, appearances, fighters, log=print) -> tuple[object, dict]:
    """Chronological split identical to the winner model; test scored once.

    C is chosen on a holdout INSIDE the training period (its last C_HOLDOUT_YEARS years,
    fit on the years before), so validation stays an honest estimate. (It used to be
    chosen on validation, from a grid whose smallest value was larger than the best one.)
    """
    t = build_method_table(fights, appearances, fighters)
    d = t["event_date"]
    train = t[(d >= METHOD_MIN_TRAIN_DATE) & (d < VALIDATION_START)]
    val = t[(d >= VALIDATION_START) & (d < TEST_START)]
    test = t[d >= TEST_START]
    log(f"method model rows: train={len(train)} ({METHOD_MIN_TRAIN_DATE}..), validation={len(val)}, test={len(test)}")

    cut = pd.Timestamp(VALIDATION_START) - pd.DateOffset(years=C_HOLDOUT_YEARS)
    inner, holdout = train[train["event_date"] < cut], train[train["event_date"] >= cut]
    holdout_scores = {C: _scores(holdout["method"], predict_method_proba(_fit(inner, C), holdout))["log_loss"]
                      for C in C_GRID}
    best_C = min(holdout_scores, key=holdout_scores.get)
    log(f"method model in-training holdout ({cut.date()}..) log loss by C: "
        f"{ {k: round(v, 4) for k, v in holdout_scores.items()} } -> C={best_C}")
    model = _fit(train, best_C)
    val_scores = {best_C: _scores(val["method"], predict_method_proba(model, val))["log_loss"]}

    base_rates = train["method"].value_counts(normalize=True).reindex(METHODS).to_numpy()
    p_test = predict_method_proba(model, test)
    p_base = np.tile(base_rates, (len(test), 1))
    ctx_rates = (train.assign(wc=train["weight_lbs"].astype(str) + train["is_female"].astype(str))
                 .groupby("wc")["method"].value_counts(normalize=True).unstack().reindex(columns=METHODS).fillna(0))
    test_wc = test["weight_lbs"].astype(str) + test["is_female"].astype(str)
    p_ctx = ctx_rates.reindex(test_wc).fillna(dict(zip(METHODS, base_rates))).to_numpy()
    p_ctx = np.clip(p_ctx, 1e-3, 1) / np.clip(p_ctx, 1e-3, 1).sum(1, keepdims=True)

    report = {
        "classes": METHODS, "C": best_C, "min_train_date": METHOD_MIN_TRAIN_DATE,
        "n_train": int(len(train)), "n_validation": int(len(val)), "n_test": int(len(test)),
        "c_selection": f"in-training holdout: last {C_HOLDOUT_YEARS} years before {VALIDATION_START}",
        "holdout_log_loss_by_C": {str(k): v for k, v in holdout_scores.items()},
        "validation_log_loss": val_scores[best_C],
        "test": {
            "model": _scores(test["method"], p_test),
            "baseline_overall_rates": _scores(test["method"], p_base),
            "baseline_weight_class_rates": _scores(test["method"], p_ctx),
        },
        "test_mean_predicted": dict(zip(METHODS, p_test.mean(0).round(4).tolist())),
        "test_actual_share": test["method"].value_counts(normalize=True).reindex(METHODS).round(4).to_dict(),
        "standardized_coefficients": {
            m: dict(zip(METHOD_FEATURES, model.named_steps["clf"].coef_[list(model.classes_).index(m)].round(4).tolist()))
            for m in METHODS},
    }
    for name, s in report["test"].items():
        log(f"method test {name:28s} log loss={s['log_loss']:.4f} accuracy={s['accuracy']:.3f}")
    return model, report


def round_split(fights: pd.DataFrame, start, end) -> dict[str, list[float]]:
    """P(finish in round r | the fight ends early), separately for 3- and 5-round bouts,
    from KO/TKO and submission finishes in [start, end), with add-one smoothing.

    Combined with the model's P(decision) this gives round-of-finish odds:
    P(ends in round r) = (1 - P(decision)) x split[r]. Two research prototypes (a Monte
    Carlo fight simulator and a competing-risks hazard model) did no better than this.
    """
    f = fights[fights["winner_side"].notna() & fights["method"].isin(["KO/TKO", "SUB"])
               & (fights["event_date"] >= pd.Timestamp(start)) & (fights["event_date"] < pd.Timestamp(end))]
    out = {}
    for k in (3, 5):
        r = f.loc[f["scheduled_rounds"] == k, "end_round"].clip(1, k).astype(int)
        counts = np.bincount(r - 1, minlength=k)[:k] + 1.0
        out[str(k)] = (counts / counts.sum()).tolist()
    return out


def refit_method_model_all(fights, appearances, fighters, C: float) -> tuple[object, dict]:
    """Production refit: the evaluated configuration (same C) trained on every method row
    from METHOD_MIN_TRAIN_DATE through the latest fight."""
    t = build_method_table(fights, appearances, fighters)
    rows = t[t["event_date"] >= METHOD_MIN_TRAIN_DATE]
    fit = {"start": str(rows["event_date"].min().date()), "end": str(rows["event_date"].max().date()),
           "n_fights": int(len(rows)), "C": C}
    return _fit(rows, C), fit


def method_given_winner(model, prof_w: pd.DataFrame, prof_l: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
    return predict_method_proba(model, method_features(prof_w, prof_l, context))
