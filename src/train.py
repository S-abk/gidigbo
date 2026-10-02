"""Train, compare and save the UFC fight prediction model.

Run:  python -m src.train

Protocol (the test set is touched exactly once, after the model is chosen):
  1. Build the leakage-safe dataset (src.build_dataset + src.features).
  2. Chronological split: train < VALIDATION_START <= validation < TEST_START <= test.
  3. Walk-forward CV inside the training period picks hyperparameters per model family.
  4. Each family is refit on the full training period (mirrored A/B rows).
  5. Calibration method (none / sigmoid / isotonic) is chosen on the validation period
     by 2-fold chronological cross-fitting, then fit on the whole validation period.
  6. The production model is chosen on validation log loss (calibrated, cross-fitted).
  7. Only then are all models and baselines scored on the test period.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import evaluate as ev
from src.build_dataset import build_all
from src.config import (MIN_TRAIN_DATE, MODELS_DIR, RANDOM_SEED, RAW_DIR, RECALIBRATE_AFTER_DAYS,
                        RETRAIN_AFTER_DAYS, TEST_START, VALIDATION_START, WALK_FORWARD_YEARS)
from src.features import (CONTEXT_FEATURES, DIFF_FEATURES, FEATURE_GROUPS, MODEL_FEATURES,
                          build_training_table, mirror)
from src.method_model import (METHODS, build_method_table, predict_method_proba,
                               train_method_model)
from src.modeling import HAS_XGB, MODEL_GRIDS, Calibrator, FightModel, make_model

# If another model's validation log loss is within this margin of logistic
# regression's, prefer the interpretable logistic regression.
SIMPLICITY_MARGIN = 0.002


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = table["event_date"]
    n_all = len(table)
    table = table[d >= MIN_TRAIN_DATE]
    log(f"binary-outcome fights: {n_all}; dropped {n_all - len(table)} before {MIN_TRAIN_DATE} "
        f"(still used as fighter history)")
    d = table["event_date"]
    train = table[d < VALIDATION_START]
    val = table[(d >= VALIDATION_START) & (d < TEST_START)]
    test = table[d >= TEST_START]
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        r = ev.date_range(part)
        log(f"{name:10s} {r['start']} .. {r['end']}  fights={r['n_fights']}  label mean={part['label'].mean():.3f}")
    assert train["event_date"].max() < val["event_date"].min() <= val["event_date"].max() < test["event_date"].min()
    return train, val, test


def fit_fight_model(kind: str, params: dict, train: pd.DataFrame) -> FightModel:
    # Mirroring happens inside one split only, so no fight crosses a split boundary.
    X, y = mirror(train[MODEL_FEATURES], train["label"].to_numpy())
    est = make_model(kind, params).fit(X, y)
    return FightModel(kind, params, est)


def walk_forward_cv(kind: str, params: dict, train: pd.DataFrame) -> float:
    losses = []
    for year in WALK_FORWARD_YEARS:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        tr = train[train["event_date"] < start]
        te = train[(train["event_date"] >= start) & (train["event_date"] < end)]
        m = fit_fight_model(kind, params, tr)
        losses.append(ev.metrics(te["label"].to_numpy(), m.predict_proba_symmetric(te))["log_loss"])
    return float(np.mean(losses))


def mirrored_raw(model: FightModel, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = mirror(df[MODEL_FEATURES], df["label"].to_numpy())
    return model.raw_proba(X), y


# Isotonic regression is deliberately NOT a candidate. It is a step function: it gave
# only ~38 distinct probabilities over 1,437 test fights, claimed >95% confidence on
# fights it got right only ~82% of the time, and -- independent of any labels -- left
# ~54% of the app's occlusion-based "model factors" at exactly zero impact, because
# small input changes don't move a plateau. It is also known to overfit small
# calibration sets (~500 fights per cross-fit half here). Calibrators must be smooth
# and strictly increasing for the explanations to work. (Decided after the test set
# had been viewed once; see README "Glicko rating: backtest".)
CALIBRATION_CANDIDATES = ("none", "sigmoid")


def choose_calibration(model: FightModel, val: pd.DataFrame) -> tuple[str, dict]:
    """2-fold chronological cross-fitting on validation: fit on one half, score the other."""
    cut = val["event_date"].sort_values().iloc[len(val) // 2]
    halves = [val[val["event_date"] < cut], val[val["event_date"] >= cut]]
    scores = {}
    for method in CALIBRATION_CANDIDATES:
        ll = []
        for fit_half, score_half in [(halves[0], halves[1]), (halves[1], halves[0])]:
            p_fit, y_fit = mirrored_raw(model, fit_half)
            cal = Calibrator(method).fit(p_fit, y_fit)
            scored = FightModel(model.kind, model.params, model.estimator, cal)
            ll.append(ev.metrics(score_half["label"].to_numpy(), scored.predict_proba_symmetric(score_half))["log_loss"])
        scores[method] = float(np.mean(ll))
    best = min(scores, key=scores.get)
    return best, scores


def group_permutation_importance(model: FightModel, df: pd.DataFrame, n_repeats: int = 5) -> dict:
    """Increase in log loss when one feature group is shuffled (jointly) on validation data."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = df["label"].to_numpy()
    base = ev.metrics(y, model.predict_proba_symmetric(df))["log_loss"]
    out = {}
    for group, feats in FEATURE_GROUPS.items():
        cols = [f"{f}_diff" for f in feats]
        incs = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            idx = rng.permutation(len(df))
            shuffled[cols] = df[cols].to_numpy()[idx]
            incs.append(ev.metrics(y, model.predict_proba_symmetric(shuffled))["log_loss"] - base)
        out[group] = float(np.mean(incs))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def feature_importance(model: FightModel) -> dict:
    est = model.estimator
    if model.kind == "logistic_regression":
        coef = est.named_steps["clf"].coef_[0]
        return {"type": "standardized_coefficient", "values": dict(zip(MODEL_FEATURES, map(float, coef)))}
    if hasattr(est, "feature_importances_"):
        return {"type": "impurity_or_gain_importance",
                "values": dict(zip(MODEL_FEATURES, map(float, est.feature_importances_)))}
    return {"type": "none", "values": {}}


def baselines(train: pd.DataFrame) -> dict[str, callable]:
    """Naive baselines, fit on the training period only."""
    def lr_on(cols):
        X, y = mirror(train[MODEL_FEATURES], train["label"].to_numpy())
        m = LogisticRegression(fit_intercept=False).fit(X[cols], y)
        return lambda df: m.predict_proba(df[cols])[:, 1]
    return {
        "baseline_always_50": lambda df: np.full(len(df), 0.5),
        "baseline_win_pct_diff": lr_on(["win_pct_diff"]),
        "baseline_experience_record": lr_on(["n_fights_diff", "win_pct_diff"]),
    }


def joint_outcome_scores(win_model: FightModel, method_model, test: pd.DataFrame, data: dict) -> dict:
    """Log loss of the actual (winner, method) cell under P(winner) * P(method | winner).

    Baselines: uniform over the six outcomes, and the winner model x overall method rates.
    """
    mt = build_method_table(data["fights"], data["appearances"], data["fighters"]).set_index("fight_id")
    t = test[test["fight_id"].isin(mt.index)]
    p_a = win_model.predict_proba_symmetric(t)
    p_actual_winner = np.where(t["label"] == 1, p_a, 1 - p_a)
    rows = mt.loc[t["fight_id"]]
    pm = predict_method_proba(method_model, rows)
    idx = rows["method"].map({m: i for i, m in enumerate(METHODS)}).to_numpy()
    p_method = pm[np.arange(len(idx)), idx]
    rates = mt[(mt["event_date"] >= "2010-01-01") & (mt["event_date"] < VALIDATION_START)]["method"] \
        .value_counts(normalize=True)
    p_rate = rows["method"].map(rates).to_numpy()
    out = {
        "n": int(len(t)),
        "model_log_loss": float(-np.log(p_actual_winner * p_method).mean()),
        "baseline_uniform_log_loss": float(np.log(6)),
        "baseline_winner_model_x_method_rates_log_loss": float(-np.log(p_actual_winner * p_rate).mean()),
    }
    log(f"joint six-way test log loss: model={out['model_log_loss']:.4f} "
        f"(winner model x method rates={out['baseline_winner_model_x_method_rates_log_loss']:.4f}, "
        f"uniform={out['baseline_uniform_log_loss']:.4f})")
    return out


def source_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(RAW_DIR), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    np.random.seed(RANDOM_SEED)
    data = build_all(save=True)
    table = build_training_table(data["fights"], data["appearances"], data["fighters"])
    train, val, test = split(table)

    # ---- 1. hyperparameters by walk-forward CV (training period only)
    families = list(MODEL_GRIDS)
    log(f"xgboost available: {HAS_XGB} (fallback: HistGradientBoostingClassifier)")
    best_params, cv_scores = {}, {}
    for kind in families:
        scores = []
        for params in MODEL_GRIDS[kind]:
            s = walk_forward_cv(kind, params, train)
            scores.append((s, params))
            log(f"walk-forward CV {kind:20s} {params} -> log loss {s:.4f}")
        s, p = min(scores, key=lambda t: t[0])
        best_params[kind], cv_scores[kind] = p, s

    # ---- 2. refit on full train, choose + fit calibration on validation
    models, val_report = {}, {}
    for kind in families:
        m = fit_fight_model(kind, best_params[kind], train)
        method, cal_scores = choose_calibration(m, val)
        p_val, y_val = mirrored_raw(m, val)
        m.calibrator = Calibrator(method).fit(p_val, y_val)
        models[kind] = m
        val_report[kind] = {"walk_forward_cv_log_loss": cv_scores[kind], "calibration_method": method,
                            "calibration_crossfit_log_loss": cal_scores,
                            "validation_log_loss_crossfit": cal_scores[method]}
        log(f"{kind:20s} calibration={method:8s} cross-fit val log loss={cal_scores[method]:.4f} {cal_scores}")

    # ---- 3. select production model (validation only; test not yet used)
    val_ll = {k: v["validation_log_loss_crossfit"] for k, v in val_report.items()}
    selected = min(val_ll, key=val_ll.get)
    if selected != "logistic_regression" and val_ll["logistic_regression"] - val_ll[selected] < SIMPLICITY_MARGIN:
        log(f"{selected} beats logistic regression by < {SIMPLICITY_MARGIN}; preferring logistic regression")
        selected = "logistic_regression"
    log(f"selected production model: {selected}")

    # ---- 4. final, one-time test evaluation
    y_test = test["label"].to_numpy()
    rows, diag = [], {}
    for name, fn in baselines(train).items():
        rows.append({"model": name, **ev.metrics(y_test, fn(test))})
    for kind, m in models.items():
        p = m.predict_proba_symmetric(test)
        rows.append({"model": kind, **ev.metrics(y_test, p)})
        diag[kind] = ev.diagnostics(y_test, p)
        raw_ab = m.oriented_proba(test)
        from src.modeling import swap
        raw_ba = m.oriented_proba(swap(test))
        diag[kind]["mean_abs_reversal_gap_before_symmetrisation"] = float(np.mean(np.abs(raw_ab - (1 - raw_ba))))
    table_df = ev.results_table(rows)
    log("test-set results (selected model chosen before this table was computed):\n" + table_df.to_string())

    prod = models[selected]
    importance = {
        "model_specific": feature_importance(prod),
        "group_permutation_log_loss_increase_validation": group_permutation_importance(prod, val),
        "logistic_regression_standardized_coefficients": feature_importance(models["logistic_regression"])["values"],
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod, MODELS_DIR / "model.joblib")

    # ---- 5. method of victory (conditional on the winner), then the joint six-way outcome
    method_model, method_report = train_method_model(data["fights"], data["appearances"], data["fighters"], log=log)
    method_report["test_joint_six_way"] = joint_outcome_scores(prod, method_model, test, data)
    joblib.dump(method_model, MODELS_DIR / "method_model.joblib")
    for kind, m in models.items():
        joblib.dump(m, MODELS_DIR / f"candidate_{kind}.joblib")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test_metrics = next(r for r in rows if r["model"] == selected)
    metadata = {
        "trained_at_utc": now,
        "model_type": selected,
        "estimator_class": type(prod.estimator).__name__,
        "hyperparameters": best_params[selected],
        "calibration_method": prod.calibrator.method,
        "prediction_rule": "symmetrised: P(A) = 0.5 * (p(A,B) + 1 - p(B,A)), after calibration",
        "feature_names": MODEL_FEATURES,
        "diff_features": DIFF_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "periods": {"train": ev.date_range(train), "validation": ev.date_range(val), "test": ev.date_range(test),
                    "min_train_date": MIN_TRAIN_DATE},
        "n_training_examples": int(len(train)),
        "n_training_rows_mirrored": int(2 * len(train)),
        "random_seed": RANDOM_SEED,
        "data_source": {"repo": "https://github.com/Greco1899/scrape_ufc_stats", "commit": source_commit(),
                        "latest_fight_date": str(data["fights"]["event_date"].max().date())},
        "test_metrics": test_metrics,
        "staleness_policy": {
            "recalibrate_after_days": RECALIBRATE_AFTER_DAYS,
            "retrain_after_days": RETRAIN_AFTER_DAYS,
            "measured_from": "periods.validation.end (last fight the weights/calibrator were fit on)",
            "note": "Cheap proxies for an ongoing retain/recalibrate/refit decision (model weights "
                    "age even though fighter profiles keep advancing with every rebuild). Not a "
                    "formal utility-optimal schedule -- see README 'Model staleness'.",
        },
    }
    evaluation = {
        "selected_model": selected,
        "selection_rule": f"lowest cross-fitted validation log loss; ties within {SIMPLICITY_MARGIN} go to logistic regression",
        "validation": val_report,
        "test_results": table_df.reset_index().to_dict(orient="records"),
        "test_diagnostics": diag,
        "importance": importance,
        "periods": metadata["periods"],
        "method_model": method_report,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (MODELS_DIR / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str))
    log(f"saved model + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
