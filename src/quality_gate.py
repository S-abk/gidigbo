"""Model-quality gate: refuse to ship a model that got clearly worse or was built on bad data.

Used by tests/test_model_quality.py (absolute checks, every CI run) and by the scheduled
refresh workflow, which also compares against the previously committed model before it
commits anything:

  python -m src.quality_gate --previous-dir /path/to/previous/models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import MODELS_DIR

MIN_FIGHTS = 8000              # the dataset had 8,909 dated fights in Oct 2026; far fewer = truncated data
BASELINE_MARGIN = 0.01         # selected model must beat the best naive baseline's test log loss by this
MAX_VALIDATION_LOG_LOSS = 0.68  # a coin flip is 0.693
MAX_TEST_LOG_LOSS_INCREASE = 0.02  # vs the previously committed model (test period grows each refresh)
MAX_REFIT_MEAN_ABS_DIFF = 0.05     # deployed refit vs evaluated model on test fights (0.019 in Oct 2026)
MIN_REFIT_SAME_FAVOURITE = 0.90    # ... and how often they pick the same favourite (95.4% in Oct 2026)


def load(models_dir: Path) -> tuple[dict, dict]:
    return (json.loads((models_dir / "model_metadata.json").read_text()),
            json.loads((models_dir / "evaluation.json").read_text()))


def check(meta: dict, evaluation: dict, prev_meta: dict | None = None, prev_eval: dict | None = None) -> list[str]:
    """Return a list of human-readable failures (empty = pass)."""
    fails = []
    sel = evaluation["selected_model"]
    results = {r["model"]: r for r in evaluation["test_results"]}
    test_ll = results[sel]["log_loss"]
    best_baseline = min(r["log_loss"] for name, r in results.items()
                        if name.startswith("baseline_") and name != "baseline_always_50")
    if test_ll > best_baseline - BASELINE_MARGIN:
        fails.append(f"selected model test log loss {test_ll:.4f} does not beat the best naive baseline "
                     f"{best_baseline:.4f} by {BASELINE_MARGIN}")
    val_ll = evaluation["validation"][sel]["validation_log_loss_crossfit"]
    if val_ll > MAX_VALIDATION_LOG_LOSS:
        fails.append(f"validation log loss {val_ll:.4f} > {MAX_VALIDATION_LOG_LOSS}")
    n = meta["data_source"].get("n_fights", 0)
    if n < MIN_FIGHTS:
        fails.append(f"only {n} fights in the dataset (< {MIN_FIGHTS}); upstream data looks truncated")
    pf = meta.get("production_fit") or {}
    if pf.get("refit_on_all_data") and pf.get("end") != meta["data_source"]["latest_fight_date"]:
        fails.append(f"production model fit ends {pf.get('end')} but data runs to "
                     f"{meta['data_source']['latest_fight_date']}")

    agree = pf.get("agreement_with_evaluated_on_test")
    if pf.get("refit_on_all_data") and agree:
        if agree["mean_abs_diff"] > MAX_REFIT_MEAN_ABS_DIFF:
            fails.append(f"deployed refit differs from the evaluated model by {agree['mean_abs_diff']:.3f} "
                         f"on average (> {MAX_REFIT_MEAN_ABS_DIFF})")
        if agree["same_favourite_rate"] < MIN_REFIT_SAME_FAVOURITE:
            fails.append(f"deployed refit picks the same favourite only {agree['same_favourite_rate']:.1%} "
                         f"of the time (< {MIN_REFIT_SAME_FAVOURITE:.0%})")
    elif pf.get("refit_on_all_data"):
        fails.append("no deployed-vs-evaluated agreement recorded; the deployed refit is unchecked")

    if prev_meta and prev_eval:
        prev_n = prev_meta["data_source"].get("n_fights", 0)
        if n < prev_n:
            fails.append(f"dataset shrank from {prev_n} to {n} fights")
        prev_sel = prev_eval["selected_model"]
        prev_ll = {r["model"]: r for r in prev_eval["test_results"]}[prev_sel]["log_loss"]
        if test_ll > prev_ll + MAX_TEST_LOG_LOSS_INCREASE:
            fails.append(f"test log loss rose from {prev_ll:.4f} to {test_ll:.4f} "
                         f"(> {MAX_TEST_LOG_LOSS_INCREASE})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--previous-dir", type=Path, help="models/ directory of the currently committed model")
    args = ap.parse_args()
    meta, ev = load(MODELS_DIR)
    prev = load(args.previous_dir) if args.previous_dir else (None, None)
    fails = check(meta, ev, *prev)
    for f in fails:
        print(f"[quality_gate] FAIL: {f}")
    if fails:
        sys.exit(1)
    print("[quality_gate] pass")


if __name__ == "__main__":
    main()
