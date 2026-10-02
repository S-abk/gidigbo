"""Model-regression gates on the trained artifacts (separate from the code-correctness tests)."""
import copy

import pytest

from src.config import MODELS_DIR
from src.quality_gate import check, load


@pytest.fixture(scope="module")
def artifacts():
    if not (MODELS_DIR / "model_metadata.json").exists():
        pytest.skip("no trained model: run `python -m src.train` first")
    return load(MODELS_DIR)


def test_current_model_passes_quality_gate(artifacts):
    meta, ev = artifacts
    assert check(meta, ev) == []


def test_gate_catches_regressions(artifacts):
    meta, ev = artifacts
    worse_meta, worse_ev = copy.deepcopy(meta), copy.deepcopy(ev)
    worse_meta["data_source"]["n_fights"] = 100  # truncated upstream data
    for r in worse_ev["test_results"]:
        if r["model"] == worse_ev["selected_model"]:
            r["log_loss"] = 0.69  # no better than baselines
    worse_meta["production_fit"]["agreement_with_evaluated_on_test"] = {"mean_abs_diff": 0.2,
                                                                       "same_favourite_rate": 0.6}
    fails = check(worse_meta, worse_ev, prev_meta=meta, prev_eval=ev)
    joined = " | ".join(fails)
    assert "baseline" in joined and "truncated" in joined and "shrank" in joined and "rose" in joined
    assert "differs from the evaluated" in joined and "same favourite" in joined
