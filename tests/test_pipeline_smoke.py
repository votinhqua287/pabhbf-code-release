"""End-to-end smoke test of training and evaluation on the smoke splits (ML smoke test of the final checklist).

Skipped when the smoke datasets are missing. Generate them with
  python -m pabhbf.data.make_datasets --config smoke.yaml --prefix smoke --far-field-twin
  python -m pabhbf.data.make_datasets --config smoke.yaml --prefix smoke --splits attack --no-teacher
"""
import numpy as np
import pytest

from pabhbf.data.dataset import DATA_DIR
from pabhbf.eval import evaluate
from pabhbf.train import baselines, pabhbf

pytestmark = pytest.mark.skipif(not all((DATA_DIR / f"smoke_{s}.npz").exists() for s in ("train", "val", "test", "attack")),
                                reason="smoke datasets not generated")


def test_pabhbf_adversarial_training_and_evaluation_run():
    log = pabhbf.train(pabhbf.parse_args(["--config", "smoke.yaml", "--prefix", "smoke", "--mode", "adversarial", "--lam", "1",
                                          "--epochs-pre", "1", "--epochs-adv", "1", "--warmup-epochs", "1",
                                          "--tag", "pytest_adv", "--val-drops", "50", "--threads", "2"]))
    assert len(log["history"]) == 2 and log["history"][1]["phase"] == "adversarial"
    val = log["history"][-1]["val"]
    assert 0.0 < val["ratio"] <= 1.5 and 0.0 <= val["adv_hit"] <= 1.0
    res = evaluate.main(["--ckpt", log["checkpoint"], "--exp", "pytest_smoke", "--epochs-att", "1", "--threads", "2",
                         "--attackers", "mlp_xy", "grid", "--no-snr-sweep"])
    s = res["privacy"]["summary"]
    assert res["payload_bits"] == 8 * 4
    assert 0.0 <= s["leakage"] <= 1.0 and np.isfinite(s["rmse_min_m"]) and np.isfinite(res["privacy"]["decoder"]["nmse_db"])
    assert res["utility"]["wsr"] > 0


def test_baseline_controller_training_runs_for_a_stochastic_representation():
    log = baselines.train(baselines.parse_args(["--config", "smoke.yaml", "--prefix", "smoke", "--rep", "gaussian_pca",
                                                "--sigma", "0.3", "--epochs", "2", "--tag", "pytest_gpca", "--threads", "2",
                                                "--val-drops", "50"]))
    assert log["payload_bits"] == 32 and len(log["history"]) == 2
