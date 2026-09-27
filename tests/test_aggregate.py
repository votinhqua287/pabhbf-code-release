"""Tests for result aggregation: grouping by evaluation group, newest-result selection and seed statistics."""
import json
import shutil

import numpy as np
import pytest

from pabhbf.eval import aggregate
from pabhbf.utils.config import ROOT


def _result(label, group, tag, seed, wsr, hits):
    return dict(label=label, group=group, tag=tag, seed=seed, payload_bits=128,
                utility=dict(wsr=wsr, ratio=wsr / 100.0),
                privacy=dict(attackers={k: dict(hit_prob=v, rmse_m=1.0 / v) for k, v in hits.items()},
                             summary=dict(leakage=max(hits.values()))))


def test_aggregate_groups_seeds_and_uses_the_common_attacker_set():
    exp = "pytest_aggregate"
    raw = ROOT / "results" / "raw" / exp
    raw.mkdir(parents=True, exist_ok=True)
    try:
        runs = [_result("lam1_s1", "lam1", "lam1", 1, 90.0, {"mlp_xy": 0.2, "grid_map": 0.3, "resmlp_xy": 0.5}),
                _result("lam1_s2", "lam1", "lam1", 2, 92.0, {"mlp_xy": 0.25, "grid_map": 0.35}),
                _result("lam1_mm_range", "lam1_mm_range", "lam1", 1, 80.0, {"mlp_xy": 0.1, "grid_map": 0.15})]
        stale = _result("lam1_s2", "lam1", "lam1", 2, 10.0, {"mlp_xy": 0.9})
        (raw / "lam1_s2_20260101-000000.json").write_text(json.dumps(stale))
        for i, r in enumerate(runs):
            (raw / f"{r['label']}_20260102-00000{i}.json").write_text(json.dumps(r))
        out = aggregate.aggregate(exp)
        assert set(out) == {"lam1", "lam1_mm_range"}
        metrics = out["lam1"]["metrics"]
        assert metrics["utility.wsr"]["n"] == 2 and metrics["utility.wsr"]["mean"] == pytest.approx(91.0)
        assert metrics["privacy.leakage_common"]["mean"] == pytest.approx(0.325)
        assert metrics["privacy.leakage"]["mean"] == pytest.approx(0.425)
        assert metrics["utility.wsr"]["ci95"] == pytest.approx(12.7062 * np.std([90.0, 92.0], ddof=1) / np.sqrt(2), rel=1e-3)
        assert out["lam1_mm_range"]["metrics"]["utility.wsr"]["n"] == 1
    finally:
        shutil.rmtree(raw, ignore_errors=True)
        (ROOT / "results" / "aggregated" / f"{exp}.json").unlink(missing_ok=True)
