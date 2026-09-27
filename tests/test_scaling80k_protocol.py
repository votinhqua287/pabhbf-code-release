"""Protocol of the post-hoc 80,000-drop attack-data scaling.

The frozen encoders are reused, the attacker architectures and the early stopping match the confirmatory stage, the
attacker-training geometry extends the confirmatory attacker set and stays disjoint from the validation and test
cohorts, and nothing in the selection depends on these results.
"""
import glob
import json
import pickle
import sys

import pytest
import yaml

from pabhbf.utils.config import ROOT
from pabhbf import revision

sys.modules.setdefault('__main__', sys.modules[__name__])
sys.modules['__main__'].TemporalSet = revision.TemporalSet  # the stored bundles reference the run module

REV = ROOT / "results" / "revision_final"
SCALING = REV / "attacks" / "scaling80k"


def load(path):
    if not path.exists():
        pytest.skip(f"{path} missing")
    return json.loads(path.read_text())


def test_scaling_split_extends_attack_train_and_stays_disjoint():
    audit = load(REV / "scaling_audit.json")
    assert audit["users"]["attack_train_80k"] == 1_280_000
    assert audit["extends_attack_train"] == 40
    assert audit["disjoint_from"] == ["attack_val", "final_test", "policy_train", "policy_val"]
    cfg = yaml.safe_load((ROOT / "configs" / "revision_final.yaml").read_text())
    assert cfg["attack_train_80k"] == {"seed": cfg["attack_train"]["seed"], "drops": 80000}
    assert cfg["scaling"]["drops"] == 80000 and cfg["scaling"]["split"] == "attack_train_80k"
    # the confirmatory attacker data and every frozen model are unchanged; only tooling moved after the freeze
    assert audit["frozen_files_unchanged"] >= 330
    assert all(name.endswith(('.py', '.yaml')) for name in audit["frozen_files_changed"])


def test_scaling_attackers_reuse_frozen_encoders_and_the_confirmatory_recipe():
    files = sorted(glob.glob(str(SCALING / "*_s*_T*" / "*.pkl")))
    if not files:
        pytest.skip("scaling fits missing")
    frozen = load(REV / "frozen.json")["files"]
    kinds = set()
    for path in files:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        assert bundle["attack_train_drops"] == 80000
        identity = bundle["model_identity"]
        assert frozen[identity["path"]] == identity["sha256"]  # the encoder is the frozen checkpoint
        kinds.add((bundle["kind"], bundle["T"]))
        if bundle["kind"] != "knn":
            assert bundle["history"]["epochs_run"] <= 40
    cfg = yaml.safe_load((ROOT / "configs" / "revision_final.yaml").read_text())
    expected = {(k, 1) for k in cfg["attacks"]["snapshot"]} | {(k, T) for k in cfg["attacks"]["temporal"] for T in (4, 8)}
    assert kinds == expected


def test_selection_and_the_frozen_test_do_not_depend_on_the_scaling_results():
    selection = load(REV / "selection.json")
    assert selection["selected"] == ["lam0.01", "dz4_lam0"]
    assert selection["rule"]["attack_drops"] == 5000
    for candidate in selection["candidates"]:
        for attack in candidate["attackers"].values():
            assert "scaling80k" not in attack["checkpoint"]
    for path in glob.glob(str(REV / "test" / "*_T*.json")):
        for attack in json.loads(open(path).read())["attackers"].values():
            assert attack["attack_train_size"] in (5000, 20000)


def test_scaling_results_use_the_frozen_test_cohort():
    files = sorted(glob.glob(str(REV / "test80k" / "*_T*.json")))
    if not files:
        pytest.skip("scaling test results missing")
    assert len(files) == 36
    for path in files:
        result = json.loads(open(path).read())
        for name, attack in result["attackers"].items():
            assert attack["n"] == 32000 and attack["attack_train_size"] == 80000
        assert result["leakage"] == max(a["hit_prob"] for a in result["attackers"].values())


def test_scaling_macros_and_wording_match_the_files():
    """The 20k/80k macros of the attack-strength paragraph, and its saturation wording, follow the test files."""
    import re
    summary = load(REV / "final_summary.json")
    decision = load(REV / "scaling_decision.json")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             (REV / "generated" / "numbers_revision.tex").read_text()))
    names = {"lam0.01": "Full", "lam0": "Task", "dz4_lam0": "Small", "dz4_lam0.03": "SmallAdv"}
    for tag, prefix in names.items():
        for T, suffix in ((1, ""), (8, "TEight")):
            per_size = {}
            for size, folder in ((20000, "test"), (80000, "test80k")):
                vals = []
                for seed in (1, 2, 3):
                    r = load(REV / folder / f"{tag}_s{seed}_T{T}.json")
                    vals.append(max(a["hit_prob"] for a in r["attackers"].values() if a["attack_train_size"] == size))
                per_size[size] = vals
            g = [b - a for a, b in zip(per_size[20000], per_size[80000])]
            assert macros[prefix + suffix + "TwentyKAll"] == f"{sum(per_size[20000]) / 3:.3f}"
            assert macros[prefix + suffix + "EightyKAll"] == f"{sum(per_size[80000]) / 3:.3f}"
            assert macros[prefix + suffix + "EightyKGain"] == f"{sum(g) / 3:.3f}"
            assert abs(decision["gain"][f"{tag}_T{T}"] - sum(g) / 3) < 1e-9
            # the reported leakage is the maximum over both registries (plus shorter windows and never less)
            reported = summary["statistics"][tag][str(T)]["Leak"]["values"]
            assert all(rep >= max(a, b) - 1e-12 for rep, a, b in zip(reported, per_size[20000], per_size[80000]))
    assert decision["small_saturated"] and decision["big_rises"]


def test_attack_family_table_and_macros_match_the_files():
    """Table of every decision rule (20k/80k cells), the temporal factors, the RMSE macros and the validation gap."""
    import re
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             (REV / "generated" / "numbers_revision.tex").read_text()))
    table = (REV / "generated" / "tab_attacks_final.tex").read_text()
    rules = {"Cartesian MLP": ("mlp_xy", 1), "Polar MLP": ("mlp_rtheta", 1), "Residual MLP": ("resmlp_xy", 1),
             "20-NN regressor": ("knn", 1), "Disc-MAP classifier": ("grid_map", 1), "Mean classifier": ("grid_mean", 1),
             "Mean-input MLP": ("mean_mlp", 8), "Mean-input disc-MAP": ("mean_grid_map", 8),
             "Mean-input mean": ("mean_grid_mean", 8), "DeepSets": ("deepsets", 8)}
    tags = ["lam0.01", "lam0", "dz4_lam0", "dz4_lam0.03"]

    def mean_of(tag, T, rule, size, field="hit_prob"):
        folder = "test" if size == 20000 else "test80k"
        return sum(load(REV / folder / f"{tag}_s{s}_T{T}.json")["attackers"][f"{rule}_{size}drops"][field] for s in (1, 2, 3)) / 3

    seen = 0
    for line in table.splitlines():
        label = line.split("&")[0].strip()
        if label not in rules:
            continue
        rule, T = rules[label]
        cells = [c.strip().rstrip("\\\\") for c in line.split("&")[1:]]
        assert len(cells) == 4
        for tag, cell in zip(tags, cells):
            small, big = cell.split("/")
            assert small == f"{mean_of(tag, T, rule, 20000):.3f}" and big == f"{mean_of(tag, T, rule, 80000):.3f}", (label, tag)
        seen += 1
    assert seen == 10
    names = {"lam0.01": "Full", "lam0": "Task", "dz4_lam0": "Small", "dz4_lam0.03": "SmallAdv"}
    for tag, P in names.items():
        assert macros[P + "ClassifierOne"] == f"{mean_of(tag, 1, 'grid_map', 80000):.3f}"
        assert macros[P + "ClassifierOneTwentyK"] == f"{mean_of(tag, 1, 'grid_map', 20000):.3f}"
        assert macros[P + "ResidualOne"] == f"{mean_of(tag, 1, 'resmlp_xy', 80000):.3f}"
        assert macros[P + "ResidualOneTwentyK"] == f"{mean_of(tag, 1, 'resmlp_xy', 20000):.3f}"
        assert macros[P + "MeanOne"] == f"{mean_of(tag, 1, 'grid_mean', 80000):.3f}"
        best = max(mean_of(tag, 1, r, 80000) for r in ("mlp_xy", "mlp_rtheta", "resmlp_xy", "knn"))
        assert macros[P + "RegressorOne"] == f"{best:.3f}"
        assert float(macros[P + "RegressorOne"]) < float(macros[P + "ClassifierOne"])  # the classifier wins at T = 1
        assert macros[P + "DeepSetsRmseEight"] == f"{mean_of(tag, 8, 'deepsets', 80000, 'rmse_m'):.2f}"
        assert macros[P + "MapRmseEight"] == f"{mean_of(tag, 8, 'mean_grid_map', 80000, 'rmse_m'):.2f}"
        assert macros[P + "TemporalFactor"] == f"{mean_of(tag, 8, 'mean_grid_map', 80000) / mean_of(tag, 1, 'grid_map', 80000):.1f}"
        # the mean-input disc-MAP rule is the strongest rule at T = 8 for every seed
        for s in (1, 2, 3):
            r = load(REV / "test80k" / f"{tag}_s{s}_T8.json")["attackers"]
            assert max(r, key=lambda k: r[k]["hit_prob"]) == "mean_grid_map_80000drops"
    for tag in ("dz4_lam0", "dz4_lam0.03"):  # DeepSets has the smallest RMSE at T = 8 for the 16-bit encoders
        for s in (1, 2, 3):
            r = load(REV / "test80k" / f"{tag}_s{s}_T8.json")["attackers"]
            assert min(r, key=lambda k: r[k]["rmse_m"]) == "deepsets_80000drops"
    knn = max(abs(load(REV / "test80k" / f"{tag}_s{s}_T1.json")["attackers"]["knn_80000drops"]["hit_prob"]
                  - load(REV / "test" / f"{tag}_s{s}_T1.json")["attackers"]["knn_20000drops"]["hit_prob"])
              for tag in tags for s in (1, 2, 3))
    assert macros["KnnMaxChange"] == f"{knn:.3f}"
    gap = max(abs(max(a["hit_prob"] for a in load(REV / "test" / f"{tag}_s{s}_T1.json")["attackers"].values()
                      if a["attack_train_size"] == 20000)
                  - load(REV / "attacks" / "confirmatory" / f"{tag}_s{s}_T1" / "validation.json")["leakage"])
              for tag in tags for s in (1, 2, 3))
    assert macros["ValTestMaxGap"] == f"{gap:.3f}"

