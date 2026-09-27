"""Numbers of the revised manuscript that come from the development runs, the selection stage and the protocol.

test_final_claims.py covers the macros generated from the frozen final test. This file covers every other number quoted
in the active results section: the development experiments (premise, teacher, mismatch, ablations, attack-data
scaling), the validation-based selection, the split sizes and the protected tool interface. Each assertion compares
the value formatted as in the manuscript with the value recomputed from the result files.
"""
import glob
import json
import pathlib

import pytest
import yaml

from pabhbf.utils.config import ROOT

AGG = ROOT / "results" / "aggregated"
RAW = ROOT / "results" / "raw"
REV = ROOT / "results" / "revision_final"
LEAK, RATIO = "privacy.leakage_common", "utility.ratio"


def load(path):
    if not path.exists():
        pytest.skip(f"{path} missing")
    return json.loads(path.read_text())


def latest(pattern):
    files = sorted(glob.glob(str(RAW / pattern)))
    if not files:
        pytest.skip(f"{pattern} missing")
    return json.loads(open(files[-1], encoding="utf-8").read())


def seed1(d, tag, key):
    g = d[tag]
    return g["metrics"][key]["values"][g["seeds"].index(1)]


def r3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


def results_text():
    return (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")


def test_premise_numbers():
    prem = latest("premise/premise_mle_nt256p_*.json")
    run = {m: next(r for r in prem["runs"] if r["model"] == m and r["attacker"] == "mle_fusion") for m in ("near_field", "far_field")}
    assert pct(run["near_field"]["hit_prob"]) == "92.1" and pct(run["far_field"]["hit_prob"]) == "20.8"
    assert min(run["near_field"]["bins"]["hit_prob"]) >= 0.83
    p64 = latest("premise/premise_mle_nt64_*.json")
    run64 = {m: next(r for r in p64["runs"] if r["model"] == m and r["attacker"] == "mle_fusion") for m in ("near_field", "far_field")}
    assert f"{run64['near_field']['bins']['hit_prob'][0]:.2f}" == "0.67" and f"{run64['far_field']['bins']['hit_prob'][0]:.2f}" == "0.14"
    assert r3(run64["near_field"]["hit_prob"]) == "0.274" and r3(run64["far_field"]["hit_prob"]) == "0.218"
    assert abs(run64["near_field"]["bins"]["edges"][2] - 7.25) < 1e-6  # quoted as 7.3 m
    rc = load(AGG / "revision_checks.json")
    bias = {(r["Nt"], r["kappa_db"]): r["median_abs_range_error_m"] for r in rc["multipath_range_error"]}
    assert f"{bias[(64, 10.0)]:.2f}" == "1.39" and f"{bias[(256, 10.0)]:.2f}" == "0.03"
    text = results_text()
    for q in (r"92.1\,\% of the users within 1\,m", r"against 20.8\,\%", r"at or above 0.83 in every distance bin",
              r"concentrated below 7.3\,m", r"0.67 against 0.14", r"0.274 and 0.218",
              r"median absolute error of the maximum-likelihood range is 1.39\,m for the 64-antenna array, against 0.03\,m"):
        assert q in text, q


def test_teacher_and_development_baseline_numbers():
    tq = load(AGG / "teacher_quality_nt256.json")
    assert sum(round(r["frac_optimal"] * 300) for r in tq["small"]) == 1799
    assert f"{100 * min(r['worst_ratio'] for r in tq['small']):.2f}" == "99.76"
    d10 = next(r for r in tq["default"] if r["snr_db"] == 10.0)
    assert f"{d10['teacher']:.2f}" == "128.46" and pct(d10["teacher"] / d10["strongest_users"] - 1) == "8.3"
    b = load(AGG / "baselines.json")
    assert r3(b["top_m_beams"]["metrics"][LEAK]["mean"]) == "0.873" and len(b["top_m_beams"]["seeds"]) == 3
    assert (pct(seed1(b, "gaussian_pca_0.2", RATIO)), r3(seed1(b, "gaussian_pca_0.2", LEAK))) == ("57.6", "0.117")
    assert (pct(seed1(b, "laplace_pca_300", RATIO)), r3(seed1(b, "laplace_pca_300", LEAK))) == ("52.8", "0.087")
    assert (pct(seed1(b, "laplace_pca_10", RATIO)), r3(seed1(b, "laplace_pca_10", LEAK))) == ("22.1", "0.008")
    e1 = load(AGG / "e1.json")
    nmse = "privacy.decoder.nmse_db"
    assert f"{b['raw_csi']['metrics'][nmse]['mean']:.1f}" == "-20.6" and f"{e1['lam0.03']['metrics'][nmse]['mean']:.2f}" == "-0.22"
    audit = {r["scheme"]: r for r in load(REV / "payload_audit.json")}
    assert audit["top_m_beams"]["actual_payload_bits"] == 128 and audit["raw_csi"]["actual_payload_bits"] == 16384
    assert audit["gaussian_pca_0.2"]["actual_payload_bits"] == 128 and audit["laplace_pca_10"]["actual_payload_bits"] == 128
    text = results_text()
    for q in (r"1,799 of 1,800", r"99.76\,\%", r"128.46\,bit/s/Hz", r"8.3\,\%", r"auxiliary leakage of 0.873 over three seeds",
              r"eight 12-bit codeword indices with 4-bit gains", r"16,384 bits per user",
              r"B5 with $\sigma_{\mathrm{B5}}=0.2$ keeps 57.6\,\% of the teacher rate with a leakage of 0.117",
              r"B9 keeps 52.8\,\% with 0.087 at $\varepsilon_{\mathrm{DP}}=300$ and 22.1\,\% with 0.008 at $\varepsilon_{\mathrm{DP}}=10$",
              r"NMSE of $-20.6$\,dB, against $-0.22$\,dB for the adversarial representation at $\mu=0.03$"):
        assert q in text, q


def test_development_ablation_numbers():
    e1, mm, ab = load(AGG / "e1.json"), load(AGG / "mismatch.json"), load(AGG / "ablations.json")
    assert r3(seed1(e1, "lam0.03", RATIO)) == "0.978" and r3(seed1(mm, "lam0.03_mm_channel", RATIO)) == "0.928"
    assert r3(seed1(e1, "lam0.03", LEAK)) == "0.174" and r3(seed1(mm, "lam0.03_mm_range", LEAK)) == "0.220"
    assert r3(seed1(ab, "abl_continuous_lam0.03", LEAK)) == "0.354"
    scaling = {(t, n): load(RAW / "attack_scaling" / f"{t}_n{n}.json")["leakage_common"] for t in ("lam0.03", "lam0") for n in (20000, 80000)}
    assert [r3(scaling[("lam0.03", n)]) for n in (20000, 80000)] == ["0.173", "0.188"]
    assert [r3(scaling[("lam0", n)]) for n in (20000, 80000)] == ["0.285", "0.355"]
    text = results_text()
    for q in (r"from 0.978 to 0.928", r"from 0.174 to 0.220", r"from 0.174 to 0.354"):
        assert q in text, q


def test_selection_and_validation_numbers():
    sel = load(REV / "selection.json")
    assert sel["selected"] == ["lam0.01", "dz4_lam0"] and len(sel["candidates"]) == 8
    assert all(len(c["attackers"]) == 6 for c in sel["candidates"])
    assert sel["rule"]["attack_drops"] == 5000 and sel["rule"]["snapshot_budget"] == 0.2
    val = [load(REV / "attacks" / "confirmatory" / f"lam0.01_s{s}_T1" / "validation.json") for s in (1, 2, 3)]
    assert [r3(v["leakage"]) for v in val] == ["0.200", "0.183", "0.201"]
    assert sum(v["ucb"] > 0.2 for v in val) == 2
    text = results_text()
    assert [c["tag"] for c in sel["candidates"]] == ["lam0", "lam0.01", "lam0.03", "lam0.1", "lam0.3", "lam1", "dz4_lam0", "dz4_lam0.03"]
    for q in (r"selects $\mu=0.01$ at 128 bits and $\mu=0$ at 16 bits", r"0.200, 0.183, and 0.201",
              r"$\mu\in\{0,0.01,0.03,0.1,0.3,1\}$ at 128 bits and $\mu\in\{0,0.03\}$ at 16 bits",
              r"adjusted upper endpoints above the screening criterion for two seeds", r"eight candidate configurations and the six decision rules",
              r"attackers trained on 5,000 fresh drops"):
        assert q in text, q


def test_protocol_sizes_and_tool_interface():
    cfg = yaml.safe_load((ROOT / "configs" / "revision_final.yaml").read_text())
    assert (cfg["attack_train"]["drops"], cfg["attack_val"]["drops"], cfg["final_test"]["drops"]) == (20000, 2000, 2000)
    assert cfg["attacks"]["epochs"] == 40 and cfg["attacks"]["patience"] == 6 and cfg["attacks"]["batch"] == 2048
    assert cfg["attacks"]["knn_rows"] == 100000 and cfg["attacks"]["threads"] == 3 and cfg["intervals"] == [1, 4, 8]
    split = load(REV / "split_audit.json")
    assert split["final_test"] == 32000 and split["attack_val"] == 32000 and split["attack_train"] == 320000
    for s in (1, 2, 3):
        for mode in load(REV / "test" / f"tools_lam0.01_s{s}.json").values():
            assert mode["ack_bits_per_batch"] == 24 and mode["threads"] == 3 and mode["batch_drops"] == 2000
            assert mode["csi_dependent_return_bits"] == 0
    text = results_text()
    for q in (r"20,000, 2,000, and 2,000 drops", r"32,000 users", r"at most 100,000 attack-training users",
              r"mini-batches of 2,048 users, at most 40 epochs, and early stopping after six epochs",
              r"batches of 2,000 drops with three CPU threads", r"24-bit acknowledgment"):
        assert q in text, q


def test_auxiliary_sweeps_and_ablations():
    """The aperture sweep, the latent-dimension sweep, the phase-resolution check and the ablations of seed 1."""
    nt, e1, b, e2 = load(AGG / "nt_sweep.json"), load(AGG / "e1.json"), load(AGG / "baselines.json"), load(AGG / "e2.json")
    raw = [nt[f"nt{n}_raw_csi"]["metrics"][LEAK]["mean"] for n in (32, 64, 128)] + [seed1(b, "raw_csi", LEAK)]
    assert [r3(x) for x in raw] == ["0.260", "0.307", "0.550", "0.774"]
    adv = [nt[f"nt{n}_lam0.03"]["metrics"][LEAK]["mean"] for n in (32, 64, 128)] + [seed1(e1, "lam0.03", LEAK)]
    assert r3(min(adv)) == "0.168" and r3(max(adv)) == "0.197"
    ratios = [nt[f"nt{n}_lam0.03"]["metrics"][RATIO]["mean"] for n in (32, 64, 128)] + [seed1(e1, "lam0.03", RATIO)]
    assert pct(min(ratios)) == "89.1" and pct(max(ratios)) == "97.8" and ratios.index(min(ratios)) == 0
    assert all(nt[f"nt{n}_raw_csi"]["seeds"] == [1] for n in (32, 64, 128))
    task = [e2[f"dz{d}_lam0"]["metrics"][LEAK]["mean"] for d in (4, 8, 16, 64)]
    advl = [e2[f"dz{d}_lam0.03"]["metrics"][LEAK]["mean"] for d in (4, 8, 16, 64)]
    assert [r3(x) for x in task] == ["0.154", "0.186", "0.218", "0.320"]
    assert [r3(x) for x in advl] == ["0.124", "0.141", "0.155", "0.201"]
    assert all(e2[f"dz{d}_{m}"]["seeds"] == ([1, 2, 3] if d in (4, 8) else [1]) for d in (4, 8, 16, 64) for m in ("lam0", "lam0.03"))
    assert min(e2[f"dz{d}_{m}"]["metrics"][RATIO]["mean"] for d in (4, 8, 16, 64) for m in ("lam0", "lam0.03")) > 0.96
    pb = load(AGG / "phase_bits.json")
    assert r3(seed1(e1, "lam0.03", RATIO)) == "0.978"
    assert [r3(pb[f"lam0.03_{k}"]["metrics"][RATIO]["mean"]) for k in ("bits3", "bits2", "bitsideal")] == ["0.971", "0.940", "0.979"]
    assert f"{100 * (pb['lam0.03_bitsideal']['metrics'][RATIO]['mean'] - seed1(e1, 'lam0.03', RATIO)):.1f}" == "0.2"
    assert "costs 0.2 percentage points against ideal phases" in results_text()
    ab = load(AGG / "ablations.json")
    assert r3(seed1(e1, "lam0.03", LEAK)) == "0.174"
    assert {k: r3(ab[f"abl_{k}_lam0.03"]["metrics"][LEAK]["mean"]) for k in ("reversal", "cnn", "angleinput", "onestep", "antennainput")} == \
        {"reversal": "0.258", "cnn": "0.263", "angleinput": "0.277", "onestep": "0.185", "antennainput": "0.133"}
    assert r3(ab["abl_onestep_lam0.03"]["metrics"][RATIO]["mean"]) == "0.960"
    assert r3(ab["abl_antennainput_lam0.03"]["metrics"][RATIO]["mean"]) == "0.972"
    text = results_text()
    for q in (r"rises from 0.260 to 0.307, 0.550, and 0.774", r"stays between 0.168 and 0.197 and retains 89.1\,\% to 97.8\,\%",
              r"from 0.154 with $d_z=4$ to 0.186, 0.218, and 0.320", r"lowers it to 0.124, 0.141, 0.155, and 0.201",
              r"from 0.978 to 0.971 and 0.940, and ideal phases give 0.979", r"from 0.174 to 0.258, a convolutional encoder gives 0.263",
              r"an angle-only input gives 0.277", r"gives 0.185 at a rate ratio of 0.960", r"leaks 0.133 at a rate ratio of 0.972"):
        assert q in text, q

def test_optimizer_and_initialization_settings():
    """Optimizer, initialization and shuffling settings, recovered from the training code itself."""
    import inspect
    import torch
    from pabhbf.attacks.models import MLPRegressor, ResidualMLPRegressor, train_regressor
    from pabhbf.models.controller import SetController
    from pabhbf.train import pabhbf as trainer

    # no custom initialization anywhere, so every layer keeps the PyTorch default
    import pabhbf
    package = pathlib.Path(pabhbf.__file__).resolve().parent
    assert not [f for f in package.rglob("*.py") if "nn.init" in f.read_text(encoding="utf-8")]
    # no dropout: the attacker perceptrons default to zero and the controller sets it explicitly
    assert inspect.signature(MLPRegressor.__init__).parameters["dropout"].default == 0.0
    assert "Dropout" not in inspect.getsource(ResidualMLPRegressor)
    assert "dropout=0.0" in inspect.getsource(SetController.__init__)
    # the only dropout layer in the package is the one the attacker perceptron leaves switched off
    assert [f.relative_to(package).as_posix() for f in package.rglob("*.py")
            if "nn.Dropout(" in f.read_text(encoding="utf-8")] == ["attacks/models.py"]

    # AdamW keeps its default moments and numerical constant everywhere
    defaults = inspect.signature(torch.optim.AdamW).parameters
    assert defaults["betas"].default == (0.9, 0.999) and defaults["eps"].default == 1e-8
    for f in package.rglob("*.py"):
        s = f.read_text(encoding="utf-8")
        assert "betas=" not in s and "eps=1e" not in s, f

    # weight decay, fixed learning rate and gradient clipping of the encoder and the controllers
    src = inspect.getsource(trainer)
    assert "torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)" in src
    assert "clip_grad_norm_(params, 5.0)" in src
    assert "lr_scheduler" not in src  # the learning rate of the encoder and the controller is constant
    assert 'ap.add_argument("--adv-lr", type=float, default=1e-3)' in src
    assert "torch.optim.Adam(adv.parameters(), lr=args.adv_lr)" in src
    from pabhbf.train import baselines
    assert "weight_decay=1e-5" in inspect.getsource(baselines)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())

    # attackers: weight decay, cosine schedule over the configured epochs, and the two starting rates
    assert inspect.signature(train_regressor).parameters["weight_decay"].default == 1e-4
    fit_src = inspect.getsource(train_regressor)
    assert "CosineAnnealingLR(opt, T_max=epochs)" in fit_src
    assert "rng.permutation(len(Xtr))" in fit_src  # reshuffled at the start of every epoch
    from pabhbf import revision
    rev_src = inspect.getsource(revision.fit)
    assert "lr=2e-3" in rev_src and "opts['lr']=1e-3" in rev_src

    assert cfg["training"]["learning_rate"] == 1e-3 and cfg["training"]["batch_size"] == 512
