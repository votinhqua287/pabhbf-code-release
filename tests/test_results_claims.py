"""Historical regression audit against the immutable pre-revision manuscript.

The original common-subset statistics are development results, not the revised
headline claims. Preserve their regression checks against the original snapshot;
confirmatory claim consistency is checked by test_final_claims.py.
"""
import glob
import json
import zipfile

import numpy as np
import pytest

from pabhbf.utils.config import ROOT

AGG = ROOT / "results" / "aggregated"
RAW = ROOT / "results" / "raw"
LEAK, RATIO, RMSE, WSR = "privacy.leakage_common", "utility.ratio", "privacy.rmse_common_m", "utility.wsr"


def historical_text(section):
    with zipfile.ZipFile(ROOT / "results/current_snapshot/original_artifacts.zip") as z:
        return z.read(f"manuscript/sections/{section}.tex").decode("utf-8")


def load(path):
    if not path.exists():
        pytest.skip(f"{path} missing")
    return json.loads(path.read_text())


def agg(exp):
    return load(AGG / f"{exp}.json")


def mean(d, tag, key):
    return d[tag]["metrics"][key]["mean"]


def seed1(d, tag, key):
    g = d[tag]
    return g["metrics"][key]["values"][g["seeds"].index(1)]


def pct(x):
    return f"{100 * x:.1f}"


def r3(x):
    return f"{x:.3f}"


def latest(pattern):
    files = sorted(glob.glob(str(RAW / pattern)))
    if not files:
        pytest.skip(f"{pattern} missing")
    return json.loads(open(files[-1], encoding="utf-8").read())


def test_teacher_check_in_the_setup():
    tq = load(AGG / "teacher_quality_nt256.json")
    assert sum(round(r["frac_optimal"] * 300) for r in tq["small"]) == 1799
    assert f"{100 * min(r['worst_ratio'] for r in tq['small']):.2f}" == "99.76"
    d10 = next(r for r in tq["default"] if r["snr_db"] == 10.0)
    assert f"{d10['teacher']:.2f}" == "128.46" and pct(d10["teacher"] / d10["strongest_users"] - 1) == "8.3"


def test_privacy_weight_sweep():
    e1 = agg("e1")
    assert all(len(e1[t]["seeds"]) == 3 for t in ("lam0", "lam0.01", "lam0.03", "lam0.1", "lam0.3", "lam1"))
    assert pct(mean(e1, "lam0", RATIO)) == "98.9" and r3(mean(e1, "lam0", LEAK)) == "0.282"
    assert r3(e1["lam0"]["metrics"][LEAK]["ci95"]) == "0.039"
    assert pct(mean(e1, "lam0.03", RATIO)) == "97.2" and r3(mean(e1, "lam0.03", LEAK)) == "0.173"
    assert r3(e1["lam0.03"]["metrics"][LEAK]["ci95"]) == "0.019"
    assert r3(mean(e1, "lam0.3", LEAK)) == "0.162" and r3(mean(e1, "lam0.03", RATIO)) == "0.972"
    assert r3(mean(e1, "lam0.3", RATIO)) == "0.938"
    assert r3(mean(e1, "lam1", LEAK)) == "0.136" and r3(mean(e1, "lam1", RATIO)) == "0.825"
    r1 = e1["lam1"]["metrics"][RATIO]["values"]
    assert r3(min(r1)) == "0.724" and r3(max(r1)) == "0.891"
    s1 = [seed1(e1, t, LEAK) for t in ("lam0.03", "lam0.1", "lam0.3")]
    assert f"{np.ceil((max(s1) - min(s1)) * 1000) / 1000:.3f}" == "0.009"


def test_comparison_with_the_baselines():
    e1, b = agg("e1"), agg("baselines")
    high = [t for t in ("raw_csi", "pca", "random_projection", "scalar_quant", "top_m_beams", "autoencoder", "angle_domain")
            if mean(b, t, RATIO) > 0.9]
    assert r3(min(mean(b, t, LEAK) for t in high)) == "0.424"
    assert f"{mean(b, 'top_m_beams', RMSE):.2f}" == "0.73" and r3(mean(b, "top_m_beams", LEAK)) == "0.873"
    assert r3(mean(b, "raw_csi", LEAK)) == "0.775"
    assert r3(mean(b, "autoencoder", LEAK)) == "0.544" and pct(mean(b, "autoencoder", RATIO)) == "97.9"
    assert r3(mean(b, "random_projection", LEAK)) == "0.518" and r3(mean(b, "angle_domain", LEAK)) == "0.424"
    assert pct(mean(b, "pca", RATIO)) == "69.8" and r3(mean(b, "pca", LEAK)) == "0.197"
    assert mean(e1, "lam0.03", RATIO) > mean(b, "pca", RATIO) and mean(e1, "lam0.03", LEAK) < mean(b, "pca", LEAK)
    assert r3(mean(b, "scalar_quant", LEAK)) == "0.136" and pct(mean(b, "scalar_quant", RATIO)) == "66.7"
    assert r3(mean(e1, "lam1", LEAK)) == r3(mean(b, "scalar_quant", LEAK)) and pct(mean(e1, "lam1", RATIO)) == "82.5"
    assert r3(seed1(b, "gaussian_pca_0.4", LEAK)) == "0.071" and pct(seed1(b, "gaussian_pca_0.4", RATIO)) == "48.4"
    assert r3(seed1(b, "laplace_pca_10", LEAK)) == "0.008" and pct(seed1(b, "laplace_pca_10", RATIO)) == "22.1"


def test_per_distance_leakage():
    pab = latest("e1/lam0.03_s1_*.json")["privacy"]["summary"]["bins_best"]["hit_prob"]
    pab0 = latest("e1/lam0_s1_*.json")["privacy"]["summary"]["bins_best"]["hit_prob"]
    assert [pct(pab[i]) for i in (0, 5, 7)] == ["43.8", "8.9", "23.6"]
    assert [pct(pab0[i]) for i in (0, 5, 7)] == ["51.7", "22.3", "38.6"]
    prem = latest("premise/premise_mle_nt256p_*.json")
    nf = next(r for r in prem["runs"] if r["model"] == "near_field" and r["attacker"] == "mle_fusion")
    assert min(nf["bins"]["hit_prob"]) >= 0.83


def test_budget_constrained_training():
    d, tr = agg("e1d"), load(AGG / "training_summary_e1d.json")
    expect = {"dual0.05": (("0.053", "0.037"), ("0.77", "0.95"), ("0.150", "0.178"), ("0.898", "0.834")),
              "dual0.1": (("0.107", "0.091"), ("0.08", "0.15"), ("0.178", "0.175"), ("0.966", "0.960")),
              "dual0.15": (("0.151", "0.145"), ("0", "0"), ("0.202", "0.211"), ("0.988", "0.989"))}
    for tag, (hits, mus, leaks, ratios) in expect.items():
        assert d[tag]["seeds"] == [1, 2]
        assert tuple(r3(tr[f"{tag}_s{s}"]["hit_last"]) for s in (1, 2)) == hits
        # weight used in the last adversarial epoch, which produced the evaluated encoder
        last = [load(RAW / "train_logs" / "pabhbf" / tr[f"{tag}_s{s}"]["file"])["history"][-1]["lam"] for s in (1, 2)]
        assert tuple("0" if v == 0 else f"{v:.2f}" for v in last) == mus
        assert tuple(r3(v) for v in d[tag]["metrics"][LEAK]["values"]) == leaks
        assert tuple(r3(v) for v in d[tag]["metrics"][RATIO]["values"]) == ratios


def test_latent_dimension():
    e2, e1, b = agg("e2"), agg("e1"), agg("baselines")
    assert r3(seed1(e2, "dz4_lam0", LEAK)) == "0.155" and r3(seed1(e2, "dz64_lam0", LEAK)) == "0.320"
    assert r3(seed1(e2, "autoencoder_dz4", LEAK)) == "0.254" and r3(seed1(e2, "autoencoder_dz64", LEAK)) == "0.582"
    assert r3(seed1(e2, "dz4_lam0.03", LEAK)) == "0.127"
    assert r3(seed1(e2, "dz16_lam0", LEAK)) == "0.218" and r3(seed1(e2, "dz16_lam0.03", LEAK)) == "0.155"
    assert r3(seed1(e2, "dz64_lam0.03", LEAK)) == "0.201"
    ratios = [seed1(e2, f"dz{d}_lam0.03", RATIO) for d in (4, 8, 16, 64)] + [seed1(e1, "lam0.03", RATIO)]
    assert pct(min(ratios)) == "96.3" and pct(max(ratios)) == "97.8"
    assert f"{seed1(e2, 'autoencoder_dz4', RMSE):.2f}" == "4.01" and f"{seed1(e2, 'autoencoder_dz64', RMSE):.2f}" == "1.54"
    rm = [seed1(e2, f"dz{d}_lam0.03", RMSE) for d in (4, 8, 16, 64)] + [seed1(e1, "lam0.03", RMSE)]
    assert f"{min(rm):.2f}" == "3.27" and f"{max(rm):.2f}" == "3.87"
    pca = {4: seed1(e2, "pca_dz4", LEAK), 8: seed1(e2, "pca_dz8", LEAK), 16: seed1(e2, "pca_dz16", LEAK),
           32: seed1(b, "pca", LEAK), 64: seed1(e2, "pca_dz64", LEAK)}
    pab = {d: seed1(e2, f"dz{d}_lam0.03", LEAK) for d in (4, 8, 16, 64)} | {32: seed1(e1, "lam0.03", LEAK)}
    assert [d for d in pca if pca[d] < pab[d]] == [4, 8, 16]
    assert pct(max(seed1(e2, f"pca_dz{d}", RATIO) for d in (4, 8, 16))) == "62.2"
    # Section VIII-C: for seed 1, the leakage of every scheme grows with the payload, and the adversary lowers it at every d_z
    dims = (4, 8, 16, 32, 64)
    series = {"pab": pab, "pab0": {d: seed1(e2, f"dz{d}_lam0", LEAK) for d in (4, 8, 16, 64)} | {32: seed1(e1, "lam0", LEAK)},
              "autoencoder": {d: seed1(e2, f"autoencoder_dz{d}", LEAK) for d in (4, 8, 16, 64)} | {32: seed1(b, "autoencoder", LEAK)},
              "pca": pca}
    assert all(all(s[a] < s[c] for a, c in zip(dims, dims[1:])) for s in series.values())
    assert all(series["pab"][d] < series["pab0"][d] for d in dims)


def test_reconstruction_attacker():
    e1, b = agg("e1"), agg("baselines")
    assert [f"{mean(b, t, 'privacy.decoder.nmse_db'):.1f}" for t in ("raw_csi", "autoencoder", "random_projection")] == ["-20.6", "-3.8", "-3.2"]
    nmse = [mean(e1, t, "privacy.decoder.nmse_db") for t in ("lam0.01", "lam0.03", "lam0.1", "lam0.3", "lam1")]
    assert f"{min(nmse):.2f}" == "-0.33" and f"{max(nmse):.2f}" == "-0.14"


def test_snr_sweep():
    from pabhbf.plots import paper_figures as pf

    ts = load(AGG / "teacher_sweep_default.json")
    snrs, n = ts["snr_db"], ts["n_drops"]
    assert n == 2000
    teacher = np.asarray(ts["teacher_wsr"])
    e1, b = agg("e1"), agg("baselines")
    ranges = {}
    for exp, a, tag in (("e1", e1, "lam0.03"), ("e1", e1, "lam0"), ("baselines", b, "raw_csi"), ("baselines", b, "autoencoder"),
                        ("baselines", b, "pca")):
        y = pf._per_snr(exp, a, tag, snrs, n)
        ranges[tag] = (pct(np.min(y / teacher)), pct(np.max(y / teacher)))
        if tag == "lam0.03":
            assert f"{y[-1]:.2f}" == "139.81"
    assert ranges == {"lam0.03": ("96.2", "97.3"), "lam0": ("98.1", "99.1"), "raw_csi": ("97.5", "98.5"),
                      "autoencoder": ("96.2", "98.5"), "pca": ("65.0", "71.5")}
    assert f"{teacher[-1]:.2f}" == "143.86" and f"{ts['strongest_wsr'][-1]:.2f}" == "132.09"


def test_phase_bits():
    pb, e1 = agg("phase_bits"), agg("e1")
    teacher = {b: load(AGG / f"teacher_sweep_default_bits{b}_stored.json")["teacher_wsr"][0] for b in ("2", "3", "ideal")}
    teacher["4"] = mean(e1, "lam0", "utility.teacher_wsr")
    assert [f"{teacher[b]:.2f}" for b in ("2", "3", "ideal", "4")] == ["100.31", "103.57", "104.62", "104.37"]

    def ratio(tag, b):
        wsr = seed1(e1, tag, WSR) if b == "4" else seed1(pb, f"{tag}_bits{b}", WSR)
        return wsr / teacher[b]

    assert sorted({pct(ratio("lam0.03", b)) for b in teacher}) == ["97.7", "97.8"]
    r0 = [ratio("lam0", b) for b in teacher]
    assert pct(min(r0)) == "98.7" and pct(max(r0)) == "98.9"


def test_temporal_attacker():
    rows = {}
    for f in sorted(glob.glob(str(RAW / "e_temporal" / "*.json"))):
        d = json.loads(open(f, encoding="utf-8").read())
        rows[d["checkpoint"].replace("\\", "/").split("/")[-1].rsplit("_s", 1)[0]] = [r3(d["results"][T]["summary"]["leakage"]) for T in ("1", "2", "4")]
    if not rows:
        pytest.skip("temporal results missing")
    assert rows["lam0.03"] == ["0.152", "0.182", "0.201"]
    assert rows["lam0"] == ["0.191", "0.247", "0.292"]
    assert rows["raw_csi"] == ["0.432", "0.479", "0.580"]


def test_mismatch():
    mm, e1, b = agg("mismatch"), agg("e1"), agg("baselines")
    assert r3(seed1(mm, "lam0.03_mm_range", LEAK)) == "0.220" and r3(seed1(e1, "lam0.03", LEAK)) == "0.174"
    assert pct(seed1(mm, "lam0.03_mm_range", RATIO)) == "97.4"
    assert r3(seed1(mm, "autoencoder_mm_range", LEAK)) == "0.609" and r3(seed1(mm, "raw_csi_mm_range", LEAK)) == "0.803"
    nominal = {"lam0.03": seed1(e1, "lam0.03", LEAK), "lam0": seed1(e1, "lam0", LEAK), "raw_csi": seed1(b, "raw_csi", LEAK),
               "pca": seed1(b, "pca", LEAK), "autoencoder": seed1(b, "autoencoder", LEAK)}
    assert all(seed1(mm, f"{t}_mm_range", LEAK) > v for t, v in nominal.items())
    assert all(seed1(mm, f"{t}_mm_channel", LEAK) < v for t, v in nominal.items())
    assert [r3(seed1(mm, f"{t}_mm_channel", LEAK)) for t in ("lam0.03", "autoencoder", "raw_csi")] == ["0.144", "0.278", "0.380"]
    assert pct(seed1(e1, "lam0.03", RATIO)) == "97.8" and pct(seed1(mm, "lam0.03_mm_channel", RATIO)) == "92.8"
    assert [pct(seed1(mm, f"{t}_mm_channel", RATIO)) for t in ("lam0", "autoencoder", "raw_csi")] == ["97.0", "96.5", "96.2"]


def test_orchestration():
    ag = latest("e13_agentic/*.json")
    p = ag["planners"]
    assert [r3(p[k]["ratio"]) for k in ("direct", "rule", "search2", "search4")] == ["0.976", "0.979", "0.983", "0.987"]
    assert round(100 * p["rule"]["calls_per_drop"]["run_wmmse_refinement"]) == 42
    assert [p[k]["exposure_bits_per_drop"] for k in ("search2", "search4")] == [16.0, 32.0]
    assert ag["learned_fit"]["label_counts"][0] == 1706 and p["learned"]["choice_counts"] == [2000, 0, 0]
    zero = [json.loads(open(f, encoding="utf-8").read()) for f in sorted(glob.glob(str(RAW / "e13_agentic_costs" / "*.json")))]
    z = next(d for d in zero if d["learned_fit"]["mu_latency"] == 0 and d["learned_fit"]["nu_bits"] == 0)["planners"]["learned"]
    assert z["choice_counts"] == [331, 1190, 479] and r3(z["ratio"]) == "0.982" and f"{z['exposure_bits_per_drop']:.1f}" == "7.7"
    others = [d for d in zero if d["learned_fit"]["nu_bits"] in (0.001, 0.005)]
    assert len(others) == 2 and all(d["planners"]["learned"]["choice_counts"] == [2000, 0, 0] for d in others)


def test_ablations():
    ab, e1 = agg("ablations"), agg("e1")
    ref_leak, ref_ratio = seed1(e1, "lam0.03", LEAK), seed1(e1, "lam0.03", RATIO)
    assert r3(ref_leak) == "0.174" and r3(ref_ratio) == "0.978"
    assert r3(seed1(ab, "abl_continuous_lam0.03", LEAK)) == "0.354" and r3(seed1(ab, "abl_continuous_lam0.03", RATIO)) == "0.965"
    assert r3(seed1(ab, "abl_reversal_lam0.03", RATIO)) == "0.989" and r3(seed1(ab, "abl_reversal_lam0.03", LEAK)) == "0.258"
    assert min(seed1(ab, t, RATIO) for t in ("abl_cnn_lam0.03", "abl_angleinput_lam0.03")) >= 0.975
    assert [r3(seed1(ab, t, LEAK)) for t in ("abl_cnn_lam0.03", "abl_angleinput_lam0.03")] == ["0.263", "0.277"]
    assert r3(seed1(ab, "abl_antennainput_lam0.03", LEAK)) == "0.133" and r3(seed1(ab, "abl_antennainput_lam0.03", RATIO)) == "0.972"
    small = ("abl_vib_lam0.03", "abl_norate_lam0.03", "abl_onestep_lam0.03")
    assert r3(max(abs(seed1(ab, t, LEAK) - ref_leak) for t in small)) == "0.011"
    assert r3(max(abs(seed1(ab, t, RATIO) - ref_ratio) for t in small)) == "0.018"


def test_training_versus_evaluation_attackers():
    e1, tr = agg("e1"), load(AGG / "training_summary_e1.json")
    hit = {mu: np.mean([tr[f"lam{mu}_s{s}"]["hit_last"] for s in (1, 2, 3)]) for mu in ("0.03", "0.3")}
    assert r3(hit["0.03"]) == "0.129" and r3(hit["0.3"]) == "0.070"
    assert r3(mean(e1, "lam0.03", LEAK)) == "0.173" and r3(mean(e1, "lam0.3", LEAK)) == "0.162"


def test_training_behaviour_of_the_privacy_weight_sweep():
    """Validation rate ratio over the 20 adversarial epochs, from the training logs in results/raw/train_logs/pabhbf."""
    tr = load(AGG / "training_summary_e1.json")

    def curve(mu, s):
        h = load(RAW / "train_logs" / "pabhbf" / tr[f"lam{mu}_s{s}"]["file"])["history"]
        return [r["val"]["ratio"] for r in h if r["phase"] == "adversarial"]

    for mu, expected, early in (("0.03", ("0.921", "0.930", "0.966", "0.978"), False),
                                ("0.3", ("0.715", "0.793", "0.917", "0.958"), True),
                                ("1", ("0.431", "0.540", "0.725", "0.884"), True)):
        curves = [curve(mu, s) for s in (1, 2, 3)]
        assert all(len(c) == 20 and min(c) < c[-1] for c in curves)
        mins, lasts = [min(c) for c in curves], [c[-1] for c in curves]
        assert (r3(min(mins)), r3(max(mins)), r3(min(lasts)), r3(max(lasts))) == expected
        if early:
            assert all(int(np.argmin(c)) < 4 for c in curves)


def test_mutual_information_bound():
    e1, b = agg("e1"), agg("baselines")
    mi = "privacy.grid_map.mi_lower_bound_bits"
    assert [f"{mean(b, t, mi):.2f}" for t in ("raw_csi", "autoencoder")] == ["7.38", "6.55"]
    assert [f"{mean(e1, t, mi):.2f}" for t in ("lam0", "lam0.03")] == ["4.57", "3.53"]
    for s in (1, 2, 3):
        g = latest(f"e1/lam0.03_s{s}_*.json")["privacy"]["attackers"]["grid_map"]
        assert g["cells"] == 1736 and f"{g['prior_entropy_nats'] / np.log(2):.2f}" == "10.71"


def test_small_latents_with_three_seeds():
    """Section VII-C: the payload and the adversary lower the leakage jointly (d_z = 4 and 8, three seeds)."""
    e1, e2, b = agg("e1"), agg("e2"), agg("baselines")
    assert all(e2[t]["seeds"] == [1, 2, 3] for t in ("dz4_lam0", "dz4_lam0.03", "dz8_lam0", "dz8_lam0.03"))

    def ci(t):
        return r3(e2[t]["metrics"][LEAK]["ci95"])

    assert (r3(mean(e2, "dz4_lam0", LEAK)), ci("dz4_lam0"), pct(mean(e2, "dz4_lam0", RATIO))) == ("0.154", "0.004", "98.6")
    assert mean(e2, "dz4_lam0", LEAK) < mean(e1, "lam0.03", LEAK) and mean(e2, "dz4_lam0", RATIO) > mean(e1, "lam0.03", RATIO)
    assert (r3(mean(e2, "dz4_lam0.03", LEAK)), ci("dz4_lam0.03"), pct(mean(e2, "dz4_lam0.03", RATIO))) == ("0.124", "0.008", "96.7")
    assert (r3(mean(e2, "dz8_lam0", LEAK)), r3(mean(e2, "dz8_lam0.03", LEAK)), ci("dz8_lam0.03"),
            pct(mean(e2, "dz8_lam0.03", RATIO))) == ("0.186", "0.141", "0.005", "96.5")
    assert mean(e2, "dz4_lam0.03", LEAK) < mean(b, "scalar_quant", LEAK) and pct(mean(b, "scalar_quant", RATIO)) == "66.7"
    text = historical_text("results")
    for q in (r"leaks $0.154\pm0.004$ at 98.6\,\% of the teacher rate, whereas the 128-bit representation with $\mu=0.03$ leaks $0.173\pm0.019$ at 97.2\,\%",
              r"falls to $0.124\pm0.008$ at 96.7\,\%, which is below the leakage of B4", r"falls from 0.186 to $0.141\pm0.005$ at 96.5\,\%",
              r"For seed 1, the leakage of PAB-HBF, B7, and B2 grows with the payload from 16 to 256 bits",
              r"the adversary lowers the leakage of PAB-HBF at every latent dimension"):
        assert q in text, q


def test_attack_data_study():
    """Section VII-E and the attack-data table: leakage of seed-1 representations versus the number of attack drops."""
    import math

    def run(tag, n):
        d = load(RAW / "attack_scaling" / f"{tag}_n{n}.json")
        assert d["epochs"] == 200 and d["patience"] == 20 and d["attack_drops"] == n
        return d["leakage_common"]

    e1, e2, b = agg("e1"), agg("e2"), agg("baselines")
    d20 = max(abs(run(t, 20000) - seed1(g, t, LEAK)) for g, t in ((e1, "lam0.03"), (e1, "lam0"), (e2, "dz4_lam0"), (b, "raw_csi")))
    assert f"{math.ceil(d20 * 1000) / 1000:.3f}" == "0.007"
    assert [r3(run("lam0.03", n)) for n in (20000, 80000)] == ["0.173", "0.188"]
    assert [r3(run("dz4_lam0", n)) for n in (5000, 20000, 80000)] == ["0.147", "0.154", "0.158"]
    assert [r3(run("lam0", n)) for n in (20000, 80000)] == ["0.285", "0.355"]
    assert all(run(t, 5000) < run(t, 20000) < run(t, 80000) for t in ("lam0.03", "lam0", "dz4_lam0"))
    assert [r3(run("raw_csi", n)) for n in (5000, 20000)] == ["0.473", "0.775"]
    text = historical_text("results")
    for q in (r"changes the leakage by at most 0.007", r"rises from 0.173 to 0.188 with $\mu=0.03$ and from 0.285 to 0.355 with $\mu=0$",
              r"stays between 0.147 and 0.158 for all three attack sets",
              r"rises from 0.473 with 5,000 drops to 0.775 with 20,000 drops", r"for at most 200 epochs with a patience of 20"):
        assert q in text, q
    assert r"so the reported values are lower bounds that depend on the size of the attack set" in text


def test_quoted_strings_appear_in_the_manuscript():
    """The expectations above must be the strings of the manuscript, not only the values of the result files."""
    text = "".join(historical_text(s) for s in ("results", "conclusion"))
    quoted = [
        r"1799 of these 1800 cases", r"99.76\,\%", r"128.46\,bit/s/Hz", r"8.3\,\%",
        r"98.9\,\% of the teacher rate with a leakage of $0.282\pm0.039$", r"$0.173\pm0.019$ at 97.2\,\%",
        r"from 0.173 to 0.162, whereas the rate ratio falls from 0.972 to 0.938", r"the leakage reaches 0.136 and the rate ratio 0.825",
        r"falls from 0.129 to 0.070", r"changes by at most 0.009 between $\mu=0.03$ and $\mu=0.3$",
        r"to no less than 0.921 for $\mu=0.03$ and to 0.431 for $\mu=1$",
        r"leak at least 0.424", r"B6 leaks 0.873, more than the 0.775 of raw CSI", r"RMSE of 0.73\,m", r"leaks 0.544 at 97.9\,\%",
        r"keeps 69.8\,\% of the teacher rate and leaks 0.197", r"0.071 at 48.4\,\% for B5 with $\sigma_{\mathrm{B5}}=0.4$",
        r"0.008 at 22.1\,\% for B9 with $\varepsilon_{\mathrm{DP}}=10$", r"1,736 cells of 0.5\,m", r"$H(c_k)=10.71$\,bits",
        r"7.38\,bits for B1, 4.57\,bits for PAB-HBF with $\mu=0$, and 3.53\,bits with $\mu=0.03$",
        r"$-20.6$\,dB on B1 and $-3.8$\,dB on B7", r"between $-0.33$ and $-0.14$\,dB",
        r"43.8\,\% of the users between 3.0 and 5.1\,m", r"against 51.7\,\% without the adversary",
        r"0.053 and 0.037, 0.107 and 0.091, and 0.151 and 0.145", r"0.150 and 0.178, 0.178 and 0.175, and 0.202 and 0.211",
        r"between 96.2\,\% and 97.3\,\%", r"leaks 0.220 instead of 0.174 and keeps 97.4\,\%", r"raw CSI leaks 0.803",
        r"from 97.8\,\% to 92.8\,\% and its leakage to 0.144",
        r"between 97.7\,\% and 97.8\,\% of the teacher rate recomputed at the same resolution",
        r"from 0.152 to 0.201", r"the 0.432 of raw CSI in a single interval",
        r"from 0.174 to 0.354", r"leaks 0.258 at a rate ratio of 0.989", r"leakage to 0.133 at a rate ratio of 0.972",
        r"from 0.976 for the direct planner to 0.979", r"0.983 with $C=2$ and 0.987 with $C=4$", r"at 16 and 32 feedback bits per drop",
        r"rate ratio of 0.982 with 7.7 feedback bits", r"selects the direct planner for every test drop",
        r"need 1.1\,ms per drop, against 36.8\,ms for the teacher",
    ]
    missing = [q for q in quoted if q not in text]
    assert not missing, missing


def test_array_size():
    nt, e1, b = agg("nt_sweep"), agg("e1"), agg("baselines")

    def val(kind, n, key):
        if n == 256:
            return seed1(e1 if kind != "raw_csi" else b, {"pab": "lam0.03", "pab0": "lam0", "raw_csi": "raw_csi"}[kind], key)
        return seed1(nt, f"nt{n}_{ {'pab': 'lam0.03', 'pab0': 'lam0', 'raw_csi': 'raw_csi'}[kind] }", key)

    nts = (32, 64, 128, 256)
    assert [r3(val("raw_csi", n, LEAK)) for n in nts] == ["0.260", "0.307", "0.550", "0.774"]
    pab = [val("pab", n, LEAK) for n in nts]
    assert r3(min(pab)) == "0.168" and r3(max(pab)) == "0.197"
    assert all(val("raw_csi", n, LEAK) - val("pab", n, LEAK) < val("raw_csi", m, LEAK) - val("pab", m, LEAK) for n, m in zip(nts, nts[1:]))
    assert [r3(val("pab", n, RATIO)) for n in nts] == ["0.891", "0.936", "0.962", "0.978"]
    pab0 = [val("pab0", n, RATIO) for n in nts]
    assert r3(min(pab0)) == "0.956" and r3(max(pab0)) == "0.989"
    prem = {}
    for n, pat in ((32, "premise_mle_nt32p_*.json"), (64, "premise_mle_nt64_*.json"), (128, "premise_mle_nt128p_*.json"), (256, "premise_mle_nt256p_*.json")):
        d = latest(f"premise/{pat}")
        prem[n] = {m: next(r["hit_prob"] for r in d["runs"] if r["model"] == m and r["attacker"] == "mle_fusion") for m in ("near_field", "far_field")}
    assert r3(prem[32]["near_field"]) == "0.214" and r3(prem[256]["near_field"]) == "0.921"
    ff = [prem[n]["far_field"] for n in nts]
    assert r3(min(ff)) == "0.208" and r3(max(ff)) == "0.218"
    text = historical_text("results")
    for q in (r"rises from 0.260 to 0.307, 0.550, and 0.774", r"from 0.214 to 0.921", r"between 0.208 and 0.218",
              r"between 0.168 and 0.197", r"so the gap to B1 widens as the aperture grows",
              r"rises from 0.891 to 0.978"):
        assert q in text, q


def test_abstract_introduction_and_conclusion_numbers():
    e1, b, nt = agg("e1"), agg("baselines"), agg("nt_sweep")
    prem = latest("premise/premise_mle_nt256p_*.json")
    fusion = {m: next(r["hit_prob"] for r in prem["runs"] if r["model"] == m and r["attacker"] == "mle_fusion") for m in ("near_field", "far_field")}
    assert pct(fusion["near_field"]) == "92.1" and pct(fusion["far_field"]) == "20.8"
    assert pct(mean(e1, "lam0.03", RATIO)) == "97.2" and pct(mean(e1, "lam0.03", LEAK)) == "17.3" and r3(mean(e1, "lam0.03", LEAK)) == "0.173"
    assert pct(mean(b, "raw_csi", LEAK)) == "77.5" and pct(mean(b, "autoencoder", LEAK)) == "54.4" and pct(mean(b, "pca", RATIO)) == "69.8"
    e2 = agg("e2")
    assert all(e2[t]["seeds"] == [1, 2, 3] for t in ("dz4_lam0", "dz4_lam0.03"))
    assert pct(mean(e2, "dz4_lam0.03", RATIO)) == "96.7" and pct(mean(e2, "dz4_lam0.03", LEAK)) == "12.4"
    assert r3(mean(e2, "dz4_lam0.03", LEAK)) == "0.124" and pct(mean(e2, "dz4_lam0", LEAK)) == "15.4"
    pab = [seed1(nt, f"nt{n}_lam0.03", LEAK) for n in (32, 64, 128)] + [seed1(e1, "lam0.03", LEAK)]
    raw = [seed1(nt, f"nt{n}_raw_csi", LEAK) for n in (32, 64, 128)] + [seed1(b, "raw_csi", LEAK)]
    assert (r3(min(pab)), r3(max(pab)), r3(raw[0]), r3(raw[-1])) == ("0.168", "0.197", "0.260", "0.774")
    abstract, intro, concl = (historical_text(s) for s in ("abstract", "introduction", "conclusion"))
    for q in (r"localize 77.5\,\% of the users within 1\,m from raw CSI", r"54.4\,\% from an autoencoder",
              r"keeps 69.8\,\% of the teacher rate", r"keeps 97.2\,\% and 96.7\,\% of this rate",
              r"localize 17.3\,\% and 12.4\,\% of the users", r"within 0.5\,m"):
        assert q in abstract, q
    for q in (r"localize 92.1\,\% of the users within 1\,m, against 20.8\,\%", r"retains 97.2\,\% of the rate", r"leakage of 0.173",
              r"leaks at least 0.424", r"lowers the leakage further to 0.124 at 96.7\,\% of the teacher rate"):
        assert q in intro, q
    for q in (r"retained 97.2\,\% of the teacher rate", r"localized 17.3\,\% of the users", r"77.5\,\% for raw CSI",
              r"localize 15.4\,\% of the users", r"fell to 12.4\,\% at 96.7\,\% of the teacher rate",
              r"between 0.168 and 0.197 from 32 to 256 antennas", r"grew from 0.260 to 0.774"):
        assert q in concl, q


def test_overhead_and_latency():
    e1, e2, b = agg("e1"), agg("e2"), agg("baselines")
    bench = load(AGG / "runtime_benchmark.json")
    ag = latest("e13_agentic/*.json")
    pab0 = [seed1(e2, f"dz{d}_lam0", RATIO) for d in (4, 8, 16, 64)] + [seed1(e1, "lam0", RATIO)]
    pca = [seed1(e2, f"pca_dz{d}", RATIO) for d in (4, 8, 16, 64)] + [seed1(b, "pca", RATIO)]
    assert (pct(min(pab0)), pct(max(pab0))) == ("98.8", "98.9")
    assert pct(seed1(e2, "autoencoder_dz4", RATIO)) == "88.0" and pct(max(pca)) == "87.8"
    assert [f"{ag['planners'][k]['latency_ms_per_drop']:.1f}" for k in ("direct", "rule", "search2", "search4")] == ["1.6", "1.8", "2.8", "3.9"]
    s = bench["schemes"]
    assert bench["threads"] == 5 and bench["n_drops"] == 1000
    assert f"{s['lam0.03']['latency_ms_per_drop']:.1f}" == "1.1" and f"{bench['teacher']['teacher_ms_per_drop']:.1f}" == "36.8"
    assert f"{bench['teacher']['decision_ms_per_drop']:.1f}" == "35.8"
    lats = {k: v["latency_ms_per_drop"] for k, v in s.items()}
    assert f"{min(lats.values()):.1f}" == "1.1" and f"{max(lats.values()):.1f}" == "2.3" and max(lats, key=lats.get) == "top_m_beams"
    text = historical_text("results")
    for q in (r"need 1.1\,ms per drop, against 36.8\,ms for the teacher",
              r"for a batch of 1,000 test drops (seed 1, median of three repeats with 5 threads)"):
        assert q in text, q


def test_premise_distance_statements():
    bins = {}
    for n, pat in ((64, "premise_mle_nt64_*.json"), (256, "premise_mle_nt256p_*.json")):
        d = latest(f"premise/{pat}")
        bins[n] = {m: next(r["bins"] for r in d["runs"] if r["model"] == m and r["attacker"] == "mle_fusion") for m in ("near_field", "far_field")}
    edges = bins[64]["near_field"]["edges"]
    near, far = bins[64]["near_field"]["hit_prob"], bins[64]["far_field"]["hit_prob"]
    beyond = [a - b for a, e, b in zip(near, edges, far) if e >= 7.25 - 1e-9]
    assert r3(max(abs(x) for x in beyond)) == "0.062" and f"{near[0]:.2f}" == "0.67" and f"{far[0]:.2f}" == "0.14"
    for n in (64, 256):
        f = bins[n]["far_field"]["hit_prob"]
        assert int(np.argmax(f)) == 5 and f"{edges[5]:.1f}" == "13.6" and f"{edges[6]:.2f}" == "15.75"
    text = historical_text("results")
    for q in (r"concentrated below 7.3\,m", r"reaches 0.67 in the closest bin against 0.14 for the plane-wave channel"):
        assert q in text, q
