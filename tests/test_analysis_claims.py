"""Audit tests: numbers stated in Sections VII and VIII of the manuscript, recomputed from the code."""
import numpy as np
import pytest

from pabhbf.attacks.crb import location_crb
from pabhbf.data.channel import response
from pabhbf.utils.config import load_config, system_params
from pabhbf.utils.units import fresnel_distance, rayleigh_distance


def fresnel_fisher(nt, lam, u, tau, snr):
    """Numerical Fisher information of (u, tau) for the Fresnel phase model with a nuisance complex gain."""
    delta = np.arange(1, nt + 1) - (nt + 1) / 2

    def b(uu, tt):
        return np.exp(1j * (np.pi * delta * uu - np.pi * lam * delta**2 * tt / 4)) / np.sqrt(nt)

    h = 1e-7
    D = np.stack([(b(u + h, tau) - b(u - h, tau)) / (2 * h), (b(u, tau + h) - b(u, tau - h)) / (2 * h)], axis=1)
    bb = b(u, tau)
    P = np.eye(nt) - np.outer(bb, bb.conj())
    return 2 * nt * snr * np.real(D.conj().T @ P @ D)


@pytest.mark.parametrize("nt", [32, 64, 256])
def test_proposition_1_closed_form_matches_numerical_fisher_information(nt):
    lam, snr = 0.0107069, 36.4
    J = fresnel_fisher(nt, lam, u=0.3, tau=0.08, snr=snr)
    assert abs(J[0, 1]) < 1e-6 * np.sqrt(J[0, 0] * J[1, 1])
    assert J[0, 0] == pytest.approx(np.pi**2 * snr * nt * (nt**2 - 1) / 6, rel=1e-4)
    assert J[1, 1] == pytest.approx(np.pi**2 * lam**2 * snr * nt * (nt**2 - 1) * (nt**2 - 4) / 1440, rel=1e-4)


def test_closed_form_range_bound_agrees_with_exact_model_at_10_m():
    p = system_params(load_config("nt64.yaml"))
    r, snr = 10.0, p.kappa / (p.kappa + 1) * (10.0 / p.r_ref) ** (-p.eta) / p.sigma_est2
    closed = np.sqrt(r**4 * 1440 / (np.pi**2 * p.lam**2 * snr * p.Nt * (p.Nt**2 - 1) * (p.Nt**2 - 4)))
    exact_phase_only = np.sqrt(location_crb("near_field", r, 0.0, p.Nt, p.lam, snr, False, p.eta, 0.0)[0][0, 0])
    exact_with_power = np.sqrt(location_crb("near_field", r, 0.0, p.Nt, p.lam, snr, True, p.eta, p.shadow_std_db)[0][0, 0])
    assert closed == pytest.approx(0.571, abs=0.002)
    assert exact_phase_only == pytest.approx(closed, rel=0.01)
    assert exact_with_power == pytest.approx(0.567, abs=0.002)


@pytest.mark.parametrize("nt,r,expected", [(32, 5.0, 0.40), (64, 10.0, 0.567), (128, 20.0, 0.805), (256, 20.0, 0.143)])
def test_table_of_range_bounds(nt, r, expected):
    p = system_params(load_config(overrides={"Nt": nt}))
    snr = p.kappa / (p.kappa + 1) * (r / p.r_ref) ** (-p.eta) / p.sigma_est2
    std = np.sqrt(location_crb("near_field", r, 0.0, nt, p.lam, snr, True, p.eta, p.shadow_std_db)[0][0, 0])
    assert std == pytest.approx(expected, rel=0.01)


def test_plane_wave_range_information_comes_only_from_power():
    p = system_params(load_config())
    r = 10.0
    crb, _ = location_crb("far_field", r, 0.0, p.Nt, p.lam, 50.0, True, p.eta, p.shadow_std_db)
    assert np.sqrt(crb[0, 0]) == pytest.approx(p.shadow_std_db * np.log(10) * r / (10 * p.eta), rel=1e-3)
    assert p.shadow_std_db * np.log(10) / (10 * p.eta) == pytest.approx(0.46, abs=0.005)


def test_geometry_numbers_of_the_setup():
    p = system_params(load_config())
    assert p.Nt == 256
    assert p.D == pytest.approx(1.365, abs=0.001)
    assert rayleigh_distance(p.D, p.lam) == pytest.approx(348.1, abs=0.1)
    assert fresnel_distance(p.D, p.lam) == pytest.approx(7.7, abs=0.05)
    assert p.area == pytest.approx(409.5, abs=0.05)
    assert np.pi * 1.0**2 / p.area == pytest.approx(0.0077, abs=0.00005)
    assert np.log2(p.area / np.pi) == pytest.approx(7.03, abs=0.005)


@pytest.mark.parametrize("nt,expected", [(64, 0.83), (256, 13.7)])
def test_curvature_phase_at_the_array_edge_for_10_m(nt, expected):
    p = system_params(load_config(overrides={"Nt": nt}))
    edge = (p.Nt - 1) / 2 * p.d
    phase = 2 * np.pi / p.lam * edge**2 / (2 * 10.0)
    assert phase == pytest.approx(expected, abs=0.01 * expected + 0.005)
    b = response("near_field", np.array([10.0]), np.array([0.0]), p.Nt, p.lam)[0]
    assert abs(np.angle(b[0] * np.conj(b[p.Nt // 2]) * np.exp(1j * 0))) > 0  # the exact response has a curvature phase


def test_overhead_ratio_and_payload():
    cfg = load_config()
    p = system_params(cfg)
    rep = cfg["representation"]
    assert rep["latent_dim"] * rep["bits_per_entry"] == 128
    assert 2 * p.Nt * rep["raw_csi_bits_per_entry"] / (rep["latent_dim"] * rep["bits_per_entry"]) == 128


def test_sector_edge_range_bound_for_256_antennas():
    p = system_params(load_config())
    r, th = 20.0, np.deg2rad(60.0)
    snr = p.kappa / (p.kappa + 1) * (r / p.r_ref) ** (-p.eta) / p.sigma_est2
    hybrid = np.sqrt(location_crb("near_field", r, th, p.Nt, p.lam, snr, True, p.eta, p.shadow_std_db)[0][0, 0])
    assert hybrid == pytest.approx(0.568, abs=0.002) and hybrid < 0.57


def test_revision_numbers_quoted_in_the_manuscript():
    """Section VII-B and VIII-B quote these values from results/aggregated/revision_checks.json."""
    import json

    from pabhbf.utils.config import ROOT

    path = ROOT / "results" / "aggregated" / "revision_checks.json"
    if not path.exists():
        pytest.skip("run python -m pabhbf.attacks.revision_checks first")
    d = json.loads(path.read_text())
    cover = d["cover_neighborhood_0p5m"]
    assert cover["discs"] == 744 and cover["bits"] == 10 and d["cover_region_1m"]["discs"] == 179
    assert round(100 * d["mle_hits_nt256"]["hit_0p5"], 1) == 81.5
    bias = {(r["Nt"], r["kappa_db"]): r["median_abs_range_error_m"] for r in d["multipath_range_error"]}
    assert round(bias[(64, 10.0)], 2) == 1.39 and bias[(64, 60.0)] < 0.01 and round(bias[(256, 10.0)], 2) == 0.03


def test_lattice_covers_are_valid_and_need_10_and_8_bits():
    """Proposition 2 uses the discs of a triangular lattice whose centers lie within delta + radius of the region. Every
    sampled point of the delta-neighborhood of the region must lie inside a kept disc, and the counts fit 10 and 8 bits."""
    from scipy.spatial import cKDTree

    from pabhbf.attacks.revision_checks import distance_to_region, hexagonal_cover

    p = system_params(load_config())
    rng = np.random.default_rng(5)
    for radius, delta, bits in ((0.5, 0.5, 10), (1.0, 0.0, 8)):
        cover = hexagonal_cover(p, radius, delta, rng, trials=300)
        assert cover["bits"] == bits and cover["discs"] <= 2 ** bits
        a, h = np.sqrt(3.0) * radius, 1.5 * radius
        ext = p.r_max + delta + 2 * radius
        ii, jj = np.meshgrid(np.arange(-int(ext / a) - 4, int(ext / a) + 5), np.arange(-int(ext / h) - 4, int(ext / h) + 5),
                             indexing="ij")
        centers = np.stack([ii * a + (jj % 2) * a / 2, jj * h], -1).reshape(-1, 2) + rng.uniform(0, [a, 2 * h], 2)
        kept = centers[distance_to_region(centers, p) <= delta + radius]
        pts = np.column_stack([rng.uniform(0.0, ext, 600000), rng.uniform(-ext, ext, 600000)])
        pts = pts[distance_to_region(pts, p) <= delta]
        d, _ = cKDTree(kept).query(pts)
        assert len(pts) > 100000 and d.max() <= radius * (1 + 1e-9)


def test_sector_edge_and_plane_wave_rows_of_the_bound_table():
    """Table tab:crb: the plane-wave bound does not depend on the angle, and the 256-antenna bound at the sector edge
    stays below 0.57 m for every range up to 20 m."""
    p = system_params(load_config())
    th = np.deg2rad(60.0)
    snr = lambda r: p.kappa / (p.kappa + 1) * (r / p.r_ref) ** (-p.eta) / p.sigma_est2  # noqa: E731
    ff = [np.sqrt(location_crb("far_field", 20.0, t, p.Nt, p.lam, snr(20.0), True, p.eta, p.shadow_std_db)[0][0, 0]) for t in (0.0, th)]
    assert ff[0] == pytest.approx(9.21, abs=0.005) and ff[1] == pytest.approx(ff[0], rel=1e-6)
    near = [np.sqrt(location_crb("near_field", r, th, p.Nt, p.lam, snr(r), True, p.eta, p.shadow_std_db)[0][0, 0])
            for r in np.linspace(p.r_min, 20.0, 35)]
    assert max(near) < 0.57 and np.all(np.diff(near) > 0)
