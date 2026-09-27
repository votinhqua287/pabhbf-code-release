"""Tests for Task 2 (units, config) and Task 3 (channel generators)."""
import numpy as np
import pytest

from pabhbf.data import channel as ch
from pabhbf.utils.config import load_config, system_params
from pabhbf.utils.units import c2r, db2lin, lin2db, r2c, rayleigh_distance, seed_everything, wavelength


@pytest.fixture(scope="module")
def p():
    return system_params(load_config())


def test_db_round_trip_and_invalid_input():
    x = np.array([-30.0, -3.0, 0.0, 7.5, 40.0])
    assert np.allclose(lin2db(db2lin(x)), x)
    with pytest.raises(ValueError):
        lin2db(0.0)


def test_wavelength_rayleigh_and_derived_parameters(p):
    assert wavelength(28e9) == pytest.approx(0.0107069, rel=1e-5)
    assert p.d == pytest.approx(p.lam / 2)
    assert p.D == pytest.approx((p.Nt - 1) * p.lam / 2)
    assert p.rayleigh == pytest.approx(rayleigh_distance(p.D, p.lam))
    q = system_params(load_config("nt64.yaml"))
    assert q.D == pytest.approx(63 * q.lam / 2) and q.rayleigh == pytest.approx(21.25, abs=0.01)
    assert system_params(load_config()).rayleigh == pytest.approx(348.1, abs=0.1) if p.Nt == 256 else True
    assert p.sigma_est2 == pytest.approx(0.1)
    assert p.area == pytest.approx(np.deg2rad(60) * (20.0**2 - 3.0**2))


def test_complex_real_round_trip():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 3, 8)) + 1j * rng.standard_normal((5, 3, 8))
    x = c2r(z)
    assert x.shape == (5, 3, 16) and np.allclose(r2c(x), z)


def test_seeding_is_reproducible():
    a = seed_everything(7).standard_normal(4)
    b = seed_everything(7).standard_normal(4)
    assert np.array_equal(a, b)


def test_one_antenna_and_norm_limits(p):
    assert np.allclose(ch.nf_response(5.0, 0.3, 1, p.lam), 1.0)
    far = ch.nf_response(1e4, 0.4, p.Nt, p.lam)
    assert np.linalg.norm(far) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(ch.ff_response(0.4, p.Nt, p.lam)) == pytest.approx(1.0)


def test_near_field_converges_to_far_field(p):
    corr = [abs(np.vdot(ch.ff_response(0.4, p.Nt, p.lam), ch.nf_response(r, 0.4, p.Nt, p.lam))) for r in (3, 10, 50, 500, 5000)]
    assert np.all(np.diff(corr) > 0) and corr[-1] > 0.9999


def test_phase_progression_matches_geometry(p):
    r, th = 7.0, -0.35
    b = ch.nf_response(r, th, p.Nt, p.lam)
    dist = ch.distances(r, th, p.Nt, p.d)
    expected = np.exp(-1j * 2 * np.pi / p.lam * (dist - r))
    assert np.allclose(b / np.abs(b), expected)
    assert np.allclose(np.abs(b) * np.sqrt(p.Nt), r / dist)


def test_positions_and_scatterers_stay_in_region(p):
    rng = np.random.default_rng(1)
    g = ch.sample_geometry(rng, 2000, p)
    assert np.all(ch.in_region(g.r, g.theta, p)) and np.all(ch.in_region(g.r_sc, g.theta_sc, p))
    xy = g.xy()
    sc_xy = np.stack([g.r_sc * np.cos(g.theta_sc), g.r_sc * np.sin(g.theta_sc)], -1)
    assert np.max(np.linalg.norm(sc_xy - xy[:, :, None, :], axis=-1)) <= p.r_sc + 1e-9
    # uniform in area: E[r^2] = (r_min^2 + r_max^2) / 2
    assert np.mean(g.r**2) == pytest.approx((p.r_min**2 + p.r_max**2) / 2, rel=0.02)


def test_channel_power_normalization(p):
    rng = np.random.default_rng(2)
    g = ch.sample_geometry(rng, 200, p)
    resp = ch.path_responses(g, p)
    acc = np.zeros_like(g.beta)
    n_draw = 400
    for _ in range(n_draw):
        acc += np.sum(np.abs(ch.draw_channels(rng, g, resp, p)) ** 2, axis=-1)
    ratio = (acc / n_draw) / (p.Nt * g.beta)
    assert np.mean(ratio) == pytest.approx(1.0, abs=0.02)


def test_far_field_model_depends_on_range_only_through_gain_and_scatterers(p):
    b1 = ch.response("far_field", 4.0, 0.2, p.Nt, p.lam)
    b2 = ch.response("far_field", 19.0, 0.2, p.Nt, p.lam)
    assert np.allclose(b1, b2)
    n1 = ch.response("near_field", 4.0, 0.2, p.Nt, p.lam)
    n2 = ch.response("near_field", 19.0, 0.2, p.Nt, p.lam)
    assert abs(np.vdot(n1, n2)) < 0.99


def test_estimation_error_variance(p):
    rng = np.random.default_rng(3)
    H = np.zeros((4000, 4, p.Nt), complex)
    Hh = ch.estimate_channels(rng, H, p.sigma_est2)
    assert np.mean(np.abs(Hh) ** 2) == pytest.approx(p.sigma_est2, rel=0.01)
    with pytest.raises(ValueError):
        ch.estimate_channels(rng, H, -1.0)
