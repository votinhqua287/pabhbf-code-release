"""Tests for the WMMSE refinement tool."""
import numpy as np
import pytest

from pabhbf.data import channel as ch
from pabhbf.phy import codebook as cbm
from pabhbf.phy import hbf
from pabhbf.phy.wmmse import wmmse_digital
from pabhbf.utils.config import load_config, system_params


@pytest.fixture(scope="module")
def setup():
    p = system_params(load_config())
    cb = cbm.polar_codebook(p)
    rng = np.random.default_rng(7)
    n = 200
    g = ch.sample_geometry(rng, n, p)
    H = ch.draw_channels(rng, g, ch.path_responses(g, p), p)
    Hh = ch.estimate_channels(rng, H, p.sigma_est2)
    sched = np.sort(np.stack([rng.permutation(p.K)[: p.Ns] for _ in range(n)]), axis=1)
    groups = hbf.gather_users(hbf.best_groups(Hh, cb), sched)
    shares = np.full((n, p.Ns), 1.0 / p.Ns)
    alpha = hbf.gather_users(rng.choice([1.0, 2.0], (n, p.K)), sched)
    return p, cb, H, Hh, sched, groups, shares, alpha


@pytest.mark.parametrize("snr_db", [0.0, 20.0])
def test_wmmse_meets_power_and_improves_estimated_weighted_sum_rate(setup, snr_db):
    p, cb, H, Hh, sched, groups, shares, alpha = setup
    snr = 10 ** (snr_db / 10)
    _, FRF, Fbar = hbf.build_precoders(Hh, sched, groups, shares, cb, snr)
    FBB0 = hbf.power_scaling(FRF, Fbar, shares)
    Hh_S = hbf.gather_users(Hh, sched)
    FBB = wmmse_digital(Hh_S, FRF, FBB0, alpha, 1.0 / snr, iters=15)
    assert np.all(np.linalg.norm(FRF @ FBB, axis=(1, 2)) ** 2 <= 1.0 + 1e-6)
    wsr0 = np.sum(alpha * np.log2(1 + hbf.sinr(Hh_S, FRF, FBB0, 1.0 / snr)), axis=1)
    wsr1 = np.sum(alpha * np.log2(1 + hbf.sinr(Hh_S, FRF, FBB, 1.0 / snr)), axis=1)
    assert wsr1.mean() >= wsr0.mean()
    assert np.mean(wsr1 >= wsr0 - 1e-6) > 0.95
