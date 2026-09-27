"""Tests for Task 5: teacher actions and their quality."""
import numpy as np
import pytest

from pabhbf.data import channel as ch
from pabhbf.phy import codebook as cbm
from pabhbf.phy import hbf, teacher
from pabhbf.utils.config import load_config, system_params


def _setup(overrides=None, n=300, seed=0, snr_db=10.0):
    p = system_params(load_config(overrides=overrides or {}))
    cb = cbm.polar_codebook(p)
    rng = np.random.default_rng(seed)
    g = ch.sample_geometry(rng, n, p)
    H = ch.draw_channels(rng, g, ch.path_responses(g, p), p)
    Hh = ch.estimate_channels(rng, H, p.sigma_est2)
    w = rng.choice([1.0, 2.0], size=(n, p.K))
    return p, cb, rng, H, Hh, w, 10 ** (snr_db / 10)


def test_teacher_outputs_are_valid_and_power_optimization_helps():
    p, cb, rng, H, Hh, w, snr = _setup()
    out = teacher.teacher_actions(H, Hh, w, snr, cb, p.Ns)
    assert out["sched"].shape == (300, p.Ns)
    assert all(len(set(r)) == p.Ns for r in out["sched"]) and np.all(np.diff(out["sched"], axis=1) > 0)
    assert np.all(out["shares"] >= 0) and np.allclose(out["shares"].sum(1), 1)
    assert np.all(np.isfinite(out["wsr_true"]))
    eq = hbf.apply_tools(Hh, Hh, out["sched"], out["groups_S"], np.full((300, p.Ns), 1 / p.Ns), cb, snr)
    wsr_eq = np.sum(hbf.gather_users(w, out["sched"]) * eq, axis=1)
    assert np.mean(out["wsr_est"]) >= np.mean(wsr_eq) - 1e-9


def test_teacher_clearly_beats_random_and_strongest_user_heuristic():
    p, cb, rng, H, Hh, w, snr = _setup(seed=1)
    out = teacher.teacher_actions(H, Hh, w, snr, cb, p.Ns)
    s, g, sh = teacher.random_actions(rng, 300, p.K, p.Ns, cb.G)
    wsr_rand = np.sum(hbf.gather_users(w, s) * hbf.apply_tools(H, Hh, s, g, sh, cb, snr), axis=1)
    s2, g2, sh2 = teacher.strongest_users_actions(Hh, w, cb, p.Ns)
    wsr_str = np.sum(hbf.gather_users(w, s2) * hbf.apply_tools(H, Hh, s2, g2, sh2, cb, snr), axis=1)
    assert out["wsr_true"].mean() > 1.5 * wsr_rand.mean()
    assert out["wsr_true"].mean() > wsr_str.mean()


def test_greedy_is_close_to_exhaustive_on_small_instances():
    over = {"K": 8, "Ns": 3, "Nrf": 3}
    p, cb, rng, H, Hh, w, snr = _setup(overrides=over, n=200, seed=2)
    gains = hbf.codeword_gains(Hh, cb)
    groups_all = hbf.best_groups(Hh, cb)
    greedy = teacher.greedy_schedule(Hh, w, snr, cb, p.Ns, groups_all, gains)
    v_greedy = teacher.estimated_wsr_equal_power(Hh, gains, w, np.full(200, snr), greedy, groups_all, cb)
    _, v_exh = teacher.exhaustive_schedule(Hh, w, snr, cb, p.Ns)
    assert np.all(v_greedy <= v_exh + 1e-9)
    assert v_greedy.mean() >= 0.97 * v_exh.mean()
