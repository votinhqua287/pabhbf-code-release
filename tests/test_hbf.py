"""Tests for Task 4: codebooks, beam groups, trusted tools, SINR and rates."""
import numpy as np
import pytest

from pabhbf.data import channel as ch
from pabhbf.phy import codebook as cbm
from pabhbf.phy import hbf
from pabhbf.utils.config import load_config, system_params


@pytest.fixture(scope="module")
def p():
    return system_params(load_config())


@pytest.fixture(scope="module")
def cb(p):
    return cbm.polar_codebook(p)


def test_beta_delta_solves_fresnel_equation():
    for level in (0.5, 0.8, 0.9):
        b = cbm.beta_delta(level)
        assert float(cbm.fresnel_correlation(b)) == pytest.approx(level, abs=1e-8)
    assert cbm.beta_delta(0.5) == pytest.approx(1.556, abs=1e-3)


def test_tau_delta_matches_exact_correlation(p):
    td = cbm.tau_delta(256, p.lam, 0.5)
    r1 = 20.0  # the second point needs 1/r1 > tau_delta (a larger r1 made its range negative)
    b1 = ch.nf_response(r1, 0.0, 256, p.lam)
    b2 = ch.nf_response(1 / (1 / r1 - td), 0.0, 256, p.lam)
    c = abs(np.vdot(b1, b2)) / (np.linalg.norm(b1) * np.linalg.norm(b2))
    assert c == pytest.approx(0.5, abs=0.03)


def test_codebook_is_feasible_distinct_and_grouped(p, cb):
    assert np.allclose(np.abs(cb.W), 1 / np.sqrt(p.Nt))
    q = np.angle(cb.W) / (2 * np.pi / 2**p.phase_bits)
    assert np.allclose(q, np.round(q), atol=1e-9)
    assert len({tuple(np.round(w, 9)) for w in cb.W}) == cb.M
    assert cb.group.min() == 0 and cb.group.max() == cb.G - 1
    assert np.all(np.diff(cb.u_edges) > 0)
    counts = np.bincount(cb.group, minlength=cb.G)
    assert counts.min() >= 1


def test_one_bit_codebook_has_no_duplicates(p):
    cb1 = cbm.polar_codebook(p, bits=1, G=7)
    assert len({tuple(np.round(w, 9)) for w in cb1.W}) == cb1.M


def _drop(p, n, seed, snr_db=10.0):
    rng = np.random.default_rng(seed)
    g = ch.sample_geometry(rng, n, p)
    H = ch.draw_channels(rng, g, ch.path_responses(g, p), p)
    Hh = ch.estimate_channels(rng, H, p.sigma_est2)
    return rng, g, H, Hh


def test_single_user_rate_matches_closed_form(p, cb):
    rng, g, H, Hh = _drop(p, 50, 1)
    sched = np.zeros((50, 1), int)
    groups = cb.group[np.argmax(np.abs(Hh[:, 0] @ cb.W.conj().T), axis=1)][:, None]
    snr = 10.0
    rates, info = hbf.apply_tools(H, Hh, sched, groups, np.ones((50, 1)), cb, snr, return_all=True)
    c = cb.W[info["idx"][:, 0]]
    closed = np.log2(1 + snr * np.abs(np.sum(H[:, 0].conj() * c, axis=1)) ** 2 / np.sum(np.abs(c) ** 2, axis=1))
    assert np.allclose(rates[:, 0], closed)


def test_power_constraint_and_assignment_rules(p, cb):
    rng, g, H, Hh = _drop(p, 300, 2)
    S = p.Ns
    sched = np.stack([rng.permutation(p.K)[:S] for _ in range(300)])
    groups = rng.integers(0, 2, (300, S))  # crowd two groups to exercise the widening rule
    shares = rng.dirichlet(np.ones(S), 300)
    rates, info = hbf.apply_tools(H, Hh, sched, groups, shares, cb, 10.0, return_all=True)
    FRF, FBB, idx = info["FRF"], info["FBB"], info["idx"]
    assert np.allclose(np.sum(np.abs(FRF @ FBB) ** 2, axis=(1, 2)), 1.0)
    assert np.allclose(np.sum(np.abs(FRF @ FBB) ** 2, axis=1), shares)
    assert all(len(set(row)) == S for row in idx)
    assert np.all(np.linalg.matrix_rank(FRF) == S)
    # the highest-power user always receives a codeword of its own group (nothing is assigned before it)
    top = np.argmax(shares, axis=1)
    assert np.all(cb.group[idx[np.arange(300), top]] == groups[np.arange(300), top])


def test_rate_increases_with_snr_in_los(p, cb):
    cfg = load_config(overrides={"channel": {"rician_k_db": 60.0}, "estimation": {"pilot_snr_db": 60.0}})
    q = system_params(cfg)
    rng, g, H, Hh = _drop(q, 40, 3)
    sched = np.tile(np.arange(q.Ns), (40, 1))
    groups = cb.group[np.argmax(np.abs(np.einsum("mt,nst->nsm", cb.W.conj(), Hh[:, : q.Ns])), axis=2)]
    shares = np.full((40, q.Ns), 1 / q.Ns)
    sums = [hbf.apply_tools(H, Hh, sched, groups, shares, cb, 10 ** (s / 10)).sum(1).mean() for s in (-10, 0, 10, 20)]
    assert np.all(np.diff(sums) > 0)


def test_zero_forcing_cancels_interference_with_perfect_csi(p, cb):
    rng, g, H, _ = _drop(p, 30, 4)
    S = p.Ns
    sched = np.tile(np.arange(S), (30, 1))
    groups = cb.group[np.argmax(np.abs(np.einsum("mt,nst->nsm", cb.W.conj(), H[:, :S])), axis=2)]
    rates, info = hbf.apply_tools(H, H, sched, groups, np.full((30, S), 1 / S), cb, 1e3, precoder="zf", return_all=True)
    G = np.einsum("nst,ntr,nrk->nsk", H[:, :S].conj(), info["FRF"], info["FBB"])
    off = np.abs(G - np.eye(S)[None] * G)
    assert off.max() < 1e-8 * np.abs(G).max()


def test_rzf_is_stable_for_nearly_collinear_users(p, cb):
    rng = np.random.default_rng(5)
    h = ch.nf_response(10.0, 0.2, p.Nt, p.lam) * np.sqrt(p.Nt)
    H = np.stack([h, h * (1 + 1e-9), ch.nf_response(12.0, -0.5, p.Nt, p.lam) * np.sqrt(p.Nt)])[None]
    sched = np.array([[0, 1, 2]])
    groups = cb.group[np.argmax(np.abs(H[0] @ cb.W.conj().T), axis=1)][None]
    rates = hbf.apply_tools(H, H, sched, groups, np.full((1, 3), 1 / 3), cb, 100.0)
    assert np.all(np.isfinite(rates)) and np.all(rates >= 0)


def test_sinr_formula_matches_signal_level_simulation():
    rng = np.random.default_rng(6)
    nt, S, trials, s2 = 16, 3, 300000, 0.2
    cplx = lambda *shape: (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    FRF = np.exp(1j * rng.uniform(0, 2 * np.pi, (1, nt, S))) / np.sqrt(nt)
    FBB = cplx(1, S, S)
    H_S = cplx(1, S, nt)
    s = cplx(trials, S)
    x = s @ (FRF[0] @ FBB[0]).T
    k = 1
    y = x @ H_S[0, k].conj() + np.sqrt(s2) * cplx(trials)
    desired = (H_S[0, k].conj() @ FRF[0] @ FBB[0, :, k]) * s[:, k]
    emp = np.mean(np.abs(desired) ** 2) / np.mean(np.abs(y - desired) ** 2)
    assert hbf.sinr(H_S, FRF, FBB, s2)[0, k] == pytest.approx(emp, rel=0.02)
