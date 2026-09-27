"""Tests for the Task 7-11 building blocks: quantizer, representations, encoders, controller, losses, grid decoding."""
import numpy as np
import pytest
import torch

from pabhbf.data import channel as ch
from pabhbf.losses import privacy as priv
from pabhbf.losses.task import wsr_from_shares
from pabhbf.models import representations as reps
from pabhbf.models.controller import SetController, decode_actions
from pabhbf.models.encoder import CNNEncoder, MLPEncoder
from pabhbf.models.quantizer import quantize_np, quantize_ste
from pabhbf.phy import codebook as cbm
from pabhbf.phy import hbf
from pabhbf.utils.config import load_config, system_params


@pytest.fixture(scope="module")
def p():
    return system_params(load_config())


@pytest.fixture(scope="module")
def csi(p):
    rng = np.random.default_rng(0)
    g = ch.sample_geometry(rng, 200, p)
    H = ch.draw_channels(rng, g, ch.path_responses(g, p), p)
    return g, H, ch.estimate_channels(rng, H, p.sigma_est2)


def test_quantizer_levels_and_straight_through_gradient():
    x = torch.linspace(-1.5, 1.5, 1001, requires_grad=True)
    q = quantize_ste(x, 3)
    assert len(torch.unique(q.detach())) == 8
    assert q.detach().abs().max() <= 1.0
    q.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    assert np.allclose(quantize_np(x.detach().numpy(), 3), q.detach().numpy(), atol=1e-6)
    assert quantize_ste(x, None) is x


@pytest.mark.parametrize("name,kw", [("raw_csi", {}), ("pca", {}), ("random_projection", {}), ("scalar_quant", {}),
                                     ("gaussian_pca", {"sigma": 0.3}), ("top_m_beams", {}), ("angle_domain", {}),
                                     ("laplace_pca", {"epsilon": 50.0})])
def test_representation_api_payload_and_shape(p, csi, name, kw):
    cfg = load_config()
    rng = np.random.default_rng(1)
    Hh = csi[2]
    rep = reps.make_representation(name, cfg, **kw).fit(Hh, p, rng)
    z = rep.transform(Hh, rng)
    assert z.shape == (Hh.shape[0] * Hh.shape[1], rep.dim) and z.dtype == np.float32 and np.all(np.isfinite(z))
    budget = cfg["representation"]["latent_dim"] * cfg["representation"]["bits_per_entry"]
    if name == "raw_csi":
        assert rep.estimated_bits() == 2 * p.Nt * cfg["representation"]["raw_csi_bits_per_entry"]
    else:
        assert rep.estimated_bits() <= budget
    if name == "scalar_quant":
        assert rep.estimated_bits() == budget and set(np.unique(z).tolist()) <= {-0.5, 0.5}


def test_beam_domain_features_keep_the_norm_and_remove_the_common_phase(p, csi):
    Hh = csi[2][:20]
    U = reps.dft_basis(p.Nt)
    assert np.allclose(U.conj().T @ U, np.eye(p.Nt), atol=1e-12)
    f = reps.csi_features(Hh, p).astype(np.float64)
    f_rot = reps.csi_features(Hh * np.exp(1j * 1.234), p).astype(np.float64)
    assert np.allclose(f, f_rot, atol=1e-5)
    scale = reps.csi_scale(p)
    assert np.allclose(np.sum(f**2, axis=1), scale**2 * np.sum(np.abs(Hh.reshape(-1, p.Nt)) ** 2, axis=1), rtol=1e-5)
    b = ch.ff_response(np.arcsin((2 * 5 - p.Nt + 1) / p.Nt), p.Nt, p.lam)
    x = reps.csi_features(b[None, :], p)[0]
    assert np.argmax(x[: p.Nt]) == 5 and abs(x[5] / scale - 1.0) < 1e-6


def test_linear_bases_are_orthonormal(p, csi):
    cfg = load_config()
    for name in ("pca", "random_projection"):
        rep = reps.make_representation(name, cfg).fit(csi[2], p, np.random.default_rng(2))
        assert np.allclose(rep.V.T @ rep.V, np.eye(rep.d), atol=1e-6)


def test_laplace_mechanism_noise_scale(p, csi):
    rep = reps.make_representation("laplace_pca", load_config(), epsilon=64.0, bits=None).fit(csi[2], p, np.random.default_rng(3))
    z = rep.transform(csi[2], np.random.default_rng(4))
    noise = z - np.clip(rep.project(csi[2]), -1.0, 1.0)
    assert np.mean(np.abs(noise)) == pytest.approx(2.0 * rep.d / 64.0, rel=0.02)


def test_encoders_output_quantized_latents(p):
    x = torch.randn(10, 2 * p.Nt)
    for enc in (MLPEncoder(2 * p.Nt, 16, bits=4), MLPEncoder(2 * p.Nt, 16, bits=4, kind="vib"), CNNEncoder(p.Nt, 16, bits=4)):
        z, kl = enc(x)
        assert z.shape == (10, 16) and kl.shape == (10,)
        levels = (z.detach().numpy().astype(np.float64) + 1.0) * 8.0 - 0.5
        assert np.allclose(levels, np.round(levels), atol=1e-5)


def test_controller_is_permutation_equivariant_and_decodes_feasible_actions(p):
    torch.manual_seed(0)
    ctrl = SetController(12, p.G, width=32, layers=2, heads=4).eval()
    Z = torch.randn(5, p.K, 12)
    w = torch.as_tensor(np.random.default_rng(0).choice([1.0, 2.0], (5, p.K)), dtype=torch.float32)
    snr = torch.full((5,), 10.0)
    perm = torch.randperm(p.K)
    with torch.no_grad():
        a = ctrl(Z, w, snr)
        b = ctrl(Z[:, perm], w[:, perm], snr)
    for x, y in zip(a, b):
        assert torch.allclose(x[:, perm], y, atol=1e-5)
    sched, groups, shares = decode_actions(*a, p.Ns)
    assert all(len(set(row.tolist())) == p.Ns for row in sched)
    assert torch.all((groups >= 0) & (groups < p.G))
    assert torch.allclose(shares.sum(1), torch.ones(5))


def test_rate_surrogate_matches_trusted_tools(p, csi):
    cb = cbm.polar_codebook(p)
    _, H, Hh = csi
    n = H.shape[0]
    rng = np.random.default_rng(4)
    sched = np.sort(np.stack([rng.permutation(p.K)[: p.Ns] for _ in range(n)]), axis=1)
    groups = hbf.gather_users(hbf.best_groups(Hh, cb), sched)
    shares = rng.dirichlet(np.ones(p.Ns), n)
    w_S = hbf.gather_users(rng.choice([1.0, 2.0], (n, p.K)), sched)
    rates = hbf.apply_tools(H, Hh, sched, groups, shares, cb, 10.0)
    _, FRF, Fbar = hbf.build_precoders(Hh, sched, groups, shares, cb, 10.0)
    A = hbf.coupling_matrix(hbf.gather_users(H, sched), FRF, Fbar)
    sur = wsr_from_shares(torch.as_tensor(A), torch.as_tensor(shares), torch.as_tensor(w_S), torch.full((n,), 0.1, dtype=torch.float64))
    assert np.allclose(sur.numpy(), np.sum(w_S * rates, axis=1))


def _toy_privacy_problem(seed):
    torch.manual_seed(seed)
    enc, att = torch.nn.Linear(3, 4).double(), torch.nn.Linear(4, 4).double()
    x = torch.randn(512, 3, dtype=torch.float64)
    cell = (x[:, 0] > 0).long() * 2 + (x[:, 1] > 0).long()
    log_prior = torch.log(torch.full((4,), 0.25, dtype=torch.float64))
    opt_a = torch.optim.Adam(att.parameters(), lr=0.05)
    for _ in range(200):
        opt_a.zero_grad()
        priv.adversary_loss(att(enc(x).detach()), cell).backward()
        opt_a.step()
    return enc, att, x, cell, log_prior


def test_adversary_learns_informative_cells():
    enc, att, x, cell, log_prior = _toy_privacy_problem(0)
    assert priv.adversary_loss(att(enc(x)), cell).item() < 0.8 * float(-log_prior[0])


@pytest.mark.parametrize("kind", ["maxent", "reversal"])
def test_encoder_privacy_gradient_has_the_right_sign(kind):
    """A small step of the encoder along the negative privacy gradient must bring the fixed adversary's posterior closer
    to the prior (maxent) or increase the adversary's cross-entropy (reversal)."""
    enc, att, x, cell, log_prior = _toy_privacy_problem(1)

    @torch.no_grad()
    def objective():
        logits = att(enc(x))
        if kind == "maxent":
            return float((log_prior.exp() * (log_prior - torch.log_softmax(logits, 1))).sum(1).mean())
        return -float(priv.adversary_loss(logits, cell))

    before = objective()
    grads = torch.autograd.grad(priv.privacy_loss(att(enc(x)), cell, log_prior, kind), list(enc.parameters()))
    with torch.no_grad():
        for prm, g in zip(enc.parameters(), grads):
            prm -= 1e-4 * g
    assert objective() < before


def test_grid_indexing_and_disc_map(p):
    rng = np.random.default_rng(5)
    r, th = ch.sample_positions(rng, 50000, p)
    xy = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    grid = priv.make_grid(xy, 0.5, 1.0, p.r_max, p.theta_max)
    assert np.isclose(grid.prior.sum(), 1.0)
    idx = grid.index(xy)
    assert np.all(np.linalg.norm(grid.centers[idx] - xy, axis=1) <= 0.5 * np.sqrt(2) / 2 + 1e-9)
    target = rng.integers(0, grid.C, 20)
    probs = np.full((20, grid.C), 1e-9)
    probs[np.arange(20), target] = 1.0
    est = priv.disc_map(probs / probs.sum(1, keepdims=True), grid)
    assert np.all(np.linalg.norm(est - grid.centers[target], axis=1) <= 1.0 + 1e-9)
