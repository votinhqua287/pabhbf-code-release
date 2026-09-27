"""Audit test: the payloads and sizes stated in Table II match the representation code."""
import numpy as np

from pabhbf.data import channel as ch
from pabhbf.models import representations as rp
from pabhbf.plots.make_tables import num, table_methods
from pabhbf.utils.config import load_config, system_params


def test_table_ii_payloads_match_the_representations():
    cfg = load_config()
    p = system_params(cfg)
    rep = cfg["representation"]
    B = rep["latent_dim"] * rep["bits_per_entry"]
    rng = np.random.default_rng(0)
    g = ch.sample_geometry(rng, 4, p)
    H = ch.draw_channels(rng, g, ch.path_responses(g, p), p)
    Hh = ch.estimate_channels(rng, H, p.sigma_est2)
    info = table_methods()

    sq = rp.ScalarQuantCSI(B).fit(Hh, p, rng)
    assert sq.estimated_bits() == B and sq.dim // 2 == info["b4_entries"]
    tm = rp.TopMBeams(B, 4).fit(Hh, p, rng)
    assert tm.estimated_bits() == B and tm.m == info["b6_codewords"]
    ad = rp.AngleDomain(rep["latent_dim"], rep["bits_per_entry"]).fit(Hh, p, rng)
    assert ad.cb_ff.M == info["plane_wave_beams"] and ad.estimated_bits() == B
    assert rp.RawCSI(rep["raw_csi_bits_per_entry"]).fit(Hh, p, rng).estimated_bits() == 2 * p.Nt * rep["raw_csi_bits_per_entry"]


def test_negative_numbers_use_a_math_minus_sign():
    assert num(-0.84) == "$-$0.84"
    assert num(0.292, 3) == "0.292"
    assert num(float("nan")) == "--"
