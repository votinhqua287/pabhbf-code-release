# Final numbers

Means and Student-t 95% CI half-widths over three fitted model seeds. Test: 2,000 drops / 32,000 users.

| Scheme | Rate ratio | Leakage T=1 | Leakage T=4 | Leakage T=8 |
|---|---:|---:|---:|---:|
| lam0.01 | 0.9848 | 0.2139 | 0.3466 | 0.4525 |
| dz4_lam0 | 0.9859 | 0.1581 | 0.1968 | 0.2226 |
| lam0 | 0.9889 | 0.3492 | 0.5324 | 0.6526 |
| dz4_lam0.03 | 0.9647 | 0.1236 | 0.1482 | 0.1599 |
| raw_csi | 0.9833 | 0.7692 | not evaluated | not evaluated |
| pca | 0.7003 | 0.1984 | not evaluated | not evaluated |
| random_projection | 0.9210 | 0.5330 | not evaluated | not evaluated |
| scalar_quant | 0.6653 | 0.1373 | not evaluated | not evaluated |
| autoencoder | 0.9793 | 0.5525 | not evaluated | not evaluated |
| angle_domain | 0.9813 | 0.4201 | not evaluated | not evaluated |

Exact inputs and all attacker outcomes: `results/revision_final/final_summary.json` and `attack_registry.json`.

Finite-test Wilson intervals are working-binomial summaries. Drop-bootstrap intervals preserve common drop context. Neither is the across-seed Student-t interval or an upper bound on an unrestricted attacker.
