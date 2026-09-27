# Evaluation protocol

The numbers reported in the paper come from one pass of the protocol below. The pass is recorded in
`results/revision_final/`, and the audit tests recompute every reported value from those files.

## Data separation

| Split | Drops | Users | Seed | Role |
|---|---|---|---|---|
| `default_train` | 20,000 | 320,000 | 20260913 | encoder and controller training |
| `default_val` | 4,000 | 64,000 | 20260914 | policy validation, tool-router fitting |
| `attack_train` | 20,000 | 320,000 | 91526011 | attacker training |
| `attack_train_80k` | 80,000 | 1,280,000 | 91526011 | attacker training of the post-hoc scaling pass |
| `attack_val` | 2,000 | 32,000 | 91526012 | attacker validation, early stopping, operating-point selection |
| `final_test` | 2,000 | 32,000 | 91526013 | one-shot confirmatory test |

User positions are disjoint across the splits. `attack_train_80k` extends `attack_train`: its first 40 chunks are
byte-identical, which `python -m pabhbf.revision scaling-audit` verifies together with the disjointness.

## Selection

Eight seed-1 configurations are screened with attackers trained on 5,000 drops. Within the 128-bit and the 16-bit
family, the configuration with the largest validation rate ratio among those whose adjusted Wilson upper endpoint is
at most 0.20 is selected. The adjustment divides the error probability 0.05 by the eight configurations and the six
decision rules. Seed 1 is the tuning seed. Seeds 2 and 3 are independent replication fits evaluated after the rule is
fixed. The screening criterion is not a privacy budget: against the 20,000-drop attackers the selected 128-bit encoder
reaches 0.200, 0.183 and 0.201 on validation.

## Freeze and one-shot test

`python -m pabhbf.revision freeze` records the SHA-256 of every checkpoint, attacker, router, configuration and source
file in `results/revision_final/frozen.json`. The final-test geometry is generated only afterwards. The first call of
`python -m pabhbf.revision final-test --final-test` writes `results/revision_final/test_access.json` and later calls
refuse to run. Delete that file only when reproducing the pipeline from scratch in a fresh copy.

## Attack registry

| Window | Decision rules |
|---|---|
| T = 1 | Cartesian multilayer perceptron, polar multilayer perceptron, residual network, 20-neighbor regressor, cell classifier with the disc-MAP decision and with the posterior mean |
| T = 4 and T = 8 | mean-input multilayer perceptron, mean-input cell classifier with both decisions, DeepSets, and every rule of the shorter windows |

The reported leakage of a representation and a window is the largest hit probability within 1 m over the registry,
computed over the 20,000-drop and the 80,000-drop attackers. The reported RMSE is the smallest over the same registry.
The 5,000-drop attackers of the selection stage are kept for the data-scaling comparison of seed 1 only.

## Statistics

Tables and figures report means over the three encoder seeds with the half-width of a Student-t 95 % interval on two
degrees of freedom. Each attack also stores its success count, a Wilson interval and a percentile interval from 1,000
bootstrap resamples of whole drops.
