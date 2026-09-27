# PAB-HBF: privacy-aware CSI abstraction for near-field hybrid beamforming

Simulation code for a near-field XL-MIMO base station that delegates scheduling, beam-group and power decisions to an
external controller while exposing only a short quantized representation of the channel state information. The package
reproduces the reported figures and tables from the frozen result files, and it runs the full pipeline from channel
generation to the confirmatory test.

## Contents

| Path | Content |
|---|---|
| `src/pabhbf/data` | near-field and plane-wave channel generators, dataset generation with teacher labels |
| `src/pabhbf/phy` | polar codebook, beam assignment, regularized zero-forcing, power scaling, greedy teacher, WMMSE |
| `src/pabhbf/models` | beam-domain encoder with the 4-bit quantizer, set-transformer controller, autoencoder baseline |
| `src/pabhbf/losses` | communication loss, prior-matching adversarial loss, leakage definitions |
| `src/pabhbf/train` | encoder and controller training, baseline training |
| `src/pabhbf/attacks` | attacker models and training, model-based near-field attacker, Cramer-Rao bound, scaling study |
| `src/pabhbf/agents` | trusted toolbox, rule-based and search planners, learned router, protected interface |
| `src/pabhbf/eval` | evaluation, aggregation, teacher quality, runtime benchmarks |
| `src/pabhbf/revision.py` | split preparation, attacker fitting, selection, freeze, one-shot confirmatory test |
| `src/pabhbf/plots` | figure and table generation |
| `configs/` | `default.yaml` with every physical and training parameter, and the sweep overrides |
| `scripts/` | experiment queues, the confirmatory pipeline, figure and table generation |
| `tests/` | unit tests and the audit tests that recompute every reported number |
| `results/` | frozen protocol files, per-attack results of the confirmatory test, aggregated development results |
| `results/checkpoints/pabhbf/` | encoder and controller checkpoints of the four reported operating points, three seeds each |
| `matlab/` | independent implementation of the channel, the precoder and the rate, used by the cross-check |
| `docs/PROTOCOL.md` | data separation, selection rule, freeze, attack registry, statistics |
| `FINAL_NUMBERS.md` | the reported rate ratios and leakage values in one table |

Large intermediates are not shipped: the channel datasets (about 25 GB), the fitted attacker models (3.3 GB) and the
checkpoints of the development sweeps. The commands below regenerate them.

## Installation

```
conda env create -f environment.yml
conda activate privacy-nf-hbf
```

or, with an existing Python 3.13 environment,

```
pip install -r requirements.txt
```

The code is CPU-only. MATLAB is optional and used by the channel cross-check in `matlab/`.

## Reproducing the reported figures and tables

The frozen result files are included, so this needs no training:

```
python scripts/make_paper_figures.py
python -m pytest
```

The first command writes `outputs/figures/` (leakage against distance for raw CSI, the privacy-utility comparison, and
leakage against the observation window) and `outputs/tables/` (final-test performance, hit probability of every
decision rule, and tool execution). The second runs the unit tests and the audit tests, which recompute every reported
number from `results/` and fail when a value drifts.

## Running the pipeline

Set `PYTHONPATH=src` and run from the package root. Runtimes are for a 32-thread CPU without a GPU.

```
# 1. channels and teacher labels (about 25 GB, several hours)
python -m pabhbf.data.make_datasets --prefix default --splits train val test --chunk 1000
python -m pabhbf.data.make_datasets --prefix default --splits attack --no-teacher --chunk 1000
python -m pabhbf.experiments datasets_extra nt_sweep --parallel 2 --threads 6

# 2. encoders, controllers and baselines
python -m pabhbf.experiments e1pre --parallel 2 --threads 6
python -m pabhbf.experiments e1 baselines e1d --lams 0 0.01 0.03 0.1 0.3 1 --parallel 4 --threads 5
python -m pabhbf.experiments e2 e2seeds ablations mismatch phase_bits agentic temporal nt_sweep --lam-main 0.03

# 3. confirmatory protocol: splits, attacker fitting, selection, routers, freeze, one-shot test
python -m pabhbf.revision prepare --split attack_train
python -m pabhbf.revision prepare --split attack_val
python scripts/run_revision.py selection
python scripts/run_revision.py confirmatory
python -m pabhbf.revision fit-tools
python scripts/audit_revision_payload.py
python -m pabhbf.revision freeze
python -m pabhbf.revision final-test --final-test

# 4. attacker-data scaling on the frozen encoders (four times the attacker data)
python -m pabhbf.revision prepare --split attack_train_80k
python -m pabhbf.revision scaling-audit
python scripts/run_scaling80k.py --worker 0            # six workers claim units through exclusive files
python scripts/run_scaling80k.py --worker 6 --tags lam0
python -m pabhbf.revision scaling-test

# 5. figures and tables
python scripts/make_paper_figures.py
```

Step 3 refuses to repeat the final test once `results/revision_final/test_access.json` exists. Remove that file only
in a fresh copy of the package.

Steps 3 and 4 reuse the checkpoints in `results/checkpoints/pabhbf/`, so the attack pipeline can be rerun without
repeating step 2. The baseline checkpoints are not shipped and come from step 2.

## Seeds

The base seed is 20260912 in `configs/default.yaml`, and the split seeds add 1 to 4. Encoder and controller training
uses seeds 1, 2 and 3 for the reported comparisons and seed 1 for the auxiliary sweeps. The attacker splits use seeds
91526011 to 91526013, attacker fitting uses 700 plus the encoder seed, and the drop bootstrap uses 91526014. The
frozen inputs and their SHA-256 hashes are listed in `results/revision_final/frozen.json`.

## Notes on the result files

- `results/revision_final/test/` and `test80k/` hold one file per representation, seed and window, with the hit
  probability, the RMSE, the success count and the intervals of every decision rule.
- `results/revision_final/final_summary.json` collects the statistics behind the tables and figures, and
  `attack_registry.json` lists every attack outcome with the winning rule.
- `results/aggregated/` holds the development results used by the auxiliary checks of the paper.
- `MANIFEST.sha256` lists the SHA-256 of every file in this package.

## License

Choose a license before publishing the package.
