#!/usr/bin/env bash
# Regenerate every figure and table from saved results. No simulation or training is rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
python -m pabhbf.eval.aggregate --exp e1 baselines e1d e2 ablations mismatch phase_bits nt_sweep
python -m pabhbf.plots.make_all --lam-main 0.03
