#!/usr/bin/env bash
# Smoke run: tiny datasets, unit and pipeline tests, one short training and evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
python -m pabhbf.data.make_datasets --config smoke.yaml --prefix smoke --far-field-twin
python -m pabhbf.data.make_datasets --config smoke.yaml --prefix smoke --splits attack --no-teacher
python -m pytest
python -m pabhbf.train.pabhbf --config smoke.yaml --prefix smoke --mode adversarial --lam 1 --tag smoke_adv --val-drops 50 --threads 4
python -m pabhbf.eval.evaluate --ckpt "results/checkpoints/pabhbf/smoke_adv_s1_*.pt" --exp smoke --epochs-att 2 --threads 4
