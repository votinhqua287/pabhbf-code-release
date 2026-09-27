#!/usr/bin/env bash
# Full experiment matrix of docs/EXPERIMENT_PLAN.md on CPU. The queue skips completed jobs, so the script can resume.
# LAM_MAIN selects the privacy weight of the main operating point used by the secondary sweeps (chosen from E1).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
LAM_MAIN="${LAM_MAIN:-1}"

# Datasets of the main scenario (Nt = 256) and of the premise study
python -m pabhbf.data.make_datasets --prefix default --splits train val test --chunk 1000
python -m pabhbf.data.make_datasets --prefix default --splits attack --no-teacher --chunk 1000
python -m pabhbf.data.make_datasets --config nt64.yaml --prefix nt64 --far-field-twin
python -m pabhbf.data.make_datasets --config nt64.yaml --prefix nt64 --splits attack --no-teacher
python -m pabhbf.data.make_datasets --config nt64.yaml --prefix los --splits test --no-teacher --far-field-twin \
  --override '{"channel": {"rician_k_db": 60.0}, "dataset": {"num_test_drops": 1000}}'
for NT in 128 256; do
  python -m pabhbf.data.make_datasets --prefix nt${NT}p --no-teacher --far-field-twin \
    --override "{\"Nt\": ${NT}, \"dataset\": {\"num_train_drops\": 5000, \"num_val_drops\": 1000, \"num_test_drops\": 1000}}"
done

# Teacher quality, CRB and premise (gate G1)
python -m pabhbf.eval.teacher_quality
python -m pabhbf.attacks.crb
python -m pabhbf.attacks.premise --config nt64.yaml --prefix nt64
python -m pabhbf.attacks.premise_mle --prefix nt64 --los-prefix los --override '{"Nt": 64}'
for NT in 128 256; do
  python -m pabhbf.attacks.premise --prefix nt${NT}p --skip-forest --override "{\"Nt\": ${NT}}"
  python -m pabhbf.attacks.premise_mle --prefix nt${NT}p --los-prefix none --fit-rows 20000 --override "{\"Nt\": ${NT}}"
done

# Experiment matrix
python -m pabhbf.experiments datasets_extra teacher_sweeps --parallel 2
python -m pabhbf.experiments g2 e1 baselines e1d --parallel 3
python -m pabhbf.experiments e2 ablations mismatch phase_bits agentic temporal nt_sweep --parallel 3 --lam-main "${LAM_MAIN}"
python -m pabhbf.eval.aggregate --exp e1 baselines e1d e2 ablations mismatch phase_bits nt_sweep
