"""Regenerate the reported figures and tables from the frozen result files.

    python scripts/make_paper_figures.py

Writes outputs/figures/fig_premise_final.pdf (Fig. 2), outputs/figures/fig_pareto_final.pdf (Fig. 3),
outputs/figures/fig_temporal_final.pdf (Fig. 4) and outputs/tables/*.tex (Tables IV, V and VI).
"""
from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    env_cmd = [sys.executable, "-m", "pabhbf.plots.fig06_nf_ff", "--raw-only", "--name", "fig_premise_final",
               "--lam-main", "0.03"]
    subprocess.run(env_cmd, cwd=ROOT, check=True, env={**__import__("os").environ,
                                                       "PYTHONPATH": str(ROOT / "src")})
    runpy.run_path(str(ROOT / "scripts" / "final_artifacts.py"), run_name="__main__")
    figures = ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    premise = ROOT / "results" / "figures" / "fig_premise_final.pdf"
    if premise.exists():
        shutil.copy2(premise, figures / premise.name)
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in ("tab_main_final.tex", "tab_attacks_final.tex", "tab_tools_final.tex", "numbers_revision.tex"):
        source = ROOT / "results" / "revision_final" / "generated" / name
        if source.exists():
            shutil.copy2(source, tables / name)
    print("figures in", figures)
    print("tables in", tables)


if __name__ == "__main__":
    main()
