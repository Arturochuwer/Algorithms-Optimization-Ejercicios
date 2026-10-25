"""Corre los 5 ejercicios en orden. Uso:  python src/run_all.py"""
import subprocess
import sys
from pathlib import Path

src = Path(__file__).resolve().parent
for f in ["ex1_speedup.py", "ex2_knapsack.py", "ex3_two_opt.py", "ex4_simulated_annealing.py", "ex5_approx_ratio.py"]:
    print(f"\n{'=' * 70}\n{f}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(src / f)], check=True)
