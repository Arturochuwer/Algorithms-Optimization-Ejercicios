"""Ejercicio 1: speedup de estimate_pi (de p2.py) con 1, 2, 4 y 8 procesos.
Uso:  python src/ex1_speedup.py"""
import json
import multiprocessing as mp
import os
import random
import time

from common import FIG, DATA


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:      # copiada de p2.py
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def run(n_procs, seeds):
    t0 = time.perf_counter()
    if n_procs == 1:
        [estimate_pi(s) for s in seeds]                  # 1 proceso = secuencial (linea base T1)
    else:
        with mp.Pool(processes=n_procs) as pool:
            pool.map(estimate_pi, seeds)
    return time.perf_counter() - t0


if __name__ == "__main__":
    seeds = list(range(16))      # 16 tareas: se reparten parejo entre 1, 2, 4 y 8 procesos
    procs = [1, 2, 4, 8]
    nucleos = os.cpu_count()
    T = {}
    for p in procs:
        tiempos = [run(p, seeds) for _ in range(3)]       # mediana de 3 repeticiones
        T[p] = sorted(tiempos)[1]
        print(f"{p} procesos: {T[p]:.3f} s")
    S = {p: T[1] / T[p] for p in procs}
    print(f"\nNucleos de esta maquina: {nucleos}")
    print(f"{'procs':>5} {'tiempo(s)':>10} {'speedup':>8} {'ideal':>6} {'eficiencia':>11}")
    for p in procs:
        print(f"{p:>5} {T[p]:>10.3f} {S[p]:>7.2f}x {p:>5}x {S[p] / p * 100:>10.0f}%")
    json.dump({"nucleos": nucleos, "tiempo": T, "speedup": S}, open(DATA / "ex1_resultados.json", "w"), indent=2)

    import matplotlib                    # se importa aqui y NO arriba: en Windows cada proceso
    matplotlib.use("Agg")                # hijo re-importa el modulo, y matplotlib tarda en cargar
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(procs, procs, "--", color="gray", label="Speedup ideal (lineal)")
    ax.plot(procs, [S[p] for p in procs], "o-", color="#d62728", label="Speedup medido")
    ax.axvline(nucleos, color="green", ls=":", label=f"Nucleos de la maquina = {nucleos}")
    ax.set_xticks(procs)
    ax.set_xlabel("Numero de procesos")
    ax.set_ylabel("Speedup = T(1) / T(p)")
    ax.set_title("Ejercicio 1: speedup de estimate_pi")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ex1_speedup.png", dpi=150)
    print("Figura: results/figures/ex1_speedup.png")
