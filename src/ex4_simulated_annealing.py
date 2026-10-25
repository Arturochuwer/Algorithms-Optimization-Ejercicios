"""Ejercicio 4: Simulated Annealing (pseudocodigo del PDF) con 2-opt como movimiento,
desde un tour inicial ALEATORIO, comparando esquemas de enfriamiento.
Uso:  python src/ex4_simulated_annealing.py"""
import random
import statistics as st
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG
from tsp import SCHEDULES, random_points, random_tour, simulated_annealing, tour_length

N_CIUDADES, N_CORRIDAS = 40, 10
T0, TF = 3.0, 0.01
N_CONV = 30_000                                  # iteraciones para las curvas de convergencia
PRESUPUESTOS = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]   # para tiempo vs calidad
COL = {"Lineal": "#2ca02c", "Exponencial": "#1f77b4", "Rapido (quench)": "#d62728"}

if __name__ == "__main__":
    pts = random_points(N_CIUDADES, random.Random(123))
    inicio = random_tour(N_CIUDADES, random.Random(5))           # solucion inicial aleatoria
    print(f"Instancia: {N_CIUDADES} ciudades, L inicial (aleatoria) = {tour_length(pts, inicio):.1f}")

    # ---------- 1) Curvas de convergencia (N_CONV iteraciones) ----------
    conv = {}
    for nombre, fab in SCHEDULES.items():
        runs = []
        for r in range(N_CORRIDAS):
            tf, hL, hT = simulated_annealing(pts, inicio, fab(T0, TF, N_CONV), random.Random(1000 + r))
            runs.append((hL, hT))
        conv[nombre] = runs
        fin = [h[-1] for h, _ in runs]
        print(f"{nombre:<16} L final: media {st.mean(fin):.2f}  mejor {min(fin):.2f}  peor {max(fin):.2f}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, (nombre, runs) in enumerate(conv.items()):
        n = len(runs[0][0])
        media = [st.mean(h[i] for h, _ in runs) for i in range(n)]
        for h, _ in runs:
            ax[0][j].plot(h, color=COL[nombre], alpha=0.15, lw=0.6)
        ax[0][j].plot(media, color=COL[nombre], lw=2)
        ax[0][j].set_title(f"{nombre}: costo vs iteraciones\n(L final media = {media[-1]:.2f})")
        ax[0][j].set_xlabel("Iteracion")
        ax[0][j].set_ylabel("Costo (longitud del tour)")
        # costo vs temperatura (T en escala log, de caliente a frio)
        hT = runs[0][1][1:]
        ax[1][j].plot(hT, media[1:], color=COL[nombre], lw=1.5)
        ax[1][j].set_xscale("log")
        ax[1][j].invert_xaxis()
        ax[1][j].set_title(f"{nombre}: costo vs temperatura")
        ax[1][j].set_xlabel("Temperatura T (log, enfriando hacia la derecha)")
        ax[1][j].set_ylabel("Costo medio")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Ejercicio 4: curvas de convergencia (media y 10 corridas)")
    fig.tight_layout()
    fig.savefig(FIG / "ex4_convergencia.png", dpi=150)

    # comparacion superpuesta + esquemas T(t)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for nombre, runs in conv.items():
        n = len(runs[0][0])
        ax[0].plot([st.mean(h[i] for h, _ in runs) for i in range(n)], color=COL[nombre], label=nombre)
        ax[1].plot([None] + runs[0][1][1:], color=COL[nombre], label=nombre)
    ax[0].set_title("Convergencia comparada (media)")
    ax[0].set_xlabel("Iteracion")
    ax[0].set_ylabel("Costo")
    ax[1].set_yscale("log")
    ax[1].set_title("Esquemas de enfriamiento T(t)")
    ax[1].set_xlabel("Iteracion")
    ax[1].set_ylabel("Temperatura (log)")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ex4_comparacion_y_schedules.png", dpi=150)

    # ---------- 2) Tiempo de ejecucion vs calidad final ----------
    print(f"\n{'Esquema':<16}{'iteraciones':>12}{'tiempo medio(s)':>17}{'L final media':>15}")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for nombre, fab in SCHEDULES.items():
        xs, ys = [], []
        for N in PRESUPUESTOS:
            tiempos, finales = [], []
            for r in range(N_CORRIDAS):
                t0 = time.perf_counter()
                tf, hL, _ = simulated_annealing(pts, inicio, fab(T0, TF, N), random.Random(2000 + r), record=False)
                tiempos.append(time.perf_counter() - t0)
                finales.append(tour_length(pts, tf))
            xs.append(st.mean(tiempos))
            ys.append(st.mean(finales))
            print(f"{nombre:<16}{N:>12,}{xs[-1]:>17.3f}{ys[-1]:>15.2f}")
        ax.plot(xs, ys, "o-", color=COL[nombre], label=nombre)
        for x, y, N in zip(xs, ys, PRESUPUESTOS):
            ax.annotate(f"{N // 1000}k", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de ejecucion medio (s, log)")
    ax.set_ylabel("Longitud final media del tour (menor = mejor)")
    ax.set_title("Ejercicio 4: tiempo vs calidad (etiqueta = iteraciones)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ex4_tiempo_vs_calidad.png", dpi=150)
    print("\nFiguras: results/figures/ex4_*.png")
