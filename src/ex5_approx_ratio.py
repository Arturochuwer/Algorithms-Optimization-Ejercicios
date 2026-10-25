"""Ejercicio 5: razon de aproximacion de MST-TSP en 100 instancias de 9 ciudades,
con y sin 2-opt, comparada con busqueda local (ej. 3) y recocido simulado (ej. 4).
Uso:  python src/ex5_approx_ratio.py"""
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, DATA
from tsp import (SCHEDULES, mst_prim, nearest_neighbour, optimal_bruteforce, preorder, random_points,
                 random_tour, simulated_annealing, tour_length, two_opt_search)

N_INST, N_CIUDADES, N_SA = 100, 9, 5_000
METODOS = ["MST puro", "MST + 2-opt", "2-opt (inicio aleatorio)", "2-opt (inicio NN)", "Recocido simulado", "Optimo"]

if __name__ == "__main__":
    rng = random.Random(7)
    sched = SCHEDULES["Exponencial"](3.0, 0.01, N_SA)
    C = {m: [] for m in METODOS}
    for _ in range(N_INST):
        pts = random_points(N_CIUDADES, rng)
        C["Optimo"].append(tour_length(pts, optimal_bruteforce(pts)))
        t_mst = preorder(mst_prim(pts))
        C["MST puro"].append(tour_length(pts, t_mst))
        C["MST + 2-opt"].append(tour_length(pts, two_opt_search(pts, t_mst)[0]))
        C["2-opt (inicio aleatorio)"].append(tour_length(pts, two_opt_search(pts, random_tour(N_CIUDADES, rng))[0]))
        C["2-opt (inicio NN)"].append(tour_length(pts, two_opt_search(pts, nearest_neighbour(pts, rng.randrange(N_CIUDADES)))[0]))
        tsa, _, _ = simulated_annealing(pts, random_tour(N_CIUDADES, rng), sched, rng, record=False)
        C["Recocido simulado"].append(tour_length(pts, tsa))

    R = {m: [c / o for c, o in zip(C[m], C["Optimo"])] for m in METODOS if m != "Optimo"}
    assert all(r <= 2 + 1e-9 for r in R["MST puro"]), "se violo la garantia teorica"
    print(f"{N_INST} instancias de {N_CIUDADES} ciudades (razon = costo / optimo)")
    print(f"{'Metodo':<26}{'media':>7}{'mediana':>9}{'max':>7}{'optimo en':>11}")
    for m, r in R.items():
        print(f"{m:<26}{st.mean(r):>7.3f}{st.median(r):>9.3f}{max(r):>7.3f}{sum(x < 1 + 1e-9 for x in r):>8}/{N_INST}")
    mej = sum(b < a - 1e-9 for a, b in zip(C["MST puro"], C["MST + 2-opt"]))
    print(f"2-opt mejoro el tour MST en {mej}/{N_INST} instancias")
    print(f"Razon maxima MST = {max(R['MST puro']):.3f}  ({max(R['MST puro']) / 2 * 100:.0f}% de la cota 2)")

    # Figura 1: razon del MST (dispersion + histograma) con linea en 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(range(1, N_INST + 1), R["MST puro"], "o", ms=4, color="#d62728")
    ax[0].axhline(2, color="black", ls="--", label="Limite teorico = 2")
    ax[0].axhline(1, color="gray", ls=":", label="Optimo = 1")
    ax[0].set_ylim(0.9, 2.1)
    ax[0].set_xlabel("Instancia")
    ax[0].set_ylabel("C_obtenido / C_optimo")
    ax[0].set_title("Razon de aproximacion MST por instancia")
    ax[0].legend(loc="center right")
    ax[1].hist(R["MST puro"], bins=[1 + 0.05 * i for i in range(22)], color="#d62728", alpha=0.8)
    ax[1].axvline(2, color="black", ls="--", label="Limite teorico = 2")
    ax[1].set_xlabel("C_obtenido / C_optimo")
    ax[1].set_ylabel("Instancias")
    ax[1].set_title("Histograma de la razon (MST puro)")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ex5_razon_mst.png", dpi=150)

    # Figura 2: comparacion de metodos (costo y razon)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].boxplot([C[m] for m in METODOS], tick_labels=METODOS)
    ax[0].set_ylabel("Costo (longitud del tour)")
    ax[0].set_title("Costo obtenido por metodo")
    ax[1].boxplot([R[m] for m in R], tick_labels=list(R))
    ax[1].axhline(1, color="gray", ls=":")
    ax[1].set_ylabel("C_obtenido / C_optimo")
    ax[1].set_title("Razon de aproximacion por metodo")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
        a.grid(alpha=0.3)
    fig.suptitle("Ejercicio 5: MST puro, MST + 2-opt y metodos previos")
    fig.tight_layout()
    fig.savefig(FIG / "ex5_comparacion_metodos.png", dpi=150)
    print("Figuras: results/figures/ex5_*.png")
