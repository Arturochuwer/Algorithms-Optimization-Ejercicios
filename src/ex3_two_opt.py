"""Ejercicio 3: 2-opt. Mejoras hasta el optimo local desde (1) tour aleatorio y
(2) Vecino Mas Cercano iniciando en un nodo ALEATORIO.
Uso:  python src/ex3_two_opt.py"""
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG
from tsp import (nearest_neighbour, random_points, random_tour, tour_length, two_opt_search)

N_CIUDADES, N_INST = 30, 200

if __name__ == "__main__":
    # Validacion con el ejemplo del PDF (5 ciudades): 19.66 -> 15.66
    c = {"A": (0, 0), "B": (0, 3), "E": (2, 5), "D": (4, 3), "C": (4, 0)}
    nom = list(c)
    pts5 = [c[k] for k in nom]
    r0 = [nom.index(x) for x in ["A", "D", "E", "B", "C"]]
    rf, m = two_opt_search(pts5, r0)
    print(f"Validacion PDF: L {tour_length(pts5, r0):.2f} -> {tour_length(pts5, rf):.2f} en {m} mejora(s)")

    rng = random.Random(42)
    R = {"Aleatorio": {"mej": [], "ini": [], "fin": []}, "Vecino mas cercano": {"mej": [], "ini": [], "fin": []}}
    hist = {}
    for t in range(N_INST):
        pts = random_points(N_CIUDADES, rng)
        inicios = {"Aleatorio": random_tour(N_CIUDADES, rng),
                   "Vecino mas cercano": nearest_neighbour(pts, rng.randrange(N_CIUDADES))}
        for nombre, t0 in inicios.items():
            tf, m, h = two_opt_search(pts, t0, return_history=True)
            R[nombre]["mej"].append(m)
            R[nombre]["ini"].append(tour_length(pts, t0))
            R[nombre]["fin"].append(tour_length(pts, tf))
            if t == 0:
                hist[nombre] = h

    print(f"\n{N_INST} instancias de {N_CIUDADES} ciudades")
    print(f"{'Inicio':<20}{'L inicial':>10}{'mejoras media':>15}{'mediana':>9}{'min-max':>9}{'L final':>9}")
    for n, r in R.items():
        print(f"{n:<20}{st.mean(r['ini']):>10.1f}{st.mean(r['mej']):>15.1f}{st.median(r['mej']):>9.0f}"
              f"{min(r['mej']):>5}-{max(r['mej']):<3}{st.mean(r['fin']):>9.2f}")
    menos = sum(a < b for a, b in zip(R["Vecino mas cercano"]["mej"], R["Aleatorio"]["mej"]))
    print(f"\nNN necesito MENOS mejoras en {menos}/{N_INST} instancias")
    dif = st.mean(R["Aleatorio"]["fin"]) - st.mean(R["Vecino mas cercano"]["fin"])
    print(f"Diferencia media de calidad final (aleatorio - NN): {dif:+.2f}")

    etq = ["Aleatorio", "Vecino mas cercano"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].boxplot([R[e]["mej"] for e in etq], tick_labels=etq)
    ax[0].set_title("Numero de mejoras hasta el optimo local")
    ax[0].set_ylabel("Mejoras 2-opt")
    ax[1].boxplot([R[e]["fin"] for e in etq], tick_labels=etq)
    ax[1].set_title("Calidad del optimo local (longitud final)")
    ax[1].set_ylabel("Longitud del tour")
    for e, col in zip(etq, ("#d62728", "#1f77b4")):
        ax[2].step(range(len(hist[e])), hist[e], where="post", color=col, label=e)
    ax[2].set_title("Convergencia en una instancia")
    ax[2].set_xlabel("Mejora #")
    ax[2].set_ylabel("Longitud del tour")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Ejercicio 3: 2-opt ({N_INST} instancias, {N_CIUDADES} ciudades)")
    fig.tight_layout()
    fig.savefig(FIG / "ex3_two_opt.png", dpi=150)
    print("Figura: results/figures/ex3_two_opt.png")
