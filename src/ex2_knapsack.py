"""Ejercicio 2: knapsack voraz por MENOR PESO vs por RAZON valor/peso (+ optimo DP del PDF).
Uso:  python src/ex2_knapsack.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, DATA


def greedy(values, weights, W, key):
    """Voraz generico: ordena por 'key' y mete cada item si todavia cabe (nunca deshace)."""
    orden = sorted(range(len(values)), key=key)
    peso = valor = 0
    elegidos = []
    for i in orden:
        if peso + weights[i] <= W:
            peso += weights[i]
            valor += values[i]
            elegidos.append(i)
    return sorted(elegidos), peso, valor


def greedy_menor_peso(v, w, W):                      # MODIFICACION pedida
    return greedy(v, w, W, key=lambda i: w[i])


def greedy_razon(v, w, W):                           # metodo original (valor/peso)
    return greedy(v, w, W, key=lambda i: -v[i] / w[i])


def knapsack_dp(values, weights, W):                 # DP del PDF / p2.py (optimo)
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    chosen, c = [], W
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen = sorted(chosen)
    return chosen, sum(weights[i] for i in chosen), K[n][W]


def comparar(nombre, items, v, w, W):
    filas = [("Menor peso", *greedy_menor_peso(v, w, W)),
             ("Valor/peso", *greedy_razon(v, w, W)),
             ("Optimo (DP)", *knapsack_dp(v, w, W))]
    print(f"\n=== {nombre}  (capacidad W = {W}) ===")
    print("Items:", ", ".join(f"{items[i]}(v={v[i]}, w={w[i]}, v/w={v[i]/w[i]:.2f})" for i in range(len(items))))
    print(f"{'Metodo':<13}{'Items elegidos':<18}{'Capacidad usada':<18}{'Valor total'}")
    for m, ch, peso, val in filas:
        print(f"{m:<13}{str([items[i] for i in ch]):<18}{f'{peso}/{W}':<18}{val}")
    return filas


def grafica(filas, items, W, titulo, archivo):
    nombres = [f[0] for f in filas]
    colores = ["#d62728", "#1f77b4", "#2ca02c"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, idx, etq in ((ax[0], 2, "Capacidad usada"), (ax[1], 3, "Valor total")):
        barras = a.bar(nombres, [f[idx] for f in filas], color=colores)
        for b, f in zip(barras, filas):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(),
                   f"{f[idx]}\n{[items[i] for i in f[1]]}", ha="center", va="bottom", fontsize=8)
        a.set_ylabel(etq)
        a.set_title(etq)
        a.set_ylim(0, max(f[idx] for f in filas) * 1.35)
        a.grid(axis="y", alpha=0.3)
    ax[0].axhline(W, color="black", ls="--", lw=1)
    ax[0].text(2.4, W, f" W={W}", va="bottom", ha="right", fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(FIG / archivo, dpi=150)


if __name__ == "__main__":
    # ENTREGABLE: instancia pequena donde menor peso rinde peor que valor/peso
    items, v, w, W = ["A", "B", "C", "D"], [10, 10, 10, 100], [10, 10, 10, 30], 30
    filas = comparar("Instancia disenada (menor peso pierde)", items, v, w, W)
    assert filas[0][3] < filas[1][3]
    grafica(filas, items, W, "Ejercicio 2: instancia disenada (W=30)", "ex2_knapsack_instancia.png")

    # Contexto: la instancia del PDF (aqui ambos voraces fallan igual; el optimo es B+C=220)
    items2, v2, w2, W2 = ["A", "B", "C"], [60, 100, 120], [10, 20, 30], 50
    filas2 = comparar("Instancia del PDF", items2, v2, w2, W2)
    grafica(filas2, items2, W2, "Instancia del PDF (W=50)", "ex2_knapsack_pdf.png")

    with open(DATA / "ex2_tabla.md", "w") as f:
        f.write("| Instancia | Metodo | Items | Capacidad usada | Valor |\n|---|---|---|---|---|\n")
        for nom, its, W_, fs in (("Disenada", items, W, filas), ("PDF", items2, W2, filas2)):
            for m, ch, peso, val in fs:
                f.write(f"| {nom} | {m} | {[its[i] for i in ch]} | {peso}/{W_} | {val} |\n")
    print("\nFiguras: results/figures/ex2_*.png  |  Tabla: results/ex2_tabla.md")
