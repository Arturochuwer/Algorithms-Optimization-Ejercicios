"""Algoritmos compartidos de TSP (ejercicios 3, 4 y 5).

DEL PDF / p2.py (copiados tal cual):  mst_prim, preorder, tour_length, two_opt_move,
                                      optimal_bruteforce (fuerza bruta de la Parte 3)
ESCRITOS PARA LA TAREA (no estan en el PDF): nearest_neighbour, two_opt_search,
                                      simulated_annealing (el PDF solo trae el pseudocodigo)
                                      y los esquemas de enfriamiento (schedules).
"""
import itertools
import math
import random


# ----------------------------- del PDF / p2.py -----------------------------
def mst_prim(points):
    """Prim desde la ciudad 0; devuelve el arbol como {nodo: hijos}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root=0):
    """Recorrido en preorden = atajo (shortcut) de ciudades repetidas."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def tour_length(points, tour):
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))


def two_opt_move(route, i, k):
    """Invierte route[i..k]: quita las aristas (i-1,i) y (k,k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def optimal_bruteforce(points):
    """Optimo por fuerza bruta (ciudad 0 fija): (n-1)! tours. Solo n <= 9-10."""
    n = len(points)
    return min(((0,) + p for p in itertools.permutations(range(1, n))),
               key=lambda t: tour_length(points, t))


# ------------------------------- escritos aqui -------------------------------
def random_points(n, rng):
    return [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)]


def random_tour(n, rng):
    t = list(range(n))
    rng.shuffle(t)
    return t


def nearest_neighbour(points, start=0):
    """Constructivo: desde 'start', ir siempre a la ciudad no visitada mas cercana."""
    unvisited = set(range(len(points))) - {start}
    tour = [start]
    while unvisited:
        ult = tour[-1]
        sig = min(unvisited, key=lambda c: math.dist(points[ult], points[c]))
        tour.append(sig)
        unvisited.remove(sig)
    return tour


def two_opt_search(points, tour, return_history=False):
    """2-opt completo: aplica two_opt_move (del PDF) a todo par (i,k) mientras alguna
    inversion acorte el tour. Cada inversion aceptada = 1 'mejora'.
    Devuelve (tour, n_mejoras[, historial de longitudes])."""
    tour = list(tour)
    n = len(tour)
    mejoras = 0
    actual = tour_length(points, tour)
    historial = [actual]
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                nuevo = two_opt_move(tour, i, k)
                L = tour_length(points, nuevo)
                if L < actual - 1e-12:
                    tour, actual = nuevo, L
                    mejoras += 1
                    historial.append(L)
                    mejoro = True
    return (tour, mejoras, historial) if return_history else (tour, mejoras)


# ----------------------- Simulated Annealing (pseudocodigo del PDF) -----------------------
# Esquemas de enfriamiento: fabrica(T0, Tf, N) -> funcion schedule(t) con t = 1..N.
# Al terminar (t > N) devuelve 0  ->  "if T = 0 then return current" del pseudocodigo.
def schedule_lineal(T0, Tf, N):
    return lambda t: T0 + (Tf - T0) * (t - 1) / (N - 1) if t <= N else 0.0


def schedule_exponencial(T0, Tf, N):
    """Geometrico: T0 -> Tf en N pasos (alpha = (Tf/T0)^(1/(N-1)))."""
    alpha = (Tf / T0) ** (1 / (N - 1))
    return lambda t: T0 * alpha ** (t - 1) if t <= N else 0.0


def schedule_rapido(T0, Tf, N):
    """Enfriamiento agresivo: exponencial que llega a Tf en solo N/10 pasos y se queda ahi."""
    alpha = (Tf / T0) ** (1 / (N / 10 - 1))
    return lambda t: max(T0 * alpha ** (t - 1), Tf) if t <= N else 0.0


SCHEDULES = {"Lineal": schedule_lineal, "Exponencial": schedule_exponencial,
             "Rapido (quench)": schedule_rapido}


def simulated_annealing(points, tour0, schedule, rng, record=True):
    """SA del PDF con 2-opt como RANDOM_SUCCESSOR.
    VALUE = -longitud  ->  delta_E = L_actual - L_vecino.
    Devuelve (tour, historial de L, historial de T)."""
    n = len(points)
    d = lambda a, b: math.dist(points[a], points[b])
    cur = list(tour0)
    L = tour_length(points, cur)
    hL, hT = [L], [None]
    t = 0
    while True:
        t += 1
        T = schedule(t)
        if T == 0:
            return cur, hL, hT
        i = rng.randint(1, n - 2)
        k = rng.randint(i + 1, n - 1)
        a, b, c, e = cur[i - 1], cur[i], cur[k], cur[(k + 1) % n]
        dL = d(a, c) + d(b, e) - d(a, b) - d(c, e)       # cambio de longitud en O(1)
        delta_E = -dL
        if delta_E > 0 or rng.random() < math.exp(delta_E / T):
            cur[i:k + 1] = cur[i:k + 1][::-1]
            L += dL
        if record:
            hL.append(L)
            hT.append(T)
