"""
result.py — Calcolo probabilità 1X2 (risultato finale) e Correct Score.

Utilizza la matrice bivariata indipendente (joint_ind) per il 1X2:
il termine Z della bivariate Poisson si cancella nella differenza (i-j),
quindi joint_ind + DC è sufficiente e più veloce della matrice full.

La matrice full è usata per il Correct Score perché include la correlazione
tra i gol rimanenti (tramite Z).
"""

from __future__ import annotations


def calcola_1x2(
    joint_ind: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
) -> tuple[float, float, float]:
    """
    Calcola le probabilità 1X2 dalla matrice indipendente + DC.

    Il termine Z si cancella nella differenza dei gol (casa - trasf),
    quindi la distribuzione del risultato finale dipende solo da joint_ind.

    Args:
        joint_ind: Matrice indipendente normalizzata con correzione DC.
            Chiavi: (gol_rimanenti_casa, gol_rimanenti_trasf).
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.

    Returns:
        (p1, px, p2): Probabilità normalizzate di 1, X, 2.
    """
    p1 = px = p2 = 0.0

    for (i, j), pij in joint_ind.items():
        diff = (gol_casa + i) - (gol_trasf + j)
        if diff > 0:
            p1 += pij
        elif diff < 0:
            p2 += pij
        else:
            px += pij

    total = p1 + px + p2
    if total > 0:
        p1 /= total
        px /= total
        p2 /= total

    return p1, px, p2


def calcola_correct_score(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    top_n: int = 5,
) -> tuple[list[tuple[tuple[int, int], float]], dict[int, float]]:
    """
    Calcola la distribuzione del Correct Score e la distribuzione dei gol totali.

    Usa la matrice full (include correlazione bivariate + DC) per una stima
    più accurata dei punteggi finali specifici.

    Args:
        full: Matrice bivariata completa normalizzata.
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.
        top_n: Numero di correct score da restituire ordinati per probabilità.

    Returns:
        (top_cs, gol_tot_dist):
          - top_cs: Lista di ((fc, ft), prob) dei top_n punteggi più probabili.
          - gol_tot_dist: Distribuzione di probabilità dei gol totali finali.
    """
    cs_final: dict[tuple[int, int], float] = {}

    for (a, b), p in full.items():
        key = (gol_casa + a, gol_trasf + b)
        cs_final[key] = cs_final.get(key, 0.0) + p

    top_cs = sorted(cs_final.items(), key=lambda x: x[1], reverse=True)[:top_n]

    gol_tot_dist: dict[int, float] = {}
    for (fc, ft), p in cs_final.items():
        tot = fc + ft
        gol_tot_dist[tot] = gol_tot_dist.get(tot, 0.0) + p

    return top_cs, gol_tot_dist
