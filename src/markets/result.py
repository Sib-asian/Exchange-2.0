"""
result.py — Calcolo probabilità 1X2 (risultato finale) e Correct Score.

La matrice full (blended) è usata per tutti i mercati per garantire
coerenza cross-market.
"""

from __future__ import annotations

import math


def apply_overdispersion(
    full: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Applica correzione overdispersion alla matrice di punteggio.

    Il modello Poisson sottostima i punteggi con molti gol futuri (a+b >= 3)
    perché la varianza reale supera la media. Questa correzione viene applicata
    UNA VOLTA alla matrice blended, prima di derivare qualsiasi mercato (1X2,
    O/U, BTTS, CS), garantendo coerenza tra tutti i mercati.

    Returns:
        Matrice corretta e rinormalizzata.
    """
    from src.config import UI as _UI

    corrected: dict[tuple[int, int], float] = {}
    for (a, b), p in full.items():
        future_goals = a + b
        if future_goals == 3:
            p *= _UI.CS_OVERDISP_3
        elif future_goals == 4:
            p *= _UI.CS_OVERDISP_4
        elif future_goals >= 5:
            p *= _UI.CS_OVERDISP_5
        corrected[(a, b)] = p

    total = sum(corrected.values())
    if total > 0:
        corrected = {k: v / total for k, v in corrected.items()}
    return corrected


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
    *,
    score_overdispersion: bool = True,
) -> tuple[list[tuple[tuple[int, int], float]], dict[int, float], float, float]:
    """
    Calcola la distribuzione del Correct Score e la distribuzione dei gol totali.

    Usa la matrice full (include correlazione bivariate + DC) per una stima
    più accurata dei punteggi finali specifici.

    Args:
        full: Matrice bivariata completa normalizzata.
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.
        top_n: Numero di correct score da restituire ordinati per probabilità.
        score_overdispersion: Se True, applica la correzione continua sui gol futuri
            (utile su matrici grezze, es. test). Se False, assume che la matrice sia
            già stata passata da ``apply_overdispersion`` (allineamento con ``compute_consensus``).

    Returns:
        (top_cs, gol_tot_dist, entropy_nats, top3_mass):
          - top_cs: Lista di ((fc, ft), prob) dei top_n punteggi più probabili.
          - gol_tot_dist: Distribuzione di probabilità dei gol totali finali.
          - entropy_nats: Entropia di Shannon sulla distribuzione completa dei CS (nats).
          - top3_mass: Somma delle probabilità dei tre score più probabili.
    """
    from src.config import UI as _UI

    cs_final: dict[tuple[int, int], float] = {}

    for (a, b), p in full.items():
        key = (gol_casa + a, gol_trasf + b)
        cs_final[key] = cs_final.get(key, 0.0) + p

    if score_overdispersion:
        # Overdispersion residua: legge continua sui gol futuri (matrici non ancora corrette).
        cs_corrected: dict[tuple[int, int], float] = {}
        for (fc, ft), p in cs_final.items():
            future_goals = (fc - gol_casa) + (ft - gol_trasf)
            if future_goals >= 3:
                _x = max(0.0, float(future_goals) - _UI.CS_OVERDISP_K0)
                _mult = 1.0 + _UI.CS_OVERDISP_ALPHA * (_x ** _UI.CS_OVERDISP_EXP)
                p *= min(_UI.CS_OVERDISP_MAX, _mult)
            cs_corrected[fc, ft] = p
        _total_p = sum(cs_corrected.values())
        cs_final = {k: v / _total_p for k, v in cs_corrected.items()} if _total_p > 0 else cs_corrected
    else:
        _total_p = sum(cs_final.values())
        if _total_p > 0:
            cs_final = {k: v / _total_p for k, v in cs_final.items()}

    # Ordinamento deterministico: probabilità decrescente, poi per punteggio crescente
    # (evita ordine non-deterministico quando due score hanno la stessa probabilità)
    top_cs = sorted(cs_final.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:top_n]

    gol_tot_dist: dict[int, float] = {}
    for (fc, ft), p in cs_final.items():
        tot = fc + ft
        gol_tot_dist[tot] = gol_tot_dist.get(tot, 0.0) + p

    entropy_nats = 0.0
    for p in cs_final.values():
        if p > 0:
            entropy_nats -= p * math.log(p)
    _sorted_probs = sorted(cs_final.values(), reverse=True)
    top3_mass = sum(_sorted_probs[:3])

    return top_cs, gol_tot_dist, entropy_nats, top3_mass
