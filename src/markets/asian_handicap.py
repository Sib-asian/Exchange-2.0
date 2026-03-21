"""
asian_handicap.py — Calcolo probabilità Asian Handicap a diversi livelli.

L'Asian Handicap è calcolato sui GOL RIMANENTI dalla matrice full:
questo è coerente con la linea AH Corrente inserita come "gol rimanenti".

Logica push:
- Se (gol_rimanenti_casa - gol_rimanenti_trasf) + handicap == 0 → push (stake restituita)
- La probabilità effettiva AH = P(win) + 0.5 * P(push)
"""

from __future__ import annotations


def calcola_asian_handicap(
    full: dict[tuple[int, int], float],
    livelli: tuple[float, ...] | list[float],
) -> list[dict]:
    """
    Calcola le probabilità Asian Handicap per una serie di livelli.

    Gli handicap sono applicati ai GOL RIMANENTI (non al risultato finale),
    coerentemente con la linea AH Corrente usata per la calibrazione.

    Args:
        full: Matrice bivariata completa normalizzata.
            Chiavi: (gol_rimanenti_casa, gol_rimanenti_trasf).
        livelli: Sequenza di handicap da calcolare (es. [-2.5, -2.0, ..., +2.5]).

    Returns:
        Lista di dict con keys:
          - level: float (handicap)
          - p_win: P(casa copre l'handicap)
          - p_push: P(push, stake restituita)
          - p_lose: P(casa non copre)
          - p_eff: Probabilità effettiva (win + 0.5 * push)
          - quota_fair: 1 / p_eff
    """
    results = []

    for level in livelli:
        win = push = lose = 0.0

        for (a, b), p in full.items():
            diff = (a - b) + level
            if diff > 1e-9:
                win += p
            elif diff < -1e-9:
                lose += p
            else:
                push += p

        p_eff = win + 0.5 * push
        quota_fair = (1.0 / p_eff) if p_eff > 1e-9 else 999.0

        results.append({
            "level": level,
            "p_win": win,
            "p_push": push,
            "p_lose": lose,
            "p_eff": p_eff,
            "quota_fair": quota_fair,
        })

    return results
