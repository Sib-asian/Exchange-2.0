"""
asian_handicap.py — Calcolo probabilità Asian Handicap a diversi livelli.

L'Asian Handicap è calcolato sui GOL RIMANENTI dalla matrice full:
questo è coerente con la linea AH Corrente inserita come "gol rimanenti".

Logica push:
- Se (gol_rimanenti_casa - gol_rimanenti_trasf) + handicap == 0 → push (stake restituita)
- La probabilità effettiva AH = P(win) + 0.5 * P(push)
"""

from __future__ import annotations

import math


def _calc_half_line(
    full: dict[tuple[int, int], float],
    level: float,
) -> tuple[float, float, float]:
    """Calcola win/push/lose per una mezza linea (o linea intera)."""
    win = push = lose = 0.0
    for (a, b), p in full.items():
        diff = (a - b) + level
        if diff > 1e-6:
            win += p
        elif diff < -1e-6:
            lose += p
        else:
            push += p
    return win, push, lose


def calcola_asian_handicap(
    full: dict[tuple[int, int], float],
    livelli: tuple[float, ...] | list[float],
) -> list[dict]:
    """
    Calcola le probabilità Asian Handicap per una serie di livelli.

    Gli handicap sono applicati ai GOL RIMANENTI (non al risultato finale),
    coerentemente con la linea AH Corrente usata per la calibrazione.

    Supporta quarter lines (es. -0.75 = split 50% su -0.5 e 50% su -1.0):
    la scommessa viene divisa a metà sulle due mezze linee adiacenti.

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
        # Determina se è una quarter line: il residuo mod 0.5 è ~0.25.
        # Linee intere (0, 1, -1) e mezze (-0.5, 1.5) hanno residuo 0.
        remainder = abs(level * 4.0 - round(level * 4.0))
        is_quarter = remainder < 1e-6 and abs((level * 2.0) - round(level * 2.0)) > 0.1

        if is_quarter:
            # Quarter line: split 50/50 sulle due mezze linee adiacenti.
            # Es: -0.75 → 50% su -0.5 + 50% su -1.0
            h_low = math.floor(level * 2.0) / 2.0
            h_high = h_low + 0.5
            w_lo, p_lo, l_lo = _calc_half_line(full, h_low)
            w_hi, p_hi, l_hi = _calc_half_line(full, h_high)
            win = 0.5 * w_lo + 0.5 * w_hi
            push = 0.5 * p_lo + 0.5 * p_hi
            lose = 0.5 * l_lo + 0.5 * l_hi
        else:
            win, push, lose = _calc_half_line(full, level)

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
