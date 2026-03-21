"""
over_under.py — Calcolo probabilità Over/Under con supporto per quarter lines.

Le quarter lines (es. 2.25, 2.75, 3.25) sono linee asiatiche che dividono
la stake al 50% su due half-lines adiacenti:
  Under 2.75 = ½×Under 2.5 + ½×Under 3.0

Questa è la meccanica standard degli Asian Handicap applicata al Total.
"""

from __future__ import annotations


def _p_under_line(
    full: dict[tuple[int, int], float],
    gol_attuali: int,
    line: float,
) -> float:
    """
    Probabilità effettiva Under per una singola linea (half o intera).

    Comportamento per tipo di linea:
    - Half-line (X.5): P(total < line) — nessun push possibile.
    - Linea intera (X.0): P(total ≤ X−1) + 0.5 × P(total = X)
      Nelle linee intere il punteggio esatto è un PUSH (stake restituita).
      La probabilità effettiva conta il push come mezzo vincita.

    Questo è essenziale per la correttezza delle quarter lines:
      Under 2.75 = ½ × P_eff(U2.5) + ½ × P_eff(U3.0)
      dove P_eff(U3.0) = P(≤2) + 0.5×P(=3), NON P(<3.0) = P(≤2).

    Args:
        full: Matrice bivariata completa normalizzata.
        gol_attuali: Gol già segnati in partita.
        line: Linea Under (es. 2.5, 3.0, 2.0).

    Returns:
        Probabilità effettiva P_eff in [0, 1].
    """
    line4 = round(line * 4)
    if line4 % 4 == 0:
        # Linea intera (es. 2.0, 3.0, 4.0): push al punteggio esatto
        int_line = int(line)
        p_win = sum(p for (a, b), p in full.items() if gol_attuali + a + b < int_line)
        p_push = sum(p for (a, b), p in full.items() if gol_attuali + a + b == int_line)
        return p_win + 0.5 * p_push
    else:
        # Half-line (es. 2.5, 3.5): strict inequality, no push
        return sum(p for (a, b), p in full.items() if gol_attuali + a + b < line)


def calcola_over_under(
    full: dict[tuple[int, int], float],
    gol_attuali: int,
    linea_ou: float,
) -> tuple[float, float]:
    """
    Calcola le probabilità Over/Under con gestione quarter lines.

    Per linee X.25 e X.75 (quarter lines):
      Under X.25 = ½×Under X.0 + ½×Under X.5
      Under X.75 = ½×Under X.5 + ½×Under X+1.0

    Usa la matrice full (include correlazione bivariate + DC) per
    una stima accurata della distribuzione dei gol totali.

    Args:
        full: Matrice bivariata completa normalizzata.
        gol_attuali: Gol totali già segnati (casa + trasf).
        linea_ou: Linea Over/Under (es. 2.5, 2.75, 3.0).

    Returns:
        (p_under, p_over): Probabilità normalizzate in [0, 1].
    """
    line4 = round(linea_ou * 4)

    if line4 % 2 != 0:
        # Quarter line (es. 2.25, 2.75, 3.25): split 50/50 sulle due semi-linee adiacenti.
        # IMPORTANTE: una delle due semi-linee può essere una linea intera con push.
        # Usa _p_under_line che gestisce correttamente sia half-line che linee intere.
        h_low = (line4 - 1) / 4.0   # es. 2.75 → 2.5 (half), 2.25 → 2.0 (intera)
        h_high = (line4 + 1) / 4.0  # es. 2.75 → 3.0 (intera), 2.25 → 2.5 (half)
        p_u_low = _p_under_line(full, gol_attuali, h_low)
        p_u_high = _p_under_line(full, gol_attuali, h_high)
        p_under = 0.5 * (p_u_low + p_u_high)
    else:
        # Half-line (es. 2.5, 3.5) o linea intera (es. 2.0, 3.0)
        p_under = _p_under_line(full, gol_attuali, linea_ou)

    p_under = min(max(p_under, 0.0), 1.0)
    p_over = 1.0 - p_under

    return p_under, p_over
