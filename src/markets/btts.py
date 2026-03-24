"""
btts.py — Calcolo probabilità BTTS (Both Teams To Score).

La probabilità BTTS è calcolata condizionalmente sullo stato attuale:
- Se entrambe hanno già segnato: BTTS Sì = 1.0 (mercato chiuso)
- Se solo una ha segnato: BTTS Sì = P(l'altra segna almeno 1 nel rimasto)
- Se nessuna ha segnato: BTTS Sì = P(entrambe segnano almeno 1 nel rimasto)

Usa la matrice full che include la correlazione bivariate + DC.
"""

from __future__ import annotations


def calcola_btts(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
) -> float:
    """
    Calcola P(BTTS Sì) tenendo conto del punteggio attuale.

    La matrice full contiene P(X_rem=a, Y_rem=b) per i GOL RIMANENTI.
    BTTS richiede che entrambe le squadre abbiano almeno 1 gol nel
    punteggio FINALE: quindi combina gol già segnati con gol rimanenti.

    Args:
        full: Matrice bivariata completa normalizzata (gol rimanenti).
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.

    Returns:
        P(BTTS Sì) in [0, 1].
    """
    if gol_casa > 0 and gol_trasf > 0:
        # Entrambe hanno già segnato: BTTS Sì garantito
        return 1.0
    elif gol_casa > 0:
        # Casa ha già segnato; BTTS iff trasferta segna almeno 1 nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if b > 0)
    elif gol_trasf > 0:
        # Trasferta ha già segnato; BTTS iff casa segna almeno 1 nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if a > 0)
    else:
        # Nessuna ha segnato; BTTS iff entrambe segnano nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if a > 0 and b > 0)

    return min(1.0, max(0.0, p_btts))
