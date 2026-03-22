"""
clean_sheet.py — Calcolo probabilità Clean Sheet (squadra non subisce gol).

La probabilità Clean Sheet è calcolata dalla matrice bivariata:
- Clean Sheet Casa: P(trasferta non segna nessun gol) = P(Y_rem = 0)
- Clean Sheet Trasferta: P(casa non segna nessun gol) = P(X_rem = 0)

In live, deve considerare i gol già segnati:
- Se la squadra avversaria ha già segnato, Clean Sheet = 0.0
- Altrimenti, calcolato dai gol rimanenti attesi
"""

from __future__ import annotations


def calcola_clean_sheet(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
) -> tuple[float, float]:
    """
    Calcola P(Clean Sheet) per entrambe le squadre.

    Args:
        full: Matrice bivariata completa normalizzata (gol rimanenti).
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.

    Returns:
        (p_clean_casa, p_clean_trasf): Probabilità che casa/trasferta
            non subiscano gol per il resto della partita.

    Note:
        - Se la trasferta ha già segnato (gol_trasf > 0), Clean Sheet Casa = 0.0
        - Se la casa ha già segnato (gol_casa > 0), Clean Sheet Trasferta = 0.0
    """
    # Se la squadra avversaria ha già segnato, clean sheet impossibile
    if gol_trasf > 0 and gol_casa > 0:
        # Entrambe hanno segnato → clean sheet impossibile per entrambe
        return 0.0, 0.0

    if gol_trasf > 0:
        # Trasferta ha segnato → casa non può avere clean sheet
        # Calcoliamo solo clean sheet trasferta
        p_clean_trasf = sum(p for (a, b), p in full.items() if a == 0)
        return 0.0, min(1.0, max(0.0, p_clean_trasf))

    if gol_casa > 0:
        # Casa ha segnato → trasferta non può avere clean sheet
        p_clean_casa = sum(p for (a, b), p in full.items() if b == 0)
        return min(1.0, max(0.0, p_clean_casa)), 0.0

    # Nessuna squadra ha segnato: calcola per entrambe
    # Clean Sheet Casa = P(trasferta non segna) = P(Y_rem = 0)
    p_clean_casa = sum(p for (a, b), p in full.items() if b == 0)

    # Clean Sheet Trasferta = P(casa non segna) = P(X_rem = 0)
    p_clean_trasf = sum(p for (a, b), p in full.items() if a == 0)

    return min(1.0, max(0.0, p_clean_casa)), min(1.0, max(0.0, p_clean_trasf))


def calcola_win_to_nil(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    p1: float,
    p2: float,
) -> tuple[float, float]:
    """
    Calcola P(Win to Nil) per entrambe le squadre.

    Win to Nil = la squadra vince senza subire gol.
    - Win to Nil Casa: casa vince E trasferta non segna
    - Win to Nil Trasferta: trasferta vince E casa non segna

    Args:
        full: Matrice bivariata completa normalizzata (gol rimanenti).
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.
        p1: Probabilità vittoria casa (già calcolata).
        p2: Probabilità vittoria trasferta (già calcolata).

    Returns:
        (p_wtn_casa, p_wtn_trasf): Probabilità Win to Nil.
    """
    # Se trasferta ha già segnato, Win to Nil casa impossibile
    if gol_trasf > 0:
        # Trasferta può ancora fare win to nil se vince senza subire altri gol
        p_wtn_trasf = sum(p for (a, b), p in full.items()
                         if gol_trasf + b > gol_casa + a and a == 0)
        return 0.0, min(1.0, max(0.0, p_wtn_trasf))

    # Se casa ha già segnato, Win to Nil trasferta impossibile
    if gol_casa > 0:
        p_wtn_casa = sum(p for (a, b), p in full.items()
                        if gol_casa + a > gol_trasf + b and b == 0)
        return min(1.0, max(0.0, p_wtn_casa)), 0.0

    # Nessuna squadra ha segnato
    # Win to Nil Casa: casa segna almeno 1, trasferta segna 0
    p_wtn_casa = sum(p for (a, b), p in full.items() if a > 0 and b == 0)

    # Win to Nil Trasferta: trasferta segna almeno 1, casa segna 0
    p_wtn_trasf = sum(p for (a, b), p in full.items() if a == 0 and b > 0)

    return min(1.0, max(0.0, p_wtn_casa)), min(1.0, max(0.0, p_wtn_trasf))
