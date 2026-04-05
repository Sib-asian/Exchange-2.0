"""
Controlli euristici sulle linee asiatiche (prematch, full-game).

Obiettivo: segnalare input sospetti prima che il motore produca numeri fuorvianti.
"""

from __future__ import annotations


def prematch_line_quality(
    *,
    ah_op: float,
    ah_cur_raw: float,
    tot_op: float,
    tot_cur_raw: float,
    linea_ou: float,
    gol_tot: int = 0,
) -> list[str]:
    """
    Restituisce messaggi di avviso (non bloccanti) per il prematch.

    I controlli bloccanti restano in ``render_linee_semplici`` (total < gol, ecc.).
    """
    if gol_tot > 0:
        return []

    out: list[str] = []

    if abs(ah_cur_raw - ah_op) >= 2.25:
        out.append(
            "AH apertura e corrente molto distanti: controlla stesso book e stessa "
            "modalità (full game a 90')."
        )

    if abs(tot_cur_raw - tot_op) >= 1.51:
        out.append(
            "Il total è cambiato molto rispetto all'apertura: verifica che le linee "
            "siano coerenti con il mercato che stai guardando."
        )

    # Avvisa su mismatch O/U vs total di mercato in entrambi i versi:
    # - linea analisi più alta del total inserito (es. 3.0 vs 2.5)
    # - linea analisi più bassa del total inserito (es. 2.25 vs 2.75)
    # Per allineare il confronto modello/mercato conviene usare la stessa linea.
    if tot_op > 0 and abs(linea_ou - tot_op) > 1e-6:
        direction = "sopra" if linea_ou > tot_op else "sotto"
        out.append(
            f"La linea Over/Under **{linea_ou:g}** è **{direction}** il total di mercato **{tot_op:.2f}** "
            "che hai inserito: controlla che entrambi si riferiscano allo stesso listino (full game, stesso book)."
        )

    if linea_ou <= 1.0:
        out.append("Linea O/U molto bassa: assicurati che sia quella su cui punti.")

    return out
