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

    if tot_op > 0 and linea_ou >= tot_op - 0.25:
        out.append(
            f"Stai analizzando Over/Under **{linea_ou:g}** con total di mercato **{tot_op:.2f}**: "
            "di solito le quote O/U sono su linee più basse del total — controlla."
        )

    if linea_ou <= 1.0:
        out.append("Linea O/U molto bassa: assicurati che sia quella su cui punti.")

    return out
