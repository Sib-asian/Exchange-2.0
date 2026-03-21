"""
inputs.py — Widget di input Streamlit per Radar Pro Live.

Centralizza tutta la logica di input dell'interfaccia utente:
  - Stato partita (minuto, gol, cartellini)
  - Linee asiatiche (apertura + corrente)
  - Selezione linea Over/Under
  - Bankroll e commissione
  - Tiri live
  - Quote exchange (opzionali)

Restituisce oggetti tipizzati (MatchState, ExchangeQuotes) pronti per il motore.
"""

from __future__ import annotations

import streamlit as st

from src.config import UI
from src.engine import ExchangeQuotes, MatchState


def render_match_state() -> dict:
    """
    Render del blocco "Stato Partita" e raccolta dei valori.

    Returns:
        Dict con i valori grezzi (minuto, gol, rossi).
    """
    st.header("1. Stato Partita")
    minuto = st.slider("Minuto Attuale", 0, 90, 0, 1)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gol_casa = st.number_input("Gol CASA", value=0, min_value=0, max_value=20)
    with col_g2:
        gol_trasf = st.number_input("Gol TRASF.", value=0, min_value=0, max_value=20)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rossi_casa = st.number_input("Rossi CASA", value=0, min_value=0, max_value=4)
    with col_r2:
        rossi_trasf = st.number_input("Rossi TRASF.", value=0, min_value=0, max_value=4)

    return {
        "minuto": minuto,
        "gol_casa": gol_casa,
        "gol_trasf": gol_trasf,
        "rossi_casa": rossi_casa,
        "rossi_trasf": rossi_trasf,
    }


def render_asian_lines() -> dict:
    """
    Render del blocco "Linee Asiatiche" e raccolta dei valori.

    Returns:
        Dict con ah_op, tot_op, ah_cur, tot_cur.
    """
    st.header("2. Linee Asiatiche")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Apertura — full 90'**")
        ah_op = st.number_input("AH Apertura", value=-0.25, step=0.25)
        tot_op = st.number_input("Totale Apertura", value=2.50, step=0.25)
    with col_a2:
        st.markdown("**Corrente — gol rimanenti**")
        ah_cur = st.number_input("AH Corrente", value=-0.75, step=0.25)
        tot_cur = st.number_input("Totale Corrente", value=2.75, step=0.25)

    return {
        "ah_op": ah_op,
        "tot_op": tot_op,
        "ah_cur": ah_cur,
        "tot_cur": tot_cur,
    }


def render_ou_selector(tot_cur: float) -> float:
    """
    Render del selettore linea Over/Under.

    Seleziona automaticamente la linea più vicina al Total corrente.

    Args:
        tot_cur: Total corrente per la selezione automatica.

    Returns:
        Linea Over/Under selezionata.
    """
    linee = list(UI.LINEE_OU)
    idx_default = min(range(len(linee)), key=lambda i: abs(linee[i] - tot_cur))
    return st.selectbox(
        "Linea U/O da analizzare:",
        linee,
        index=idx_default,
        help=(
            "Selezionata automaticamente in base al Totale Corrente. "
            "Le linee X.75/X.25 sono Asian quarter lines (mezzo stake su ogni semi-linea)."
        ),
    )


def render_bankroll() -> tuple[float, float, float]:
    """
    Render dei widget bankroll e commissione.

    Returns:
        (bankroll, comm_pct, comm_rate)
    """
    col_bk, col_cm = st.columns(2)
    with col_bk:
        bankroll = st.number_input(
            "Bankroll (€)", value=UI.BANKROLL_DEFAULT, step=UI.BANKROLL_STEP, min_value=1.0,
        )
    with col_cm:
        comm_pct = st.number_input(
            "Commissione exchange (%)",
            value=UI.COMM_DEFAULT,
            min_value=0.0,
            max_value=UI.COMM_MAX,
            step=UI.COMM_STEP,
            help="Betfair Standard: 2-5%. Usata per calcolare l'edge netto.",
        )
    return bankroll, comm_pct, comm_pct / 100.0


def render_shots(minuto: int) -> tuple[int, int, int, int]:
    """
    Render dei widget per i tiri live con validazione.

    Args:
        minuto: Minuto attuale (per validazione).

    Returns:
        (sot_h, soff_h, sot_a, soff_a)
    """
    st.header("3. Tiri (Live)")
    st.caption("Lascia a 0 per analisi prematch. Live: inserisci i tiri totali da inizio gara.")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        sot_h = st.number_input("In porta CASA", min_value=0, value=0, step=1)
    with col_t2:
        soff_h = st.number_input("Fuori CASA", min_value=0, value=0, step=1)
    with col_t3:
        sot_a = st.number_input("In porta TRASF.", min_value=0, value=0, step=1)
    with col_t4:
        soff_a = st.number_input("Fuori TRASF.", min_value=0, value=0, step=1)

    # Validazione coerenza tiri / minuto
    n_tiri = sot_h + soff_h + sot_a + soff_a
    if n_tiri > 0 and minuto == 0:
        st.warning(
            "⚠️ Tiri inseriti ma minuto = 0: per analisi prematch lascia tutti i tiri a 0, "
            "altrimenti il modello proietta il rate su 90' interi."
        )
    elif n_tiri > 0 and minuto > 0:
        tiri_max = max(UI.TIRI_MIN_BASE, int(minuto * UI.TIRI_PER_MINUTO) + UI.TIRI_WARNING_BUFFER)
        if n_tiri > tiri_max:
            st.warning(
                f"⚠️ {n_tiri} tiri totali al {minuto}' sembra elevato "
                f"(atteso ≤ ~{tiri_max}) — verifica che siano tiri totali da inizio gara, "
                "non dell'ultimo periodo."
            )

    return sot_h, soff_h, sot_a, soff_a


def render_exchange_quotes(linea_ou: float) -> "ExchangeQuotes":
    """
    Render del pannello quote exchange (opzionale, in expander).

    Args:
        linea_ou: Linea Over/Under selezionata (per le label).

    Returns:
        ExchangeQuotes con le quote inserite.
    """
    with st.expander("⚙️ Analisi avanzata con quote exchange (opzionale)"):
        st.caption(
            "Inserisci le quote che vedi sull'exchange per ottenere edge preciso, "
            "stake Kelly in € e EV atteso. Lascia a 0 i mercati che non ti interessano."
        )
        eq1, eqx, eq2 = st.columns(3)
        q_1 = eq1.number_input("Quota 1 (Casa)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_x = eqx.number_input("Quota X (Pareggio)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_2 = eq2.number_input("Quota 2 (Trasf.)", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        equ, eqo, eqb = st.columns(3)
        q_under = equ.number_input(f"Quota Under {linea_ou}", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_over = eqo.number_input(f"Quota Over {linea_ou}", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_btts_si = eqb.number_input("Quota BTTS Sì", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        eqbn_col, _, _ = st.columns(3)
        q_btts_no = eqbn_col.number_input("Quota BTTS No", min_value=0.0, value=0.0, step=0.05, format="%.2f")

    return ExchangeQuotes(
        q_1=q_1, q_x=q_x, q_2=q_2,
        q_over=q_over, q_under=q_under,
        q_btts_si=q_btts_si, q_btts_no=q_btts_no,
    )


def build_match_state(
    match: dict,
    lines: dict,
    linea_ou: float,
    bankroll: float,
    comm_rate: float,
    shots: tuple[int, int, int, int],
) -> MatchState:
    """
    Costruisce il MatchState validato dai valori dei widget.

    Args:
        match: Dict con minuto, gol, rossi.
        lines: Dict con linee asiatiche.
        linea_ou: Linea Over/Under.
        bankroll: Capitale.
        comm_rate: Commissione.
        shots: (sot_h, soff_h, sot_a, soff_a).

    Returns:
        MatchState validato.
    """
    sot_h, soff_h, sot_a, soff_a = shots
    return MatchState(
        minuto=match["minuto"],
        gol_casa=match["gol_casa"],
        gol_trasf=match["gol_trasf"],
        rossi_casa=match["rossi_casa"],
        rossi_trasf=match["rossi_trasf"],
        ah_op=lines["ah_op"],
        tot_op=lines["tot_op"],
        ah_cur=lines["ah_cur"],
        tot_cur=lines["tot_cur"],
        linea_ou=linea_ou,
        sot_h=sot_h, soff_h=soff_h,
        sot_a=sot_a, soff_a=soff_a,
        bankroll=bankroll,
        comm_rate=comm_rate,
    )
