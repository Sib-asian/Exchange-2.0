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


def render_asian_lines(gol_casa: int = 0, gol_trasf: int = 0, minuto: int = 0, tot_op: float = 2.5) -> dict:
    """
    Render del blocco "Linee Asiatiche" con selettore modalità (full-game o rimanenti).

    MODALITÀ FULL GAME (Betfair Exchange, Pinnacle, la maggior parte degli exchange):
      Le linee live quotano il risultato COMPLETO a 90 minuti.
      Esempio: score 1-0 al 60' → AH live "Home -1.0" significa che la casa
      deve vincere di 1+ goal sull'intera partita (non solo nei 30' rimanenti).

    MODALITÀ GOL RIMANENTI (Asian market puri, alcuni operatori):
      Le linee live quotano solo i GOL RIMANENTI da ora al 90'.
      Esempio: score 1-0 al 60' → AH live "-0.0" potrebbe significare
      che entrambe le squadre hanno 50/50 di segnare di più nei 30' rimanenti.

    Il modello lavora internamente in "gol rimanenti". In modalità full-game
    la conversione avviene automaticamente:
      ah_rimanenti  = ah_full  + (gol_casa − gol_trasf)
      tot_rimanenti = tot_full − (gol_casa + gol_trasf)

    Returns:
        Dict con ah_op, tot_op, ah_cur, tot_cur (sempre in "gol rimanenti").
        Include anche "fullgame_mode" (bool) e i valori raw inseriti dall'utente.
        Include "validation_errors" con eventuali errori di validazione.
    """
    st.header("2. Linee Asiatiche")

    # Validazione input: variabile per raccogliere errori
    validation_errors: list[str] = []

    # Selettore modalità — impatta direttamente la semantica delle linee correnti
    fullgame_mode = st.radio(
        "Modalità linee live:",
        options=["Full Game (Betfair, Pinnacle, exchange standard)", "Gol Rimanenti (mercati asiatici puri)"],
        index=0,
        horizontal=True,
        help=(
            "**Full Game** (raccomandato per Betfair/Pinnacle): le linee live quotano "
            "il risultato finale a 90'. Esempio: Over 2.5 = 2.5 gol totali intera partita.\n\n"
            "**Gol Rimanenti**: le linee live quotano solo i gol da ora al fischio finale. "
            "Più raro, usato da alcuni operatori asiatici."
        ),
    ) == "Full Game (Betfair, Pinnacle, exchange standard)"

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Apertura — full 90'**")
        ah_op  = st.number_input("AH Apertura",     value=-0.25, step=0.25,
                                  help="Handicap asiatico alla quota di apertura (intera partita).")
        tot_op = st.number_input("Totale Apertura", value=2.50,  step=0.25,
                                  help="Over/Under alla quota di apertura (gol totali intera partita).")

    with col_a2:
        if fullgame_mode:
            st.markdown("**Corrente — full game live** *(opzionale, auto-converte in rimanenti)*")
            ah_cur_raw  = st.number_input(
                "AH Corrente (full game)",
                value=ah_op, step=0.25,
                help=(
                    "AH live come quotato sull'exchange (full 90'). "
                    "Se non hai dati live, lascia uguale all'AH Apertura: il modello scala "
                    "automaticamente al tempo rimanente. "
                    "Aggiorna solo se il mercato si è mosso significativamente."
                ),
            )
            tot_cur_raw = st.number_input(
                "Totale Corrente (full game)",
                value=tot_op, step=0.25,
                help=(
                    "Linea Over/Under live come quotata sull'exchange (gol totali da inizio partita). "
                    "Se non hai dati live, lascia uguale al Totale Apertura: il modello scala "
                    "automaticamente al tempo rimanente. "
                    "Aggiorna solo se il mercato si è mosso significativamente."
                ),
            )
            gol_diff = gol_casa - gol_trasf
            gol_tot  = gol_casa + gol_trasf
            ah_cur   = ah_cur_raw  + gol_diff
            tot_cur  = max(0.10, tot_cur_raw - gol_tot)

            if gol_tot > 0:
                st.caption(
                    f"📐 Conversione automatica — AH rimanenti: **{ah_cur:+.2f}** "
                    f"(= {ah_cur_raw:+.2f} + {gol_diff:+d} gol di vantaggio)  "
                    f"· Total rimanenti: **{tot_cur:.2f}** (= {tot_cur_raw:.2f} − {gol_tot} gol segnati)"
                )
            if tot_cur_raw < gol_tot:
                st.error(
                    f"⛔ Impossibile: il Totale Corrente ({tot_cur_raw}) è inferiore ai gol già segnati "
                    f"({gol_tot}). Inserisci la linea full game corretta (es. >{gol_tot + 0.5:.1f})."
                )
        else:
            st.markdown("**Corrente — gol rimanenti** *(opzionale)*")
            ah_cur_raw = st.number_input(
                "AH Corrente (rimanenti)",
                value=ah_op, step=0.25,
                help=(
                    "Handicap asiatico live riferito ai soli gol rimanenti da ora al 90'. "
                    "Se non hai dati live, lascia uguale all'apertura."
                ),
            )
            tot_cur_raw = st.number_input(
                "Totale Corrente (rimanenti)",
                value=tot_op, step=0.25,
                help=(
                    "Gol attesi rimanenti da ora al 90' secondo il mercato. "
                    "Se non hai dati live, lascia uguale all'apertura."
                ),
            )
            ah_cur  = ah_cur_raw
            tot_cur = tot_cur_raw

    # ===================== VALIDAZIONE INPUT CRITICA =====================
    # FIX: Verifica che tot_cur sia plausibile per il minuto corrente.
    # Questo previene risultati sbagliati quando l'utente cambia minuto/punteggio
    # ma dimentica di aggiornare le linee live.
    TOT_TEMPORAL_MAX = 4.0  # massimo realistico per gol/90'
    TOT_BAYES_MIN = 0.20    # minimo per calcoli

    if minuto > 0:
        mins_rem = max(1, 90 - minuto)
        tot_cap = max(TOT_BAYES_MIN, mins_rem / 90.0 * TOT_TEMPORAL_MAX)

        if tot_cur > tot_cap * 1.5:
            # ERRORE BLOCCANTE: tot_cur troppo alto per il minuto
            st.error(
                f"⛔ **LINEE NON AGGIORNATE!**\n\n"
                f"Hai inserito **{tot_cur:.2f} gol rimanenti** al minuto **{minuto}'**, "
                f"ma il massimo realistico è **~{tot_cap:.2f} gol**.\n\n"
                f"**Devi aggiornare le linee live dall'exchange!**\n\n"
                f"Suggerimento: al {minuto}' rimangono ~{mins_rem} minuti, "
                f"quindi cerca linee Total intorno a **{tot_cap:.1f}** gol rimanenti."
            )
            validation_errors.append(f"tot_cur={tot_cur:.2f} > tot_cap={tot_cap:.2f} al minuto {minuto}")

            # Pulsante auto-correzione
            if st.button("🔧 Auto-correggi Total", help=f"Imposta Total a {tot_cap:.2f} gol rimanenti"):
                st.rerun()  # Forza refresh - l'utente dovrà inserire il valore corretto
        elif tot_cur > tot_cap * 1.2:
            # WARNING: tot_cur sospettosamente alto
            st.warning(
                f"⚠️ **Attenzione**: Total rimanenti ({tot_cur:.2f}) sembra alto per il minuto {minuto}'. "
                f"Massimo realistico: {tot_cap:.2f} gol. Verifica le linee live."
            )

    # Validazione AH: se |ah_cur| > tot_cur, è geometricamente impossibile
    if abs(ah_cur) > tot_cur + 0.5:
        st.error(
            f"⛔ **AH impossibile!**\n\n"
            f"Hai inserito AH = **{ah_cur:+.2f}** con Total = **{tot_cur:.2f}**.\n\n"
            f"L'handicap non può superare il totale dei gol attesi. "
            f"Verifica i valori inseriti."
        )
        validation_errors.append(f"|ah_cur|={abs(ah_cur):.2f} > tot_cur={tot_cur:.2f}")

    return {
        "ah_op":        ah_op,
        "tot_op":       tot_op,
        "ah_cur":       ah_cur,
        "tot_cur":      tot_cur,
        "fullgame_mode": fullgame_mode,
        "tot_cur_raw":  tot_cur_raw,   # valore grezzo pre-conversione (per OU default)
        "validation_errors": validation_errors,  # errori di validazione (vuoto = OK)
    }


def render_ou_selector(
    tot_cur_raw: float,
    gol_attuali: int,
    fullgame_mode: bool,
) -> float:
    """
    Render del selettore linea Over/Under.

    La linea analizzata è sempre una linea FULL GAME (gol totali partita).
    Il default viene calcolato come:
      - Full game mode:   closest a tot_cur_raw          (già full game)
      - Remaining mode:   closest a gol_attuali + tot_cur  (remaining → full game)

    Args:
        tot_cur_raw: Valore inserito dall'utente (full game o rimanenti a seconda della modalità).
        gol_attuali: Gol totali già segnati.
        fullgame_mode: True = modalità full game, False = modalità gol rimanenti.

    Returns:
        Linea Over/Under selezionata (sempre full game).
    """
    linee = list(UI.LINEE_OU)

    target = tot_cur_raw if fullgame_mode else gol_attuali + tot_cur_raw

    idx_default = min(range(len(linee)), key=lambda i: abs(linee[i] - target))
    return st.selectbox(
        "Linea U/O da analizzare (full game):",
        linee,
        index=idx_default,
        help=(
            "Linea full game (include i gol già segnati). "
            "Selezionata automaticamente come la più vicina al totale atteso. "
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


def render_exchange_quotes(linea_ou: float) -> ExchangeQuotes:
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
