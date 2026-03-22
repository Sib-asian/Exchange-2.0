"""
app.py — Radar Pro Live v2.0
Entry point Streamlit. Orchestra input → engine → output.
"""

import streamlit as st

from src.config import BAYES, SIGNALS, UI
from src.engine import analizza
from src.models.kelly import calcola_kelly_fraction
from src.signals import (
    calcola_soglie,
    genera_segnali_avanzati,
    genera_segnali_rapidi,
)
from src.ui.inputs import (
    build_match_state,
    render_asian_lines,
    render_bankroll,
    render_exchange_quotes,
    render_match_state,
    render_ou_selector,
    render_shots,
)
from src.ui.outputs import (
    render_allineamento_mercato,
    render_asian_handicap,
    render_avvisi_affidabilita,
    render_avvisi_incoerenza,
    render_coerenza_mercati,
    render_confidence_bands,
    render_correct_score,
    render_debug,
    render_mercati_chiusi,
    render_model_confidence,
    render_momentum,
    render_quote_fair,
    render_red_card_impact,
    render_segnali_avanzati,
    render_segnali_rapidi,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=UI.PAGE_TITLE,
    page_icon=UI.PAGE_ICON,
    layout=UI.LAYOUT,
)
st.title(f"{UI.PAGE_ICON} {UI.PAGE_TITLE}")
st.caption(f"v{UI.VERSION} · Modello bivariate Poisson + Dixon-Coles + Kelly frazionato")

# ── Input ────────────────────────────────────────────────────────────────────
match = render_match_state()
st.divider()

lines = render_asian_lines(match["gol_casa"], match["gol_trasf"])
gol_attuali = match["gol_casa"] + match["gol_trasf"]
linea_ou = render_ou_selector(lines["tot_cur_raw"], gol_attuali, lines["fullgame_mode"])
bankroll, comm_pct, comm_rate = render_bankroll()
st.divider()

shots = render_shots(match["minuto"])
st.divider()

# ── Analisi ──────────────────────────────────────────────────────────────────
if st.button("ANALIZZA", use_container_width=True, type="primary"):

    try:
        state = build_match_state(match, lines, linea_ou, bankroll, comm_rate, shots)
    except AssertionError as e:
        st.error(f"❌ Input non valido: {e}")
        st.stop()

    # Avviso linee live non aggiornate: se tot_cur (gol rimanenti) è implausibilmente
    # alto per il minuto giocato, l'utente probabilmente non ha aggiornato le linee.
    # Il motore applica comunque un cap automatico, ma meglio avvisare esplicitamente.
    if state.minuto >= 15:
        _mins_rem_warn = max(1, 90 - state.minuto)
        _expected_max = max(0.25, _mins_rem_warn / 90.0 * BAYES.TOT_TEMPORAL_MAX)
        if state.tot_cur > _expected_max * 1.5:
            st.warning(
                f"⚠️ **Linee live non aggiornate?** "
                f"Il Total Rimanente ({state.tot_cur:.2f} gol) è implausibile al {state.minuto}' "
                f"(massimo realistico ≈ {_expected_max:.2f} gol). "
                f"Aggiorna **AH Corrente** e **Totale Corrente** con i valori live prima di analizzare."
            )

    with st.spinner("Calcolo matrice bivariata..."):
        risultati = analizza(state)

    n_shots_tot = sum(shots)
    gol_attuali = state.gol_casa + state.gol_trasf  # recompute from validated state

    # ── Quote Fair ───────────────────────────────────────────────────────────
    render_quote_fair(risultati, state.minuto, state.gol_casa, state.gol_trasf, linea_ou)
    render_model_confidence(risultati)
    render_confidence_bands(risultati, linea_ou)
    render_coerenza_mercati(risultati, linea_ou)
    render_correct_score(risultati)
    render_asian_handicap(risultati.full_matrix)

    # ── Red card impact (Fix #16) ─────────────────────────────────────────────
    render_red_card_impact(state.rossi_casa, state.rossi_trasf, state.minuto)

    # ── Momentum con decomposizione AH/Total (Fix #14) ───────────────────────
    st.divider()
    render_momentum(risultati.momentum, risultati.delta_ah, risultati.delta_tot)

    # ── Fine partita ─────────────────────────────────────────────────────────
    if state.minuto >= SIGNALS.GAME_END_THRESHOLD:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()

    # ── Segnali rapidi ───────────────────────────────────────────────────────
    st.divider()
    st.header("Segnali rapidi")
    st.caption("Nessuna quota da inserire — confronta a occhio con l'exchange.")

    segnali_rapidi = genera_segnali_rapidi(
        risultati.p1, risultati.px, risultati.p2,
        risultati.p_over, risultati.p_under,
        risultati.p_btts,
        state.minuto, linea_ou, gol_attuali,
        model_confidence=risultati.model_confidence,
        model_agreement=risultati.model_agreement,
    )

    # Mostra mercati chiusi prima dei segnali
    settled = render_mercati_chiusi(
        gol_attuali, linea_ou,
        state.gol_casa, state.gol_trasf,
        state.minuto, risultati.p_btts,
    )

    # Filtra i segnali per mercati chiusi (usa .upper() per robustezza)
    if settled.get("ou_vinto"):
        segnali_rapidi = [s for s in segnali_rapidi if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
    if settled.get("btts_si_settled") or settled.get("btts_no_settled"):
        segnali_rapidi = [s for s in segnali_rapidi if "BTTS" not in s.mercato.upper()]

    render_segnali_rapidi(segnali_rapidi)
    render_avvisi_affidabilita(risultati.flat_lines, n_shots_tot, state.minuto)

    # ── Analisi avanzata con quote exchange ──────────────────────────────────
    st.divider()
    quotes = render_exchange_quotes(linea_ou)

    kelly_frac = calcola_kelly_fraction(state.minuto, n_shots_tot, risultati.model_confidence)
    momentum_factor = max(
        SIGNALS.MOMENTUM_STAKE_FLOOR,
        1.0 - SIGNALS.MOMENTUM_STAKE_REDUCTION_RATE * max(0.0, risultati.momentum - SIGNALS.MOMENTUM_STAKE_THRESHOLD),
    )

    segnali_avanzati = genera_segnali_avanzati(
        risultati.p1, risultati.px, risultati.p2,
        risultati.p_over, risultati.p_under,
        risultati.p_btts,
        quotes,
        state.minuto, linea_ou, gol_attuali,
        bankroll, comm_rate, n_shots_tot,
        risultati.momentum,
        model_confidence=risultati.model_confidence,
        model_agreement=risultati.model_agreement,
    )

    # Filtra segnali avanzati per mercati già chiusi (stessa logica dei rapidi)
    if settled.get("ou_vinto"):
        segnali_avanzati = [s for s in segnali_avanzati if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
    if settled.get("btts_si_settled") or settled.get("btts_no_settled"):
        segnali_avanzati = [s for s in segnali_avanzati if "BTTS" not in s.mercato.upper()]

    render_segnali_avanzati(segnali_avanzati, quotes.any_active)
    render_allineamento_mercato(risultati, quotes)

    # Fix #17: calcola e mostra market_divergence se ci sono quote exchange
    if quotes.any_active:
        from src.models.consensus import compute_model_market_divergence
        model_probs_map = {
            "p1": risultati.p1, "px": risultati.px, "p2": risultati.p2,
            "p_over": risultati.p_over, "p_under": risultati.p_under,
            "p_btts": risultati.p_btts,
        }
        market_probs_map: dict[str, float] = {}
        if quotes.q_1 > 1.0:
            market_probs_map["p1"] = 1.0 / quotes.q_1
        if quotes.q_x > 1.0:
            market_probs_map["px"] = 1.0 / quotes.q_x
        if quotes.q_2 > 1.0:
            market_probs_map["p2"] = 1.0 / quotes.q_2
        if quotes.q_over > 1.0:
            market_probs_map["p_over"] = 1.0 / quotes.q_over
        if quotes.q_under > 1.0:
            market_probs_map["p_under"] = 1.0 / quotes.q_under
        if quotes.q_btts_si > 1.0:
            market_probs_map["p_btts"] = 1.0 / quotes.q_btts_si
        risultati.market_divergence = compute_model_market_divergence(model_probs_map, market_probs_map)

    # ── Avvisi incoerenza ────────────────────────────────────────────────────
    render_avvisi_incoerenza(
        risultati.p_btts, risultati.p_under, risultati.p_over,
        linea_ou, gol_attuali,
        btts_settled=settled.get("btts_si_settled", False),
    )

    # ── Debug ────────────────────────────────────────────────────────────────
    st.divider()
    soglie = calcola_soglie(state.minuto, linea_ou, gol_attuali)
    render_debug(
        risultati, linea_ou, state.minuto,
        soglie, comm_pct, momentum_factor, n_shots_tot,
    )
