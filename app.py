"""
app.py — Radar Pro Live v2.0
Entry point Streamlit. Orchestra input → engine → output.
"""

import streamlit as st

from src.config import INPUT_VALIDATION, SIGNALS, UI
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
    render_extracted_data_panel,
    render_image_upload,
    render_match_state_live,
    render_ou_selector,
)
from src.ui.outputs import (
    render_allineamento_mercato,
    render_anomalies,
    render_asian_handicap,
    render_avvisi_affidabilita,
    render_avvisi_incoerenza,
    render_clean_sheet,
    render_coerenza_mercati,
    render_confidence_bands,
    render_correct_score,
    render_debug,
    render_lines_need_update,
    render_mercati_chiusi,
    render_model_confidence,
    render_momentum,
    render_ocr_market_divergence,
    render_quote_fair,
    render_red_card_impact,
    render_risk_metrics,
    render_segnali_avanzati,
    render_segnali_rapidi,
    render_session_stats,
    render_value_bets,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=UI.PAGE_TITLE,
    page_icon=UI.PAGE_ICON,
    layout=UI.LAYOUT,
)
st.title(f"{UI.PAGE_ICON} {UI.PAGE_TITLE}")
st.caption(f"v{UI.VERSION} · Modello bivariate Poisson + Dixon-Coles + Kelly frazionato")

# ── Step 1: Linee Asiatiche (Spread / Total) ─────────────────────────────────
# Le linee vanno inserite per prime: servono sia al prematch sia al live.
# Leggiamo minuto/gol dal session state (impostati dalla sezione live al render precedente).
_live_min = st.session_state.get("live_minuto", 0)
_live_gh = st.session_state.get("live_gol_casa", 0)
_live_ga = st.session_state.get("live_gol_trasf", 0)

lines = render_asian_lines(_live_gh, _live_ga, _live_min)
gol_attuali = _live_gh + _live_ga
linea_ou = render_ou_selector(lines["tot_cur_raw"], gol_attuali, lines["fullgame_mode"])
bankroll, comm_pct, comm_rate = render_bankroll()

# Mostra avviso se ci sono errori di validazione
if lines.get("validation_errors"):
    st.warning(f"⚠️ Rilevati {len(lines['validation_errors'])} problemi di validazione. "
               f"Correggi i valori prima di analizzare.")

# FIX: Blocca l'analisi se ci sono errori critici
if lines.get("blocking_errors") and INPUT_VALIDATION.BLOCK_ON_CRITICAL_ERRORS:
    st.error(f"🚫 **ANALISI BLOCCATA**: Rilevati {len(lines['blocking_errors'])} errori critici nei dati. "
             f"Correggi i valori evidenziati prima di procedere.")
    st.stop()
st.divider()

# ── Step 2: Screenshot Quote (Prematch) ──────────────────────────────────────
with st.expander("📷 Carica Screenshot Quote (prematch)", expanded=False):
    extracted_data = render_image_upload()
    if extracted_data is not None:
        _ocr_data = render_extracted_data_panel(extracted_data)  # noqa: F841

st.divider()

# ── Step 3: Stato Partita Live ───────────────────────────────────────────────
match = render_match_state_live()
st.divider()

# ── Contesto Pre-Partita (badge visibile prima di ANALIZZA) ──────────────────
# La ricerca è opzionale: se fatta dalla pagina 🔍 Ricerca Partita, i dati
# entrano automaticamente nel calcolo senza nessuna azione aggiuntiva.
_ricerca = st.session_state.get("ricerca_risultato")
_ricerca_ok = _ricerca is not None and _ricerca.success

if _ricerca_ok:
    # Badge compatto sempre visibile — l'utente vede esattamente cosa entra nel calcolo
    _aff_icon = {"alta": "🟢", "media": "🟡", "bassa": "🔴"}.get(_ricerca.affidabilita, "⚪")
    _badge_parts = [f"{_ricerca.squadra_casa} vs {_ricerca.squadra_trasf} {_aff_icon}"]
    if _ricerca.h2h_media_gol > 0:
        _badge_parts.append(f"H2H {_ricerca.h2h_media_gol:.1f} gol/p")
    if _ricerca.adj_tot != 0:
        _badge_parts.append(f"Δtot {_ricerca.adj_tot:+.2f}")
    if _ricerca.adj_ah != 0:
        _badge_parts.append(f"ΔAH {_ricerca.adj_ah:+.2f}")
    _abs_tot = len(_ricerca.assenze_casa) + len(_ricerca.assenze_trasf)
    if _abs_tot:
        _badge_parts.append(f"{_abs_tot} assenz{'a' if _abs_tot == 1 else 'e'}")

    _bcol, _xcol = st.columns([6, 1])
    with _bcol:
        st.success("🔍 Ricerca attiva · " + " · ".join(_badge_parts))
    with _xcol:
        if st.button("✕ Rimuovi", key="remove_ricerca"):
            del st.session_state["ricerca_risultato"]
            st.rerun()

    with st.expander("Vedi dettagli ricerca", expanded=False):
        _dc1, _dc2 = st.columns(2)
        with _dc1:
            if _ricerca.assenze_casa:
                st.markdown("**Assenze " + _ricerca.squadra_casa + "**")
                for a in _ricerca.assenze_casa:
                    st.markdown(f"- ❌ {a}")
            if _ricerca.forma_casa:
                st.markdown(f"**Forma {_ricerca.squadra_casa}**: `{_ricerca.forma_casa}`")
        with _dc2:
            if _ricerca.assenze_trasf:
                st.markdown("**Assenze " + _ricerca.squadra_trasf + "**")
                for a in _ricerca.assenze_trasf:
                    st.markdown(f"- ❌ {a}")
            if _ricerca.forma_trasf:
                st.markdown(f"**Forma {_ricerca.squadra_trasf}**: `{_ricerca.forma_trasf}`")
        if _ricerca.contesto:
            st.info(f"**Contesto**: {_ricerca.contesto}")
        if _ricerca.note_aggiustamento:
            st.caption(f"Aggiustamenti: {_ricerca.note_aggiustamento}")
else:
    st.page_link("pages/2_🔍_Ricerca_Partita.py", label="💡 Aggiungi contesto pre-partita (assenze, H2H, forma)", icon="🔍")

st.divider()

# ── Analisi ──────────────────────────────────────────────────────────────────
if st.button("ANALIZZA", use_container_width=True, type="primary"):

    # Linea O/U estratta da OCR: usata come soft prior Bayesiano in prematch.
    _ocr_total = 0.0
    if extracted_data is not None and extracted_data.extraction_success:
        if extracted_data.linea_ou > 0.5:
            _ocr_total = extracted_data.linea_ou

    _ocr_q1    = extracted_data.quota_1    if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_qx    = extracted_data.quota_x    if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_q2    = extracted_data.quota_2    if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_qover = extracted_data.quota_over if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_qund  = extracted_data.quota_under if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_qgg   = extracted_data.quota_gg   if extracted_data is not None and extracted_data.extraction_success else 0.0
    _ocr_qng   = extracted_data.quota_ng   if extracted_data is not None and extracted_data.extraction_success else 0.0

    # ── Arricchimento Gemini (3 call in parallelo) ────────────────────────────
    # Eseguite automaticamente se:
    #   1. C'è uno screenshot OCR con nomi squadre riconosciuti
    #   2. La partita è in prematch (minuto == 0): dati H2H e validazione quote sono prematch-only
    #   3. C'è almeno un movimento di linea (per interpreta_movimento) oppure quote OCR (per valida/prior)
    # I risultati sono cachati in session_state per evitare re-chiamate se l'utente
    # preme ANALIZZA più volte senza cambiare screenshot o linee.
    _fixture_prior = 0.0
    _movement_quality = 1.0
    _ocr_confidence_scale = 1.0
    _ricerca_adj_tot = 0.0
    _ricerca_adj_ah = 0.0

    # Usa dati dalla Ricerca Pre-Partita se disponibili
    if _ricerca_ok:
        if _ricerca.h2h_media_gol > 0:
            _fixture_prior = _ricerca.h2h_media_gol
        _ricerca_adj_tot = _ricerca.adj_tot
        _ricerca_adj_ah = _ricerca.adj_ah

    _has_ocr_teams = (
        extracted_data is not None
        and extracted_data.extraction_success
        and bool(extracted_data.squadra_casa)
        and bool(extracted_data.squadra_trasf)
    )
    # Fallback: usa nomi squadre dalla ricerca se OCR non li ha riconosciuti
    _has_ricerca_teams = _ricerca_ok and bool(_ricerca.squadra_casa) and bool(_ricerca.squadra_trasf)
    _has_teams = _has_ocr_teams or _has_ricerca_teams
    _is_prematch = _live_min == 0

    if _has_teams:
        from concurrent.futures import ThreadPoolExecutor

        from src.research import _get_gemini_api_key as _check_key
        from src.research import (
            cerca_prior_storico,
            interpreta_movimento_linee,
            valida_quote_ocr,
        )

        _squadra_casa = extracted_data.squadra_casa if _has_ocr_teams else _ricerca.squadra_casa
        _squadra_trasf = extracted_data.squadra_trasf if _has_ocr_teams else _ricerca.squadra_trasf

        # Delta AH puro: rimuove l'offset meccanico del punteggio
        _gol_diff_live = _live_gh - _live_ga
        _gol_tot_live  = _live_gh + _live_ga
        _ah_cur_full   = lines["ah_cur"] - _gol_diff_live  # back to full-game
        _delta_ah_pure = _ah_cur_full - lines["ah_op"]
        _delta_tot_pure = lines["tot_cur_raw"] - lines["tot_op"]  # già in full-game

        # Cache key: si ricalcola solo se squadre, linee o screenshot cambiano
        _gemini_cache_key = (
            _squadra_casa, _squadra_trasf,
            round(_delta_ah_pure, 2), round(_delta_tot_pure, 2),
            st.session_state.get("last_ocr_file_id", ""),
        )
        _cached = st.session_state.get("gemini_enrichment")

        if _cached is None or _cached.get("_key") != _gemini_cache_key:
            if _check_key():
                with st.spinner("Gemini analizza il contesto partita..."):
                    try:
                        with ThreadPoolExecutor(max_workers=3) as _executor:
                            # A: validatore quote OCR (solo se ci sono quote e prematch)
                            if _is_prematch and (_ocr_q1 > 1.0 or _ocr_qover > 1.0):
                                _f_ocr = _executor.submit(
                                    valida_quote_ocr,
                                    _squadra_casa, _squadra_trasf,
                                    _ocr_q1, _ocr_qx, _ocr_q2,
                                    _ocr_qover, _ocr_qund,
                                    linea_ou,
                                )
                            else:
                                _f_ocr = None

                            # B: prior storico H2H (solo prematch, skip se già dalla Ricerca)
                            if _is_prematch and _fixture_prior <= 0:
                                _f_prior = _executor.submit(
                                    cerca_prior_storico,
                                    _squadra_casa, _squadra_trasf,
                                )
                            else:
                                _f_prior = None

                            # C: interpretazione movimento linee (se c'è movimento)
                            if abs(_delta_ah_pure) >= 0.15 or abs(_delta_tot_pure) >= 0.15:
                                _f_mov = _executor.submit(
                                    interpreta_movimento_linee,
                                    _squadra_casa, _squadra_trasf,
                                    _delta_ah_pure, _delta_tot_pure,
                                )
                            else:
                                _f_mov = None

                        # Raccogli risultati
                        _ocr_val = _f_ocr.result() if _f_ocr else {"confidence_scale": 1.0, "flags": []}
                        _fix_val = _f_prior.result() if _f_prior else _fixture_prior
                        _mov_val = _f_mov.result() if _f_mov else 1.0

                        _new_cache = {
                            "_key": _gemini_cache_key,
                            "ocr_confidence_scale": _ocr_val["confidence_scale"],
                            "ocr_flags": _ocr_val["flags"],
                            "fixture_prior": _fix_val,
                            "movement_quality": _mov_val,
                        }
                        st.session_state["gemini_enrichment"] = _new_cache
                    except Exception:
                        _new_cache = {
                            "_key": _gemini_cache_key,
                            "ocr_confidence_scale": 1.0,
                            "ocr_flags": [],
                            "fixture_prior": 0.0,
                            "movement_quality": 1.0,
                        }
                        st.session_state["gemini_enrichment"] = _new_cache

        _cached = st.session_state.get("gemini_enrichment", {})
        _ocr_confidence_scale = _cached.get("ocr_confidence_scale", 1.0)
        _fixture_prior = _cached.get("fixture_prior", 0.0)
        _movement_quality = _cached.get("movement_quality", 1.0)

        # Mostra avvisi Gemini se rilevanti
        _ocr_flags = _cached.get("ocr_flags", [])
        if _ocr_flags:
            st.warning(
                "⚠️ **Gemini: anomalie quote OCR** — "
                + " | ".join(_ocr_flags[:3])
                + f" (confidenza ridotta: {_ocr_confidence_scale:.0%})"
            )
        if _fixture_prior > 0.5:
            st.info(f"📊 **Prior H2H Gemini**: media storica {_fixture_prior:.1f} gol/partita blendato nel modello.")
        if abs(_movement_quality - 1.0) > 0.05:
            _mov_desc = "sharp/notizie" if _movement_quality > 1.0 else "rumore/pubblico"
            st.info(f"📈 **Movimento linee Gemini**: segnale {_mov_desc} (qualità: {_movement_quality:.2f}×).")

    # ── Applica aggiustamenti dalla Ricerca Pre-Partita ─────────────────────
    _adj_lines = dict(lines)  # copia per non mutare l'originale
    if _ricerca_adj_tot != 0 or _ricerca_adj_ah != 0:
        _adj_lines["tot_op"] = lines["tot_op"] + _ricerca_adj_tot
        _adj_lines["ah_op"] = lines["ah_op"] + _ricerca_adj_ah

    try:
        state = build_match_state(
            match, _adj_lines, linea_ou, bankroll, comm_rate,
            ocr_imp_total=_ocr_total,
            ocr_quota_1=_ocr_q1,
            ocr_quota_x=_ocr_qx,
            ocr_quota_2=_ocr_q2,
            ocr_quota_over=_ocr_qover,
            ocr_quota_under=_ocr_qund,
            ocr_quota_gg=_ocr_qgg,
            ocr_quota_ng=_ocr_qng,
            fixture_historical_total=_fixture_prior,
            movement_quality=_movement_quality,
            ocr_confidence_scale=_ocr_confidence_scale,
        )
    except (AssertionError, ValueError) as e:
        st.error(f"❌ Input non valido: {e}")
        st.stop()

    # ── Anomaly Detection (automatico, nessun input aggiuntivo) ─────────────
    anomalies = render_anomalies(state)

    with st.spinner("Calcolo matrice bivariata..."):
        risultati = analizza(state)

    shots = (state.sot_h, state.soff_h, state.sot_a, state.soff_a)

    # #12: Warning se le quote OCR divergono >15% dal modello
    if extracted_data is not None and extracted_data.extraction_success:
        render_ocr_market_divergence(
            risultati,
            ocr_quota_1=extracted_data.quota_1,
            ocr_quota_x=extracted_data.quota_x,
            ocr_quota_2=extracted_data.quota_2,
            ocr_quota_over=extracted_data.quota_over,
            ocr_quota_under=extracted_data.quota_under,
        )
    n_shots_tot = sum(shots)
    gol_attuali = state.gol_casa + state.gol_trasf  # recompute from validated state

    # ── Verifica linee non aggiornate (FIX) ───────────────────────────────────
    render_lines_need_update(risultati)

    # ── Quote Fair ───────────────────────────────────────────────────────────
    render_quote_fair(risultati, state.minuto, state.gol_casa, state.gol_trasf, linea_ou)
    render_model_confidence(risultati)
    render_confidence_bands(risultati, linea_ou)
    render_coerenza_mercati(risultati, linea_ou)
    render_correct_score(risultati)
    render_asian_handicap(risultati.full_matrix)
    render_clean_sheet(risultati.full_matrix, state.gol_casa, state.gol_trasf)  # Nuovo mercato!

    # ── Red card impact (Fix #16) ─────────────────────────────────────────────
    render_red_card_impact(state.rossi_casa, state.rossi_trasf, state.minuto)

    # ── Momentum con decomposizione AH/Total (Fix #14) ───────────────────────
    st.divider()
    render_momentum(risultati.momentum, risultati.delta_ah, risultati.delta_tot)

    # ── Fine partita ─────────────────────────────────────────────────────────
    if state.minuto >= SIGNALS.GAME_END_THRESHOLD:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()

    # FIX: Gestione recovery time (minuti > 90)
    if state.minuto > SIGNALS.RECOVERY_TIME_WARNING:
        if state.minuto >= SIGNALS.RECOVERY_TIME_HARD_CAP:
            st.error(f"🚫 **ANALISI BLOCCATA**: Minuto {state.minuto}' supera il limite massimo ({SIGNALS.RECOVERY_TIME_HARD_CAP}').")
            st.stop()
        else:
            st.warning(
                f"⚠️ **RECOVERY TIME**: Minuto {state.minuto}' — il modello opera in zona recupero. "
                f"Le previsioni sono meno affidabili e gli spread più ampi."
            )

    # ── Mercati chiusi (PRIMA dei segnali) ───────────────────────────────────
    st.divider()
    settled = render_mercati_chiusi(
        gol_attuali, linea_ou,
        state.gol_casa, state.gol_trasf,
        state.minuto, risultati.p_btts,
    )

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

    # ── Value Bet Detection (automatico) ────────────────────────────────────
    if quotes.any_active:
        st.divider()
        render_value_bets(risultati, quotes)

    # ── Risk Metrics (automatico) ───────────────────────────────────────────
    if segnali_avanzati:
        render_risk_metrics(segnali_avanzati, bankroll)

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

    # ── Debug & Session Stats ──────────────────────────────────────────────
    st.divider()
    soglie = calcola_soglie(state.minuto, linea_ou, gol_attuali)
    render_debug(
        risultati, linea_ou, state.minuto,
        soglie, comm_pct, momentum_factor, n_shots_tot,
    )
    render_session_stats()
