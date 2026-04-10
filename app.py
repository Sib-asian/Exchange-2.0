"""
app.py — Radar Pro Live v2.0
Flusso: Linee → ANALIZZA  |  ▶ Dati Live → ANALIZZA
"""

import streamlit as st

from src.config import ENGINE, SIGNALS, UI
from src.tracking.prediction_log import tot_op_band
from src.session_storage import (
    PartitaSalvata,
    build_saved_at_label,
    collect_prematch_analysis,
    collect_widget_state,
    delete_partita,
    load_partite,
    restore_prematch_analysis,
    restore_widget_state,
    save_partita,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title=UI.PAGE_TITLE, page_icon=UI.PAGE_ICON, layout=UI.LAYOUT)
st.title(f"{UI.PAGE_ICON} {UI.PAGE_TITLE}")
st.caption(f"v{UI.VERSION} · Bivariate Poisson + Dixon-Coles + Kelly frazionato")

# ── Match Memory ──────────────────────────────────────────────────────────────
_saved = load_partite()
if _saved:
    _opts = {f"{p.nome}  [{p.saved_at}]": p for p in _saved}
    _cs, _cl, _cd = st.columns([5, 1, 1])
    with _cs:
        _sel = st.selectbox("Partite salvate", list(_opts.keys()), index=None,
                            placeholder="Seleziona partita salvata...",
                            key="match_memory_selector", label_visibility="collapsed")
    with _cl:
        if _sel and st.button("Carica", key="btn_load", use_container_width=True):
            _p = _opts[_sel]
            restore_widget_state(st.session_state, _p.widget_state)
            restore_prematch_analysis(st.session_state, _p.prematch_analysis)
            st.session_state["active_match_id"] = _p.id
            st.rerun()
    with _cd:
        if _sel and st.button("✕", key="btn_del", use_container_width=True):
            _p = _opts[_sel]
            delete_partita(_p.id)
            st.session_state.pop("active_match_id", None)
            st.rerun()

    _active_p = next((p for p in _saved if p.id == st.session_state.get("active_match_id")), None)
    if _active_p:
        st.caption(f"Attiva: **{_active_p.nome}** · {_active_p.saved_at}")

st.divider()

# ── Imports UI ────────────────────────────────────────────────────────────────
from src.ui.inputs import (
    build_match_state,
    render_bankroll,
    render_linee_semplici,
    render_live_semplice,
    render_prematch_analysis_screen,
)
from src.prematch_app_bridge import build_match_state_from_prematch_analysis

# ── Linee ─────────────────────────────────────────────────────────────────────
_live_gh = st.session_state.get("live_gol_casa", 0)
_live_ga = st.session_state.get("live_gol_trasf", 0)

lines = render_linee_semplici(gol_casa=_live_gh, gol_trasf=_live_ga)
if lines.get("blocking_errors"):
    st.stop()

st.divider()

# ── Bankroll ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Bankroll e commissione", expanded=False):
    render_bankroll()

bankroll  = float(st.session_state.get("bankroll_value", UI.BANKROLL_DEFAULT))
comm_pct  = float(st.session_state.get("comm_pct_value", UI.COMM_DEFAULT))
comm_rate = comm_pct / 100.0

st.divider()

# ── Analisi Prematch (Nowgoal Analysis tab) ───────────────────────────────────
render_prematch_analysis_screen()

st.divider()

# ── ANALIZZA Prematch ─────────────────────────────────────────────────────────
_btn_prematch = st.button("ANALIZZA", use_container_width=True, type="primary", key="btn_prematch")

st.divider()

# ── Dati Live ─────────────────────────────────────────────────────────────────
# _live_expander_open viene impostato a True dopo l'upload dello screenshot
# per mantenere l'expander aperto dopo il rerun. Pop lo consuma una volta sola.
_live_exp_open = st.session_state.pop("_live_expander_open", False)
with st.expander("📺 Dati Live", expanded=_live_exp_open):
    _match_live = render_live_semplice()
    _btn_live   = st.button("ANALIZZA", use_container_width=True, type="primary", key="btn_live")

# ── Prediction Tracker (sempre visibile) ───────────────────────────────────────
st.divider()
st.markdown("### 📋 Prediction Tracker")
try:
    from src.tracking.ui import render_tracking_tab
    render_tracking_tab()
except Exception as _e:
    import traceback
    st.error(f"Errore Prediction Tracker: {_e}")
    st.code(traceback.format_exc())

# ── Pipeline ──────────────────────────────────────────────────────────────────
if _btn_prematch or _btn_live:

    _live_min = st.session_state.get("live_minuto", 0)

    if _btn_prematch and _live_min == 0:
        # Prematch puro
        _match = {
            "minuto": 0, "gol_casa": 0, "gol_trasf": 0,
            "rossi_casa": 0, "rossi_trasf": 0,
            "gialli_casa": 0, "gialli_trasf": 0,
            "sot_h": 0, "soff_h": 0, "sot_a": 0, "soff_a": 0,
            "blk_h": 0, "blk_a": 0,
            "corner_h": 0, "corner_a": 0,
            "possesso_h": 0.0, "possesso_a": 0.0,
            "att_pericolosi_h": 0, "att_pericolosi_a": 0,
            "att_h": 0, "att_a": 0,
            "falli_casa": 0, "falli_trasf": 0,
        }
    else:
        _match = _match_live

    _lou = float(lines["linea_ou"])

    _pa = st.session_state.get("prematch_analysis")
    try:
        state, _lega, _coverage = build_match_state_from_prematch_analysis(
            _pa,
            match=_match,
            lines=lines,
            linea_ou=_lou,
            bankroll=bankroll,
            comm_rate=comm_rate,
        )
    except (AssertionError, ValueError) as e:
        st.error(f"❌ Input non valido: {e}")
        st.stop()

    with st.spinner("Calcolo..."):
        from dataclasses import replace as _dc_replace
        from src.engine import analizza
        from src.pipeline import run_analysis_pipeline

        _cov_pipe = float(_coverage) if (state.minuto == 0 and _pa) else 1.0
        try:
            risultati, _cal_sig, _pipe_trace = run_analysis_pipeline(
                state,
                league=_lega,
                apply_prematch_calibration=(state.minuto == 0),
                extraction_coverage=_cov_pipe,
            )
        except Exception as _pipe_err:
            import traceback
            st.error(
                f"❌ Errore nel motore di analisi: {_pipe_err}\n\n"
                "Controlla i dati inseriti e riprova."
            )
            st.code(traceback.format_exc())
            st.stop()

        # Trasparenza pipeline prematch: mostra se e come ha agito la calibrazione storica.
        if state.minuto == 0:
            if _cal_sig is not None:
                if _cal_sig.weight > 0:
                    st.caption(
                        f"Calibrazione prematch applicata: scope={_cal_sig.scope}, "
                        f"campioni={_cal_sig.samples}, peso={_cal_sig.weight:.1%}"
                    )
                else:
                    st.caption(
                        f"Calibrazione prematch disponibile ma non attiva "
                        f"(campioni={_cal_sig.samples}, soglia minima non raggiunta)."
                    )
            else:
                st.caption("Calibrazione prematch non applicata (lega non disponibile).")

            if _pipe_trace is not None:
                with st.expander("🔬 Traccia pipeline prematch", expanded=False):
                    for line in _pipe_trace.summary_lines():
                        st.caption(line)

        if state.minuto == 0:
            st.session_state["prematch_last_model_1x2"] = {
                "p1": float(risultati.p1),
                "px": float(risultati.px),
                "p2": float(risultati.p2),
            }
        # Scenari "se segna subito" (solo live)
        _scen_h: object = None
        _scen_a: object = None
        if state.minuto > 0:
            try:
                _scen_h = analizza(_dc_replace(state, gol_casa=state.gol_casa + 1))
                _scen_a = analizza(_dc_replace(state, gol_trasf=state.gol_trasf + 1))
            except Exception:
                _scen_h = _scen_a = None

    # Auto-save (usa un ID fisso per sessione basato sulle linee)
    _pid = st.session_state.get("active_match_id") or str(
        abs(hash((lines["ah_op"], lines["tot_op"]))) % (10 ** 9)
    )
    save_partita(PartitaSalvata(
        id=_pid, nome=f"AH {lines['ah_op']:+.2f} · Tot {lines['tot_op']:.2f}",
        saved_at=build_saved_at_label(),
        widget_state=collect_widget_state(st.session_state),
        prematch_analysis=collect_prematch_analysis(st.session_state),
    ))
    st.session_state["active_match_id"] = _pid

    # ── Prediction Logging (salvataggio automatico) ───────────────────────────────
    from src.tracking.prediction_log import get_prediction_log, create_record_from_analysis

    # I nomi dei campi in PrematchAnalysisExtracted sono home_team/away_team/league_name
    _squadra_casa = getattr(_pa, "home_team", "") if _pa else ""
    _squadra_trasf = getattr(_pa, "away_team", "") if _pa else ""

    # Se non ci sono i nomi, usa placeholder
    if not _squadra_casa:
        _squadra_casa = "Casa"
    if not _squadra_trasf:
        _squadra_trasf = "Trasferta"

    _input_data = {
        "ah_op": lines["ah_op"],
        "tot_op": lines["tot_op"],
        "linea_ou": _lou,
        "minuto": state.minuto,
        "xg_h": risultati.xg_h_final,
        "xg_a": risultati.xg_a_final,
    }
    _predictions = {
        "p1": risultati.p1,
        "px": risultati.px,
        "p2": risultati.p2,
        "p_over": risultati.p_over,
        "p_under": risultati.p_under,
        "p_btts": risultati.p_btts,
        "model_confidence": risultati.model_confidence,
    }
    # Quote nel log: da estrazione prematch / stato (nessun input manuale richiesto).
    # 1X2: preferisci mkt_init_* (Nowgoal); se mancano, usa ocr_quota_* (form manuale).
    _q1 = float(state.mkt_init_1) or float(state.ocr_quota_1)
    _qx = float(state.mkt_init_x) or float(state.ocr_quota_x)
    _q2 = float(state.mkt_init_2) or float(state.ocr_quota_2)
    _market_quotes = {
        "quota_1": _q1,
        "quota_x": _qx,
        "quota_2": _q2,
        "quota_over": float(state.ocr_quota_over),
        "quota_under": float(state.ocr_quota_under),
        "quota_btts_si": float(state.ocr_quota_gg),
        "quota_btts_no": float(state.ocr_quota_ng),
    }
    _quote_source = "unknown"
    if _pa:
        _has_init = (
            float(getattr(_pa, "mkt_init_1", 0.0) or 0.0) > 1.0
            and float(getattr(_pa, "mkt_init_x", 0.0) or 0.0) > 1.0
            and float(getattr(_pa, "mkt_init_2", 0.0) or 0.0) > 1.0
        )
        _has_live = (
            float(getattr(_pa, "mkt_live_1", 0.0) or 0.0) > 1.0
            and float(getattr(_pa, "mkt_live_x", 0.0) or 0.0) > 1.0
            and float(getattr(_pa, "mkt_live_2", 0.0) or 0.0) > 1.0
        )
        if _has_init:
            _quote_source = "initial"
        elif _has_live:
            _quote_source = "live_fallback"

    _tracking_meta = {
        "extraction_coverage": float(_coverage),
        "league_source": str(getattr(_pa, "league_source", "")) if _pa else "",
        "model_agreement": float(risultati.model_agreement),
        "quality_score": float(getattr(risultati, "quality_score", 0.0)),
        "signals_blocked": bool(getattr(risultati, "signals_blocked", False)),
        "signals_block_reason": str(getattr(risultati, "signals_block_reason", "")),
        "tot_band": tot_op_band(float(lines["tot_op"])),
        "software_version": str(UI.VERSION),
        "model_revision": str(ENGINE.MODEL_REVISION),
        "consensus_w_bp": float(risultati.consensus_w_bp),
        "consensus_w_cop": float(risultati.consensus_w_cop),
        "consensus_w_mk": float(risultati.consensus_w_mk),
        "p1_bp": float(risultati.p1_bp),
        "px_bp": float(risultati.px_bp),
        "p2_bp": float(risultati.p2_bp),
        "p1_cop": float(risultati.p1_cop),
        "px_cop": float(risultati.px_cop),
        "p2_cop": float(risultati.p2_cop),
        "p1_mk": float(risultati.p1_mk),
        "px_mk": float(risultati.px_mk),
        "p2_mk": float(risultati.p2_mk),
        "p_over_bp": float(risultati.p_over_bp),
        "p_over_cop": float(risultati.p_over_cop),
        "p_over_mk": float(risultati.p_over_mk),
        "p_btts_bp": float(risultati.p_btts_bp),
        "p_btts_cop": float(risultati.p_btts_cop),
        "p_btts_mk": float(risultati.p_btts_mk),
        "xg_h_pre_prev": float(risultati.xg_h_pre_prev_blend),
        "xg_a_pre_prev": float(risultati.xg_a_pre_prev_blend),
        "prev_lambda_h": float(risultati.prev_lambda_h),
        "prev_lambda_a": float(risultati.prev_lambda_a),
        "quote_source": _quote_source,
    }
    _tracking_record = create_record_from_analysis(
        squadra_casa=_squadra_casa,
        squadra_trasf=_squadra_trasf,
        lega=_lega,
        input_data=_input_data,
        predictions=_predictions,
        market_quotes=_market_quotes,
        metadata=_tracking_meta,
    )

    _pred_log = get_prediction_log()
    _pred_log.add(_tracking_record)
    st.session_state["last_prediction_id"] = _tracking_record.id

    # ── Output ────────────────────────────────────────────────────────────────
    from src.signals import calcola_soglie, genera_segnali_rapidi
    from src.ui.outputs import (
        render_avvisi_affidabilita,
        render_avvisi_incoerenza,
        render_ocr_market_divergence,
        render_mercati_chiusi,
        render_lines_need_update,
        render_pronostici_rapidi,
        render_analisi_dinamica,
        render_segnali_rapidi,
    )

    _minuto  = state.minuto
    _gol_h   = state.gol_casa
    _gol_a   = state.gol_trasf
    _gol_tot = _gol_h + _gol_a
    _n_shots = state.sot_h + state.soff_h + state.sot_a + state.soff_a

    # Stop se fine partita
    if _minuto >= SIGNALS.GAME_END_THRESHOLD:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()
    if _minuto > SIGNALS.RECOVERY_TIME_WARNING:
        if _minuto >= SIGNALS.RECOVERY_TIME_HARD_CAP:
            st.error(f"🚫 Minuto {_minuto}' oltre il limite ({SIGNALS.RECOVERY_TIME_HARD_CAP}').")
            st.stop()
        else:
            st.warning(f"⚠️ Recovery time ({_minuto}'): previsioni meno affidabili.")

    st.divider()
    render_pronostici_rapidi(
        risultati,
        _lou,
        _minuto,
        _gol_h,
        _gol_a,
        linea_ah=lines["ah_cur"],
        prematch=_pa,
        match_state=state,
    )

    render_analisi_dinamica(risultati, state, _gol_tot, _scen_h, _scen_a)

    _settled = render_mercati_chiusi(_gol_tot, _lou, _gol_h, _gol_a, _minuto, risultati.p_btts)

    segnali = genera_segnali_rapidi(
        risultati.p1, risultati.px, risultati.p2,
        risultati.p_over, risultati.p_under, risultati.p_btts,
        _minuto, _lou, _gol_tot,
        model_confidence=risultati.model_confidence,
        model_agreement=risultati.model_agreement,
        gol_casa=_gol_h,
        gol_trasf=_gol_a,
        top_cs=risultati.top_cs,
        signals_blocked=bool(getattr(risultati, "signals_blocked", False)),
    )
    if _settled.get("ou_vinto"):
        segnali = [s for s in segnali if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
    if _settled.get("btts_si_settled") or _settled.get("btts_no_settled"):
        segnali = [s for s in segnali if "BTTS" not in s.mercato.upper()]

    render_segnali_rapidi(segnali)
    if getattr(risultati, "signals_blocked", False):
        _reason = getattr(risultati, "signals_block_reason", "")
        st.warning(
            "No-Bet firewall attivo: qualità operativa insufficiente."
            + (f" Motivo: {_reason}." if _reason else "")
        )
    render_avvisi_affidabilita(risultati.flat_lines, _n_shots, _minuto)
    render_lines_need_update(risultati)
    render_avvisi_incoerenza(
        risultati.p_btts, risultati.p_under, risultati.p_over,
        _lou, _gol_tot,
        btts_settled=_settled.get("btts_si_settled", False),
    )
    render_ocr_market_divergence(
        risultati,
        state.ocr_quota_1,
        state.ocr_quota_x,
        state.ocr_quota_2,
        state.ocr_quota_over,
        state.ocr_quota_under,
    )

    # ── Analisi avanzata ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("📊 Analisi avanzata", expanded=False):
        from src.ui.outputs import (
            render_asian_handicap,
            render_clean_sheet,
            render_coerenza_mercati,
            render_confidence_bands,
            render_correct_score,
            render_debug,
            render_momentum,
            render_red_card_impact,
            render_session_stats,
        )
        render_confidence_bands(risultati, _lou)
        render_coerenza_mercati(risultati, _lou)
        render_correct_score(risultati)
        render_asian_handicap(risultati.full_matrix)
        render_clean_sheet(risultati.full_matrix, _gol_h, _gol_a)
        render_red_card_impact(state.rossi_casa, state.rossi_trasf, _minuto)
        render_momentum(risultati.momentum, risultati.delta_ah, risultati.delta_tot)
        _soglie = calcola_soglie(_minuto, _lou, _gol_tot)
        render_debug(risultati, _lou, _minuto, _soglie, comm_pct, 1.0, _n_shots)
        render_session_stats()

    # ── Segnali con quote exchange ────────────────────────────────────────────
    with st.expander("💰 Segnali con quote exchange", expanded=False):
        from src.signals import genera_segnali_avanzati
        from src.ui.inputs import render_exchange_quotes
        from src.ui.outputs import (
            render_allineamento_mercato,
            render_risk_metrics,
            render_value_bets,
            render_segnali_avanzati,
        )

        quotes = render_exchange_quotes(_lou)
        segnali_av = genera_segnali_avanzati(
            risultati.p1, risultati.px, risultati.p2,
            risultati.p_over, risultati.p_under, risultati.p_btts,
            quotes, _minuto, _lou, _gol_tot,
            bankroll, comm_rate, _n_shots, risultati.momentum,
            model_confidence=risultati.model_confidence,
            model_agreement=risultati.model_agreement,
            gol_casa=_gol_h,
            gol_trasf=_gol_a,
            signals_blocked=bool(getattr(risultati, "signals_blocked", False)),
            ci_tightness=float(getattr(risultati, "pipeline_ci_tightness", 0.55)),
            credible_intervals=getattr(risultati, "credible_intervals", None) or None,
        )
        if _settled.get("ou_vinto"):
            segnali_av = [s for s in segnali_av if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
        if _settled.get("btts_si_settled") or _settled.get("btts_no_settled"):
            segnali_av = [s for s in segnali_av if "BTTS" not in s.mercato.upper()]

        render_segnali_avanzati(segnali_av, quotes.any_active)
        render_allineamento_mercato(risultati, quotes)
        if quotes.any_active:
            render_value_bets(risultati, quotes)
        if segnali_av:
            render_risk_metrics(segnali_av, bankroll)
