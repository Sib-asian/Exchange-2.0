"""
app.py — Radar Pro Live v2.0
Flusso: Linee → ANALIZZA  |  ▶ Dati Live → ANALIZZA
"""

import streamlit as st

from src.config import SIGNALS, UI
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

# ── Linee ─────────────────────────────────────────────────────────────────────
_live_gh = st.session_state.get("live_gol_casa", 0)
_live_ga = st.session_state.get("live_gol_trasf", 0)

lines = render_linee_semplici(gol_casa=_live_gh, gol_trasf=_live_ga)

# Linea O/U: sempre 2.5 (standard per pronostici)
def _linea_ou(tot_raw: float) -> float:
    # Sempre 2.5 come richiesto dall'utente
    return 2.5

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

    _lou = _linea_ou(lines["tot_cur_raw"])

    _pa = st.session_state.get("prematch_analysis")
    _fm_raw_h = getattr(_pa, "forma_mult_h", None) if _pa else None
    _fm_raw_a = getattr(_pa, "forma_mult_a", None) if _pa else None
    _forma_h  = float(_fm_raw_h) if _fm_raw_h is not None else 1.0
    _forma_a  = float(_fm_raw_a) if _fm_raw_a is not None else 1.0
    _hist_tot = float(getattr(_pa, "fixture_historical_total", 0.0)) if _pa else 0.0
    _mkt1     = float(getattr(_pa, "mkt_init_1", 0.0)) if _pa else 0.0
    _mktx     = float(getattr(_pa, "mkt_init_x", 0.0)) if _pa else 0.0
    _mkt2     = float(getattr(_pa, "mkt_init_2", 0.0)) if _pa else 0.0
    _h2h_home = float(getattr(_pa, "h2h_home_win_pct", 0.0)) if _pa else 0.0
    _h2h_draw = float(getattr(_pa, "h2h_draw_pct", 0.0)) if _pa else 0.0
    _h2h_away = float(getattr(_pa, "h2h_away_win_pct", 0.0)) if _pa else 0.0
    # Nuovi parametri per miglioramenti
    _h2h_over = float(getattr(_pa, "h2h_over_pct", 0.0)) if _pa else 0.0
    _str_home = int(getattr(_pa, "strength_home", 0)) if _pa else 0
    _str_away = int(getattr(_pa, "strength_away", 0)) if _pa else 0
    
    # Previous Scores (da sezione H2H)
    _prev_sc_h = float(getattr(_pa, "home_prev_avg_scored", 0.0)) if _pa else 0.0
    _prev_co_h = float(getattr(_pa, "home_prev_avg_conceded", 0.0)) if _pa else 0.0
    _prev_sc_a = float(getattr(_pa, "away_prev_avg_scored", 0.0)) if _pa else 0.0
    _prev_co_a = float(getattr(_pa, "away_prev_avg_conceded", 0.0)) if _pa else 0.0
    
    # Team Statistics (da tabella con possesso/corner - più affidabile, Recent 10 Matches)
    _ts_sc_h = float(getattr(_pa, "team_stats_home_goals", 0.0)) if _pa else 0.0
    _ts_co_h = float(getattr(_pa, "team_stats_home_conceded", 0.0)) if _pa else 0.0
    _ts_sc_a = float(getattr(_pa, "team_stats_away_goals", 0.0)) if _pa else 0.0
    _ts_co_a = float(getattr(_pa, "team_stats_away_conceded", 0.0)) if _pa else 0.0
    
    # Form Analysis (da standings Nowgoal) — prima non venivano passati al motore!
    _st_rank_h = int(getattr(_pa, "home_rank", 0)) if _pa else 0
    _st_rank_a = int(getattr(_pa, "away_rank", 0)) if _pa else 0
    _st_pts_h = int(getattr(_pa, "home_points", 0)) if _pa else 0
    _st_pts_a = int(getattr(_pa, "away_points", 0)) if _pa else 0
    _st_played_h = int(getattr(_pa, "home_matches", 0)) if _pa else 0
    _st_played_a = int(getattr(_pa, "away_matches", 0)) if _pa else 0
    # Last 6: calcola punti (W=3, D=1, L=0)
    _l6w_h = int(getattr(_pa, "home_last6_win", 0)) if _pa else 0
    _l6d_h = int(getattr(_pa, "home_last6_draw", 0)) if _pa else 0
    _l6w_a = int(getattr(_pa, "away_last6_win", 0)) if _pa else 0
    _l6d_a = int(getattr(_pa, "away_last6_draw", 0)) if _pa else 0
    _l6_pts_h = _l6w_h * 3 + _l6d_h
    _l6_pts_a = _l6w_a * 3 + _l6d_a
    # Home/Away PPG e gol medi
    _h_home_w = int(getattr(_pa, "home_home_win", 0)) if _pa else 0
    _h_home_d = int(getattr(_pa, "home_home_draw", 0)) if _pa else 0
    _h_home_m = int(getattr(_pa, "home_home_win", 0)) + int(getattr(_pa, "home_home_draw", 0)) + int(getattr(_pa, "home_home_lose", 0)) if _pa else 0
    _a_away_w = int(getattr(_pa, "away_away_win", 0)) if _pa else 0
    _a_away_d = int(getattr(_pa, "away_away_draw", 0)) if _pa else 0
    _a_away_m = int(getattr(_pa, "away_away_win", 0)) + int(getattr(_pa, "away_away_draw", 0)) + int(getattr(_pa, "away_away_lose", 0)) if _pa else 0
    _h_ppg = (_h_home_w * 3 + _h_home_d) / max(1, _h_home_m) if _h_home_m > 0 else 0.0
    _a_ppg = (_a_away_w * 3 + _a_away_d) / max(1, _a_away_m) if _a_away_m > 0 else 0.0
    _h_gf = float(getattr(_pa, "home_home_scored", 0)) / max(1, _h_home_m) if _pa and _h_home_m > 0 else 0.0
    _h_ga = float(getattr(_pa, "home_home_conceded", 0)) / max(1, _h_home_m) if _pa and _h_home_m > 0 else 0.0
    _a_gf = float(getattr(_pa, "away_away_scored", 0)) / max(1, _a_away_m) if _pa and _a_away_m > 0 else 0.0
    _a_ga = float(getattr(_pa, "away_away_conceded", 0)) / max(1, _a_away_m) if _pa and _a_away_m > 0 else 0.0
    
    # FUSIONE: media di Previous Scores e Team Statistics quando entrambi disponibili
    # Questo riduce errori di estrazione e rende i dati più affidabili
    def _fuse(prev_val: float, ts_val: float) -> float:
        """Fonde due fonti. Se entrambe disponibili, media. Se una sola, usa quella."""
        if prev_val > 0 and ts_val > 0:
            return (prev_val + ts_val) / 2.0  # Media
        elif ts_val > 0:
            return ts_val  # Fallback su Team Statistics
        else:
            return prev_val  # Fallback su Previous Scores (o 0)
    
    _final_sc_h = _fuse(_prev_sc_h, _ts_sc_h)
    _final_co_h = _fuse(_prev_co_h, _ts_co_h)
    _final_sc_a = _fuse(_prev_sc_a, _ts_sc_a)
    _final_co_a = _fuse(_prev_co_a, _ts_co_a)

    try:
        state = build_match_state(
            _match, lines, _lou, bankroll, comm_rate,
            forma_mult_h=_forma_h,
            forma_mult_a=_forma_a,
            fixture_historical_total=_hist_tot,
            mkt_init_1=_mkt1,
            mkt_init_x=_mktx,
            mkt_init_2=_mkt2,
            h2h_home_win_pct=_h2h_home,
            h2h_draw_pct=_h2h_draw,
            h2h_away_win_pct=_h2h_away,
            h2h_over_pct=_h2h_over,
            strength_home=_str_home,
            strength_away=_str_away,
            prev_avg_scored_h=_final_sc_h,
            prev_avg_conceded_h=_final_co_h,
            prev_avg_scored_a=_final_sc_a,
            prev_avg_conceded_a=_final_co_a,
            # Form Analysis
            standings_rank_h=_st_rank_h,
            standings_rank_a=_st_rank_a,
            standings_points_h=_st_pts_h,
            standings_points_a=_st_pts_a,
            standings_played_h=_st_played_h,
            standings_played_a=_st_played_a,
            last6_points_h=_l6_pts_h,
            last6_points_a=_l6_pts_a,
            home_ppg_h=_h_ppg,
            away_ppg_a=_a_ppg,
            home_gf_h=_h_gf,
            home_ga_h=_h_ga,
            away_gf_a=_a_gf,
            away_ga_a=_a_ga,
        )
    except (AssertionError, ValueError) as e:
        st.error(f"❌ Input non valido: {e}")
        st.stop()

    with st.spinner("Calcolo..."):
        from dataclasses import replace as _dc_replace
        from src.engine import analizza
        risultati = analizza(state)
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
    _lega = getattr(_pa, "league_name", "") if _pa else ""

    # Se non ci sono i nomi, usa placeholder
    if not _squadra_casa:
        _squadra_casa = "Casa"
    if not _squadra_trasf:
        _squadra_trasf = "Trasferta"

    _input_data = {
        "ah_op": lines["ah_op"],
        "tot_op": lines["tot_op"],
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
    _market_quotes = {
        "quota_1": _mkt1,
        "quota_x": _mktx,
        "quota_2": _mkt2,
    }

    _tracking_record = create_record_from_analysis(
        squadra_casa=_squadra_casa,
        squadra_trasf=_squadra_trasf,
        lega=_lega,
        input_data=_input_data,
        predictions=_predictions,
        market_quotes=_market_quotes,
    )

    _pred_log = get_prediction_log()
    _pred_log.add(_tracking_record)
    st.session_state["last_prediction_id"] = _tracking_record.id

    # ── Output ────────────────────────────────────────────────────────────────
    from src.signals import calcola_soglie, genera_segnali_rapidi
    from src.ui.outputs import (
        render_avvisi_affidabilita,
        render_avvisi_incoerenza,
        render_lines_need_update,
        render_mercati_chiusi,
        render_pronostici_rapidi,
        render_analisi_dinamica,
        render_segnali_rapidi,
    )

    _minuto  = state.minuto
    _gol_h   = state.gol_casa
    _gol_a   = state.gol_trasf
    _gol_tot = _gol_h + _gol_a
    _n_shots = state.sot_h + state.soff_h + state.sot_a + state.soff_a

    render_lines_need_update(risultati)

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
    render_pronostici_rapidi(risultati, _lou, _minuto, _gol_h, _gol_a, linea_ah=lines["ah_cur"], prematch=_pa)

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
    )
    if _settled.get("ou_vinto"):
        segnali = [s for s in segnali if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
    if _settled.get("btts_si_settled") or _settled.get("btts_no_settled"):
        segnali = [s for s in segnali if "BTTS" not in s.mercato.upper()]

    render_segnali_rapidi(segnali)
    render_avvisi_affidabilita(risultati.flat_lines, _n_shots, _minuto)
    render_avvisi_incoerenza(
        risultati.p_btts, risultati.p_under, risultati.p_over,
        _lou, _gol_tot,
        btts_settled=_settled.get("btts_si_settled", False),
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
