"""
app.py — Radar Pro Live v2.0
Flusso: Linee → ANALIZZA  |  ▶ Dati Live → ANALIZZA
"""

import streamlit as st

from src.config import SIGNALS, UI
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
from src.models.ai_adjustments import calcola_assenze_mult

# ── Linee ─────────────────────────────────────────────────────────────────────
_live_gh = st.session_state.get("live_gol_casa", 0)
_live_ga = st.session_state.get("live_gol_trasf", 0)

lines = render_linee_semplici(gol_casa=_live_gh, gol_trasf=_live_ga)
if lines.get("blocking_errors"):
    st.stop()

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
    _lega = getattr(_pa, "league_name", "") if _pa else ""
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
    _h2h_btts = float(getattr(_pa, "h2h_btts_pct", 0.0)) if _pa else 0.0
    _str_home = int(getattr(_pa, "strength_home", 0)) if _pa else 0
    _str_away = int(getattr(_pa, "strength_away", 0)) if _pa else 0
    _weather_impact = float(getattr(_pa, "weather_impact", 0.0)) if _pa else 0.0
    _sharp_sig = float(getattr(_pa, "odds_sharp_signal", 0.0)) if _pa else 0.0
    _coverage = float(getattr(_pa, "extraction_coverage", 0.0)) if _pa else 0.0
    _movement_quality = 1.0 + min(0.30, _sharp_sig * 0.08)
    _ocr_conf_scale = 0.70 + min(0.30, _coverage)
    _streak_score_h = int(getattr(_pa, "scoring_streak_h", 0)) if _pa else 0
    _streak_score_a = int(getattr(_pa, "scoring_streak_a", 0)) if _pa else 0
    _streak_cs_h = int(getattr(_pa, "clean_sheet_streak_h", 0)) if _pa else 0
    _streak_cs_a = int(getattr(_pa, "clean_sheet_streak_a", 0)) if _pa else 0
    _h2h_n = int(getattr(_pa, "h2h_matches_count", 0)) if _pa else 0
    _abs_h = int(getattr(_pa, "home_absences_count", 0)) if _pa else 0
    _abs_a = int(getattr(_pa, "away_absences_count", 0)) if _pa else 0
    _abs_h_list = [
        x.strip()
        for x in (getattr(_pa, "home_absences_players", None) or [])
        if isinstance(x, str) and x.strip()
    ][:8]
    _abs_a_list = [
        x.strip()
        for x in (getattr(_pa, "away_absences_players", None) or [])
        if isinstance(x, str) and x.strip()
    ][:8]
    _hg1 = float(getattr(_pa, "home_goals_1h", 0.0)) if _pa else 0.0
    _hg2 = float(getattr(_pa, "home_goals_2h", 0.0)) if _pa else 0.0
    _ag1 = float(getattr(_pa, "away_goals_1h", 0.0)) if _pa else 0.0
    _ag2 = float(getattr(_pa, "away_goals_2h", 0.0)) if _pa else 0.0
    _late_pct_h = (_hg2 / max(1e-9, _hg1 + _hg2)) * 100.0 if (_hg1 + _hg2) > 0 else 0.0
    _late_pct_a = (_ag2 / max(1e-9, _ag1 + _ag2)) * 100.0 if (_ag1 + _ag2) > 0 else 0.0
    if not _abs_h_list and _abs_h > 0:
        _abs_h_list = ["Unknown (MID, PROBABLE)"] * max(0, min(8, _abs_h))
    if not _abs_a_list and _abs_a > 0:
        _abs_a_list = ["Unknown (MID, PROBABLE)"] * max(0, min(8, _abs_a))
    _absence_mult_h = calcola_assenze_mult(_abs_h_list) * calcola_assenze_mult(_abs_a_list, per_avversario=True)
    _absence_mult_a = calcola_assenze_mult(_abs_a_list) * calcola_assenze_mult(_abs_h_list, per_avversario=True)

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
    _l6_gf_h = float(getattr(_pa, "home_last6_scored", 0)) if _pa else 0.0
    _l6_ga_h = float(getattr(_pa, "home_last6_conceded", 0)) if _pa else 0.0
    _l6_gf_a = float(getattr(_pa, "away_last6_scored", 0)) if _pa else 0.0
    _l6_ga_a = float(getattr(_pa, "away_last6_conceded", 0)) if _pa else 0.0
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

    # Quality gate graduale: riduce i prior in modo continuo invece di on/off.
    _key_fields_ok = int(_hist_tot > 0) + int(_h2h_home + _h2h_draw + _h2h_away > 0) + int(_mkt1 > 1.0 and _mktx > 1.0 and _mkt2 > 1.0)
    _quality_gate = 0.50 if _key_fields_ok >= 2 else 0.62
    _quality_ok = _coverage >= _quality_gate
    _sec_scores = getattr(_pa, "extraction_section_scores", {}) if _pa else {}

    def _sec(name: str, default: float = 0.0) -> float:
        if not _sec_scores:
            return default
        try:
            return float(_sec_scores.get(name, default))
        except (TypeError, ValueError):
            return default

    _global_w = 1.0 if _quality_ok else max(0.25, min(1.0, _coverage / max(1e-9, _quality_gate)))
    _w_identity = _sec("identity", _global_w)
    _w_h2h = _global_w * _sec("h2h_core", _global_w)
    _w_prev = _global_w * _sec("previous_scores", _global_w)
    _w_stats = _global_w * _sec("team_stats", _global_w)
    _w_weather = _global_w * _sec("weather", _global_w)
    _w_inj = _global_w * _sec("injuries", _global_w)

    _h2h_home *= _w_h2h
    _h2h_draw *= _w_h2h
    _h2h_away *= _w_h2h
    _h2h_over *= _w_h2h
    _h2h_btts *= _w_h2h
    _hist_tot *= _w_h2h
    _final_sc_h *= max(_w_prev, _w_stats)
    _final_co_h *= max(_w_prev, _w_stats)
    _final_sc_a *= max(_w_prev, _w_stats)
    _final_co_a *= max(_w_prev, _w_stats)
    _weather_impact *= _w_weather
    _streak_score_h = int(round(_streak_score_h * _w_prev))
    _streak_score_a = int(round(_streak_score_a * _w_prev))
    _streak_cs_h = int(round(_streak_cs_h * _w_prev))
    _streak_cs_a = int(round(_streak_cs_a * _w_prev))
    _late_pct_h *= _w_stats
    _late_pct_a *= _w_stats
    _absence_mult_h = 1.0 + (_absence_mult_h - 1.0) * _w_inj
    _absence_mult_a = 1.0 + (_absence_mult_a - 1.0) * _w_inj
    _movement_quality = 1.0 + (_movement_quality - 1.0) * _global_w
    _ocr_conf_scale = 0.70 + (min(1.0, _coverage) * 0.30 * max(_global_w, _w_identity))

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
            h2h_btts_pct=_h2h_btts,
            h2h_matches_count=_h2h_n,
            strength_home=_str_home,
            strength_away=_str_away,
            weather_xg_impact=_weather_impact,
            scoring_streak_h=_streak_score_h,
            scoring_streak_a=_streak_score_a,
            clean_sheet_streak_h=_streak_cs_h,
            clean_sheet_streak_a=_streak_cs_a,
            late_goals_pct_h=_late_pct_h,
            late_goals_pct_a=_late_pct_a,
            movement_quality=_movement_quality,
            ocr_confidence_scale=_ocr_conf_scale,
            absence_mult_h=_absence_mult_h,
            absence_mult_a=_absence_mult_a,
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
            last6_gf_h=_l6_gf_h,
            last6_ga_h=_l6_ga_h,
            last6_gf_a=_l6_gf_a,
            last6_ga_a=_l6_ga_a,
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
        from src.pipeline import run_analysis_pipeline

        _cov_pipe = float(_coverage) if (state.minuto == 0 and _pa) else 1.0
        risultati, _cal_sig = run_analysis_pipeline(
            state,
            league=_lega,
            apply_prematch_calibration=(state.minuto == 0),
            extraction_coverage=_cov_pipe,
        )
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
    _market_quotes = {
        "quota_1": _mkt1,
        "quota_x": _mktx,
        "quota_2": _mkt2,
    }

    _tracking_meta = {
        "extraction_coverage": float(_coverage),
        "league_source": str(getattr(_pa, "league_source", "")) if _pa else "",
        "model_agreement": float(risultati.model_agreement),
        "tot_band": tot_op_band(float(lines["tot_op"])),
        "software_version": str(UI.VERSION),
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
