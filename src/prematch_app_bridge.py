"""
Ponte tra PrematchAnalysisExtracted e MatchState (stessa logica del flusso Streamlit prematch).
"""

from __future__ import annotations

from src.engine import MatchState
from src.models.ai_adjustments import calcola_assenze_mult
from src.ocr import PrematchAnalysisExtracted
from src.ui.inputs import build_match_state


def build_match_state_from_prematch_analysis(
    pa: PrematchAnalysisExtracted | None,
    *,
    match: dict,
    lines: dict,
    linea_ou: float,
    bankroll: float,
    comm_rate: float,
) -> tuple[MatchState, str, float]:
    """
    Costruisce MatchState come nel blocco prematch di app.py.

    Returns:
        (state, league_name, extraction_coverage)
    """
    _lega = getattr(pa, "league_name", "") if pa else ""
    _fm_raw_h = getattr(pa, "forma_mult_h", None) if pa else None
    _fm_raw_a = getattr(pa, "forma_mult_a", None) if pa else None
    _forma_h = float(_fm_raw_h) if _fm_raw_h is not None else 1.0
    _forma_a = float(_fm_raw_a) if _fm_raw_a is not None else 1.0
    _hist_tot = float(getattr(pa, "fixture_historical_total", 0.0)) if pa else 0.0
    _mkt1 = float(getattr(pa, "mkt_init_1", 0.0)) if pa else 0.0
    _mktx = float(getattr(pa, "mkt_init_x", 0.0)) if pa else 0.0
    _mkt2 = float(getattr(pa, "mkt_init_2", 0.0)) if pa else 0.0
    _h2h_home = float(getattr(pa, "h2h_home_win_pct", 0.0)) if pa else 0.0
    _h2h_draw = float(getattr(pa, "h2h_draw_pct", 0.0)) if pa else 0.0
    _h2h_away = float(getattr(pa, "h2h_away_win_pct", 0.0)) if pa else 0.0
    _h2h_over = float(getattr(pa, "h2h_over_pct", 0.0)) if pa else 0.0
    _h2h_btts = float(getattr(pa, "h2h_btts_pct", 0.0)) if pa else 0.0
    _h2h_ht_home = float(getattr(pa, "h2h_ht_home_win_pct", 0.0)) if pa else 0.0
    _h2h_ht_draw = float(getattr(pa, "h2h_ht_draw_pct", 0.0)) if pa else 0.0
    _h2h_ht_away = float(getattr(pa, "h2h_ht_away_win_pct", 0.0)) if pa else 0.0
    _str_home = int(getattr(pa, "strength_home", 0)) if pa else 0
    _str_away = int(getattr(pa, "strength_away", 0)) if pa else 0
    _weather_impact = float(getattr(pa, "weather_impact", 0.0)) if pa else 0.0
    _sharp_sig = float(getattr(pa, "odds_sharp_signal", 0.0)) if pa else 0.0
    _coverage = float(getattr(pa, "extraction_coverage", 0.0)) if pa else 0.0
    _movement_quality = 1.0 + min(0.30, _sharp_sig * 0.08)
    _ocr_conf_scale = 0.70 + min(0.30, _coverage)
    _streak_score_h = int(getattr(pa, "scoring_streak_h", 0)) if pa else 0
    _streak_score_a = int(getattr(pa, "scoring_streak_a", 0)) if pa else 0
    _streak_cs_h = int(getattr(pa, "clean_sheet_streak_h", 0)) if pa else 0
    _streak_cs_a = int(getattr(pa, "clean_sheet_streak_a", 0)) if pa else 0
    _h2h_n = int(getattr(pa, "h2h_matches_count", 0)) if pa else 0
    _abs_h = int(getattr(pa, "home_absences_count", 0)) if pa else 0
    _abs_a = int(getattr(pa, "away_absences_count", 0)) if pa else 0
    _abs_h_list = [
        x.strip()
        for x in (getattr(pa, "home_absences_players", None) or [])
        if isinstance(x, str) and x.strip()
    ][:8]
    _abs_a_list = [
        x.strip()
        for x in (getattr(pa, "away_absences_players", None) or [])
        if isinstance(x, str) and x.strip()
    ][:8]
    _hg1 = float(getattr(pa, "home_goals_1h", 0.0)) if pa else 0.0
    _hg2 = float(getattr(pa, "home_goals_2h", 0.0)) if pa else 0.0
    _ag1 = float(getattr(pa, "away_goals_1h", 0.0)) if pa else 0.0
    _ag2 = float(getattr(pa, "away_goals_2h", 0.0)) if pa else 0.0
    _late_pct_h = (_hg2 / max(1e-9, _hg1 + _hg2)) * 100.0 if (_hg1 + _hg2) > 0 else 0.0
    _late_pct_a = (_ag2 / max(1e-9, _ag1 + _ag2)) * 100.0 if (_ag1 + _ag2) > 0 else 0.0
    _h_ht_lose = int(getattr(pa, "home_ht_lose", 0)) if pa else 0
    _a_ht_lose = int(getattr(pa, "away_ht_lose", 0)) if pa else 0
    _h_matches = int(getattr(pa, "home_matches", 0)) if pa else 0
    _a_matches = int(getattr(pa, "away_matches", 0)) if pa else 0
    _early_conc_h = round(_h_ht_lose / _h_matches * 100, 1) if _h_matches > 0 else 0.0
    _early_conc_a = round(_a_ht_lose / _a_matches * 100, 1) if _a_matches > 0 else 0.0
    if not _abs_h_list and _abs_h > 0:
        _abs_h_list = ["Unknown (MID, PROBABLE)"] * max(0, min(8, _abs_h))
    if not _abs_a_list and _abs_a > 0:
        _abs_a_list = ["Unknown (MID, PROBABLE)"] * max(0, min(8, _abs_a))
    _absence_mult_h = calcola_assenze_mult(_abs_h_list) * calcola_assenze_mult(_abs_a_list, per_avversario=True)
    _absence_mult_a = calcola_assenze_mult(_abs_a_list) * calcola_assenze_mult(_abs_h_list, per_avversario=True)

    _prev_sc_h = float(getattr(pa, "home_prev_avg_scored", 0.0)) if pa else 0.0
    _prev_co_h = float(getattr(pa, "home_prev_avg_conceded", 0.0)) if pa else 0.0
    _prev_sc_a = float(getattr(pa, "away_prev_avg_scored", 0.0)) if pa else 0.0
    _prev_co_a = float(getattr(pa, "away_prev_avg_conceded", 0.0)) if pa else 0.0

    _ts_sc_h = float(getattr(pa, "team_stats_home_goals", 0.0)) if pa else 0.0
    _ts_co_h = float(getattr(pa, "team_stats_home_conceded", 0.0)) if pa else 0.0
    _ts_sc_a = float(getattr(pa, "team_stats_away_goals", 0.0)) if pa else 0.0
    _ts_co_a = float(getattr(pa, "team_stats_away_conceded", 0.0)) if pa else 0.0
    _ts3_sc_h = float(getattr(pa, "team_stats3_home_goals", 0.0)) if pa else 0.0
    _ts3_co_h = float(getattr(pa, "team_stats3_home_conceded", 0.0)) if pa else 0.0
    _ts3_sc_a = float(getattr(pa, "team_stats3_away_goals", 0.0)) if pa else 0.0
    _ts3_co_a = float(getattr(pa, "team_stats3_away_conceded", 0.0)) if pa else 0.0
    _fix_d_h = int(getattr(pa, "fixture_next_days_home", 0) or 0) if pa else 0
    _fix_d_a = int(getattr(pa, "fixture_next_days_away", 0) or 0) if pa else 0

    _st_rank_h = int(getattr(pa, "home_rank", 0)) if pa else 0
    _st_rank_a = int(getattr(pa, "away_rank", 0)) if pa else 0
    _st_pts_h = int(getattr(pa, "home_points", 0)) if pa else 0
    _st_pts_a = int(getattr(pa, "away_points", 0)) if pa else 0
    _st_played_h = int(getattr(pa, "home_matches", 0)) if pa else 0
    _st_played_a = int(getattr(pa, "away_matches", 0)) if pa else 0
    _st_total_teams = int(getattr(pa, "standings_total_teams", 0)) if pa else 0
    _l6w_h = int(getattr(pa, "home_last6_win", 0)) if pa else 0
    _l6d_h = int(getattr(pa, "home_last6_draw", 0)) if pa else 0
    _l6w_a = int(getattr(pa, "away_last6_win", 0)) if pa else 0
    _l6d_a = int(getattr(pa, "away_last6_draw", 0)) if pa else 0
    _l6_pts_h = _l6w_h * 3 + _l6d_h
    _l6_pts_a = _l6w_a * 3 + _l6d_a
    _l6_gf_h = float(getattr(pa, "home_last6_scored", 0)) if pa else 0.0
    _l6_ga_h = float(getattr(pa, "home_last6_conceded", 0)) if pa else 0.0
    _l6_gf_a = float(getattr(pa, "away_last6_scored", 0)) if pa else 0.0
    _l6_ga_a = float(getattr(pa, "away_last6_conceded", 0)) if pa else 0.0
    _h_home_w = int(getattr(pa, "home_home_win", 0)) if pa else 0
    _h_home_d = int(getattr(pa, "home_home_draw", 0)) if pa else 0
    _h_home_m = int(getattr(pa, "home_home_win", 0)) + int(getattr(pa, "home_home_draw", 0)) + int(getattr(pa, "home_home_lose", 0)) if pa else 0
    _a_away_w = int(getattr(pa, "away_away_win", 0)) if pa else 0
    _a_away_d = int(getattr(pa, "away_away_draw", 0)) if pa else 0
    _a_away_m = int(getattr(pa, "away_away_win", 0)) + int(getattr(pa, "away_away_draw", 0)) + int(getattr(pa, "away_away_lose", 0)) if pa else 0
    _h_ppg = (_h_home_w * 3 + _h_home_d) / max(1, _h_home_m) if _h_home_m > 0 else 0.0
    _a_ppg = (_a_away_w * 3 + _a_away_d) / max(1, _a_away_m) if _a_away_m > 0 else 0.0
    _h_gf = float(getattr(pa, "home_home_scored", 0)) / max(1, _h_home_m) if pa and _h_home_m > 0 else 0.0
    _h_ga = float(getattr(pa, "home_home_conceded", 0)) / max(1, _h_home_m) if pa and _h_home_m > 0 else 0.0
    _a_gf = float(getattr(pa, "away_away_scored", 0)) / max(1, _a_away_m) if pa and _a_away_m > 0 else 0.0
    _a_ga = float(getattr(pa, "away_away_conceded", 0)) / max(1, _a_away_m) if pa and _a_away_m > 0 else 0.0

    def _fuse(prev_val: float, ts_val: float) -> float:
        if prev_val > 0 and ts_val > 0:
            return (prev_val + ts_val) / 2.0
        if ts_val > 0:
            return ts_val
        return prev_val

    def _fuse_with_recent3(prev_val: float, ts10: float, ts3: float) -> float:
        core = _fuse(prev_val, ts10)
        if ts3 <= 0:
            return core
        return 0.70 * core + 0.30 * ts3

    def _fixture_schedule_mult(days: int) -> float:
        if days <= 0:
            return 1.0
        if days <= 3:
            return 0.96
        if days <= 4:
            return 0.98
        if days >= 10:
            return 1.015
        return 1.0

    _final_sc_h = _fuse_with_recent3(_prev_sc_h, _ts_sc_h, _ts3_sc_h)
    _final_co_h = _fuse_with_recent3(_prev_co_h, _ts_co_h, _ts3_co_h)
    _final_sc_a = _fuse_with_recent3(_prev_sc_a, _ts_sc_a, _ts3_sc_a)
    _final_co_a = _fuse_with_recent3(_prev_co_a, _ts_co_a, _ts3_co_a)

    _forma_h *= _fixture_schedule_mult(_fix_d_h)
    _forma_a *= _fixture_schedule_mult(_fix_d_a)

    _key_fields_ok = int(_hist_tot > 0) + int(_h2h_home + _h2h_draw + _h2h_away > 0) + int(_mkt1 > 1.0 and _mktx > 1.0 and _mkt2 > 1.0)
    _quality_gate = 0.50 if _key_fields_ok >= 2 else 0.62
    _quality_ok = _coverage >= _quality_gate
    _sec_scores = getattr(pa, "extraction_section_scores", {}) if pa else {}

    def _sec(name: str, default: float = 0.0) -> float:
        if not _sec_scores:
            return default
        try:
            return float(_sec_scores.get(name, default))
        except (TypeError, ValueError):
            return default

    _global_w = 1.0 if _quality_ok else max(0.25, min(1.0, _coverage / max(1e-9, _quality_gate)))
    _w_h2h = _global_w * _sec("h2h_core", _global_w)
    _w_prev = _global_w * _sec("previous_scores", _global_w)
    _w_stats = _global_w * _sec("team_stats", _global_w)
    _w_weather = _global_w * _sec("weather", _global_w)
    _w_inj = _global_w * _sec("injuries", _global_w)
    _w_identity = _sec("identity", _global_w)

    _h2h_home *= _w_h2h
    _h2h_draw *= _w_h2h
    _h2h_away *= _w_h2h
    _h2h_over *= _w_h2h
    _h2h_btts *= _w_h2h
    _h2h_ht_home *= _w_h2h
    _h2h_ht_draw *= _w_h2h
    _h2h_ht_away *= _w_h2h
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

    _ocr_q1 = _mkt1 if _mkt1 > 1.0 else 0.0
    _ocr_qx = _mktx if _mktx > 1.0 else 0.0
    _ocr_q2 = _mkt2 if _mkt2 > 1.0 else 0.0
    _ocr_qo = float(getattr(pa, "total_over_odds_open", 0.0)) if pa else 0.0
    _ocr_qu = float(getattr(pa, "total_under_odds_open", 0.0)) if pa else 0.0
    _ocr_imp = float(getattr(pa, "total_line_open", 0.0)) if pa else 0.0
    _ocr_qgg = float(getattr(pa, "mkt_init_gg", 0.0)) if pa else 0.0
    _ocr_qng = float(getattr(pa, "mkt_init_ng", 0.0)) if pa else 0.0

    state = build_match_state(
        match, lines, linea_ou, bankroll, comm_rate,
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
        early_conceded_pct_h=_early_conc_h,
        early_conceded_pct_a=_early_conc_a,
        h2h_ht_home_win_pct=_h2h_ht_home,
        h2h_ht_draw_pct=_h2h_ht_draw,
        h2h_ht_away_win_pct=_h2h_ht_away,
        movement_quality=_movement_quality,
        ocr_confidence_scale=_ocr_conf_scale,
        absence_mult_h=_absence_mult_h,
        absence_mult_a=_absence_mult_a,
        prev_avg_scored_h=_final_sc_h,
        prev_avg_conceded_h=_final_co_h,
        prev_avg_scored_a=_final_sc_a,
        prev_avg_conceded_a=_final_co_a,
        standings_rank_h=_st_rank_h,
        standings_rank_a=_st_rank_a,
        standings_points_h=_st_pts_h,
        standings_points_a=_st_pts_a,
        standings_played_h=_st_played_h,
        standings_played_a=_st_played_a,
        standings_total_teams=_st_total_teams if _st_total_teams > 0 else 20,
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
        ocr_quota_1=_ocr_q1,
        ocr_quota_x=_ocr_qx,
        ocr_quota_2=_ocr_q2,
        ocr_quota_over=_ocr_qo,
        ocr_quota_under=_ocr_qu,
        ocr_imp_total=_ocr_imp,
        ocr_quota_gg=_ocr_qgg,
        ocr_quota_ng=_ocr_qng,
    )
    return state, _lega, _coverage
