"""
Ponte tra PrematchAnalysisExtracted e MatchState (stessa logica del flusso Streamlit prematch).
"""

from __future__ import annotations

import math

from src.config import FORM_ANALYSIS, OCR_QUOTES
from src.engine import MatchState
from src.models.ai_adjustments import calcola_assenze_mult
from src.ocr import PrematchAnalysisExtracted
from src.ui.inputs import build_match_state


def _valid_1x2_triplet(q1: float, qx: float, q2: float) -> bool:
    if not (1.01 < q1 < 100 and 1.01 < qx < 100 and 1.01 < q2 < 100):
        return False
    overround = 1.0 / q1 + 1.0 / qx + 1.0 / q2
    return 1.0 <= overround <= OCR_QUOTES.MAX_OVERROUND_3WAY


def _apply_ah_consistency_guard(q1: float, qx: float, q2: float, ah_op: float) -> tuple[float, float, float]:
    """
    Protezione anti-inversione Home/Away:
    - AH negativo => casa favorita (atteso q1 < q2)
    - AH positivo => trasferta favorita (atteso q2 < q1)
    Se il segnale 1X2 è forte ma opposto all'AH, scambia 1<->2.
    """
    if abs(ah_op) < 0.25:
        return q1, qx, q2
    skew = abs(math.log(max(1e-9, q1 / q2)))
    if skew < 0.18:
        return q1, qx, q2
    home_fav_ah = ah_op < 0
    home_fav_1x2 = q1 < q2
    if home_fav_ah != home_fav_1x2:
        return q2, qx, q1
    return q1, qx, q2


def _choose_market_1x2(pa: PrematchAnalysisExtracted | None, lines: dict) -> tuple[float, float, float]:
    if not pa:
        return 0.0, 0.0, 0.0
    init_1 = float(getattr(pa, "mkt_init_1", 0.0) or 0.0)
    init_x = float(getattr(pa, "mkt_init_x", 0.0) or 0.0)
    init_2 = float(getattr(pa, "mkt_init_2", 0.0) or 0.0)
    live_1 = float(getattr(pa, "mkt_live_1", 0.0) or 0.0)
    live_x = float(getattr(pa, "mkt_live_x", 0.0) or 0.0)
    live_2 = float(getattr(pa, "mkt_live_2", 0.0) or 0.0)

    if _valid_1x2_triplet(init_1, init_x, init_2):
        q1, qx, q2 = init_1, init_x, init_2
    elif _valid_1x2_triplet(live_1, live_x, live_2):
        # Prematch: fallback live solo se initial manca/non valido.
        q1, qx, q2 = live_1, live_x, live_2
    else:
        return 0.0, 0.0, 0.0

    ah_op = float(lines.get("ah_op", 0.0) or 0.0)
    return _apply_ah_consistency_guard(q1, qx, q2, ah_op)


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
    _mkt1, _mktx, _mkt2 = _choose_market_1x2(pa, lines)
    if not OCR_QUOTES.USE_EXTRACTED_1X2_PRIOR:
        _mkt1 = _mktx = _mkt2 = 0.0
    _h2h_home = float(getattr(pa, "h2h_home_win_pct", 0.0)) if pa else 0.0
    _h2h_draw = float(getattr(pa, "h2h_draw_pct", 0.0)) if pa else 0.0
    _h2h_away = float(getattr(pa, "h2h_away_win_pct", 0.0)) if pa else 0.0
    _h2h_ah_cov = float(getattr(pa, "h2h_ah_home_cover_pct", 0.0)) if pa else 0.0
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
    # Se abbiamo solo il conteggio assenze ma non i dettagli ruolo/status,
    # non applicare impatti sintetici: meglio restare neutri (1.0) che introdurre bias.
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
    _h2h_ah_cov *= _w_h2h
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
    _line_move_ah_raw = float(getattr(pa, "line_movement_ah", 0.0)) if pa else 0.0
    _line_move_total_raw = float(getattr(pa, "line_movement_total", 0.0)) if pa else 0.0
    _ts_sh_h = float(getattr(pa, "team_stats_home_shots", 0.0)) if pa else 0.0
    _ts_sh_a = float(getattr(pa, "team_stats_away_shots", 0.0)) if pa else 0.0
    _ts_cor_h = float(getattr(pa, "team_stats_home_corners", 0.0)) if pa else 0.0
    _ts_cor_a = float(getattr(pa, "team_stats_away_corners", 0.0)) if pa else 0.0
    _ts_pos_h = float(getattr(pa, "team_stats_home_possession", 0.0)) if pa else 0.0
    _ts_pos_a = float(getattr(pa, "team_stats_away_possession", 0.0)) if pa else 0.0
    _ts_yel_h = float(getattr(pa, "team_stats_home_yellows", 0.0)) if pa else 0.0
    _ts_yel_a = float(getattr(pa, "team_stats_away_yellows", 0.0)) if pa else 0.0
    _ts_foul_h = float(getattr(pa, "team_stats_home_fouls", 0.0)) if pa else 0.0
    _ts_foul_a = float(getattr(pa, "team_stats_away_fouls", 0.0)) if pa else 0.0
    _prev_over_h = float(getattr(pa, "home_prev_over_pct", 0.0)) if pa else 0.0
    _prev_over_a = float(getattr(pa, "away_prev_over_pct", 0.0)) if pa else 0.0
    _prev_win_h = float(getattr(pa, "home_prev_win_pct", 0.0)) if pa else 0.0
    _prev_win_a = float(getattr(pa, "away_prev_win_pct", 0.0)) if pa else 0.0
    _rxg_h = float(getattr(pa, "home_xg_from_recent", 0.0)) if pa else 0.0
    _rxg_a = float(getattr(pa, "away_xg_from_recent", 0.0)) if pa else 0.0
    _mot_h = str(getattr(pa, "home_motivation", "normal") or "normal").strip().lower()
    _mot_a = str(getattr(pa, "away_motivation", "normal") or "normal").strip().lower()
    if _mot_h not in ("high", "normal", "low"):
        _mot_h = "normal"
    if _mot_a not in ("high", "normal", "low"):
        _mot_a = "normal"

    # Upgrade 8-4: HT/FT transition counts (18 campi)
    _htft_h_hw_fw = int(getattr(pa, "htft_home_htw_ftw", 0)) if pa else 0
    _htft_h_hw_fd = int(getattr(pa, "htft_home_htw_ftd", 0)) if pa else 0
    _htft_h_hw_fl = int(getattr(pa, "htft_home_htw_ftl", 0)) if pa else 0
    _htft_h_hd_fw = int(getattr(pa, "htft_home_htd_ftw", 0)) if pa else 0
    _htft_h_hd_fd = int(getattr(pa, "htft_home_htd_ftd", 0)) if pa else 0
    _htft_h_hd_fl = int(getattr(pa, "htft_home_htd_ftl", 0)) if pa else 0
    _htft_h_hl_fw = int(getattr(pa, "htft_home_htl_ftw", 0)) if pa else 0
    _htft_h_hl_fd = int(getattr(pa, "htft_home_htl_ftd", 0)) if pa else 0
    _htft_h_hl_fl = int(getattr(pa, "htft_home_htl_ftl", 0)) if pa else 0
    _htft_a_hw_fw = int(getattr(pa, "htft_away_htw_ftw", 0)) if pa else 0
    _htft_a_hw_fd = int(getattr(pa, "htft_away_htw_ftd", 0)) if pa else 0
    _htft_a_hw_fl = int(getattr(pa, "htft_away_htw_ftl", 0)) if pa else 0
    _htft_a_hd_fw = int(getattr(pa, "htft_away_htd_ftw", 0)) if pa else 0
    _htft_a_hd_fd = int(getattr(pa, "htft_away_htd_ftd", 0)) if pa else 0
    _htft_a_hd_fl = int(getattr(pa, "htft_away_htd_ftl", 0)) if pa else 0
    _htft_a_hl_fw = int(getattr(pa, "htft_away_htl_ftw", 0)) if pa else 0
    _htft_a_hl_fd = int(getattr(pa, "htft_away_htl_ftd", 0)) if pa else 0
    _htft_a_hl_fl = int(getattr(pa, "htft_away_htl_ftl", 0)) if pa else 0

    _h2h_avg_gh = float(getattr(pa, "h2h_avg_goals_home", 0.0)) if pa else 0.0
    _h2h_avg_ga = float(getattr(pa, "h2h_avg_goals_away", 0.0)) if pa else 0.0
    _h2h_ht_n_raw = int(getattr(pa, "h2h_ht_matches_count", 0) or 0) if pa else 0
    _h2h_ht_n = max(0, int(round(_h2h_ht_n_raw * min(1.0, _w_h2h))))
    _sharp_for_state = max(0.0, _sharp_sig * _global_w)
    _h2h_core_w = 1.0
    if _global_w > 1e-9:
        _h2h_core_w = max(0.0, min(1.0, _w_h2h / _global_w))

    _bad_notes = frozenset({
        "market_1x2_missing_or_unreadable",
        "league_missing",
        "team_names_partial",
    })
    _extraction_trust = 1.0
    for _note in getattr(pa, "extraction_notes", None) or []:
        if str(_note).strip() in _bad_notes:
            _extraction_trust *= FORM_ANALYSIS.EXTRACTION_NOTE_TRUST_PENALTY
    _extraction_trust = max(
        FORM_ANALYSIS.EXTRACTION_TRUST_FLOOR,
        min(1.0, _extraction_trust),
    )
    # Qualità sezione: degrada in modo continuo se più blocchi sono deboli.
    _section_keys = ("identity", "h2h_core", "previous_scores", "team_stats", "injuries", "weather")
    _sec_vals = []
    for _k in _section_keys:
        _sv = _sec(_k, _global_w)
        _sec_vals.append(max(0.30, min(1.0, float(_sv))))
    if _sec_vals:
        _sec_avg = sum(_sec_vals) / len(_sec_vals)
        _extraction_trust *= 0.72 + 0.28 * _sec_avg
    _extraction_trust = max(
        FORM_ANALYSIS.EXTRACTION_TRUST_FLOOR,
        min(1.0, _extraction_trust),
    )

    state = build_match_state(
        match, lines, linea_ou, bankroll, comm_rate,
        forma_mult_h=_forma_h,
        forma_mult_a=_forma_a,
        fixture_historical_total=_hist_tot,
        h2h_avg_goals_home=_h2h_avg_gh,
        h2h_avg_goals_away=_h2h_avg_ga,
        h2h_ht_matches_count=_h2h_ht_n,
        odds_sharp_signal=_sharp_for_state,
        h2h_core_weight=_h2h_core_w,
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
        line_movement_ah_raw=_line_move_ah_raw,
        line_movement_total_raw=_line_move_total_raw,
        extraction_coverage=_coverage,
        extraction_trust_factor=_extraction_trust,
        team_stats_home_shots=_ts_sh_h,
        team_stats_away_shots=_ts_sh_a,
        team_stats_home_corners=_ts_cor_h,
        team_stats_away_corners=_ts_cor_a,
        team_stats_home_possession=_ts_pos_h,
        team_stats_away_possession=_ts_pos_a,
        team_stats_home_yellows=_ts_yel_h,
        team_stats_away_yellows=_ts_yel_a,
        team_stats_home_fouls=_ts_foul_h,
        team_stats_away_fouls=_ts_foul_a,
        prev_over_pct_h=_prev_over_h,
        prev_over_pct_a=_prev_over_a,
        h2h_ah_home_cover_pct=_h2h_ah_cov,
        prev_win_pct_h=_prev_win_h,
        prev_win_pct_a=_prev_win_a,
        recent_xg_prior_h=_rxg_h,
        recent_xg_prior_a=_rxg_a,
        motivation_home=_mot_h,
        motivation_away=_mot_a,
        htft_home_htw_ftw=_htft_h_hw_fw,
        htft_home_htw_ftd=_htft_h_hw_fd,
        htft_home_htw_ftl=_htft_h_hw_fl,
        htft_home_htd_ftw=_htft_h_hd_fw,
        htft_home_htd_ftd=_htft_h_hd_fd,
        htft_home_htd_ftl=_htft_h_hd_fl,
        htft_home_htl_ftw=_htft_h_hl_fw,
        htft_home_htl_ftd=_htft_h_hl_fd,
        htft_home_htl_ftl=_htft_h_hl_fl,
        htft_away_htw_ftw=_htft_a_hw_fw,
        htft_away_htw_ftd=_htft_a_hw_fd,
        htft_away_htw_ftl=_htft_a_hw_fl,
        htft_away_htd_ftw=_htft_a_hd_fw,
        htft_away_htd_ftd=_htft_a_hd_fd,
        htft_away_htd_ftl=_htft_a_hd_fl,
        htft_away_htl_ftw=_htft_a_hl_fw,
        htft_away_htl_ftd=_htft_a_hl_fd,
        htft_away_htl_ftl=_htft_a_hl_fl,
    )
    return state, _lega, _coverage
