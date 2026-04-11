"""Tests for prematch app bridge field mapping."""

from types import SimpleNamespace

from src.prematch_app_bridge import build_match_state_from_prematch_analysis


def _base_lines() -> dict:
    return {
        "ah_op": -0.25,
        "tot_op": 2.5,
        "ah_cur": -0.25,
        "tot_cur": 2.5,
        "tot_cur_raw": 2.5,
        "linea_ou": 2.5,
        "fullgame_mode": True,
        "validation_errors": [],
        "blocking_errors": [],
    }


def _base_match() -> dict:
    return {
        "minuto": 0,
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
    }


def _base_pa(**kwargs) -> SimpleNamespace:
    payload = {
        "league_name": "Test League",
        "extraction_coverage": 0.80,
        "extraction_section_scores": {},
        "odds_sharp_signal": 0.40,
        "line_movement_ah": -0.50,
        "line_movement_total": 0.25,
        "mkt_init_1": 2.10,
        "mkt_init_x": 3.20,
        "mkt_init_2": 3.60,
        "total_line_open": 2.25,
        "total_over_odds_open": 1.92,
        "total_under_odds_open": 1.92,
        "mkt_init_gg": 1.85,
        "mkt_init_ng": 1.95,
        "fixture_historical_total": 2.7,
        "forma_mult_h": 1.02,
        "forma_mult_a": 0.99,
        "h2h_home_win_pct": 42.0,
        "h2h_draw_pct": 30.0,
        "h2h_away_win_pct": 28.0,
        "h2h_over_pct": 58.0,
        "h2h_btts_pct": 56.0,
        "h2h_matches_count": 8,
        "team_stats_home_shots": 13.0,
        "team_stats_away_shots": 9.0,
        "team_stats_home_corners": 6.0,
        "team_stats_away_corners": 4.0,
        "team_stats_home_possession": 55.0,
        "team_stats_away_possession": 45.0,
        "home_absences_count": 0,
        "away_absences_count": 0,
        "home_absences_players": [],
        "away_absences_players": [],
    }
    payload.update(kwargs)
    return SimpleNamespace(**payload)


def test_bridge_maps_new_premarket_fields_to_state() -> None:
    pa = _base_pa()
    state, league, coverage = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert league == "Test League"
    assert abs(coverage - 0.80) < 1e-9
    assert state.line_movement_ah_raw == -0.50
    assert state.line_movement_total_raw == 0.25
    assert state.team_stats_home_shots == 13.0
    assert state.team_stats_away_shots == 9.0
    assert state.team_stats_home_corners == 6.0
    assert state.team_stats_away_corners == 4.0
    assert state.team_stats_home_possession == 55.0
    assert state.team_stats_away_possession == 45.0


def test_bridge_uses_valid_initial_1x2_prior_by_default() -> None:
    pa = _base_pa(mkt_init_1=2.00, mkt_init_x=3.30, mkt_init_2=3.80)
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.mkt_init_1 == 2.0
    assert state.mkt_init_x == 3.3
    assert state.mkt_init_2 == 3.8
    assert state.ocr_quota_1 == 2.0
    assert state.ocr_quota_x == 3.3
    assert state.ocr_quota_2 == 3.8


def test_bridge_falls_back_to_live_1x2_when_initial_missing() -> None:
    pa = _base_pa(
        mkt_init_1=0.0,
        mkt_init_x=0.0,
        mkt_init_2=0.0,
        mkt_live_1=2.22,
        mkt_live_x=3.30,
        mkt_live_2=3.10,
    )
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.mkt_init_1 == 2.22
    assert state.mkt_init_x == 3.30
    assert state.mkt_init_2 == 3.10


def test_bridge_extraction_trust_factor_drops_on_critical_notes() -> None:
    pa = _base_pa(extraction_notes=["market_1x2_missing_or_unreadable", "ok_note"])
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.extraction_trust_factor < 1.0
    assert state.extraction_trust_factor >= 0.5


def test_bridge_h2h_core_weight_follows_section_score() -> None:
    pa = _base_pa(extraction_section_scores={"h2h_core": 0.28})
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert abs(state.h2h_core_weight - 0.28) < 1e-6


def test_bridge_maps_h2h_marginals_ht_count_and_sharp_signal() -> None:
    pa = _base_pa(
        h2h_avg_goals_home=1.35,
        h2h_avg_goals_away=0.95,
        h2h_ht_matches_count=7,
    )
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.h2h_avg_goals_home == 1.35
    assert state.h2h_avg_goals_away == 0.95
    assert state.h2h_ht_matches_count == 7
    assert state.odds_sharp_signal > 0


def test_bridge_maps_url_derived_signals_to_match_state() -> None:
    pa = _base_pa(
        h2h_ah_home_cover_pct=68.0,
        home_prev_win_pct=55.0,
        away_prev_win_pct=40.0,
        home_xg_from_recent=1.45,
        away_xg_from_recent=1.05,
        home_motivation="high",
        away_motivation="low",
    )
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.h2h_ah_home_cover_pct > 0
    assert state.prev_win_pct_h > 0
    assert state.recent_xg_prior_h > 0.1
    assert state.motivation_home == "high"
    assert state.motivation_away == "low"


def test_bridge_swaps_inverted_1x2_when_ah_says_home_favorite() -> None:
    # AH apertura negativa => casa favorita, ma quote invertite (q1 molto > q2).
    pa = _base_pa(mkt_init_1=4.82, mkt_init_x=3.50, mkt_init_2=1.70)
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=_base_match(),
        lines=_base_lines(),
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.mkt_init_1 == 1.70
    assert state.mkt_init_x == 3.50
    assert state.mkt_init_2 == 4.82

