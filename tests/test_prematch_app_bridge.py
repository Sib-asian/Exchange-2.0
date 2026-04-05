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
    payload = dict(
        league_name="Test League",
        extraction_coverage=0.80,
        extraction_section_scores={},
        odds_sharp_signal=0.40,
        line_movement_ah=-0.50,
        line_movement_total=0.25,
        mkt_init_1=2.10,
        mkt_init_x=3.20,
        mkt_init_2=3.60,
        total_line_open=2.25,
        total_over_odds_open=1.92,
        total_under_odds_open=1.92,
        mkt_init_gg=1.85,
        mkt_init_ng=1.95,
        fixture_historical_total=2.7,
        forma_mult_h=1.02,
        forma_mult_a=0.99,
        h2h_home_win_pct=42.0,
        h2h_draw_pct=30.0,
        h2h_away_win_pct=28.0,
        h2h_over_pct=58.0,
        h2h_btts_pct=56.0,
        h2h_matches_count=8,
        team_stats_home_shots=13.0,
        team_stats_away_shots=9.0,
        team_stats_home_corners=6.0,
        team_stats_away_corners=4.0,
        team_stats_home_possession=55.0,
        team_stats_away_possession=45.0,
        home_absences_count=0,
        away_absences_count=0,
        home_absences_players=[],
        away_absences_players=[],
    )
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

