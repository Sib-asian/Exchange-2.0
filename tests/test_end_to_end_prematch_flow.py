"""End-to-end prematch flow checks: lines -> bridge -> pipeline."""

from types import SimpleNamespace

from src.pipeline import run_analysis_pipeline
from src.prematch_app_bridge import build_match_state_from_prematch_analysis


def _fake_pa(**kwargs):
    base = dict(
        league_name="Test League",
        # Market + OCR
        mkt_init_1=2.10,
        mkt_init_x=3.30,
        mkt_init_2=3.60,
        total_line_open=2.25,
        total_over_odds_open=1.92,
        total_under_odds_open=1.92,
        mkt_init_gg=1.85,
        mkt_init_ng=1.95,
        # H2H and form
        h2h_home_win_pct=42.0,
        h2h_draw_pct=30.0,
        h2h_away_win_pct=28.0,
        h2h_over_pct=58.0,
        h2h_btts_pct=56.0,
        h2h_matches_count=8,
        forma_mult_h=1.02,
        forma_mult_a=0.99,
        fixture_historical_total=2.7,
        extraction_coverage=0.80,
        extraction_section_scores={},
        odds_sharp_signal=0.4,
        weather_impact=0.0,
        # Empty absences list with zero counts should stay neutral
        home_absences_count=0,
        away_absences_count=0,
        home_absences_players=[],
        away_absences_players=[],
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_full_prematch_flow_returns_normalized_probabilities() -> None:
    lines = {
        "ah_op": -0.25,
        "tot_op": 2.25,
        "ah_cur": -0.25,
        "tot_cur": 2.25,
        "tot_cur_raw": 2.25,
        "linea_ou": 2.25,
        "fullgame_mode": True,
        "validation_errors": [],
        "blocking_errors": [],
    }
    match = {
        "minuto": 0,
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
    }
    pa = _fake_pa()
    state, league, coverage = build_match_state_from_prematch_analysis(
        pa,
        match=match,
        lines=lines,
        linea_ou=2.25,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    out, cal = run_analysis_pipeline(
        state,
        league=league,
        apply_prematch_calibration=True,
        extraction_coverage=coverage,
    )

    assert abs((out.p1 + out.px + out.p2) - 1.0) < 1e-9
    assert abs((out.p_over + out.p_under) - 1.0) < 1e-9
    assert 0.0 <= out.p_btts <= 1.0
    assert out.xg_h_final > 0.0 and out.xg_a_final > 0.0
    assert cal is not None


def test_bridge_absence_multiplier_neutral_without_player_details() -> None:
    lines = {
        "ah_op": 0.0,
        "tot_op": 2.5,
        "ah_cur": 0.0,
        "tot_cur": 2.5,
        "tot_cur_raw": 2.5,
        "linea_ou": 2.5,
        "fullgame_mode": True,
        "validation_errors": [],
        "blocking_errors": [],
    }
    match = {
        "minuto": 0,
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
    }
    pa = _fake_pa(
        home_absences_count=3,
        away_absences_count=2,
        home_absences_players=[],
        away_absences_players=[],
    )
    state, _, _ = build_match_state_from_prematch_analysis(
        pa,
        match=match,
        lines=lines,
        linea_ou=2.5,
        bankroll=1000.0,
        comm_rate=0.025,
    )
    assert state.absence_mult_h == 1.0
    assert state.absence_mult_a == 1.0
