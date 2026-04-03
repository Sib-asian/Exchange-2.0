from pathlib import Path

from src.ocr import _extract_all_with_regex


def _fixture(name: str) -> str:
    p = Path(__file__).parent / "fixtures" / name
    return p.read_text(encoding="utf-8")


def test_snapshot_1_identity_and_h2h_btts():
    parsed = _extract_all_with_regex(_fixture("nowgoal_snapshot_1.txt"))
    assert parsed["home_team"] == "Adelaide United"
    assert parsed["away_team"] == "Auckland FC"
    assert parsed["league_name"] == "Australia A-League"
    assert parsed["h2h_matches_count"] == 3
    assert parsed["h2h_btts_pct"] == 100.0
    # W/D/L e medie dalla tabella punteggi (prospettiva Adelaide in casa vs Auckland)
    assert parsed["h2h_home_win_pct"] == 0.0
    assert parsed["h2h_draw_pct"] == 66.7
    assert parsed["h2h_away_win_pct"] == 33.3
    assert parsed["h2h_avg_goals_home"] == 2.33
    assert parsed["h2h_avg_goals_away"] == 2.67


def test_snapshot_2_small_sample_h2h_count():
    parsed = _extract_all_with_regex(_fixture("nowgoal_snapshot_2.txt"))
    assert parsed["home_team"] == "Team Alpha"
    assert parsed["away_team"] == "Team Beta"
    assert parsed["h2h_matches_count"] == 2
    assert parsed["h2h_btts_pct"] == 0.0
    assert parsed["h2h_home_win_pct"] == 50.0
    assert parsed["h2h_draw_pct"] == 50.0
    assert parsed["h2h_away_win_pct"] == 0.0


def test_snapshot_3_league_fallback_from_code_and_section_scores():
    parsed = _extract_all_with_regex(_fixture("nowgoal_snapshot_3.txt"))
    assert parsed["league_name"] == "Australia A-League"
    assert parsed["extraction_section_scores"]["identity"] == 1.0
    assert parsed["extraction_section_scores"]["market_1x2"] == 0.0


def test_snapshot_4_plain_code_and_injuries_section():
    parsed = _extract_all_with_regex(_fixture("nowgoal_snapshot_4.txt"))
    assert parsed["league_name"] == "Australia A-League"
    assert parsed["home_absences_count"] == 2
    assert parsed["away_absences_count"] == 1
    assert parsed["extraction_section_scores"]["injuries"] == 1.0
