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


def test_snapshot_2_small_sample_h2h_count():
    parsed = _extract_all_with_regex(_fixture("nowgoal_snapshot_2.txt"))
    assert parsed["home_team"] == "Team Alpha"
    assert parsed["away_team"] == "Team Beta"
    assert parsed["h2h_matches_count"] == 2
    assert parsed["h2h_btts_pct"] == 0.0
