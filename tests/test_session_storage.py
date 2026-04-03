from src.ocr import PrematchAnalysisExtracted
from src.session_storage import collect_prematch_analysis, restore_prematch_analysis


def test_collect_restore_prematch_analysis_roundtrip():
    session = {
        "prematch_analysis": PrematchAnalysisExtracted(
            extraction_success=True,
            home_team="Team A",
            away_team="Team B",
            league_name="League X",
            home_points=42,
            away_points=39,
            home_matches=20,
            away_matches=20,
            team_stats_home_goals=1.8,
            team_stats_away_goals=1.4,
            extraction_coverage=0.9,
            extraction_notes=["ok"],
        )
    }
    saved = collect_prematch_analysis(session)
    restored_session = {}
    restore_prematch_analysis(restored_session, saved)
    pa = restored_session["prematch_analysis"]
    assert pa.home_points == 42
    assert pa.away_points == 39
    assert pa.team_stats_home_goals == 1.8
    assert pa.extraction_coverage == 0.9
