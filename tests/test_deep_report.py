"""Report a segmenti (linea O/U, lega)."""

from src.tracking.deep_report import build_segment_rows, print_deep_report
from src.tracking.prediction_log import PredictionRecord
from src.tracking.stats import PerformanceStats


def _completed(**kwargs: object) -> PredictionRecord:
    base = {
        "id": "t",
        "timestamp": "2026-01-01T12:00:00",
        "squadra_casa": "A",
        "squadra_trasf": "B",
        "lega": "Test League",
        "minuto": 0,
        "is_prematch": True,
        "p1": 0.4,
        "px": 0.3,
        "p2": 0.3,
        "p_over_25": 0.55,
        "p_under_25": 0.45,
        "ou_line": 2.5,
        "gol_casa": 2,
        "gol_trasf": 1,
        "status": "COMPLETED",
        "risultato_1x2": "1",
        "over_25_hit": True,
        "btts_hit": True,
    }
    base.update(kwargs)
    return PredictionRecord(**base)  # type: ignore[arg-type]


def test_segment_by_ou_line_groups() -> None:
    a = _completed(id="1", ou_line=1.5, p_over_25=0.6, over_25_hit=True)
    b = _completed(id="2", ou_line=1.5, p_over_25=0.4, over_25_hit=False)
    c = _completed(id="3", ou_line=2.5, p_over_25=0.5, over_25_hit=True)
    done = [a, b, c]
    seg = PerformanceStats.segment_by_ou_line(done)
    assert set(seg.keys()) == {"1.5", "2.5"}
    assert len(seg["1.5"]) == 2
    assert len(seg["2.5"]) == 1


def test_build_segment_rows_respects_min_n() -> None:
    a = _completed(id="1", ou_line=2.5)
    b = _completed(id="2", ou_line=2.5)
    seg = PerformanceStats.segment_by_ou_line([a, b])
    rows = build_segment_rows([a, b], seg, min_n=3)
    assert rows == []
    rows2 = build_segment_rows([a, b], seg, min_n=2)
    assert len(rows2) == 1
    assert rows2[0]["N"] == 2
    assert "ECE 1X2" in rows2[0]
    assert "CLV 1X2" in rows2[0]


def test_print_deep_report_no_crash(capsys: object) -> None:
    print_deep_report([], min_n=3)
    captured = capsys.readouterr()
    assert "0" in captured.out or "nessun" in captured.out.lower()

    recs = [_completed(id=str(i)) for i in range(3)]
    print_deep_report(recs, min_n=3)
    out = capsys.readouterr().out
    assert "2.5" in out or "O/U" in out
