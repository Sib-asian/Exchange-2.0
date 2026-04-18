"""Tests for empirical correct-score blending from prediction log."""

from __future__ import annotations

from src.models.correct_score_history import blend_top_cs_with_history
from src.tracking.prediction_log import PredictionRecord


def test_blend_top_cs_no_history_returns_unchanged(monkeypatch):
    monkeypatch.setattr(
        "src.models.correct_score_history.get_prediction_log",
        lambda: type("_L", (), {"get_completed": staticmethod(lambda: [])})(),
    )
    top = [((1, 1), 0.25), ((1, 0), 0.20)]
    out = blend_top_cs_with_history(top, league="Serie A", tot_band="2.25-2.75")
    assert out == top


def test_blend_top_cs_moves_toward_empirical(monkeypatch):
    records: list[PredictionRecord] = []
    for i in range(20):
        records.append(
            PredictionRecord(
                id=f"c{i}",
                timestamp=f"2026-01-{1 + i % 28:02d}T00:00:00",
                lega="Serie A",
                tot_band="2.25-2.75",
                is_prematch=True,
                status="COMPLETED",
                gol_casa=1,
                gol_trasf=1,
                risultato_1x2="X",
                over_25_hit=True,
                btts_hit=True,
            )
        )

    class _Log:
        def get_completed(self):
            return records

    monkeypatch.setattr("src.models.correct_score_history.get_prediction_log", lambda: _Log())
    top = [((2, 2), 0.40), ((1, 1), 0.15)]
    out = blend_top_cs_with_history(
        top,
        league="Serie A",
        tot_band="2.25-2.75",
        extraction_trust=1.0,
        model_agreement=1.0,
    )
    d1 = dict(top)
    d2 = dict(out)
    assert d2[(1, 1)] > d1[(1, 1)]
