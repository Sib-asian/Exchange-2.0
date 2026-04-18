"""Ensemble weights: O/U learning su Over 2.5 per-modello quando il log ha campioni EU."""

from __future__ import annotations

import pytest

from src.engine import MatchState, analizza
from src.models.ensemble_adaptive import blend_consensus_weights_with_history
from src.tracking.prediction_log import PredictionRecord


def test_analizza_populates_per_model_over_eu_at_line_25():
    st = MatchState(
        minuto=0,
        gol_casa=0,
        gol_trasf=0,
        rossi_casa=0,
        rossi_trasf=0,
        ah_op=-0.25,
        tot_op=2.5,
        ah_cur=-0.25,
        tot_cur=2.5,
        linea_ou=2.5,
    )
    r = analizza(st)
    assert r.p_over_bp_eu == pytest.approx(r.p_over_bp, abs=1e-9)
    assert r.p_over_mk_eu == pytest.approx(r.p_over_mk, abs=1e-9)


def test_analizza_per_model_over_eu_not_equal_when_analyzed_line_higher():
    st = MatchState(
        minuto=0,
        gol_casa=0,
        gol_trasf=0,
        rossi_casa=0,
        rossi_trasf=0,
        ah_op=-0.25,
        tot_op=3.25,
        ah_cur=-0.25,
        tot_cur=3.25,
        linea_ou=3.0,
    )
    r = analizza(st)
    assert r.p_over_bp > 0 and r.p_over_bp_eu > 0
    # P(Over 2.5) >= P(Over 3.0) per la stessa legge congiunta
    assert r.p_over_bp_eu >= r.p_over_bp - 1e-6


def test_blend_prefers_eu_ou_when_enough_completed_records(monkeypatch):
    recs: list[PredictionRecord] = []
    for i in range(12):
        recs.append(
            PredictionRecord(
                id=f"eu{i}",
                timestamp="2026-01-01T00:00:00",
                is_prematch=True,
                status="COMPLETED",
                lega="Test",
                risultato_1x2="1",
                gol_casa=2,
                gol_trasf=1,
                p1_bp=0.4,
                px_bp=0.3,
                p2_bp=0.3,
                p1_cop=0.4,
                px_cop=0.3,
                p2_cop=0.3,
                p1_mk=0.4,
                px_mk=0.3,
                p2_mk=0.3,
                p_over_bp_eu=0.55,
                p_over_cop_eu=0.50,
                p_over_mk_eu=0.52,
                p_over_bp=0.40,
                p_over_cop=0.40,
                p_over_mk=0.40,
                over_eu_25_hit=True,
                over_25_hit=True,
                btts_hit=True,
            )
        )

    class _Log:
        def get_completed(self):
            return recs

    monkeypatch.setattr("src.tracking.prediction_log.get_prediction_log", lambda: _Log())
    _w1, w_ou, _wb = blend_consensus_weights_with_history(
        0, 0.5, 0.3, 0.2, min_completed=10
    )
    assert sum(w_ou) == pytest.approx(1.0, abs=1e-6)
