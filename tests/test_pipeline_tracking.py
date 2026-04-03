"""Test pipeline centralizzata, JSON prediction log retrocompatibile, shrink, stats."""

from __future__ import annotations

import pytest

from src.engine import MatchState
from src.models.uncertainty_shrink import shrink_outcome_probs
from src.pipeline import run_analysis_pipeline
from src.tracking.prediction_log import (
    PredictionRecord,
    record_from_dict,
    tot_op_band,
)
from src.tracking.stats import PerformanceStats


def test_tot_op_band():
    assert tot_op_band(0) == "unknown"
    assert tot_op_band(2.0) == "<2.25"
    assert tot_op_band(2.5) == "2.25-2.75"
    assert tot_op_band(3.0) == ">2.75"


def test_record_from_dict_ignores_unknown_keys():
    raw = {
        "id": "x",
        "timestamp": "2026-01-01T00:00:00",
        "p1": 0.4,
        "future_field_xyz": 123,
        "status": "PENDING",
    }
    r = record_from_dict(raw)
    assert r.id == "x"
    assert r.p1 == 0.4
    assert not hasattr(r, "future_field_xyz")


def test_shrink_moves_toward_uniform_when_weak_signal():
    p1, px, p2 = 0.7, 0.15, 0.15
    q1, qx, q2, qo, qu, qb = shrink_outcome_probs(
        p1, px, p2, 0.6, 0.4, 0.5,
        extraction_coverage=0.1,
        model_agreement=0.1,
    )
    assert abs((q1 + qx + q2) - 1.0) < 1e-6
    assert q1 < p1
    assert qx > px


def test_run_analysis_pipeline_prematch_runs():
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
    out, sig = run_analysis_pipeline(
        st,
        league="",
        apply_prematch_calibration=False,
        extraction_coverage=1.0,
    )
    assert out.p1 + out.px + out.p2 == pytest.approx(1.0, abs=1e-3)
    assert out.consensus_w_bp > 0
    assert out.p1_bp > 0.0
    assert sig is None


def test_multiclass_brier_and_segments():
    recs = [
        PredictionRecord(
            id="a",
            timestamp="t",
            p1=0.5,
            px=0.3,
            p2=0.2,
            lega="Test League",
            tot_band="2.25-2.75",
            risultato_1x2="1",
            gol_casa=1,
            gol_trasf=0,
            status="COMPLETED",
        ),
        PredictionRecord(
            id="b",
            timestamp="t2",
            p1=0.2,
            px=0.3,
            p2=0.5,
            lega="Test League",
            tot_band="2.25-2.75",
            risultato_1x2="2",
            gol_casa=0,
            gol_trasf=1,
            status="COMPLETED",
        ),
    ]
    b = PerformanceStats.compute_multiclass_brier_1x2(recs)
    assert b is not None and b >= 0
    ll = PerformanceStats.compute_log_loss_1x2(recs)
    assert ll is not None and ll > 0
    assert "Test League" in PerformanceStats.segment_by_league(recs)
