"""Test pipeline centralizzata, JSON prediction log retrocompatibile, shrink, stats."""

from __future__ import annotations

import pytest

from src.engine import MatchState
from src.models.uncertainty_shrink import shrink_outcome_probs
from src.pipeline import run_analysis_pipeline
from src.tracking.prediction_log import (
    PredictionRecord,
    assess_quote_quality,
    create_record_from_analysis,
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
    q1, qx, q2, qo, qu, qb, qo15, qu15 = shrink_outcome_probs(
        p1, px, p2, 0.6, 0.4, 0.5,
        extraction_coverage=0.1,
        model_agreement=0.1,
    )
    assert qo15 is None
    assert qu15 is None
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
    assert sig is None or hasattr(sig, "weight")
    assert out.p_over_15 + out.p_under_15 == pytest.approx(1.0, abs=1e-3)
    assert out.p_over_15 > out.p_over


def test_run_analysis_pipeline_sets_quality_firewall_fields():
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
    out, _ = run_analysis_pipeline(
        st,
        league="",
        apply_prematch_calibration=False,
        extraction_coverage=0.10,
    )
    assert 0.0 <= out.quality_score <= 1.0
    assert isinstance(out.signals_blocked, bool)
    assert isinstance(out.signals_block_reason, str)


def test_run_analysis_pipeline_blocks_signals_when_quality_firewall_triggers():
    st = MatchState(
        minuto=0,
        gol_casa=0,
        gol_trasf=0,
        rossi_casa=0,
        rossi_trasf=0,
        ah_op=0.0,
        tot_op=2.5,
        ah_cur=0.0,
        tot_cur=2.5,
        linea_ou=2.5,
    )
    out, _ = run_analysis_pipeline(
        st,
        league="",
        apply_prematch_calibration=False,
        extraction_coverage=0.0,
    )
    assert out.signals_blocked is True
    assert out.signals_block_reason != ""


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


def test_market_stats_ece_and_clv_proxy():
    recs = [
        PredictionRecord(
            id="c1",
            timestamp="t",
            p1=0.60,
            px=0.25,
            p2=0.15,
            quota_1=2.10,
            quota_x=3.40,
            quota_2=3.80,
            quota_1_close=2.00,
            quota_x_close=3.50,
            quota_2_close=4.10,
            quote_quality="trusted",
            risultato_1x2="1",
            gol_casa=2,
            gol_trasf=1,
            status="COMPLETED",
        ),
        PredictionRecord(
            id="c2",
            timestamp="t2",
            p1=0.45,
            px=0.30,
            p2=0.25,
            quota_1=2.40,
            quota_x=3.10,
            quota_2=2.90,
            quota_1_close=2.50,
            quota_x_close=3.00,
            quota_2_close=2.80,
            quote_quality="trusted",
            risultato_1x2="2",
            gol_casa=0,
            gol_trasf=1,
            status="COMPLETED",
        ),
    ]
    s = PerformanceStats.compute_market_stats(recs, "1X2_1")
    assert s.total_predictions == 2
    assert 0.0 <= s.ece_score <= 1.0
    assert s._clv_n == 2
    clv = PerformanceStats.compute_clv_proxy_1x2(recs)
    assert clv is not None


def test_ece_binary_returns_value():
    ece = PerformanceStats.compute_ece_binary(
        probs=[0.1, 0.2, 0.8, 0.9],
        outcomes=[0, 0, 1, 1],
        bins=5,
    )
    assert ece is not None
    assert 0.0 <= ece <= 1.0


def test_champion_challenger_gate_promotes_when_better():
    # challenger nettamente migliore del champion su stessi id.
    champion = []
    challenger = []
    for i in range(35):
        rid = f"id{i}"
        out = "1" if i % 3 != 0 else "2"
        champion.append(
            PredictionRecord(
                id=rid,
                timestamp="t",
                p1=0.40,
                px=0.30,
                p2=0.30,
                risultato_1x2=out,
                status="COMPLETED",
                gol_casa=1 if out == "1" else 0,
                gol_trasf=0 if out == "1" else 1,
            )
        )
        challenger.append(
            PredictionRecord(
                id=rid,
                timestamp="t",
                p1=0.96 if out == "1" else 0.02,
                px=0.02,
                p2=0.02 if out == "1" else 0.96,
                risultato_1x2=out,
                status="COMPLETED",
                gol_casa=1 if out == "1" else 0,
                gol_trasf=0 if out == "1" else 1,
            )
        )
    ev = PerformanceStats.evaluate_champion_challenger(champion, challenger)
    assert ev.samples >= 30
    assert ev.promote is True


def test_create_record_id_includes_microseconds():
    r = create_record_from_analysis(
        "Alpha",
        "Beta",
        "Test",
        {"tot_op": 2.5},
        {"p1": 0.34, "px": 0.33, "p2": 0.33},
    )
    parts = r.id.split("_")
    assert len(parts) >= 5
    assert parts[2].isdigit() and len(parts[2]) == 6


def test_market_stats_avg_edge_only_uses_rows_with_quota():
    recs = [
        PredictionRecord(
            id="a",
            timestamp="t",
            p1=0.6,
            px=0.25,
            p2=0.15,
            quota_1=2.0,
            quota_x=3.5,
            quota_2=3.5,
            risultato_1x2="1",
            gol_casa=1,
            gol_trasf=0,
            status="COMPLETED",
        ),
        PredictionRecord(
            id="b",
            timestamp="t2",
            p1=0.6,
            px=0.25,
            p2=0.15,
            quota_1=0.0,
            quota_x=0.0,
            quota_2=0.0,
            risultato_1x2="2",
            gol_casa=0,
            gol_trasf=1,
            status="COMPLETED",
        ),
    ]
    s = PerformanceStats.compute_market_stats(recs, "1X2_1")
    assert s.total_predictions == 2
    assert s.predictions_with_quote == 1
    implied = 1.0 / 2.0
    assert s.avg_edge == pytest.approx(0.6 - implied)


def test_assess_quote_quality_trusted_and_untrusted():
    q_ok = {"quota_1": 2.20, "quota_x": 3.30, "quota_2": 3.40}
    q_bad = {"quota_1": 2.20, "quota_x": 0.0, "quota_2": 3.40}
    quality_ok, _ = assess_quote_quality(q_ok, {"quote_source": "initial"})
    quality_bad, _ = assess_quote_quality(q_bad, {"quote_source": "initial"})
    assert quality_ok == "trusted"
    assert quality_bad == "untrusted"


def test_create_record_stores_quote_quality():
    r = create_record_from_analysis(
        "Alpha",
        "Beta",
        "Test",
        {"tot_op": 2.5},
        {"p1": 0.34, "px": 0.33, "p2": 0.33},
        {"quota_1": 2.1, "quota_x": 3.3, "quota_2": 3.4},
        {"quote_source": "initial"},
    )
    assert r.quote_quality == "trusted"
    assert "ok" in r.quote_quality_reason


def test_trusted_only_stats_ignore_untrusted_quotes_for_edge():
    recs = [
        PredictionRecord(
            id="t1",
            timestamp="t1",
            p1=0.6,
            px=0.25,
            p2=0.15,
            quota_1=2.0,
            quota_x=3.5,
            quota_2=3.5,
            quote_quality="trusted",
            risultato_1x2="1",
            gol_casa=1,
            gol_trasf=0,
            status="COMPLETED",
        ),
        PredictionRecord(
            id="t2",
            timestamp="t2",
            p1=0.6,
            px=0.25,
            p2=0.15,
            quota_1=2.0,
            quota_x=3.5,
            quota_2=3.5,
            quote_quality="untrusted",
            risultato_1x2="2",
            gol_casa=0,
            gol_trasf=1,
            status="COMPLETED",
        ),
    ]
    s_all = PerformanceStats.compute_market_stats(recs, "1X2_1")
    s_trusted = PerformanceStats.compute_market_stats(recs, "1X2_1", trusted_only_quotes=True)
    assert s_all.predictions_with_quote == 2
    assert s_trusted.predictions_with_quote == 1


def test_pick_best_falls_back_to_brier_without_quotes():
    """Con poche quote, pick_best usa Brier (più basso = meglio)."""
    stats = {
        "1X2_1": PerformanceStats.compute_market_stats(
            [
                PredictionRecord(
                    id="1",
                    timestamp="t",
                    p1=0.5,
                    px=0.3,
                    p2=0.2,
                    risultato_1x2="1",
                    gol_casa=1,
                    gol_trasf=0,
                    status="COMPLETED",
                ),
            ]
            * 5,
            "1X2_1",
        ),
        "1X2_X": PerformanceStats.compute_market_stats(
            [
                PredictionRecord(
                    id="x",
                    timestamp="t",
                    p1=0.33,
                    px=0.34,
                    p2=0.33,
                    risultato_1x2="X",
                    gol_casa=0,
                    gol_trasf=0,
                    status="COMPLETED",
                ),
            ]
            * 5,
            "1X2_X",
        ),
    }
    best, how = PerformanceStats.pick_best_market(stats, min_n=5, min_with_quote=3)
    assert best is not None
    assert how == "brier"


def test_sort_completed_newest_first():
    a = PredictionRecord(
        id="old",
        timestamp="2020-01-01T10:00:00",
        completed_at="2020-01-02T10:00:00",
        risultato_1x2="1",
        gol_casa=1,
        gol_trasf=0,
        status="COMPLETED",
    )
    b = PredictionRecord(
        id="new",
        timestamp="2021-01-01T10:00:00",
        completed_at="2021-01-03T10:00:00",
        risultato_1x2="1",
        gol_casa=1,
        gol_trasf=0,
        status="COMPLETED",
    )
    out = PerformanceStats.sort_completed_newest_first([a, b])
    assert out[0].id == "new"
