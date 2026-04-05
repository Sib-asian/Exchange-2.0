from src.models.prematch_history_calibration import (
    calibrate_prematch_probs,
    estimate_calibration_signals,
    estimate_calibration_signals_segmented,
)
from src.tracking.prediction_log import PredictionRecord


def test_calibration_no_history_keeps_probabilities():
    p1, px, p2, po, pu, pb, po15, pu15, sig = calibrate_prematch_probs(
        0.45, 0.27, 0.28, 0.54, 0.46, 0.52,
    )
    assert po15 is None
    assert pu15 is None
    assert abs((p1 + px + p2) - 1.0) < 1e-9
    assert abs((po + pu) - 1.0) < 1e-9
    assert 0.0 <= pb <= 1.0
    assert sig.samples >= 0


def test_calibration_prefers_league_scope_when_enough_samples(monkeypatch):
    class _FakeLog:
        def get_completed(self):
            records = []
            for i in range(25):
                records.append(
                    PredictionRecord(
                        id=f"a{i}",
                        timestamp="2026-01-01T00:00:00",
                        lega="Serie A",
                        is_prematch=True,
                        p1=0.45,
                        px=0.27,
                        p2=0.28,
                        p_over_25=0.55,
                        p_under_25=0.45,
                        p_btts=0.52,
                        risultato_1x2="1",
                        over_25_hit=True,
                        btts_hit=True,
                        status="COMPLETED",
                        gol_casa=2,
                        gol_trasf=1,
                    )
                )
            for i in range(25):
                records.append(
                    PredictionRecord(
                        id=f"b{i}",
                        timestamp="2026-01-01T00:00:00",
                        lega="Premier League",
                        is_prematch=True,
                        p1=0.45,
                        px=0.27,
                        p2=0.28,
                        p_over_25=0.55,
                        p_under_25=0.45,
                        p_btts=0.52,
                        risultato_1x2="2",
                        over_25_hit=False,
                        btts_hit=False,
                        status="COMPLETED",
                        gol_casa=0,
                        gol_trasf=1,
                    )
                )
            return records

    monkeypatch.setattr("src.models.prematch_history_calibration.get_prediction_log", lambda: _FakeLog())
    sig = estimate_calibration_signals(league="Serie A")
    assert sig.scope.startswith("league:")
    assert sig.samples == 25


def test_calibration_prefers_league_plus_band_when_available(monkeypatch):
    class _FakeLog:
        def get_completed(self):
            records = []
            for i in range(28):
                records.append(
                    PredictionRecord(
                        id=f"a{i}",
                        timestamp="2026-01-01T00:00:00",
                        lega="Serie A",
                        tot_band="2.25-2.75",
                        is_prematch=True,
                        p1=0.48,
                        px=0.27,
                        p2=0.25,
                        p_over_25=0.56,
                        p_under_25=0.44,
                        p_btts=0.53,
                        risultato_1x2="1",
                        over_25_hit=True,
                        btts_hit=True,
                        status="COMPLETED",
                        gol_casa=2,
                        gol_trasf=1,
                    )
                )
            for i in range(20):
                records.append(
                    PredictionRecord(
                        id=f"b{i}",
                        timestamp="2026-01-01T00:00:00",
                        lega="Serie A",
                        tot_band=">2.75",
                        is_prematch=True,
                        p1=0.44,
                        px=0.30,
                        p2=0.26,
                        p_over_25=0.60,
                        p_under_25=0.40,
                        p_btts=0.55,
                        risultato_1x2="2",
                        over_25_hit=False,
                        btts_hit=False,
                        status="COMPLETED",
                        gol_casa=0,
                        gol_trasf=1,
                    )
                )
            return records

    monkeypatch.setattr("src.models.prematch_history_calibration.get_prediction_log", lambda: _FakeLog())
    sig = estimate_calibration_signals_segmented(league="Serie A", tot_band="2.25-2.75")
    assert sig.scope.startswith("league+band:")
    assert sig.samples == 28
