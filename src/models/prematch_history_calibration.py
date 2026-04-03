"""
prematch_history_calibration.py — calibrazione leggera basata su storico chiuso.

Usa solo partite prematch completate dal Prediction Tracker:
- stima bias medio per 1X2, Over 2.5 e BTTS
- applica un piccolo nudge conservativo (mai aggressivo)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tracking.prediction_log import PredictionRecord, get_prediction_log

_MIN_SAMPLES = 20
_MAX_WEIGHT = 0.18
_SCALE_MIN = 0.85
_SCALE_MAX = 1.15


@dataclass(frozen=True)
class CalibrationSignals:
    p1_scale: float = 1.0
    px_scale: float = 1.0
    p2_scale: float = 1.0
    over_scale: float = 1.0
    btts_scale: float = 1.0
    weight: float = 0.0
    samples: int = 0
    scope: str = "global"


def _safe_scale(avg_pred: float, avg_outcome: float) -> float:
    if avg_pred <= 1e-9:
        return 1.0
    raw = avg_outcome / avg_pred
    return max(_SCALE_MIN, min(_SCALE_MAX, raw))


def _prematch_completed_records() -> list[PredictionRecord]:
    records = get_prediction_log().get_completed()
    return [r for r in records if getattr(r, "is_prematch", False)]


def _normalize_league(league: str) -> str:
    return (league or "").strip().lower()


def estimate_calibration_signals(league: str = "") -> CalibrationSignals:
    records = _prematch_completed_records()
    target_league = _normalize_league(league)
    league_records = [r for r in records if _normalize_league(getattr(r, "lega", "")) == target_league] if target_league else []

    # Preferisci calibrazione per lega se ci sono abbastanza campioni.
    if len(league_records) >= max(12, _MIN_SAMPLES // 2):
        records = league_records
        scope = f"league:{target_league}"
    else:
        scope = "global"

    n = len(records)
    if n < _MIN_SAMPLES:
        return CalibrationSignals(samples=n, scope=scope)

    avg_p1 = sum(r.p1 for r in records) / n
    avg_px = sum(r.px for r in records) / n
    avg_p2 = sum(r.p2 for r in records) / n
    avg_over = sum(r.p_over_25 for r in records) / n
    avg_btts = sum(r.p_btts for r in records) / n

    avg_o1 = sum(1.0 for r in records if r.risultato_1x2 == "1") / n
    avg_ox = sum(1.0 for r in records if r.risultato_1x2 == "X") / n
    avg_o2 = sum(1.0 for r in records if r.risultato_1x2 == "2") / n
    avg_over_hit = sum(1.0 for r in records if bool(r.over_25_hit)) / n
    avg_btts_hit = sum(1.0 for r in records if bool(r.btts_hit)) / n

    # Peso cresce con i campioni ma resta conservativo.
    weight = min(_MAX_WEIGHT, (n - _MIN_SAMPLES) / 100.0 * _MAX_WEIGHT)
    return CalibrationSignals(
        p1_scale=_safe_scale(avg_p1, avg_o1),
        px_scale=_safe_scale(avg_px, avg_ox),
        p2_scale=_safe_scale(avg_p2, avg_o2),
        over_scale=_safe_scale(avg_over, avg_over_hit),
        btts_scale=_safe_scale(avg_btts, avg_btts_hit),
        weight=max(0.0, weight),
        samples=n,
        scope=scope,
    )


def calibrate_prematch_probs(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    league: str = "",
) -> tuple[float, float, float, float, float, float, CalibrationSignals]:
    signals = estimate_calibration_signals(league=league)
    if signals.weight <= 0:
        return p1, px, p2, p_over, p_under, p_btts, signals

    p1_adj = (1.0 - signals.weight) * p1 + signals.weight * min(1.0, p1 * signals.p1_scale)
    px_adj = (1.0 - signals.weight) * px + signals.weight * min(1.0, px * signals.px_scale)
    p2_adj = (1.0 - signals.weight) * p2 + signals.weight * min(1.0, p2 * signals.p2_scale)
    s = max(1e-9, p1_adj + px_adj + p2_adj)
    p1_adj, px_adj, p2_adj = p1_adj / s, px_adj / s, p2_adj / s

    over_adj = (1.0 - signals.weight) * p_over + signals.weight * min(1.0, p_over * signals.over_scale)
    over_adj = max(0.0, min(1.0, over_adj))
    under_adj = 1.0 - over_adj

    btts_adj = (1.0 - signals.weight) * p_btts + signals.weight * min(1.0, p_btts * signals.btts_scale)
    btts_adj = max(0.0, min(1.0, btts_adj))

    return p1_adj, px_adj, p2_adj, over_adj, under_adj, btts_adj, signals
