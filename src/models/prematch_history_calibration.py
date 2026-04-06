"""
prematch_history_calibration.py — calibrazione leggera basata su storico chiuso.

Usa solo partite prematch completate dal Prediction Tracker:
- stima bias medio per 1X2, Over 2.5 e BTTS
- applica un piccolo nudge conservativo (mai aggressivo)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.tracking.prediction_log import PredictionRecord, get_prediction_log

_MIN_SAMPLES = 20
_MAX_WEIGHT = 0.18
_MIN_WEIGHT_AT_THRESHOLD = 0.04
_RAMP_MATCHES = 120
_SCALE_MIN = 0.85
_SCALE_MAX = 1.15
_RECENCY_TAU_MATCHES = 35.0


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


def _learning_weight(n: int) -> float:
    """
    Peso calibrazione:
    - 0 fino a MIN_SAMPLES-1
    - >0 già a MIN_SAMPLES (inizio apprendimento percepibile)
    - ramp graduale fino al cap
    """
    if n < _MIN_SAMPLES:
        return 0.0
    progress = min(1.0, max(0.0, (n - _MIN_SAMPLES) / max(1.0, _RAMP_MATCHES)))
    return _MIN_WEIGHT_AT_THRESHOLD + progress * (_MAX_WEIGHT - _MIN_WEIGHT_AT_THRESHOLD)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / denom


def _recency_weights(n: int) -> list[float]:
    # records arrivano in ordine cronologico; più recente = peso maggiore.
    return [math.exp(-(n - 1 - i) / _RECENCY_TAU_MATCHES) for i in range(n)]


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

    base = _estimate_from_records(records)
    return CalibrationSignals(
        p1_scale=base.p1_scale,
        px_scale=base.px_scale,
        p2_scale=base.p2_scale,
        over_scale=base.over_scale,
        btts_scale=base.btts_scale,
        weight=base.weight,
        samples=base.samples,
        scope=scope,
    )


def estimate_calibration_signals_segmented(
    *,
    league: str = "",
    tot_band: str = "",
) -> CalibrationSignals:
    """
    Calibrazione gerarchica:
    1) league + tot_band
    2) league
    3) global
    """
    records = _prematch_completed_records()
    target_league = _normalize_league(league)
    target_band = (tot_band or "").strip()

    league_records = [
        r for r in records if _normalize_league(getattr(r, "lega", "")) == target_league
    ] if target_league else []

    league_band_records = [
        r for r in league_records if str(getattr(r, "tot_band", "")).strip() == target_band
    ] if target_band else []

    if len(league_band_records) >= max(10, _MIN_SAMPLES // 2):
        base = _estimate_from_records(league_band_records)
        return CalibrationSignals(
            p1_scale=base.p1_scale,
            px_scale=base.px_scale,
            p2_scale=base.p2_scale,
            over_scale=base.over_scale,
            btts_scale=base.btts_scale,
            weight=base.weight,
            samples=base.samples,
            scope=f"league+band:{target_league}|{target_band}",
        )

    if len(league_records) >= max(12, _MIN_SAMPLES // 2):
        base = _estimate_from_records(league_records)
        return CalibrationSignals(
            p1_scale=base.p1_scale,
            px_scale=base.px_scale,
            p2_scale=base.p2_scale,
            over_scale=base.over_scale,
            btts_scale=base.btts_scale,
            weight=base.weight,
            samples=base.samples,
            scope=f"league:{target_league}",
        )

    base = _estimate_from_records(records)
    return CalibrationSignals(
        p1_scale=base.p1_scale,
        px_scale=base.px_scale,
        p2_scale=base.p2_scale,
        over_scale=base.over_scale,
        btts_scale=base.btts_scale,
        weight=base.weight,
        samples=base.samples,
        scope="global",
    )


def _estimate_from_records(records: list[PredictionRecord]) -> CalibrationSignals:
    n = len(records)
    if n < _MIN_SAMPLES:
        return CalibrationSignals(samples=n, scope="global")

    ws = _recency_weights(n)
    avg_p1 = _weighted_mean([float(r.p1) for r in records], ws)
    avg_px = _weighted_mean([float(r.px) for r in records], ws)
    avg_p2 = _weighted_mean([float(r.p2) for r in records], ws)
    avg_over = _weighted_mean([float(r.p_over_25) for r in records], ws)
    avg_btts = _weighted_mean([float(r.p_btts) for r in records], ws)

    avg_o1 = _weighted_mean([1.0 if r.risultato_1x2 == "1" else 0.0 for r in records], ws)
    avg_ox = _weighted_mean([1.0 if r.risultato_1x2 == "X" else 0.0 for r in records], ws)
    avg_o2 = _weighted_mean([1.0 if r.risultato_1x2 == "2" else 0.0 for r in records], ws)
    avg_over_hit = _weighted_mean([1.0 if bool(r.over_25_hit) else 0.0 for r in records], ws)
    avg_btts_hit = _weighted_mean([1.0 if bool(r.btts_hit) else 0.0 for r in records], ws)

    weight = _learning_weight(n)
    return CalibrationSignals(
        p1_scale=_safe_scale(avg_p1, avg_o1),
        px_scale=_safe_scale(avg_px, avg_ox),
        p2_scale=_safe_scale(avg_p2, avg_o2),
        over_scale=_safe_scale(avg_over, avg_over_hit),
        btts_scale=_safe_scale(avg_btts, avg_btts_hit),
        weight=max(0.0, weight),
        samples=n,
        scope="global",
    )


def calibrate_prematch_probs(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    league: str = "",
    tot_band: str = "",
    *,
    p_over_15: float | None = None,
    p_under_15: float | None = None,
) -> tuple[float, float, float, float, float, float, float | None, float | None, CalibrationSignals]:
    signals = estimate_calibration_signals_segmented(league=league, tot_band=tot_band)
    if signals.weight <= 0:
        return p1, px, p2, p_over, p_under, p_btts, p_over_15, p_under_15, signals

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

    o15, u15 = p_over_15, p_under_15
    if p_over_15 is not None and p_under_15 is not None:
        o15 = (1.0 - signals.weight) * p_over_15 + signals.weight * min(1.0, p_over_15 * signals.over_scale)
        o15 = max(0.0, min(1.0, o15))
        u15 = 1.0 - o15

    return p1_adj, px_adj, p2_adj, over_adj, under_adj, btts_adj, o15, u15, signals
