"""
Pesi consensus adattivi da storico (solo prematch).

Usa i Brier marginali per modello registrati nei PredictionRecord completati:
- 1X2: come prima
- Over/Under (linea salvata): da p_over_* per modello se disponibili
- BTTS: da p_btts_* per modello se disponibili
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tracking.prediction_log import PredictionRecord


def _multiclass_brier_1x2(p1: float, px: float, p2: float, outcome: str) -> float:
    o1, ox, o2 = (1.0, 0.0, 0.0) if outcome == "1" else (
        (0.0, 1.0, 0.0) if outcome == "X" else (0.0, 0.0, 1.0)
    )
    return float((p1 - o1) ** 2 + (px - ox) ** 2 + (p2 - o2) ** 2)


def _binary_brier(p: float, y: float) -> float:
    return float((p - y) ** 2)


def _record_has_model_probs(r: PredictionRecord) -> bool:
    return (
        r.p1_bp > 0.0 or r.px_bp > 0.0 or r.p2_bp > 0.0
    ) and (
        r.p1_cop > 0.0 or r.px_cop > 0.0 or r.p2_cop > 0.0
    ) and (
        r.p1_mk > 0.0 or r.px_mk > 0.0 or r.p2_mk > 0.0
    )


def _record_has_ou_probs(r: PredictionRecord) -> bool:
    return r.p_over_bp > 1e-9 and r.p_over_cop > 1e-9 and r.p_over_mk > 1e-9


def _record_has_btts_probs(r: PredictionRecord) -> bool:
    return r.p_btts_bp > 1e-9 and r.p_btts_cop > 1e-9 and r.p_btts_mk > 1e-9


def _blend_one_market(
    w_bp: float,
    w_cop: float,
    w_mk: float,
    losses: list[tuple[float, float, float]],
    *,
    max_blend: float,
    min_completed: int,
) -> tuple[float, float, float]:
    if len(losses) < min_completed:
        return w_bp, w_cop, w_mk

    mean_bp = sum(t[0] for t in losses) / len(losses)
    mean_cop = sum(t[1] for t in losses) / len(losses)
    mean_mk = sum(t[2] for t in losses) / len(losses)

    eps = 1e-4
    raw_bp = 1.0 / (eps + mean_bp)
    raw_cop = 1.0 / (eps + mean_cop)
    raw_mk = 1.0 / (eps + mean_mk)
    s = raw_bp + raw_cop + raw_mk
    if s <= 0:
        return w_bp, w_cop, w_mk
    lw_bp, lw_cop, lw_mk = raw_bp / s, raw_cop / s, raw_mk / s

    alpha = min(max_blend, (len(losses) / 50.0) * max_blend)

    b = (1.0 - alpha) * w_bp + alpha * lw_bp
    c = (1.0 - alpha) * w_cop + alpha * lw_cop
    m = (1.0 - alpha) * w_mk + alpha * lw_mk
    tot = b + c + m
    if tot <= 0:
        return w_bp, w_cop, w_mk
    return b / tot, c / tot, m / tot


def blend_consensus_weights_with_history(
    minuto: int,
    w_bp: float,
    w_cop: float,
    w_mk: float,
    *,
    min_completed: int = 10,
    max_blend: float = 0.45,
    max_blend_ou: float = 0.36,
    max_blend_btts: float = 0.34,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """
    Ritorna (pesi_1x2, pesi_ou, pesi_btts) fra baseline di fase e stima da storico.

    In live (`minuto > 0`) non modifica (le distribuzioni sono path-dependent).
    """
    base = (w_bp, w_cop, w_mk)
    if minuto != 0:
        return base, base, base

    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = [
            r
            for r in log.get_completed()
            if r.is_prematch and r.is_completed() and r.risultato_1x2 in ("1", "X", "2")
        ]
        usable_1x2 = [r for r in completed if _record_has_model_probs(r)]
        usable_ou = [
            r
            for r in completed
            if _record_has_ou_probs(r) and r.over_25_hit is not None
        ]
        usable_btts = [
            r
            for r in completed
            if _record_has_btts_probs(r) and r.btts_hit is not None
        ]
    except Exception:
        return base, base, base

    losses_1x2: list[tuple[float, float, float]] = []
    for r in usable_1x2:
        out = r.risultato_1x2
        losses_1x2.append(
            (
                _multiclass_brier_1x2(r.p1_bp, r.px_bp, r.p2_bp, out),
                _multiclass_brier_1x2(r.p1_cop, r.px_cop, r.p2_cop, out),
                _multiclass_brier_1x2(r.p1_mk, r.px_mk, r.p2_mk, out),
            )
        )

    w1 = _blend_one_market(
        w_bp, w_cop, w_mk, losses_1x2, max_blend=max_blend, min_completed=min_completed
    )

    losses_ou: list[tuple[float, float, float]] = []
    for r in usable_ou:
        y_ou = 1.0 if r.over_25_hit else 0.0
        losses_ou.append(
            (
                _binary_brier(float(r.p_over_bp), y_ou),
                _binary_brier(float(r.p_over_cop), y_ou),
                _binary_brier(float(r.p_over_mk), y_ou),
            )
        )
    w_ou = _blend_one_market(
        w_bp, w_cop, w_mk, losses_ou, max_blend=max_blend_ou, min_completed=min_completed
    )

    losses_bt: list[tuple[float, float, float]] = []
    for r in usable_btts:
        y_b = 1.0 if r.btts_hit else 0.0
        losses_bt.append(
            (
                _binary_brier(float(r.p_btts_bp), y_b),
                _binary_brier(float(r.p_btts_cop), y_b),
                _binary_brier(float(r.p_btts_mk), y_b),
            )
        )
    w_bt = _blend_one_market(
        w_bp, w_cop, w_mk, losses_bt, max_blend=max_blend_btts, min_completed=min_completed
    )

    return w1, w_ou, w_bt


__all__ = ["blend_consensus_weights_with_history"]
