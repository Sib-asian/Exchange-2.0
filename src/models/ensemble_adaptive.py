"""
Pesi consensus adattivi da storico (solo prematch).

Usa i Brier 1X2 marginali per modello (bp / copula / markov) registrati nei
PredictionRecord completati, e sposta leggermente i pesi dalla baseline di fase.
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


def _record_has_model_probs(r: PredictionRecord) -> bool:
    return (
        r.p1_bp > 0.0 or r.px_bp > 0.0 or r.p2_bp > 0.0
    ) and (
        r.p1_cop > 0.0 or r.px_cop > 0.0 or r.p2_cop > 0.0
    ) and (
        r.p1_mk > 0.0 or r.px_mk > 0.0 or r.p2_mk > 0.0
    )


def blend_consensus_weights_with_history(
    minuto: int,
    w_bp: float,
    w_cop: float,
    w_mk: float,
    *,
    min_completed: int = 10,
    max_blend: float = 0.28,
) -> tuple[float, float, float]:
    """
    Ritorna pesi (bp, cop, mk) fra baseline di fase e stima da storico.

    In live (`minuto > 0`) non modifica (le distribuzioni sono path-dependent).
    """
    if minuto != 0:
        return w_bp, w_cop, w_mk

    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = [
            r
            for r in log.get_completed()
            if r.is_prematch and r.is_completed() and r.risultato_1x2 in ("1", "X", "2")
        ]
        usable = [r for r in completed if _record_has_model_probs(r)]
    except Exception:
        return w_bp, w_cop, w_mk

    if len(usable) < min_completed:
        return w_bp, w_cop, w_mk

    losses: list[tuple[float, float, float]] = []
    for r in usable:
        out = r.risultato_1x2
        losses.append(
            (
                _multiclass_brier_1x2(r.p1_bp, r.px_bp, r.p2_bp, out),
                _multiclass_brier_1x2(r.p1_cop, r.px_cop, r.p2_cop, out),
                _multiclass_brier_1x2(r.p1_mk, r.px_mk, r.p2_mk, out),
            )
        )

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

    alpha = min(max_blend, (len(usable) / 50.0) * max_blend)

    b = (1.0 - alpha) * w_bp + alpha * lw_bp
    c = (1.0 - alpha) * w_cop + alpha * lw_cop
    m = (1.0 - alpha) * w_mk + alpha * lw_mk
    tot = b + c + m
    if tot <= 0:
        return w_bp, w_cop, w_mk
    return b / tot, c / tot, m / tot


__all__ = ["blend_consensus_weights_with_history"]
