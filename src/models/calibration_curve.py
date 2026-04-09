"""
calibration_curve.py — Calibrazione cross-validated dalle previsioni storiche.

Costruisce una curva di calibrazione (predicted → observed) dal prediction_log
e la applica per correggere bias sistematici del modello.

Se il modello predice P(Home)=70% per 100 partite e la casa vince solo il 55%,
la curva rivela un bias di +15% e corregge automaticamente le nuove previsioni.

Approccio: Platt scaling (sigmoide), più robusto di isotonic regression
con campioni limitati (<200).

Riferimenti:
  Platt (1999), "Probabilistic Outputs for Support Vector Machines"
  Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities With Supervised Learning"
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

MIN_RECORDS_FOR_CALIBRATION: int = 25
MAX_PLATT_SHIFT: float = 0.08  # Massimo shift dalla curva Platt (conservativo)


# ---------------------------------------------------------------------------
# Platt scaling (logistic calibration)
# ---------------------------------------------------------------------------

def _fit_platt_params(
    predictions: list[float],
    outcomes: list[float],
) -> tuple[float, float] | None:
    """
    Fit dei parametri Platt (a, b) tramite minimizzazione.

    Modello: P_calibrated = sigmoid(a * logit(P_raw) + b)

    Con a=1.0, b=0.0 → nessuna correzione.
    a > 1 → sharpening, a < 1 → smoothing.
    b > 0 → shift verso l'alto, b < 0 → shift verso il basso.

    Usa grid search semplice (robusto con pochi campioni).

    Returns:
        (a, b) o None se non abbastanza dati.
    """
    if len(predictions) < MIN_RECORDS_FOR_CALIBRATION:
        return None

    # Filtra predizioni degeneri
    valid = [(p, o) for p, o in zip(predictions, outcomes) if 0.01 < p < 0.99]
    if len(valid) < MIN_RECORDS_FOR_CALIBRATION:
        return None

    preds_v = [v[0] for v in valid]
    outs_v = [v[1] for v in valid]

    best_a, best_b = 1.0, 0.0
    best_loss = _log_loss(preds_v, outs_v, 1.0, 0.0)

    # Grid search su a ∈ [0.85, 1.15], b ∈ [-0.15, 0.15]
    for a_int in range(85, 116, 3):
        a = a_int / 100.0
        for b_int in range(-15, 16, 3):
            b = b_int / 100.0
            loss = _log_loss(preds_v, outs_v, a, b)
            if loss < best_loss:
                best_loss = loss
                best_a, best_b = a, b

    return best_a, best_b


def _log_loss(
    predictions: list[float],
    outcomes: list[float],
    a: float,
    b: float,
) -> float:
    """Log-loss con parametri Platt (a, b)."""
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p_cal = _apply_platt(p, a, b)
        p_cal = max(1e-9, min(1.0 - 1e-9, p_cal))
        total -= o * math.log(p_cal) + (1.0 - o) * math.log(1.0 - p_cal)
    return total / max(1, len(predictions))


def _apply_platt(p: float, a: float, b: float) -> float:
    """Applica Platt scaling: sigmoid(a * logit(p) + b)."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


# ---------------------------------------------------------------------------
# Calibrazione per mercato
# ---------------------------------------------------------------------------

def build_calibration_maps() -> dict[str, tuple[float, float]]:
    """
    Costruisce le mappe di calibrazione Platt per ogni mercato.

    Usa i PredictionRecord completati dal prediction_log.

    Returns:
        Dict {mercato: (a, b)} per ogni mercato con dati sufficienti.
        Mercati senza dati sufficienti non appaiono nel dict.
    """
    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = [r for r in log.get_completed() if r.is_prematch and r.is_completed()]
    except Exception:
        return {}

    if len(completed) < MIN_RECORDS_FOR_CALIBRATION:
        return {}

    maps: dict[str, tuple[float, float]] = {}

    # 1X2 — Home win
    preds_1 = [float(r.p1) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_1 = [1.0 if r.risultato_1x2 == "1" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_1, outs_1)
    if params:
        maps["p1"] = params

    # 1X2 — Draw
    preds_x = [float(r.px) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_x = [1.0 if r.risultato_1x2 == "X" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_x, outs_x)
    if params:
        maps["px"] = params

    # 1X2 — Away win
    preds_2 = [float(r.p2) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_2 = [1.0 if r.risultato_1x2 == "2" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_2, outs_2)
    if params:
        maps["p2"] = params

    # Over/Under
    preds_over = [float(r.p_over_25) for r in completed if r.over_25_hit is not None]
    outs_over = [1.0 if r.over_25_hit else 0.0 for r in completed if r.over_25_hit is not None]
    params = _fit_platt_params(preds_over, outs_over)
    if params:
        maps["p_over"] = params

    # BTTS
    preds_btts = [float(r.p_btts) for r in completed if r.btts_hit is not None]
    outs_btts = [1.0 if r.btts_hit else 0.0 for r in completed if r.btts_hit is not None]
    params = _fit_platt_params(preds_btts, outs_btts)
    if params:
        maps["p_btts"] = params

    return maps


def apply_calibration(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    cal_maps: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    """
    Applica la calibrazione Platt alle probabilità.

    La correzione è limitata a ±MAX_PLATT_SHIFT per evitare
    overcorrection con pochi campioni.

    Returns:
        (p1, px, p2, p_over, p_under, p_btts) calibrate.
    """
    if not cal_maps:
        return p1, px, p2, p_over, p_under, p_btts

    def _safe_apply(p: float, key: str) -> float:
        if key not in cal_maps or p <= 0.001 or p >= 0.999:
            return p
        a, b = cal_maps[key]
        p_cal = _apply_platt(p, a, b)
        # Limita lo shift massimo
        shift = p_cal - p
        shift = max(-MAX_PLATT_SHIFT, min(MAX_PLATT_SHIFT, shift))
        return max(0.001, min(0.999, p + shift))

    p1_cal = _safe_apply(p1, "p1")
    px_cal = _safe_apply(px, "px")
    p2_cal = _safe_apply(p2, "p2")

    # Rinormalizza 1X2
    s = p1_cal + px_cal + p2_cal
    if s > 0:
        p1_cal /= s
        px_cal /= s
        p2_cal /= s

    p_over_cal = _safe_apply(p_over, "p_over")
    p_under_cal = 1.0 - p_over_cal

    p_btts_cal = _safe_apply(p_btts, "p_btts")

    return p1_cal, px_cal, p2_cal, p_over_cal, p_under_cal, p_btts_cal


__all__ = ["build_calibration_maps", "apply_calibration"]
