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
    valid = [(p, o) for p, o in zip(predictions, outcomes, strict=True) if 0.01 < p < 0.99]
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
    for p, o in zip(predictions, outcomes, strict=True):
        p_cal = _apply_platt(p, a, b)
        p_cal = max(1e-9, min(1.0 - 1e-9, p_cal))
        total -= o * math.log(p_cal) + (1.0 - o) * math.log(1.0 - p_cal)
    return total / max(1, len(predictions))


def _apply_platt(p: float, a: float, b: float) -> float:
    """Applica Platt scaling: sigmoid(a * logit(p) + b)."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


def _mean_binary_logloss(pairs: list[tuple[float, float]], a: float, b: float) -> float:
    if not pairs:
        return float("inf")
    tot = 0.0
    for p, o in pairs:
        pc = _apply_platt(p, a, b)
        pc = max(1e-9, min(1.0 - 1e-9, pc))
        tot += -(o * math.log(pc) + (1.0 - o) * math.log(1.0 - pc))
    return tot / len(pairs)


def _platt_improves_on_holdout(
    test_pairs: list[tuple[float, float]],
    a: float,
    b: float,
) -> bool:
    """True se Platt riduce log-loss rispetto all'identità sui dati test."""
    if len(test_pairs) < 5:
        return True
    ll_id = _mean_binary_logloss(test_pairs, 1.0, 0.0)
    ll_pl = _mean_binary_logloss(test_pairs, a, b)
    return ll_pl < ll_id - 1e-5


def _maps_from_record_list(completed: list) -> dict[str, tuple[float, float]]:
    maps: dict[str, tuple[float, float]] = {}

    preds_1 = [float(r.p1) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_1 = [1.0 if r.risultato_1x2 == "1" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_1, outs_1)
    if params:
        maps["p1"] = params

    preds_x = [float(r.px) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_x = [1.0 if r.risultato_1x2 == "X" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_x, outs_x)
    if params:
        maps["px"] = params

    preds_2 = [float(r.p2) for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    outs_2 = [1.0 if r.risultato_1x2 == "2" else 0.0 for r in completed if r.risultato_1x2 in ("1", "X", "2")]
    params = _fit_platt_params(preds_2, outs_2)
    if params:
        maps["p2"] = params

    preds_over = [float(r.p_over_25) for r in completed if r.over_25_hit is not None]
    outs_over = [1.0 if r.over_25_hit else 0.0 for r in completed if r.over_25_hit is not None]
    params = _fit_platt_params(preds_over, outs_over)
    if params:
        maps["p_over"] = params

    preds_btts = [float(r.p_btts) for r in completed if r.btts_hit is not None]
    outs_btts = [1.0 if r.btts_hit else 0.0 for r in completed if r.btts_hit is not None]
    params = _fit_platt_params(preds_btts, outs_btts)
    if params:
        maps["p_btts"] = params

    return maps


# ---------------------------------------------------------------------------
# Calibrazione per mercato
# ---------------------------------------------------------------------------

def build_calibration_maps() -> dict[str, tuple[float, float]]:
    """
    Costruisce le mappe di calibrazione Platt per ogni mercato.

    Con abbastanza storico, fit su porzione temporale anteriore e valida sull'holdout
    recente (ordine timestamp); tiene solo mercati che migliorano log-loss sul test.

    Returns:
        Dict {mercato: (a, b)} per ogni mercato con dati sufficienti.
    """
    try:
        from src.config import PRECISION
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = sorted(
            (r for r in log.get_completed() if r.is_prematch and r.is_completed()),
            key=lambda r: r.timestamp,
        )
    except Exception:
        return {}

    if len(completed) < MIN_RECORDS_FOR_CALIBRATION:
        return {}

    n = len(completed)
    min_tot = int(PRECISION.CALIBRATION_TEMPORAL_MIN_TOTAL)
    min_test = int(PRECISION.CALIBRATION_TEMPORAL_MIN_TEST)
    train_frac = float(PRECISION.CALIBRATION_TEMPORAL_TRAIN_FRAC)

    if n < min_tot or n - int(n * train_frac) < min_test:
        return _maps_from_record_list(list(completed))

    split = max(MIN_RECORDS_FOR_CALIBRATION, int(n * train_frac))
    train = list(completed[:split])
    test = list(completed[split:])
    if len(test) < min_test:
        return _maps_from_record_list(list(completed))

    maps_train = _maps_from_record_list(train)
    out: dict[str, tuple[float, float]] = {}

    def _test_pairs_p1() -> list[tuple[float, float]]:
        return [
            (float(r.p1), 1.0 if r.risultato_1x2 == "1" else 0.0)
            for r in test
            if r.risultato_1x2 in ("1", "X", "2")
        ]

    def _test_pairs_px() -> list[tuple[float, float]]:
        return [
            (float(r.px), 1.0 if r.risultato_1x2 == "X" else 0.0)
            for r in test
            if r.risultato_1x2 in ("1", "X", "2")
        ]

    def _test_pairs_p2() -> list[tuple[float, float]]:
        return [
            (float(r.p2), 1.0 if r.risultato_1x2 == "2" else 0.0)
            for r in test
            if r.risultato_1x2 in ("1", "X", "2")
        ]

    def _test_pairs_over() -> list[tuple[float, float]]:
        return [
            (float(r.p_over_25), 1.0 if r.over_25_hit else 0.0)
            for r in test
            if r.over_25_hit is not None
        ]

    def _test_pairs_btts() -> list[tuple[float, float]]:
        return [
            (float(r.p_btts), 1.0 if r.btts_hit else 0.0)
            for r in test
            if r.btts_hit is not None
        ]

    checks = [
        ("p1", _test_pairs_p1),
        ("px", _test_pairs_px),
        ("p2", _test_pairs_p2),
        ("p_over", _test_pairs_over),
        ("p_btts", _test_pairs_btts),
    ]

    for key, pair_fn in checks:
        pr = maps_train.get(key)
        if not pr:
            continue
        a, b = pr
        pairs = [
            (p, o)
            for p, o in pair_fn()
            if 0.01 < p < 0.99
        ]
        if _platt_improves_on_holdout(pairs, a, b):
            out[key] = pr

    return out if out else _maps_from_record_list(list(completed))


def apply_calibration(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    cal_maps: dict[str, tuple[float, float]],
    *,
    strength: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Applica la calibrazione Platt alle probabilità.

    La correzione è limitata a ±MAX_PLATT_SHIFT per evitare
    overcorrection con pochi campioni.
    `strength` ∈ [0,1] scala l'entità dello shift (utile se altre calibrazioni hanno già agito).

    Returns:
        (p1, px, p2, p_over, p_under, p_btts) calibrate.
    """
    if not cal_maps:
        return p1, px, p2, p_over, p_under, p_btts

    st = max(0.0, min(1.0, float(strength)))

    def _safe_apply(p: float, key: str) -> float:
        if key not in cal_maps or p <= 0.001 or p >= 0.999:
            return p
        a, b = cal_maps[key]
        p_cal = _apply_platt(p, a, b)
        shift = (p_cal - p) * st
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
