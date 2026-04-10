"""
hyperparameter_tuning.py — Ottimizzazione automatica degli iperparametri.

Molti parametri in config.py sono scelti a mano (DRAW_SHRINKAGE_BASE, blend weights,
calibration thresholds). Questo modulo li ottimizza automaticamente usando i dati
storici del prediction_log.

Approccio:
  - Grid search su un sottoinsieme di parametri chiave.
  - Obiettivo: minimizzare il Brier score multiclass sulle previsioni completate.
  - Ogni parametro ha un range ragionevole definito.
  - I parametri vengono ottimizzati uno alla volta (coordinate descent)
    per semplicità e robustezza con pochi campioni.

Nota: non usiamo scipy/optuna per minimizzare le dipendenze.
Il coordinate descent con grid search è semplice e robusto.

Riferimenti:
  Brier (1950), "Verification of Forecasts Expressed in Terms of Probability"
"""

from __future__ import annotations

MIN_RECORDS_FOR_TUNING: int = 40


def _multiclass_brier(p1: float, px: float, p2: float, outcome: str) -> float:
    """Brier score per una singola previsione 1X2."""
    o1 = 1.0 if outcome == "1" else 0.0
    ox = 1.0 if outcome == "X" else 0.0
    o2 = 1.0 if outcome == "2" else 0.0
    return float((p1 - o1) ** 2 + (px - ox) ** 2 + (p2 - o2) ** 2)


def _ou_brier(p_over: float, over_hit: bool) -> float:
    """Brier score binario per Over/Under."""
    target = 1.0 if over_hit else 0.0
    return float((p_over - target) ** 2)


def tune_hyperparameters() -> dict[str, float] | None:
    """
    Esegue l'ottimizzazione degli iperparametri chiave.

    Returns:
        Dict con parametri ottimizzati, o None se non abbastanza dati.
        Chiavi: "draw_shrinkage", "h2h_1x2_alpha", "h2h_over_alpha",
                "prev_over_alpha", "logistic_alpha_over".
    """
    try:
        from src.tracking.prediction_log import get_prediction_log
    except ImportError:
        return None

    log = get_prediction_log()
    completed = [
        r for r in log.get_completed()
        if r.is_prematch and r.risultato_1x2 in ("1", "X", "2")
    ]

    if len(completed) < MIN_RECORDS_FOR_TUNING:
        return None

    results: dict[str, float] = {}

    # 1. Draw shrinkage: ottimizza il fattore di riduzione del pareggio
    results["draw_shrinkage"] = _optimize_draw_shrinkage(completed)

    # 2. H2H 1X2 blend alpha
    results["h2h_1x2_alpha"] = _optimize_h2h_alpha(completed)

    # 3. Logistic sharpening alpha
    results["logistic_alpha_over"] = _optimize_logistic_alpha(completed)

    return results


def _optimize_draw_shrinkage(records: list) -> float:
    """Grid search per draw shrinkage ottimale."""
    best_val = 0.97
    best_score = float("inf")

    for ds_int in range(92, 101):
        ds = ds_int / 100.0
        total_brier = 0.0
        for r in records:
            # Simula draw shrinkage
            px_adj = r.px * ds
            surplus = r.px * (1.0 - ds)
            p1p2 = r.p1 + r.p2
            if p1p2 > 1e-9:
                p1_adj = r.p1 + surplus * (r.p1 / p1p2)
                p2_adj = r.p2 + surplus * (r.p2 / p1p2)
            else:
                p1_adj = r.p1 + surplus * 0.5
                p2_adj = r.p2 + surplus * 0.5
            s = p1_adj + px_adj + p2_adj
            if s > 0:
                p1_adj /= s
                px_adj /= s
                p2_adj /= s
            total_brier += _multiclass_brier(p1_adj, px_adj, p2_adj, r.risultato_1x2)

        avg_brier = total_brier / len(records)
        if avg_brier < best_score:
            best_score = avg_brier
            best_val = ds

    return best_val


def _optimize_h2h_alpha(records: list) -> float:
    """Grid search per H2H 1X2 blend weight ottimale."""
    # Filtra solo record con dati H2H (p1 > 0 significa che il modello ha prodotto output)
    usable = [r for r in records if r.p1 > 0 and r.p2 > 0]
    if len(usable) < MIN_RECORDS_FOR_TUNING:
        return 0.05

    best_val = 0.05
    best_score = float("inf")

    for alpha_int in range(0, 16, 2):
        alpha = alpha_int / 100.0
        total_brier = 0.0
        for r in usable:
            total_brier += _multiclass_brier(r.p1, r.px, r.p2, r.risultato_1x2)
        avg = total_brier / len(usable)
        if avg < best_score:
            best_score = avg
            best_val = alpha

    return best_val


def _optimize_logistic_alpha(records: list) -> float:
    """Grid search per logistic sharpening alpha."""
    import math

    usable = [r for r in records if r.p_over_25 > 0.01 and r.over_25_hit is not None]
    if len(usable) < MIN_RECORDS_FOR_TUNING:
        return 1.03

    best_val = 1.03
    best_score = float("inf")

    for a_int in range(97, 112, 2):
        alpha = a_int / 100.0
        total_brier = 0.0
        for r in usable:
            p = max(1e-9, min(1.0 - 1e-9, r.p_over_25))
            logit = math.log(p / (1.0 - p))
            p_cal = 1.0 / (1.0 + math.exp(-(alpha * logit)))
            total_brier += _ou_brier(p_cal, r.over_25_hit)
        avg = total_brier / len(usable)
        if avg < best_score:
            best_score = avg
            best_val = alpha

    return best_val


__all__ = ["tune_hyperparameters"]
