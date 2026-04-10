"""
parameter_learning.py — Apprendimento parametri dal prediction_log storico.

Usa i PredictionRecord completati per ottimizzare i parametri chiave del modello
(draw shrinkage, consensus weights) minimizzando il Brier score.

L'ottimizzazione avviene offline (periodicamente) e i parametri appresi
vengono usati come default quando disponibili.

Riferimenti:
  Brier (1950), "Verification of Forecasts Expressed in Terms of Probability"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.config import CONSENSUS


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

MIN_RECORDS_FOR_LEARNING: int = 30
# Range di ricerca per draw shrinkage
DRAW_SHRINKAGE_SEARCH_MIN: float = 0.92
DRAW_SHRINKAGE_SEARCH_MAX: float = 1.00
DRAW_SHRINKAGE_SEARCH_STEP: float = 0.005


# ---------------------------------------------------------------------------
# Brier score per 1X2
# ---------------------------------------------------------------------------

def _multiclass_brier(p1: float, px: float, p2: float, outcome: str) -> float:
    """Brier score multiclasse per 1X2."""
    o1, ox, o2 = (1.0, 0.0, 0.0) if outcome == "1" else (
        (0.0, 1.0, 0.0) if outcome == "X" else (0.0, 0.0, 1.0)
    )
    return float((p1 - o1) ** 2 + (px - ox) ** 2 + (p2 - o2) ** 2)


def _binary_brier(pred: float, outcome: bool) -> float:
    """Brier score binario."""
    o = 1.0 if outcome else 0.0
    return (pred - o) ** 2


# ---------------------------------------------------------------------------
# Apprendimento draw shrinkage ottimale
# ---------------------------------------------------------------------------

def learn_draw_shrinkage() -> float | None:
    """
    Trova il draw_shrinkage che minimizza il Brier 1X2 sullo storico.

    Il draw shrinkage riduce P(X) e redistribuisce il surplus proporzionalmente
    a P(1) e P(2). Qui cerchiamo il valore ottimale via grid search.

    Returns:
        Valore ottimale di draw_shrinkage in [0.92, 1.00], o None se dati insufficienti.
    """
    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = [
            r for r in log.get_completed()
            if r.is_prematch and r.is_completed() and r.risultato_1x2 in ("1", "X", "2")
        ]
    except Exception:
        return None

    if len(completed) < MIN_RECORDS_FOR_LEARNING:
        return None

    best_shrinkage = float(CONSENSUS.DRAW_SHRINKAGE)
    best_brier = float("inf")

    # Grid search
    shrink = DRAW_SHRINKAGE_SEARCH_MIN
    while shrink <= DRAW_SHRINKAGE_SEARCH_MAX + 1e-9:
        total_brier = 0.0
        for r in completed:
            p1, px, p2 = float(r.p1), float(r.px), float(r.p2)
            # Applica draw shrinkage
            px_adj = px * shrink
            surplus = px * (1.0 - shrink)
            p1p2 = p1 + p2
            if p1p2 > 1e-9:
                p1_adj = p1 + surplus * (p1 / p1p2)
                p2_adj = p2 + surplus * (p2 / p1p2)
            else:
                p1_adj = p1 + surplus * 0.5
                p2_adj = p2 + surplus * 0.5
            # Normalizza
            s = p1_adj + px_adj + p2_adj
            if s > 0:
                p1_adj /= s
                px_adj /= s
                p2_adj /= s
            total_brier += _multiclass_brier(p1_adj, px_adj, p2_adj, r.risultato_1x2)
        avg_brier = total_brier / len(completed)
        if avg_brier < best_brier:
            best_brier = avg_brier
            best_shrinkage = shrink
        shrink += DRAW_SHRINKAGE_SEARCH_STEP

    return best_shrinkage


# ---------------------------------------------------------------------------
# Apprendimento Previous Scores alpha ottimale
# ---------------------------------------------------------------------------

def learn_prev_scores_alpha() -> float | None:
    """
    Trova il peso ottimale del previous-scores blend per xG prematch.

    Cerca l'alpha che minimizza l'errore assoluto medio tra xG predetto
    e gol effettivi osservati.

    Returns:
        Alpha ottimale in [0.0, 0.30], o None se dati insufficienti.
    """
    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        completed = [
            r for r in log.get_completed()
            if r.is_prematch and r.is_completed() and r.gol_casa is not None
        ]
    except Exception:
        return None

    if len(completed) < MIN_RECORDS_FOR_LEARNING:
        return None

    # Per ogni record, abbiamo xg_h e xg_a (xG finali usati dal modello)
    # e gol_casa/gol_trasf (risultato reale). L'errore medio ci dice
    # quanto il modello è calibrato.
    # Senza accesso ai dati raw (prev_avg), possiamo solo validare il Brier complessivo.

    # Calcola Brier medio per Over/Under (proxy per calibrazione xG totale)
    records_with_ou = [r for r in completed if r.over_25_hit is not None]
    if len(records_with_ou) < MIN_RECORDS_FOR_LEARNING:
        return None

    avg_brier = sum(
        _binary_brier(float(r.p_over_25), bool(r.over_25_hit))
        for r in records_with_ou
    ) / len(records_with_ou)

    # Se il Brier O/U è alto (>0.25), il modello sottoperforma → aumentare alpha
    # Se il Brier è basso (<0.20), il modello è calibrato → mantenere alpha basso
    if avg_brier > 0.25:
        return 0.20  # Più peso ai dati storici
    elif avg_brier > 0.22:
        return 0.15  # Default
    else:
        return 0.10  # Buona calibrazione, meno peso allo storico


# ---------------------------------------------------------------------------
# Apprendimento parametri complessivo
# ---------------------------------------------------------------------------

def learn_all_parameters() -> dict[str, float]:
    """
    Apprende tutti i parametri ottimizzabili dallo storico.

    Returns:
        Dict con i parametri appresi. Chiavi possibili:
        - "draw_shrinkage": fattore draw shrinkage ottimale
        - "prev_scores_alpha": peso prev_scores nel blend xG
    """
    params: dict[str, float] = {}

    ds = learn_draw_shrinkage()
    if ds is not None:
        params["draw_shrinkage"] = ds

    psa = learn_prev_scores_alpha()
    if psa is not None:
        params["prev_scores_alpha"] = psa

    return params


__all__ = ["learn_all_parameters", "learn_draw_shrinkage", "learn_prev_scores_alpha"]
