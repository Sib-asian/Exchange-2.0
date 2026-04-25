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

import hashlib
import logging
from typing import TYPE_CHECKING

_LOG = logging.getLogger("exchange.parameter_learning")

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

# Previous-scores alpha: grid e default motore
PREV_SCORES_ALPHA_DEFAULT: float = 0.15
PREV_SCORES_ALPHA_MIN: float = 0.05
PREV_SCORES_ALPHA_MAX: float = 0.30
PREV_SCORES_ALPHA_STEP: float = 0.025

# (len_usable, fingerprint record ids), alpha — fingerprint evita cache stale a parità di n.
_prev_alpha_cache: tuple[tuple[int, str], float | None] | None = None


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


def clear_prev_scores_alpha_cache() -> None:
    global _prev_alpha_cache
    _prev_alpha_cache = None


def _prev_alpha_usable_fingerprint(usable: list) -> tuple[int, str]:
    blob = "|".join(sorted(r.id for r in usable)).encode("utf-8", errors="replace")
    return len(usable), hashlib.sha256(blob).hexdigest()[:32]


def _p_over_indep_poisson(lh: float, la: float, line: float) -> float:
    """P(gol totali > linea) con due Poisson indipendenti (prematch, linea tipica 2.5)."""
    from src.models.poisson import poisson_pmf

    ph = poisson_pmf(max(1e-6, lh))
    pa = poisson_pmf(max(1e-6, la))
    p = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            if i + j > line:
                p += pi * pj
    return min(1.0, max(0.0, p))


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
    except Exception as _dle:
        _LOG.debug("parameter learning data load failed: %s", _dle)
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
    Grid search su alpha del blend previous-scores → xG prematch.

    Usa i record con snapshot pre-blend (xg_*_pre_prev, prev_lambda_*) salvati nel log
    e minimizza il Brier binario su P(Over linea) da Poisson indipendenti sui lambda blendati.

    Returns:
        Alpha in [PREV_SCORES_ALPHA_MIN, PREV_SCORES_ALPHA_MAX], o None se dati insufficienti.
    """
    global _prev_alpha_cache

    try:
        from src.tracking.prediction_log import get_prediction_log

        log = get_prediction_log()
        usable = [
            r
            for r in log.get_completed()
            if r.is_prematch
            and r.is_completed()
            and r.over_25_hit is not None
            and r.xg_h_pre_prev > 1e-6
            and r.xg_a_pre_prev > 1e-6
            and r.prev_lambda_h > 1e-6
            and r.prev_lambda_a > 1e-6
        ]
    except Exception as _ale:
        _LOG.debug("previous scores alpha data load failed: %s", _ale)
        return None

    if len(usable) < MIN_RECORDS_FOR_LEARNING:
        return None

    cache_key = _prev_alpha_usable_fingerprint(usable)
    if _prev_alpha_cache is not None and _prev_alpha_cache[0] == cache_key:
        return _prev_alpha_cache[1]

    best_a = PREV_SCORES_ALPHA_DEFAULT
    best_brier = float("inf")
    a = PREV_SCORES_ALPHA_MIN
    while a <= PREV_SCORES_ALPHA_MAX + 1e-9:
        tot = 0.0
        for r in usable:
            lh = (1.0 - a) * float(r.xg_h_pre_prev) + a * float(r.prev_lambda_h)
            la = (1.0 - a) * float(r.xg_a_pre_prev) + a * float(r.prev_lambda_a)
            line = float(r.ou_line)
            p_hat = _p_over_indep_poisson(lh, la, line)
            tot += _binary_brier(p_hat, bool(r.over_25_hit))
        avg = tot / len(usable)
        if avg < best_brier:
            best_brier = avg
            best_a = a
        a += PREV_SCORES_ALPHA_STEP

    _prev_alpha_cache = (cache_key, float(best_a))
    return best_a


def effective_prev_scores_alpha() -> float:
    """Alpha da usare in motore: appreso se possibile, altrimenti default."""
    v = learn_prev_scores_alpha()
    return float(v) if v is not None else PREV_SCORES_ALPHA_DEFAULT


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


__all__ = [
    "clear_prev_scores_alpha_cache",
    "effective_prev_scores_alpha",
    "learn_all_parameters",
    "learn_draw_shrinkage",
    "learn_prev_scores_alpha",
]
