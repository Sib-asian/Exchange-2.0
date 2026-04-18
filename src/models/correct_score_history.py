"""
correct_score_history.py — affina i top correct score usando lo storico chiuso.

Piccolo blend verso le frequenze empiriche per (gol_casa, gol_trasf) nella stessa
lega e fascia di tot_op di apertura, così i risultati esatti riflettono meglio
lo stile reale del campionamento quando il log ha abbastanza partite.
"""

from __future__ import annotations

import math

from src.config import PRECISION
from src.tracking.prediction_log import PredictionRecord, get_prediction_log


def _normalize_league(league: str) -> str:
    return (league or "").strip().lower()


def _recency_weight(idx: int, n: int, tau: float = 40.0) -> float:
    return math.exp(-(n - 1 - idx) / max(1e-9, tau))


def estimate_scoreline_empirical(
    league: str,
    tot_band: str,
    *,
    max_goals: int = 4,
    tau_matches: float = 40.0,
) -> tuple[dict[tuple[int, int], float], int, float]:
    """
    Distribuzione empirica sui final score (bucketati a max_goals per lato).

    Returns:
        (probs, n_weighted, raw_count) — probs sommano ~1 sul sottoinsieme bucketato;
        n_weighted è la somma dei pesi; raw_count partite usate.
    """
    target_league = _normalize_league(league)
    target_band = (tot_band or "").strip()
    if not target_league or not target_band:
        return {}, 0, 0

    try:
        records = get_prediction_log().get_completed()
    except Exception:
        return {}, 0, 0

    usable: list[PredictionRecord] = []
    for r in records:
        if not getattr(r, "is_prematch", False) or not r.is_completed():
            continue
        if _normalize_league(getattr(r, "lega", "")) != target_league:
            continue
        if str(getattr(r, "tot_band", "")).strip() != target_band:
            continue
        gh, ga = r.gol_casa, r.gol_trasf
        if gh is None or ga is None:
            continue
        usable.append(r)

    if len(usable) < int(PRECISION.CORRECT_SCORE_HISTORY_MIN_SAMPLES):
        return {}, 0, len(usable)

    n = len(usable)
    cells: dict[tuple[int, int], float] = {}
    total_w = 0.0
    for i, r in enumerate(usable):
        w = _recency_weight(i, n, tau=tau_matches)
        qq = str(getattr(r, "quote_quality", "") or "").strip().lower()
        if qq == "trusted":
            w *= 1.12
        elif qq == "untrusted":
            w *= 0.88
        gh = min(int(r.gol_casa or 0), max_goals)
        ga = min(int(r.gol_trasf or 0), max_goals)
        cells[(gh, ga)] = cells.get((gh, ga), 0.0) + w
        total_w += w

    if total_w <= 1e-12:
        return {}, 0, n

    # Dirichlet smoothing sul griglia (max_goals+1)^2
    dim = (max_goals + 1) ** 2
    alpha = float(PRECISION.CORRECT_SCORE_HISTORY_DIRICHLET_ALPHA)
    smooth_den = total_w + alpha * dim
    out: dict[tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            cnt = cells.get((h, a), 0.0)
            out[(h, a)] = (cnt + alpha) / smooth_den
    return out, total_w, n


def blend_top_cs_with_history(
    top_cs: list[tuple[tuple[int, int], float]],
    league: str,
    tot_band: str,
    *,
    extraction_trust: float = 1.0,
    model_agreement: float = 1.0,
) -> list[tuple[tuple[int, int], float]]:
    """
    Blend conservativo delle probabilità nei top score verso l'empirico storico.
    """
    if not top_cs or not league.strip() or not (tot_band or "").strip():
        return top_cs

    emp, _wsum, n_raw = estimate_scoreline_empirical(league, tot_band)
    if not emp or n_raw < int(PRECISION.CORRECT_SCORE_HISTORY_MIN_SAMPLES):
        return top_cs

    trust = max(0.0, min(1.0, float(extraction_trust)))
    agree = max(0.0, min(1.0, float(model_agreement)))
    alpha = float(PRECISION.CORRECT_SCORE_HISTORY_BLEND_MAX)
    alpha *= trust * (0.72 + 0.28 * agree)
    alpha *= min(1.0, n_raw / 72.0)

    new_list: list[tuple[tuple[int, int], float]] = []
    for sc, p in top_cs:
        pe = emp.get(sc, 0.0)
        new_list.append((sc, (1.0 - alpha) * p + alpha * pe))
    s = sum(pr for _, pr in new_list)
    if s <= 1e-15:
        return top_cs
    new_list = [(sc, pr / s) for sc, pr in new_list]
    new_list.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
    return new_list


__all__ = ["blend_top_cs_with_history", "estimate_scoreline_empirical"]
