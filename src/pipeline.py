"""
Pipeline analisi: motore + calibrazione prematch storica + shrink incertezza.

Isolata per test e per evitare duplicazione tra pagine Streamlit.
"""

from __future__ import annotations

from dataclasses import replace

from src.engine import MatchState, ProbabilitaModello, analizza
from src.models.prematch_history_calibration import calibrate_prematch_probs
from src.models.uncertainty_shrink import shrink_outcome_probs


def run_analysis_pipeline(
    state: MatchState,
    *,
    league: str = "",
    apply_prematch_calibration: bool = True,
    extraction_coverage: float = 1.0,
) -> tuple[ProbabilitaModello, str | None]:
    """
    Esegue analizza → (opz.) calibrazione prematch per lega → shrink probabilità.

    Returns:
        (risultati, calibration_signature o None)
    """
    risultati = analizza(state)
    cal_sig: str | None = None

    if apply_prematch_calibration and state.minuto == 0 and league.strip():
        p1, px, p2, p_over, p_under, p_btts, o15, u15, cal_sig = calibrate_prematch_probs(
            risultati.p1,
            risultati.px,
            risultati.p2,
            risultati.p_over,
            risultati.p_under,
            risultati.p_btts,
            league=league,
            p_over_15=risultati.p_over_15,
            p_under_15=risultati.p_under_15,
        )
        risultati = replace(
            risultati,
            p1=p1,
            px=px,
            p2=p2,
            p_over=p_over,
            p_under=p_under,
            p_btts=p_btts,
            p_over_15=o15 if o15 is not None else risultati.p_over_15,
            p_under_15=u15 if u15 is not None else risultati.p_under_15,
        )

    q1, qx, q2, qo, qu, qb, qo15, qu15 = shrink_outcome_probs(
        risultati.p1,
        risultati.px,
        risultati.p2,
        risultati.p_over,
        risultati.p_under,
        risultati.p_btts,
        extraction_coverage=extraction_coverage,
        model_agreement=risultati.model_agreement,
        p_over_15=risultati.p_over_15,
        p_under_15=risultati.p_under_15,
    )
    risultati = replace(
        risultati,
        p1=q1,
        px=qx,
        p2=q2,
        p_over=qo,
        p_under=qu,
        p_btts=qb,
        p_over_15=qo15 if qo15 is not None else risultati.p_over_15,
        p_under_15=qu15 if qu15 is not None else risultati.p_under_15,
    )

    return risultati, cal_sig


__all__ = ["run_analysis_pipeline"]
