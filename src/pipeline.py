"""
Pipeline analisi: motore + calibrazione prematch storica + shrink incertezza.

Isolata per test e per evitare duplicazione tra pagine Streamlit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.config import PRECISION
from src.engine import MatchState, ProbabilitaModello, analizza
from src.models.prematch_history_calibration import calibrate_prematch_probs
from src.models.uncertainty_shrink import shrink_outcome_probs

if TYPE_CHECKING:
    from src.models.prematch_history_calibration import CalibrationSignals


def run_analysis_pipeline(
    state: MatchState,
    *,
    league: str = "",
    apply_prematch_calibration: bool = True,
    extraction_coverage: float = 1.0,
) -> tuple[ProbabilitaModello, CalibrationSignals | None]:
    """
    Esegue analizza → (opz.) calibrazione prematch per lega → shrink probabilità.

    Returns:
        (risultati, segnali_calibrazione o None)
    """
    risultati = analizza(state)
    cal_sig: CalibrationSignals | None = None

    if apply_prematch_calibration and state.minuto == 0 and league.strip():
        p1, px, p2, p_over, p_under, p_btts, o15, u15, cal_sig = calibrate_prematch_probs(
            risultati.p1,
            risultati.px,
            risultati.p2,
            risultati.p_over,
            risultati.p_under,
            risultati.p_btts,
            league=league,
            tot_band=f"{float(state.linea_ou):g}",
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

    # Fase 4/5: No-Bet + Data Quality Firewall.
    # Blocca i segnali operativi in condizioni di bassa affidabilità strutturale.
    _signals_blocked = False
    _block_reasons: list[str] = []

    _quality_score = max(
        0.0,
        min(
            1.0,
            0.45 * float(risultati.model_confidence)
            + 0.35 * float(risultati.model_agreement)
            + 0.20 * float(max(0.0, 1.0 - risultati.market_divergence)),
        ),
    )

    if _quality_score < PRECISION.QUALITY_SCORE_MIN:
        _signals_blocked = True
        _block_reasons.append(f"quality<{PRECISION.QUALITY_SCORE_MIN:.2f}")
    if float(risultati.model_confidence) < PRECISION.MODEL_CONFIDENCE_MIN:
        _signals_blocked = True
        _block_reasons.append(f"conf<{PRECISION.MODEL_CONFIDENCE_MIN:.2f}")
    if float(risultati.model_agreement) < PRECISION.MODEL_AGREEMENT_MIN:
        _signals_blocked = True
        _block_reasons.append(f"agreement<{PRECISION.MODEL_AGREEMENT_MIN:.2f}")
    if float(risultati.market_divergence) > PRECISION.MARKET_DIVERGENCE_MAX:
        _signals_blocked = True
        _block_reasons.append(f"divergence>{PRECISION.MARKET_DIVERGENCE_MAX:.2f}")
    if state.minuto == 0 and float(extraction_coverage) < PRECISION.PREMATCH_COVERAGE_MIN:
        _signals_blocked = True
        _block_reasons.append(f"coverage<{PRECISION.PREMATCH_COVERAGE_MIN:.2f}")
    if state.minuto >= PRECISION.STALE_LINE_MINUTE_BLOCK and bool(risultati.stale_line):
        _signals_blocked = True
        _block_reasons.append("stale_line")

    if PRECISION.HARD_BLOCK_ON_FIREWALL and _signals_blocked:
        risultati = replace(
            risultati,
            quality_score=_quality_score,
            signals_blocked=True,
            signals_block_reason=", ".join(_block_reasons),
        )
    else:
        risultati = replace(
            risultati,
            quality_score=_quality_score,
            signals_blocked=False,
            signals_block_reason="",
        )

    return risultati, cal_sig


__all__ = ["run_analysis_pipeline"]
