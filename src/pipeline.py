"""
Pipeline analisi: motore + calibrazione prematch storica + shrink incertezza.

Isolata per test e per evitare duplicazione tra pagine Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from src.config import CONSENSUS, PRECISION
from src.engine import MatchState, ProbabilitaModello, analizza
from src.models.prematch_diagnostics import (
    PrematchPipelineTrace,
    ci_tightness_score,
    line_coherence_warnings,
)
from src.models.prematch_history_calibration import calibrate_prematch_probs
from src.models.uncertainty_shrink import shrink_outcome_probs

if TYPE_CHECKING:
    from src.models.prematch_history_calibration import CalibrationSignals

_LOG = logging.getLogger("exchange.pipeline")


def run_analysis_pipeline(
    state: MatchState,
    *,
    league: str = "",
    apply_prematch_calibration: bool = True,
    extraction_coverage: float = 1.0,
) -> tuple[ProbabilitaModello, CalibrationSignals | None, PrematchPipelineTrace | None]:
    """
    Esegue analizza → (opz.) calibrazione prematch per lega → Platt → draw learning → shrink.

    Returns:
        (risultati, segnali_calibrazione o None, traccia prematch o None se live)
    """
    risultati = analizza(state)
    cal_sig: CalibrationSignals | None = None
    trace: PrematchPipelineTrace | None = None
    plog: list[str] = []

    if state.minuto == 0:
        trace = PrematchPipelineTrace(
            p1_engine=risultati.p1,
            px_engine=risultati.px,
            p2_engine=risultati.p2,
            p_over_engine=risultati.p_over,
        )

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
        if trace is not None:
            trace.p_after_league_p1 = p1
            trace.league_cal_weight = float(cal_sig.weight) if cal_sig else 0.0

    platt_strength = 1.0
    if state.minuto == 0 and cal_sig is not None and cal_sig.weight > 0:
        platt_strength = max(
            float(PRECISION.PLATT_STRENGTH_FLOOR),
            1.0
            - float(PRECISION.PLATT_STRENGTH_DAMP_PER_LEAGUE_WEIGHT)
            * min(1.0, float(cal_sig.weight)),
        )

    if state.minuto == 0:
        try:
            from src.models.calibration_curve import apply_calibration, build_calibration_maps

            _cal_maps = build_calibration_maps()
            if _cal_maps:
                _cp1, _cpx, _cp2, _cpo, _cpu, _cpb = apply_calibration(
                    risultati.p1,
                    risultati.px,
                    risultati.p2,
                    risultati.p_over,
                    risultati.p_under,
                    risultati.p_btts,
                    _cal_maps,
                    strength=platt_strength,
                )
                risultati = replace(
                    risultati,
                    p1=_cp1,
                    px=_cpx,
                    p2=_cp2,
                    p_over=_cpo,
                    p_under=_cpu,
                    p_btts=_cpb,
                )
                if trace is not None:
                    trace.p_after_platt_p1 = _cp1
                    trace.platt_applied = True
                    trace.platt_strength = platt_strength
            elif trace is not None:
                plog.append("Platt: mappe vuote (storico insufficiente o validazione temporale).")
        except Exception as e:
            _LOG.warning("pipeline platt calibration skipped: %s", e, exc_info=True)
            plog.append(f"Platt: errore ({type(e).__name__}), ignorato.")

    if state.minuto == 0:
        try:
            from src.models.parameter_learning import learn_draw_shrinkage

            _learned_ds = learn_draw_shrinkage()
            _ds_base = float(CONSENSUS.DRAW_SHRINKAGE)
            if _learned_ds is not None and abs(_learned_ds - _ds_base) > 0.006:
                _ds_delta = (_learned_ds - _ds_base) * float(
                    PRECISION.PARAMETER_LEARNING_DRAW_MICRO_SCALE
                )
                _px_adj = risultati.px * (1.0 + _ds_delta)
                _surplus = risultati.px - _px_adj
                _p1p2 = risultati.p1 + risultati.p2
                if _p1p2 > 1e-9:
                    _p1_adj = risultati.p1 + _surplus * (risultati.p1 / _p1p2)
                    _p2_adj = risultati.p2 + _surplus * (risultati.p2 / _p1p2)
                else:
                    _p1_adj = risultati.p1 + _surplus * 0.5
                    _p2_adj = risultati.p2 + _surplus * 0.5
                _s = _p1_adj + _px_adj + _p2_adj
                if _s > 0:
                    risultati = replace(
                        risultati,
                        p1=_p1_adj / _s,
                        px=_px_adj / _s,
                        p2=_p2_adj / _s,
                    )
                    if trace is not None:
                        trace.p_after_drawlearn_p1 = _p1_adj / _s
                        trace.draw_learning_applied = True
        except Exception as e:
            _LOG.warning("pipeline draw learning skipped: %s", e, exc_info=True)
            plog.append(f"Draw learning: errore ({type(e).__name__}), ignorato.")

    # Upgrade 8-7: Ottimizzazione automatica iperparametri.
    # Se il prediction_log ha abbastanza dati, ottimizza i parametri chiave
    # (draw shrinkage, H2H alpha, logistic alpha) via grid search sul Brier score.
    if state.minuto == 0:
        try:
            from src.models.hyperparameter_tuning import tune_hyperparameters
            _tuned = tune_hyperparameters()
            if _tuned and "logistic_alpha_over" in _tuned:
                import math as _hp_math
                _la = _tuned["logistic_alpha_over"]
                if abs(_la - 1.03) > 0.005:
                    # Applica logistic sharpening ottimizzato su O/U
                    _p = max(1e-9, min(1.0 - 1e-9, risultati.p_over))
                    _logit = _hp_math.log(_p / (1.0 - _p))
                    _p_cal = 1.0 / (1.0 + _hp_math.exp(-(_la * _logit)))
                    _delta = _p_cal - risultati.p_over
                    # Cap the adjustment to ±5%
                    _delta = max(-0.05, min(0.05, _delta))
                    risultati = replace(
                        risultati,
                        p_over=risultati.p_over + _delta,
                        p_under=risultati.p_under - _delta,
                    )
        except Exception:
            pass  # Hyperparameter tuning is best-effort

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

    if trace is not None:
        trace.final_p1 = q1
        trace.final_px = qx
        trace.final_p2 = q2
        trace.final_p_over = qo
        trace.line_coherence_warnings = line_coherence_warnings(
            ah_op=float(state.ah_op),
            tot_op=float(state.tot_op),
            linea_ou=float(state.linea_ou),
            p1=q1,
            p2=q2,
            p_over=qo,
        )
        trace.pipeline_log = tuple(plog)

    _signals_blocked = False
    _block_reasons: list[str] = []

    _conf = float(risultati.model_confidence)
    _agree = float(risultati.model_agreement)
    _div = float(risultati.market_divergence)

    _ci_tight = ci_tightness_score(risultati.credible_intervals)

    _quality_score = max(
        0.0,
        min(
            1.0,
            0.45 * _conf
            + 0.35 * _agree
            + 0.20 * max(0.0, 1.0 - _div)
            + float(PRECISION.CI_QUALITY_FIREWALL_WEIGHT) * _ci_tight,
        ),
    )

    if _conf > 0.70 and _agree < 0.50:
        _quality_score *= 0.75

    if _quality_score < PRECISION.QUALITY_HARD_BLOCK_MIN:
        _signals_blocked = True
        _block_reasons.append(f"quality<{PRECISION.QUALITY_HARD_BLOCK_MIN:.2f}")
    elif _quality_score < PRECISION.QUALITY_SCORE_MIN:
        _q_range = PRECISION.QUALITY_SCORE_MIN - PRECISION.QUALITY_HARD_BLOCK_MIN
        _q_mult = (_quality_score - PRECISION.QUALITY_HARD_BLOCK_MIN) / max(_q_range, 1e-9)
        _conf *= _q_mult
        _block_reasons.append(f"quality_degraded({_quality_score:.2f})")

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

    _effective_conf = min(_conf, float(risultati.model_confidence))
    # Incertezza da CI: riduce confidence operativa in prematch (fluisce con Kelly via app).
    if state.minuto == 0:
        _effective_conf *= max(0.55, 0.62 + 0.38 * _ci_tight)

    if PRECISION.HARD_BLOCK_ON_FIREWALL and _signals_blocked:
        risultati = replace(
            risultati,
            quality_score=_quality_score,
            model_confidence=_effective_conf,
            signals_blocked=True,
            signals_block_reason=", ".join(_block_reasons),
        )
    else:
        risultati = replace(
            risultati,
            quality_score=_quality_score,
            model_confidence=_effective_conf,
            signals_blocked=False,
            signals_block_reason=", ".join([r for r in _block_reasons if "degraded" in r])
            if _block_reasons
            else "",
        )

    return risultati, cal_sig, trace


__all__ = ["run_analysis_pipeline"]
