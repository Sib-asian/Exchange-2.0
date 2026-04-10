"""
Pipeline analisi: motore + calibrazione prematch storica + shrink incertezza.

Isolata per test e per evitare duplicazione tra pagine Streamlit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.config import CONSENSUS, PRECISION
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

    # Upgrade 2: Calibrazione cross-validated (Platt scaling) dal prediction_log.
    # Corregge bias sistematici del modello usando la curva di calibrazione storica.
    if state.minuto == 0:
        try:
            from src.models.calibration_curve import apply_calibration, build_calibration_maps
            _cal_maps = build_calibration_maps()
            if _cal_maps:
                _cp1, _cpx, _cp2, _cpo, _cpu, _cpb = apply_calibration(
                    risultati.p1, risultati.px, risultati.p2,
                    risultati.p_over, risultati.p_under, risultati.p_btts,
                    _cal_maps,
                )
                risultati = replace(
                    risultati, p1=_cp1, px=_cpx, p2=_cp2,
                    p_over=_cpo, p_under=_cpu, p_btts=_cpb,
                )
        except Exception:
            pass  # Calibration is best-effort; don't break pipeline

    # Upgrade 6: Parametri appresi dallo storico (draw shrinkage).
    # I record nel log sono già post-engine (draw dinamico + isotonica); learn_draw_shrinkage
    # stima un fattore globale aggiuntivo sulle px salvate. La micro-correzione usa il
    # baseline CONSENSUS.DRAW_SHRINKAGE (fallback test) e una scala conservativa.
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
                        p1=_p1_adj / _s, px=_px_adj / _s, p2=_p2_adj / _s,
                    )
        except Exception:
            pass  # Parameter learning is best-effort

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

    _conf = float(risultati.model_confidence)
    _agree = float(risultati.model_agreement)
    _div = float(risultati.market_divergence)

    _quality_score = max(
        0.0,
        min(
            1.0,
            0.45 * _conf
            + 0.35 * _agree
            + 0.20 * max(0.0, 1.0 - _div),
        ),
    )

    # Penalità interazione: alta confidenza + basso accordo = i modelli sono
    # "sicuri" di cose diverse → scenario pericoloso che la somma pesata non
    # cattura. Senza questo termine, conf=0.8 + agree=0.3 dà quality=0.465
    # (vicino alla soglia), rischiando falsi positivi.
    if _conf > 0.70 and _agree < 0.50:
        _quality_score *= 0.75

    # Firewall graduato: hard block solo sotto QUALITY_HARD_BLOCK_MIN (0.20).
    # Tra HARD_BLOCK e QUALITY_SCORE_MIN: degrada model_confidence proporzionalmente
    # → soglie segnali più alte + Kelly fraction ridotta, ma non silenzio totale.
    # Sopra QUALITY_SCORE_MIN: nessuna penalità.
    if _quality_score < PRECISION.QUALITY_HARD_BLOCK_MIN:
        _signals_blocked = True
        _block_reasons.append(f"quality<{PRECISION.QUALITY_HARD_BLOCK_MIN:.2f}")
    elif _quality_score < PRECISION.QUALITY_SCORE_MIN:
        # Degradazione graduata: scala confidence linearmente
        # quality=0.20 → multiplier=0.0, quality=0.50 → multiplier=1.0
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

    # Applica confidence degradata al risultato se quality è intermedia.
    # La confidence ridotta fluisce nei segnali → Kelly fraction minore e soglie più alte.
    _effective_conf = min(_conf, float(risultati.model_confidence))

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
            signals_block_reason=", ".join([r for r in _block_reasons if "degraded" in r]) if _block_reasons else "",
        )

    return risultati, cal_sig


__all__ = ["run_analysis_pipeline"]
