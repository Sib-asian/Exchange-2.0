"""Regressione modifiche motore fase 2.0 (Markov DC, CS overdisp, sharpening)."""

from __future__ import annotations

import math

from src.config import CONSENSUS, UI
from src.models.consensus import _logistic_sharpen, logistic_sharpen_over
from src.models.markov import markov_score_distribution


def test_markov_negative_rho_still_boosts_zero_zero():
    base = markov_score_distribution(1.2, 1.0, 0, 0, 0, rho_dc=0.0)
    adj = markov_score_distribution(1.2, 1.0, 0, 0, 0, rho_dc=-0.13)
    assert adj.get((0, 0), 0.0) > base.get((0, 0), 0.0)


def test_logistic_sharpen_over_high_tail_stronger_than_mid():
    p = 0.88
    a_mid = CONSENSUS.LOGISTIC_ALPHA_OVER
    mid_only = _logistic_sharpen(p, alpha=a_mid)
    asym = logistic_sharpen_over(p)
    assert asym > mid_only or math.isclose(asym, mid_only, rel_tol=0, abs_tol=1e-6)


def test_cs_overdisp_mult_monotone_in_future_goals():
    """Stesso moltiplicatore implicito cresce con future_goals (k>=3)."""
    k0, alpha, exp, mx = UI.CS_OVERDISP_K0, UI.CS_OVERDISP_ALPHA, UI.CS_OVERDISP_EXP, UI.CS_OVERDISP_MAX

    def mult(k: int) -> float:
        if k < 3:
            return 1.0
        x = max(0.0, float(k) - k0)
        return min(mx, 1.0 + alpha * (x**exp))

    assert mult(4) >= mult(3)
    assert mult(6) >= mult(4)
