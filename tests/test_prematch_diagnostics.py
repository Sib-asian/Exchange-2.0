"""Test diagnostica prematch (coerenza linee, traccia pipeline, CI tightness)."""

from src.models.prematch_diagnostics import (
    ci_tightness_score,
    line_coherence_warnings,
)


def test_line_coherence_warnings_detect_ah_mismatch():
    w = line_coherence_warnings(
        ah_op=-0.5,
        tot_op=2.5,
        linea_ou=2.5,
        p1=0.25,
        p2=0.55,
        p_over=0.48,
    )
    assert len(w) >= 1


def test_ci_tightness_score_bounds():
    tight = ci_tightness_score(
        {"p1": (0.45, 0.55), "p_over": (0.45, 0.55), "p_btts": (0.45, 0.55)}
    )
    assert 0.0 <= tight <= 1.0
    wide = ci_tightness_score(
        {"p1": (0.2, 0.9), "p_over": (0.2, 0.9), "p_btts": (0.2, 0.9)}
    )
    assert wide < tight
