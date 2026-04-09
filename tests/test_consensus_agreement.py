"""Accordo 1X2 tra modelli (pre-isotonica) per draw shrinkage."""

from src.models.consensus import agreement_1x2_from_per_raw


def test_agreement_1x2_identical_models_is_one():
    pr = {
        "bp": {"p1": 0.42, "px": 0.29, "p2": 0.29},
        "copula": {"p1": 0.42, "px": 0.29, "p2": 0.29},
        "markov": {"p1": 0.42, "px": 0.29, "p2": 0.29},
    }
    assert agreement_1x2_from_per_raw(pr) == 1.0


def test_agreement_1x2_divergent_models_below_one():
    pr = {
        "bp": {"p1": 0.55, "px": 0.25, "p2": 0.20},
        "copula": {"p1": 0.38, "px": 0.32, "p2": 0.30},
        "markov": {"p1": 0.22, "px": 0.28, "p2": 0.50},
    }
    ag = agreement_1x2_from_per_raw(pr)
    assert 0.0 <= ag < 0.92
