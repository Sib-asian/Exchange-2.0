"""
correlation_ensemble.py — Ensemble correlation-aware weighting.

Il consensus standard usa media pesata lineare dei 3 modelli, trattandoli come
indipendenti. Ma condividono gli stessi input (xG, rho_dc, tot_cur) → quando
concordano, non è diversificazione ma ripetizione.

Questo modulo:
  1. Misura la distanza pairwise tra le previsioni 1X2 dei 3 modelli.
  2. Se 2 modelli sono troppo simili (|p1_A - p1_B| < soglia), riduce il loro
     peso combinato e redistribuisce al terzo (meno correlato).
  3. Il risultato è un ensemble che premia opinioni genuinamente indipendenti.

Riferimenti:
  Kuncheva & Whitaker (2003), "Measures of Diversity in Classifier Ensembles"
"""

from __future__ import annotations

# Soglia sotto la quale due modelli sono considerati "troppo concordanti"
CONCORDANCE_THRESHOLD: float = 0.025

# Fattore di dampening per modelli concordanti (20% riduzione peso)
CONCORDANCE_DAMPEN: float = 0.80

# Peso minimo per qualsiasi modello (non azzerare mai un modello)
MIN_MODEL_WEIGHT: float = 0.10


def adjust_weights_for_correlation(
    w_bp: float,
    w_cop: float,
    w_mk: float,
    probs_bp: dict[str, float],
    probs_copula: dict[str, float],
    probs_markov: dict[str, float],
) -> tuple[float, float, float]:
    """
    Aggiusta i pesi ensemble in base alla correlazione tra modelli.

    Se due modelli danno previsioni quasi identiche su p1/px/p2, il loro
    peso combinato viene ridotto e il surplus va al terzo modello.

    Args:
        w_bp, w_cop, w_mk: Pesi base dal phase-based scheduling.
        probs_bp, probs_copula, probs_markov: Probabilità per-model.

    Returns:
        (w_bp_adj, w_cop_adj, w_mk_adj) normalizzati a somma 1.
    """
    # Distanza pairwise media su p1, px, p2
    d_bp_cop = _model_distance(probs_bp, probs_copula)
    d_bp_mk = _model_distance(probs_bp, probs_markov)
    d_cop_mk = _model_distance(probs_copula, probs_markov)

    w_bp_adj = w_bp
    w_cop_adj = w_cop
    w_mk_adj = w_mk

    # Se BP e Copula sono troppo simili → dampening su entrambi, surplus a Markov
    if d_bp_cop < CONCORDANCE_THRESHOLD:
        w_bp_adj *= CONCORDANCE_DAMPEN
        w_cop_adj *= CONCORDANCE_DAMPEN
        surplus = w_bp * (1.0 - CONCORDANCE_DAMPEN) + w_cop * (1.0 - CONCORDANCE_DAMPEN)
        w_mk_adj += surplus

    # Se BP e Markov sono troppo simili → surplus a Copula
    if d_bp_mk < CONCORDANCE_THRESHOLD:
        w_bp_adj *= CONCORDANCE_DAMPEN
        w_mk_adj *= CONCORDANCE_DAMPEN
        surplus = w_bp * (1.0 - CONCORDANCE_DAMPEN) + w_mk * (1.0 - CONCORDANCE_DAMPEN)
        w_cop_adj += surplus

    # Se Copula e Markov sono troppo simili → surplus a BP
    if d_cop_mk < CONCORDANCE_THRESHOLD:
        w_cop_adj *= CONCORDANCE_DAMPEN
        w_mk_adj *= CONCORDANCE_DAMPEN
        surplus = w_cop * (1.0 - CONCORDANCE_DAMPEN) + w_mk * (1.0 - CONCORDANCE_DAMPEN)
        w_bp_adj += surplus

    # Floor minimo
    w_bp_adj = max(MIN_MODEL_WEIGHT, w_bp_adj)
    w_cop_adj = max(MIN_MODEL_WEIGHT, w_cop_adj)
    w_mk_adj = max(MIN_MODEL_WEIGHT, w_mk_adj)

    # Normalizza
    total = w_bp_adj + w_cop_adj + w_mk_adj
    if total > 0:
        return w_bp_adj / total, w_cop_adj / total, w_mk_adj / total
    return w_bp, w_cop, w_mk


def _model_distance(probs_a: dict[str, float], probs_b: dict[str, float]) -> float:
    """Distanza media assoluta su p1, px, p2."""
    keys = ("p1", "px", "p2")
    diffs = [abs(probs_a.get(k, 0.0) - probs_b.get(k, 0.0)) for k in keys]
    return sum(diffs) / len(diffs)


__all__ = ["adjust_weights_for_correlation"]
