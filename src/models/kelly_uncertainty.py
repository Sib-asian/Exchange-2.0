"""
kelly_uncertainty.py — Kelly discount basato sull'incertezza dell'edge.

Il Kelly standard assume che l'edge sia noto con certezza, ma è stimato
dal modello con un intervallo di confidenza. Un edge di 4% ± 0.5% (stretto)
merita uno stake diverso da 4% ± 3% (largo).

Questo modulo usa gli intervalli credibili del consensus per stimare
l'incertezza sull'edge e applicare un discount proporzionale.

Riferimenti:
  Thorp (2006), "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
  Baker & McHale (2013), "Optimal Betting Under Parameter Uncertainty"
"""

from __future__ import annotations

# Discount massimo per incertezza (non ridurre mai più del 50%)
MAX_UNCERTAINTY_DISCOUNT: float = 0.50

# Larghezza CI baseline (sotto questa → nessun discount)
CI_BASELINE_WIDTH: float = 0.03


def compute_edge_uncertainty_discount(
    edge: float,
    ci_width: float,
) -> float:
    """
    Calcola il fattore di discount per lo stake Kelly basato sull'incertezza.

    Args:
        edge: Edge stimato dal modello (es. 0.04 = 4%).
        ci_width: Larghezza dell'intervallo credibile per la probabilità
                  (es. ci_high - ci_low per il mercato in questione).

    Returns:
        Fattore moltiplicativo in [1 - MAX_UNCERTAINTY_DISCOUNT, 1.0].
        1.0 = nessun discount (CI stretto), 0.5 = max discount (CI largo).
    """
    if edge <= 0 or ci_width <= 0:
        return 1.0

    # Se il CI è stretto (< baseline), nessun discount
    excess_width = max(0.0, ci_width - CI_BASELINE_WIDTH)
    if excess_width <= 0:
        return 1.0

    # Rapporto incertezza/edge: quanto è largo il CI rispetto all'edge?
    # Se ci_width = 2 * edge, l'edge potrebbe facilmente essere zero.
    uncertainty_ratio = excess_width / max(0.001, abs(edge))

    # Discount sigmoide: cresce dolcemente con l'incertezza
    # ratio=0 → discount=0, ratio=1 → discount~0.25, ratio=2 → discount~0.40
    discount = MAX_UNCERTAINTY_DISCOUNT * (1.0 - 1.0 / (1.0 + uncertainty_ratio))

    return max(1.0 - MAX_UNCERTAINTY_DISCOUNT, 1.0 - discount)


def kelly_with_uncertainty(
    kelly_stake: float,
    edge: float,
    credible_intervals: dict[str, tuple[float, float]],
    market_key: str,
) -> float:
    """
    Applica il discount di incertezza allo stake Kelly.

    Args:
        kelly_stake: Stake calcolato dal Kelly standard.
        edge: Edge netto per questo mercato.
        credible_intervals: Dict {mercato: (ci_low, ci_high)}.
        market_key: Chiave del mercato (es. "p1", "p_over").

    Returns:
        Stake aggiustato (ridotto se incertezza alta, invariato se bassa).
    """
    ci = credible_intervals.get(market_key)
    if ci is None:
        return kelly_stake

    ci_width = ci[1] - ci[0]
    discount = compute_edge_uncertainty_discount(edge, ci_width)
    return kelly_stake * discount


__all__ = ["compute_edge_uncertainty_discount", "kelly_with_uncertainty"]
