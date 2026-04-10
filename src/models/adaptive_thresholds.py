"""
adaptive_thresholds.py — Soglie segnali adattive alla ricchezza informativa.

Le soglie attuali crescono con √(minuto/90) per compensare l'incertezza
temporale. Ma non considerano la ricchezza informativa del momento:
al minuto 80' con 15 tiri e score chiaro, il modello è PIÙ calibrato
che al minuto 20' con 2 tiri — ma le soglie sono PIÙ alte.

Questo modulo calcola un "information richness index" che modula le soglie:
  - Più informazione → soglie più basse → più segnali quando sei più sicuro
  - Meno informazione → soglie più alte → filtro più stretto

L'indice combina:
  1. Shot saturation: quanti tiri rispetto al minuto? (>1 tiro/6min = saturo)
  2. Model confidence: quanto il modello si fida di sé?
  3. Market activity: il mercato si è mosso? (linee attive = più informazione)
"""

from __future__ import annotations

# Bonus massimo (riduzione soglia) per alta ricchezza informativa
MAX_RICHNESS_BONUS: float = 0.04  # -4% alla soglia

# Penalità massima per bassa ricchezza informativa
MAX_POVERTY_PENALTY: float = 0.03  # +3% alla soglia

# Soglia di saturazione tiri (sopra → massimo bonus)
SHOT_SATURATION_TARGET: float = 12.0


def compute_information_richness(
    minuto: int,
    n_shots_tot: int,
    model_confidence: float,
    momentum: float,
) -> float:
    """
    Calcola l'indice di ricchezza informativa [0, 1].

    Args:
        minuto: Minuto attuale.
        n_shots_tot: Tiri totali registrati.
        model_confidence: Confidenza del modello [0, 1].
        momentum: Momentum di mercato (attività delle linee).

    Returns:
        Indice in [0, 1]. 0 = informazione minima, 1 = massima.
    """
    # Shot density: tiri normalizzati per il tempo giocato
    if minuto <= 0:
        shot_score = 0.5  # Prematch: valore neutro
    else:
        expected_shots = minuto / 6.0  # ~1 tiro ogni 6 min per squadra
        shot_score = min(1.0, n_shots_tot / max(1.0, expected_shots * 2.0))

    # Confidence score (già calibrata)
    conf_score = max(0.0, min(1.0, model_confidence))

    # Market activity: più movimento = più informazione (fino a un punto)
    # momentum > 2 = molto attivo, momentum < 0.5 = dormiente
    market_score = min(1.0, abs(momentum) / 3.0) if minuto > 0 else 0.5

    # Media pesata: tiri pesano di più (dati reali > proxy)
    richness = 0.45 * shot_score + 0.35 * conf_score + 0.20 * market_score
    return max(0.0, min(1.0, richness))


def compute_threshold_adjustment(
    minuto: int,
    n_shots_tot: int,
    model_confidence: float,
    momentum: float,
) -> float:
    """
    Calcola l'aggiustamento della soglia basato sulla ricchezza informativa.

    Returns:
        Offset in punti percentuali da applicare alla soglia.
        Negativo = riduce soglia (più segnali), Positivo = alza soglia.
    """
    richness = compute_information_richness(
        minuto, n_shots_tot, model_confidence, momentum,
    )

    # Neutro a richness=0.50, bonus sotto 0.50, penalità sopra
    # Ma invertiamo: alta ricchezza = bonus (riduzione soglia)
    if richness >= 0.50:
        # Alta ricchezza → riduzione soglia (più segnali quando sicuri)
        bonus = (richness - 0.50) / 0.50 * MAX_RICHNESS_BONUS
        return -bonus
    else:
        # Bassa ricchezza → alza soglia (meno segnali quando incerti)
        penalty = (0.50 - richness) / 0.50 * MAX_POVERTY_PENALTY
        return penalty


__all__ = [
    "compute_information_richness",
    "compute_threshold_adjustment",
]
