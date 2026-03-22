"""
cmp.py — Conway-Maxwell-Poisson (CMP) per overdispersion nel calcio.

La Poisson standard assume var = mean. Nel calcio la varianza dei gol è
~15-20% superiore alla media (overdispersion, Karlis 2003).

La CMP generalizza la Poisson con un parametro ν:
    P(X=k) = μ^k / (k!)^ν / Z(μ,ν)

dove Z è la costante di normalizzazione.
    ν < 1 → overdispersion (calcio: ν ≈ 0.92)
    ν = 1 → Poisson standard
    ν > 1 → underdispersion

Questo migliora:
  - P(0-0): la Poisson standard la sottostima
  - Code alte (3+, 4+): la Poisson le sottostima
  - Over/Under su linee alte (3.5+)
  - Correct Score per punteggi estremi
"""

from __future__ import annotations

import math

from src.config import POISSON


def cmp_pmf(
    mu: float,
    nu: float = 0.92,
    tail_mass: float = POISSON.TAIL_MASS,
) -> list[float]:
    """
    PMF Conway-Maxwell-Poisson con troncatura adattiva e normalizzazione.

    Args:
        mu: Tasso atteso (lambda). Se <= 0 restituisce [1.0].
        nu: Parametro di dispersione. 0.92 = overdispersion tipica calcio.
        tail_mass: Soglia coda sotto cui troncare.

    Returns:
        Lista normalizzata di probabilità [P(X=0), P(X=1), ...].
    """
    if mu <= 0:
        return [1.0]

    log_mu = math.log(max(mu, 1e-300))
    max_k = max(20, int(mu + 6.0 * math.sqrt(max(mu, 1.0)) + 10))

    # Calcolo in log-space per stabilità numerica
    log_pmf: list[float] = []
    for k in range(max_k + 1):
        log_p = k * log_mu - nu * math.lgamma(k + 1)
        log_pmf.append(log_p)

    # Normalizzazione: sottrai il massimo prima di exp() per evitare overflow
    max_log = max(log_pmf)
    pmf = [math.exp(lp - max_log) for lp in log_pmf]

    # Troncatura adattiva
    total = sum(pmf)
    cumsum = 0.0
    for i, p in enumerate(pmf):
        cumsum += p
        if cumsum >= total * (1.0 - tail_mass):
            pmf = pmf[: i + 1]
            break

    # Normalizzazione finale
    total = sum(pmf)
    return [p / total for p in pmf] if total > 0 else [1.0]
