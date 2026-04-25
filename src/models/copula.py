"""
copula.py — Matrice bivariata via Frank copula.

La copula Frank è una copula archimedea con parametro θ:
    C(u,v) = -1/θ × ln(1 + (e^(-θu) - 1)(e^(-θv) - 1) / (e^(-θ) - 1))

    θ > 0: dipendenza positiva (gol di una squadra → più probabili gol dell'altra)
    θ = 0: indipendenza
    θ < 0: dipendenza negativa

Vantaggi rispetto alla bivariate Poisson:
  - Struttura di dipendenza più flessibile
  - Può catturare dipendenza asimmetrica nelle code
  - Un singolo parametro θ codifica la correlazione

Usata come modello alternativo nel consensus multi-modello.

Correzione Dixon-Coles:
  Dopo l'inclusione-esclusione, si applica la tau-correction di Dixon-Coles
  ai punteggi bassi (0-0, 1-0, 0-1, 1-1) per coerenza con i modelli
  Poisson e Markov nel consensus blend. Senza questa correzione il modello
  Copula avrebbe P(0-0) sistematicamente diversa, creando bias nel consensus.
"""

from __future__ import annotations

import math

from src.config import COPULA, POISSON
from src.models.cmp import cmp_pmf
from src.models.poisson import dixon_coles_tau


def _frank_C(u: float, v: float, theta: float) -> float:
    """
    CDF della copula Frank.

    Usa expm1 per stabilità numerica con θ piccoli.
    Fix #1.11: Soglie numeriche dal config invece di hardcoded.
    """
    # Se |θ| ≈ 0, la copula degenera in indipendenza: C(u,v) = u*v
    if abs(theta) < COPULA.THETA_NEAR_ZERO:
        return u * v
    eu = math.expm1(-theta * u)
    ev = math.expm1(-theta * v)
    et = math.expm1(-theta)
    # Se |exp(-θ) - 1| ≈ 0, il denominatore è ~0 → indipendenza
    if abs(et) < COPULA.INNER_NEAR_ZERO:
        return u * v
    inner = 1.0 + eu * ev / et
    if inner <= 0:
        # Fallback per casi numerici estremi: Fréchet-Hoeffding bounds
        return max(0.0, min(u + v - 1.0, min(u, v)))
    return -math.log(inner) / theta


def build_copula_matrix(
    mu_h: float,
    mu_a: float,
    theta: float,
    nu: float = 0.92,
    rho_dc: float = -0.13,
) -> dict[tuple[int, int], float]:
    """
    Costruisce la matrice bivariata usando CMP marginali + Frank copula.

    La probabilità congiunta P(X=i, Y=j) si ottiene dall'inclusione-esclusione:
        P(i,j) = C(F_X(i), F_Y(j)) - C(F_X(i-1), F_Y(j))
                 - C(F_X(i), F_Y(j-1)) + C(F_X(i-1), F_Y(j-1))

    dove F_X, F_Y sono le CDF marginali CMP.

    Dopo il calcolo congiunto, si applica la correzione Dixon-Coles (tau)
    ai punteggi bassi (0-0, 1-0, 0-1, 1-1) per coerenza con i modelli
    Poisson e Markov nel consensus blend.

    Args:
        mu_h: Lambda casa (gol rimanenti).
        mu_a: Lambda trasferta.
        theta: Parametro copula Frank.
        nu: Parametro dispersione CMP.
        rho_dc: Coefficiente Dixon-Coles (negativo = correlazione negativa).
            Default -0.13. Passare 0.0 per disabilitare la correzione.

    Returns:
        Matrice bivariata normalizzata {(i,j): prob}.
    """
    mu_h = max(POISSON.EPS, float(mu_h))
    mu_a = max(POISSON.EPS, float(mu_a))

    pmf_h = cmp_pmf(mu_h, nu=nu)
    pmf_a = cmp_pmf(mu_a, nu=nu)

    # CDF marginali
    cdf_h = []
    cum = 0.0
    for p in pmf_h:
        cum += p
        cdf_h.append(cum)
    cdf_a = []
    cum = 0.0
    for p in pmf_a:
        cum += p
        cdf_a.append(cum)

    # Matrice congiunta via inclusione-esclusione
    joint: dict[tuple[int, int], float] = {}
    for i in range(len(pmf_h)):
        if pmf_h[i] < POISSON.PROB_SKIP_THRESHOLD:
            continue
        u1 = cdf_h[i]
        u0 = cdf_h[i - 1] if i > 0 else 0.0
        for j in range(len(pmf_a)):
            if pmf_a[j] < POISSON.PROB_SKIP_THRESHOLD:
                continue
            v1 = cdf_a[j]
            v0 = cdf_a[j - 1] if j > 0 else 0.0
            p = (
                _frank_C(u1, v1, theta)
                - _frank_C(u0, v1, theta)
                - _frank_C(u1, v0, theta)
                + _frank_C(u0, v0, theta)
            )
            if p > POISSON.PROB_SKIP_THRESHOLD:
                joint[(i, j)] = max(0.0, p)

    # Applica correzione Dixon-Coles per coerenza con Poisson/Markov.
    # Modifica P(0-0), P(1-0), P(0-1), P(1-1) per evitare che il Copula
    # abbia probabilità sistematicamente diverse nel consensus blend.
    if abs(rho_dc) > 1e-12:
        dc_joint: dict[tuple[int, int], float] = {}
        dc_sum = 0.0
        for (i, j), p in joint.items():
            tau = dixon_coles_tau(i, j, mu_h, mu_a, rho_dc=rho_dc)
            val = max(0.0, p * tau)
            dc_joint[(i, j)] = val
            dc_sum += val
        if dc_sum > 1e-18:
            joint = {k: v / dc_sum for k, v in dc_joint.items()}

    return joint
