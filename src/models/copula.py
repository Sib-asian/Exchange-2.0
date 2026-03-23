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
"""

from __future__ import annotations

import math

from src.config import COPULA, POISSON
from src.models.cmp import cmp_pmf


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
) -> dict[tuple[int, int], float]:
    """
    Costruisce la matrice bivariata usando CMP marginali + Frank copula.

    La probabilità congiunta P(X=i, Y=j) si ottiene dall'inclusione-esclusione:
        P(i,j) = C(F_X(i), F_Y(j)) - C(F_X(i-1), F_Y(j))
                 - C(F_X(i), F_Y(j-1)) + C(F_X(i-1), F_Y(j-1))

    dove F_X, F_Y sono le CDF marginali CMP.

    Args:
        mu_h: Lambda casa (gol rimanenti).
        mu_a: Lambda trasferta.
        theta: Parametro copula Frank.
        nu: Parametro dispersione CMP.

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

    total = sum(joint.values())
    if total > 0:
        joint = {k: v / total for k, v in joint.items()}

    return joint
