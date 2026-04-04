"""
Riduzione conservativa delle probabilità quando copertura estrazione o accordo
tra modelli è basso (evita overconfidence sui margini/kelly impliciti).
"""

from __future__ import annotations


def shrink_outcome_probs(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    *,
    extraction_coverage: float,
    model_agreement: float,
    max_mass_pull_1x2: float = 0.22,
    max_mass_pull_binary: float = 0.18,
    p_over_15: float | None = None,
    p_under_15: float | None = None,
) -> tuple[float, float, float, float, float, float, float | None, float | None]:
    """
    Tira leggermente verso uniforme (1X2) e verso 0.5 (O/U, BTTS).

    `extraction_coverage` e `model_agreement` attesi in [0, 1].
    """
    cov = max(0.0, min(1.0, float(extraction_coverage)))
    ag = max(0.0, min(1.0, float(model_agreement)))
    strength = 0.45 * cov + 0.55 * ag
    lam_1x2 = max(0.0, 1.0 - strength) * max_mass_pull_1x2
    lam_bi = max(0.0, 1.0 - strength) * max_mass_pull_binary

    u = 1.0 / 3.0
    q1 = (1.0 - lam_1x2) * p1 + lam_1x2 * u
    qx = (1.0 - lam_1x2) * px + lam_1x2 * u
    q2 = (1.0 - lam_1x2) * p2 + lam_1x2 * u
    s12 = q1 + qx + q2
    if s12 > 0:
        q1, qx, q2 = q1 / s12, qx / s12, q2 / s12

    po = (1.0 - lam_bi) * p_over + lam_bi * 0.5
    pu = (1.0 - lam_bi) * p_under + lam_bi * 0.5
    sou = po + pu
    if sou > 0:
        po, pu = po / sou, pu / sou
    else:
        po, pu = 0.5, 0.5

    pb = (1.0 - lam_bi) * p_btts + lam_bi * 0.5
    pb = max(0.0, min(1.0, pb))

    qo15: float | None = None
    qu15: float | None = None
    if p_over_15 is not None and p_under_15 is not None:
        qo15 = (1.0 - lam_bi) * p_over_15 + lam_bi * 0.5
        qu15 = (1.0 - lam_bi) * p_under_15 + lam_bi * 0.5
        sou2 = qo15 + qu15
        if sou2 > 0:
            qo15, qu15 = qo15 / sou2, qu15 / sou2
        else:
            qo15, qu15 = 0.5, 0.5

    return q1, qx, q2, po, pu, pb, qo15, qu15


__all__ = ["shrink_outcome_probs"]
