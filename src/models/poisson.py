"""
poisson.py — Distribuzioni di Poisson e matrice bivariata per il calcio.

Implementa:
  - PMF di Poisson normalizzata con gestione della coda
  - Correzione Dixon-Coles per punteggi bassi (tau-correction)
  - Correlazione dinamica rho per il modello bivariato
  - Costruzione della matrice bivariata completa con convoluzione Z

Riferimenti:
  Dixon & Coles (1997), "Modelling Association Football Scores"
  Karlis & Ntzoufras (2003), "Analysis of sports data by using bivariate Poisson models"
  Brechot & Flepp (2020), "Dealing With Randomness in Soccer"
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.config import DC, POISSON, RHO

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# PMF di Poisson
# ---------------------------------------------------------------------------

def poisson_pmf(mu: float, tail_mass: float = POISSON.TAIL_MASS) -> list[float]:
    """
    Calcola la PMF di Poisson con troncatura adattiva e normalizzazione.

    La coda viene troncata quando P(X > k) < tail_mass.
    max_k = max(PMF_MIN_K, mu + PMF_SIGMA * sqrt(mu) + PMF_EXTRA_BUFFER)
    garantisce la copertura per qualsiasi mu realistico:
      - PMF_MIN_K=20: minimo assoluto per mu bassi
      - PMF_SIGMA=6: 6σ coprono >99.9999998% della distribuzione
      - PMF_EXTRA_BUFFER=10: margine per overdispersion

    Args:
        mu: Tasso atteso (lambda). Se <= 0 restituisce [1.0].
        tail_mass: Soglia di coda sotto cui fermare l'accumulo.

    Returns:
        Lista normalizzata di probabilità [P(X=0), P(X=1), ...].
    """
    if mu <= 0:
        return [1.0]

    max_k = max(
        POISSON.PMF_MIN_K,
        int(mu + POISSON.PMF_SIGMA * math.sqrt(max(mu, 1.0)) + POISSON.PMF_EXTRA_BUFFER)
    )
    p0 = math.exp(-mu)
    pmf: list[float] = [p0]
    cumsum = p0
    p = p0

    for k in range(1, max_k + 1):
        p = p * mu / k
        pmf.append(p)
        cumsum += p
        if cumsum >= (1.0 - tail_mass):
            break

    # Normalizzazione: garantisce che la somma sia esattamente 1.0
    if cumsum > 0:
        return [x / cumsum for x in pmf]
    return pmf


# ---------------------------------------------------------------------------
# Correzione Dixon-Coles
# ---------------------------------------------------------------------------

def dixon_coles_tau(
    i: int,
    j: int,
    mu_h: float,
    mu_a: float,
    rho_dc: float = DC.RHO_DC,
) -> float:
    """
    Fattore correttivo Dixon-Coles per i 4 punteggi bassi (i+j <= 1).

    Poisson indipendente sovrastima P(0-0) e sottostima P(1-0)/P(0-1)/P(1-1).
    La correzione si applica SOLO a i,j in {0,1} con i+j <= 2:

        tau(0,0) = 1 - mu_h * mu_a * rho_dc
        tau(1,0) = 1 + mu_a * rho_dc
        tau(0,1) = 1 + mu_h * rho_dc
        tau(1,1) = 1 - rho_dc
        tau(i,j) = 1  per tutti gli altri (i+j > 2 o i>1 o j>1)

    rho_dc < 0 (tipicamente -0.13): le squadre tendono a NON segnare
    simultaneamente (effetto difensivo).

    Il clamp [TAU_MIN, TAU_MAX] previene l'azzeramento (rho troppo estremo)
    o l'amplificazione eccessiva.

    Args:
        i: Gol casa (punteggio considerato).
        j: Gol trasferta.
        mu_h: Lambda casa per la distribuzione indipendente.
        mu_a: Lambda trasferta.
        rho_dc: Coefficiente Dixon-Coles (negativo = correlazione negativa).

    Returns:
        Fattore tau in [TAU_MIN, TAU_MAX].
    """
    if i == 0 and j == 0:
        tau = 1.0 - mu_h * mu_a * rho_dc
    elif i == 1 and j == 0:
        tau = 1.0 + mu_a * rho_dc
    elif i == 0 and j == 1:
        tau = 1.0 + mu_h * rho_dc
    elif i == 1 and j == 1:
        tau = 1.0 - rho_dc
    else:
        return 1.0

    return max(DC.TAU_MIN, min(tau, DC.TAU_MAX))


# ---------------------------------------------------------------------------
# rho_DC dinamico
# ---------------------------------------------------------------------------

def rho_dc_dinamico(
    tot_cur: float,
    minuto: int,
    gol_totali: int = 0,
) -> float:
    """
    Coefficiente Dixon-Coles dinamico, contestualizzato alla partita.

    La correlazione negativa tra gol bassi dipende dal contesto:
    - Partite difensive (Total < 2.0): struttura difensiva → rho_DC più negativo
    - Partite aperte (Total > 3.0): difese aperte → rho_DC meno negativo
    - Late game con vantaggio: parking the bus → rho_DC più negativo
    - Alto punteggio: partita aperta → rho_DC meno negativo

    Args:
        tot_cur: Total corrente (gol rimanenti).
        minuto: Minuto attuale [0, 90].
        gol_totali: Gol già segnati.

    Returns:
        rho_DC in [RHO_DC_MIN, RHO_DC_MAX].
    """
    # Effetto total: basso total → struttura difensiva → più negativo
    tot_factor = max(0.0, DC.RHO_DC_TOT_REF - min(tot_cur, DC.RHO_DC_TOT_REF)) / DC.RHO_DC_TOT_REF
    # Effetto tempo: late game → più negativo
    time_factor = minuto / 90.0
    # Effetto gol: alto punteggio → partita aperta → meno negativo
    goal_factor = max(0.0, 1.0 - gol_totali * DC.RHO_DC_GOAL_DAMPEN)

    rho_dc = DC.RHO_DC_BASE + DC.RHO_DC_TOT_SCALE * tot_factor * goal_factor \
        + DC.RHO_DC_TIME_SCALE * time_factor
    return max(DC.RHO_DC_MIN, min(DC.RHO_DC_MAX, rho_dc))


# ---------------------------------------------------------------------------
# Correlazione dinamica rho
# ---------------------------------------------------------------------------

def rho_dinamico(
    tot_cur: float,
    minuto: int,
    shot_dom: float = 0.0,
    gol_totali: int = 0,
) -> float:
    """
    Coefficiente di correlazione bivariate Poisson, dinamico in base al contesto.

    Logica:
    - tot_cur alto → partita aperta → squadre segnano più indipendentemente → rho ↓
    - Avanzare del minuto → gol già osservati dominano → rho ↓
    - gol_totali alti → correlazione residua tra gol futuri minore → rho ↓
    - shot_dom alto → partita unidirezionale → quasi indipendenza → rho ↓

    Range risultante: ~0.02 (dominio totale, gara avanzata) → ~0.14 (equilibrio, inizio)

    Args:
        tot_cur: Linea Total corrente (gol rimanenti attesi dal mercato).
        minuto: Minuto attuale [0, 90].
        shot_dom: Indice dominio tiri [0, 1] = |sot_h - sot_a| / (sot_h + sot_a).
        gol_totali: Numero di gol già segnati in partita.

    Returns:
        Coefficiente rho in [RHO_MIN, ~0.14].
    """
    base = max(RHO.RHO_MIN, RHO.BASE_MAX - RHO.BASE_DECAY_RATE * min(tot_cur, RHO.BASE_TOT_CAP))
    frac_giocata = max(0.0, min(minuto / 90.0, 1.0))
    time_factor = 1.0 - RHO.TIME_DECAY_FACTOR * frac_giocata
    goal_decay = max(RHO.GOAL_DECAY_FLOOR, 1.0 - RHO.GOAL_DECAY_RATE * gol_totali)
    rho_base = base * time_factor * goal_decay

    # shot_dom × tempo: early game il dominio è instabile → effetto ridotto.
    # Late game il dominio è predittivo → effetto pieno.
    shot_dom_time = RHO.SHOT_DOM_TIME_FLOOR + (1.0 - RHO.SHOT_DOM_TIME_FLOOR) * frac_giocata
    # shot_dom × totale: in partite aperte (alto totale) il dominio tiri è meno
    # informativo (entrambe le squadre tirano molto → shot_dom meno discriminante).
    shot_dom_tot = 1.0 - min(RHO.SHOT_DOM_TOT_REDUCTION, tot_cur / RHO.SHOT_DOM_TOT_CAP * RHO.SHOT_DOM_TOT_REDUCTION)
    shot_dom_adj = shot_dom * shot_dom_time * shot_dom_tot
    shot_factor = 1.0 - RHO.SHOT_DOM_REDUCTION * shot_dom_adj

    return max(RHO.RHO_MIN, rho_base * shot_factor)


# ---------------------------------------------------------------------------
# Matrice bivariata
# ---------------------------------------------------------------------------

def build_bivariate_matrix(
    mu_h: float,
    mu_a: float,
    minuto: int,
    tot_cur: float,
    shot_dom: float = 0.0,
    gol_totali: int = 0,
    rho_dc_preset: float | None = None,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], float]:
    """
    Costruisce la matrice di probabilità bivariata completa.

    Modello bivariate Poisson (Karlis 2003):
        X_rem = X_ind + Z    (gol rimanenti casa)
        Y_rem = Y_ind + Z    (gol rimanenti trasferta)
        X_ind ~ Poisson(mu_h - lambda0)
        Y_ind ~ Poisson(mu_a - lambda0)
        Z     ~ Poisson(lambda0)

    dove lambda0 = rho * sqrt(mu_h * mu_a), cappato a 75% del min(mu_h, mu_a).

    La matrice indipendente viene corretta con Dixon-Coles prima della
    convoluzione con Z.

    Args:
        mu_h: Lambda atteso casa (gol rimanenti).
        mu_a: Lambda atteso trasferta.
        minuto: Minuto attuale [0, 90].
        tot_cur: Linea Total corrente (per il calcolo di rho).
        shot_dom: Indice dominio tiri [0, 1].
        gol_totali: Gol già segnati (per il calcolo di rho).
        rho_dc_preset: Se fornito, usa questo valore di rho_dc invece di calcolarlo.
            Fix #2.4: Permette di passare un valore precalcolato per coerenza tra modelli.

    Returns:
        Tupla (joint_ind, full, rho_used):
          - joint_ind: Matrice indipendente con DC, normalizzata.
          - full: Matrice bivariata completa (con Z), normalizzata.
          - rho_used: Valore di rho effettivamente usato.
    """
    mu_h = max(POISSON.EPS, float(mu_h))
    mu_a = max(POISSON.EPS, float(mu_a))

    rho = rho_dinamico(tot_cur, minuto, shot_dom, gol_totali)
    # Fix #2.4: Usa rho_dc precalcolato se fornito, altrimenti calcolalo
    rho_dc_val = rho_dc_preset if rho_dc_preset is not None else rho_dc_dinamico(tot_cur, minuto, gol_totali)

    # lambda0: correlazione tra gol delle due squadre (componente comune Z)
    # Media geometrica: sqrt(mu_h * mu_a) è più robusta di min() per partite sbilanciate
    # Cap: lambda0 <= LAMBDA0_CAP_RATIO * min(mu_h, mu_a) (Karlis & Ntzoufras 2003)
    # Ulteriore sicurezza: lambda0 non può mai superare min(mu_h, mu_a)
    geom_mu = math.sqrt(mu_h * mu_a)
    mu_min = min(mu_h, mu_a)
    lambda0 = min(rho * geom_mu, POISSON.LAMBDA0_CAP_RATIO * mu_min, mu_min)
    lambda0 = max(0.0, lambda0)

    mu_h_ind = max(POISSON.EPS, mu_h - lambda0)
    mu_a_ind = max(POISSON.EPS, mu_a - lambda0)

    pmf_h = poisson_pmf(mu_h_ind)
    pmf_a = poisson_pmf(mu_a_ind)
    pmf_z = poisson_pmf(lambda0)

    # Matrice indipendente con correzione Dixon-Coles
    joint_ind: dict[tuple[int, int], float] = {}
    dc_sum = 0.0

    for i, pi in enumerate(pmf_h):
        if pi < POISSON.PROB_SKIP_THRESHOLD:
            continue
        for j, pj in enumerate(pmf_a):
            if pj < POISSON.PROB_SKIP_THRESHOLD:
                continue
            tau = dixon_coles_tau(i, j, mu_h_ind, mu_a_ind, rho_dc=rho_dc_val)
            val = pi * pj * tau
            joint_ind[(i, j)] = val
            dc_sum += val

    if dc_sum > 0:
        joint_ind = {k: v / dc_sum for k, v in joint_ind.items()}

    # Matrice full: convoluzione con Z
    # P(X_rem=a, Y_rem=b) = sum_z P(X_ind=a-z, Y_ind=b-z) * P(Z=z)
    full: dict[tuple[int, int], float] = {}
    for (i, j), pij in joint_ind.items():
        for z, pz in enumerate(pmf_z):
            if pz < POISSON.PROB_SKIP_THRESHOLD:
                continue
            a, b = i + z, j + z
            full[(a, b)] = full.get((a, b), 0.0) + pij * pz

    fj_sum = sum(full.values())
    if fj_sum > 0:
        full = {k: v / fj_sum for k, v in full.items()}

    return joint_ind, full, rho
