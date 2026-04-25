"""
markov.py — Score-state Markov chain per distribuzione gol rimanenti.

Modella le transizioni minuto-per-minuto tra stati di punteggio:
    P(t+1 = (i+1,j) | t = (i,j)) = λ_h(score_state) × dt
    P(t+1 = (i,j+1) | t = (i,j)) = λ_a(score_state) × dt

La catena cattura l'effetto cascata: un gol cambia il punteggio → le rates
cambiano → la probabilità dei gol successivi cambia. Questo è diverso dal
modello Poisson che applica un singolo aggiustamento statico.

Vantaggi:
  - Score effect evolve naturalmente (non serve residuo hardcoded)
  - Cattura clustering dei gol (dopo un gol è più probabile un altro)
  - BTTS e O/U diventano derivate naturali della catena
"""

from __future__ import annotations

from src.config import DECAY
from typing import Dict, Tuple

# Dixon–Coles tau applicato una solta sulla distribuzione finale (non per minuto):
# evita la compounding ~90× sugli stati bassi (0,0)/(1,0)/(0,1).
from src.models.poisson import dixon_coles_tau


# Moltiplicatore per l'effetto pressing (coerente con time_decay.py)
# La squadra in svantaggio preme con più volume ma minore qualità
SCORE_PRESS_MULTIPLIER = DECAY.SCORE_DOWN_MULTIPLIER  # 0.65


# ======================================================================
# Time-varying goal rate multiplier (empirical goal distribution)
# ======================================================================
#
# Empirical data from top-5 European leagues (Premier League, La Liga,
# Bundesliga, Serie A, Ligue 1) over multiple seasons shows that goals
# are NOT uniformly distributed across 90 minutes:
#
#   - 0-15 min:  ~8%  of goals (settling period, teams cautious)
#   - 16-30 min: ~13%  of goals (game develops,节奏 normalises)
#   - 31-45 min: ~14%  of goals (approaching halftime, urgency rises)
#   - 46-60 min: ~13%  of goals (post-halftime tactical reset)
#   - 61-75 min: ~14%  of goals (substitutions, tactical shifts)
#   - 76-90 min: ~20%  of goals (desperation, fatigue, added time)
#
# 1st half (0-45'):   ~45% of goals
# 2nd half (46-90'):  ~55% of goals
# Last 15 min spike:  ~20% of all goals (highest concentration)
#
# Per-minute rate relative to uniform (each 15-min bucket has 100%/6 ≈ 16.7%):
#   -  0-15':  8/16.7 ≈ 0.48  but we use 0.85 (less extreme, avoids
#               distortion; the full Markov propagation amplifies differences)
#   - 16-30': 13/16.7 ≈ 0.78 → 0.95
#   - 31-45': 14/16.7 ≈ 0.84 → 1.05
#   - 46-60': 13/16.7 ≈ 0.78 → 1.00
#   - 61-75': 14/16.7 ≈ 0.84 → 1.05
#   - 76-90': 20/16.7 ≈ 1.20 → 1.25
#
# Values are deliberately conservative (less extreme than raw ratios) to
# avoid overfitting to league averages when applied to individual matches.
# The overall integral still sums to ~1.0× over 90 minutes, preserving
# the total expected goals while redistributing across time.
#
# Sources: Opta/StatsBomb internal data, plus public analyses by
#   - Michael Caley (2014-2020)
#   - Mark Taylor (ThePowerRank)
#   - Understat / FBref aggregated league data
# ======================================================================

# Knots for linear interpolation: (minute, multiplier)
# Chosen at 15-min intervals with values reflecting empirical rates.
# Linear interpolation between knots avoids discontinuities at boundaries.
_GOAL_RATE_KNOTS: list[Tuple[int, float]] = [
    (0,  0.85),   # Settling period — teams cautious, fewer chances
    (15, 0.85),   # End of settling block
    (16, 0.95),   # Game develops — transition point
    (30, 0.95),   # End of development block
    (31, 1.05),   # Approaching halftime pressure
    (45, 1.05),   # End of first half
    (46, 1.00),   # Post-halftime reset
    (60, 1.00),   # Normal second-half rhythm
    (61, 1.05),   # Tactical substitutions begin
    (75, 1.05),   # End of sub-block
    (76, 1.25),   # Desperation / fatigue / late pressure spike
    (90, 1.25),   # End of match (including added time)
]


def _goal_rate_multiplier(minute: int) -> float:
    """
    Return a per-minute goal rate multiplier based on empirical goal distribution.

    Uses linear interpolation between 15-minute interval knots. The multiplier
    adjusts the flat base rate to reflect real-world goal timing patterns
    observed in top-5 European leagues.

    Args:
        minute: Absolute match minute [0, 90]. Values outside [0, 90] are
                clamped to the nearest boundary.

    Returns:
        Multiplier in range ~[0.85, 1.25]. Multiply base per-minute rate
        by this value to get the time-adjusted rate.

    Example:
        >>> _goal_rate_multiplier(5)   # settling period
        0.85
        >>> _goal_rate_multiplier(80)  # late pressure
        1.25
        >>> _goal_rate_multiplier(45)  # halftime boundary
        1.05
    """
    if minute <= 0:
        return _GOAL_RATE_KNOTS[0][1]
    if minute >= 90:
        return _GOAL_RATE_KNOTS[-1][1]

    # Find the two bracketing knots and linearly interpolate
    for k in range(len(_GOAL_RATE_KNOTS) - 1):
        m0, v0 = _GOAL_RATE_KNOTS[k]
        m1, v1 = _GOAL_RATE_KNOTS[k + 1]
        if m0 <= minute <= m1:
            if m1 == m0:  # avoid division by zero (shouldn't happen)
                return v0
            frac = (minute - m0) / (m1 - m0)
            return v0 + frac * (v1 - v0)

    # Fallback (should never reach here)
    return 1.0


def markov_score_distribution(
    mu_h: float,
    mu_a: float,
    minuto: int,
    gol_h: int,
    gol_a: int,
    score_effect_rate: float = 0.03,
    max_goals: int = 8,
    rho_dc: float = -0.13,
) -> dict[tuple[int, int], float]:
    """
    Calcola la distribuzione dei gol rimanenti via Markov chain.

    Propaga minuto per minuto dalla situazione attuale, con rates che
    dipendono dallo stato del punteggio (score-dependent).

    Correlazione Dixon–Coles: dopo la propagazione, si applica la tau-correction
    standard (come in poisson.dixon_coles_tau) una sola volta sulla PMF finale.
    Non più a ogni minuto sugli stati bassi — evitava compounding eccessivo su P(0,0).

    Args:
        mu_h: Lambda casa (gol rimanenti attesi).
        mu_a: Lambda trasferta (gol rimanenti attesi).
        minuto: Minuto attuale [0, 90].
        gol_h: Gol casa attuali.
        gol_a: Gol trasferta attuali.
        score_effect_rate: Intensità del score effect per gol di vantaggio.
        max_goals: Massimo gol rimanenti per squadra (troncatura).
        rho_dc: Coefficiente Dixon-Coles (negativo = correlazione negativa).

    Returns:
        Distribuzione {(rem_h, rem_a): prob} dei gol rimanenti.
    """
    minutes_remaining = max(1, 90 - minuto)
    dt = 1.0  # step di 1 minuto

    # Rates base per minuto (uniforme — sarà poi modificato dal moltiplicatore temporale)
    base_rate_h = max(0.0, mu_h / minutes_remaining)
    base_rate_a = max(0.0, mu_a / minutes_remaining)

    # Stato iniziale: (0, 0) gol rimanenti
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}

    # Itera minuto per minuto, tracciando il minuto assoluto per applicare
    # il moltiplicatore temporale delle probabilità di gol.
    # Se siamo al minuto 60, iteriamo da 60 a 89 (minuti rimanenti).
    for abs_min in range(minuto, 90):
        # Moltiplicatore per la distribuzione temporale empirica dei gol.
        # Vedi docstring di _goal_rate_multiplier per fonti e dati.
        time_mult = _goal_rate_multiplier(abs_min)

        new_states: dict[tuple[int, int], float] = {}
        for (gh, ga), prob in states.items():
            if prob < 1e-15:
                continue

            # Upgrade 4: Score effect rimosso dal Markov per eliminare double-counting.
            # Il time_decay.py è l'unica sorgente di score effect e già riduce il
            # residuo tramite market_absorption. Gli xG passati a questa funzione
            # (mu_h, mu_a) sono già corretti da time_decay.
            # Il Markov propaga i rate puri, senza duplicare l'aggiustamento tattico.
            lh = base_rate_h * time_mult
            la = base_rate_a * time_mult

            # Probabilità di transizione con moltiplicatore temporale.
            # Cap a 0.20 per squadra garantisce che P(h) + P(a) ≤ 0.40,
            # lasciando sempre P(nessun gol) ≥ 0.60 anche nei minuti di picco (76-90).
            p_h_raw = min(0.20, lh * dt)
            p_a_raw = min(0.20, la * dt)

            p_h = p_h_raw
            p_a = p_a_raw
            p_none = max(0.0, 1.0 - p_h - p_a)

            # Nessun gol
            key = (gh, ga)
            new_states[key] = new_states.get(key, 0.0) + prob * p_none

            # Gol casa
            if gh + 1 <= max_goals:
                key = (gh + 1, ga)
                new_states[key] = new_states.get(key, 0.0) + prob * p_h

            # Gol trasferta
            if ga + 1 <= max_goals:
                key = (gh, ga + 1)
                new_states[key] = new_states.get(key, 0.0) + prob * p_a

        states = new_states

    # Normalizzazione
    total = sum(states.values())
    if total > 0:
        states = {k: v / total for k, v in states.items()}

    # Dixon–Coles (tau) una sola volta sulla PMF dei gol rimanenti, coerente con poisson.py.
    if abs(rho_dc) > 1e-12:
        _adj: dict[tuple[int, int], float] = {}
        for (i, j), pij in states.items():
            tau_ij = dixon_coles_tau(i, j, mu_h, mu_a, rho_dc=rho_dc)
            _adj[(i, j)] = max(0.0, pij * tau_ij)
        _s = sum(_adj.values())
        if _s > 1e-18:
            states = {k: v / _s for k, v in _adj.items()}

    return states
