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

# Dixon–Coles tau applicato una solta sulla distribuzione finale (non per minuto):
# evita la compounding ~90× sugli stati bassi (0,0)/(1,0)/(0,1).
from src.models.poisson import dixon_coles_tau


# Moltiplicatore per l'effetto pressing (coerente con time_decay.py)
# La squadra in svantaggio preme con più volume ma minore qualità
SCORE_PRESS_MULTIPLIER = DECAY.SCORE_DOWN_MULTIPLIER  # 0.65


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

    # Rates base per minuto
    base_rate_h = max(0.0, mu_h / minutes_remaining)
    base_rate_a = max(0.0, mu_a / minutes_remaining)

    # Stato iniziale: (0, 0) gol rimanenti
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}

    for _ in range(minutes_remaining):
        new_states: dict[tuple[int, int], float] = {}
        for (gh, ga), prob in states.items():
            if prob < 1e-15:
                continue

            # Upgrade 4: Score effect rimosso dal Markov per eliminare double-counting.
            # Il time_decay.py è l'unica sorgente di score effect e già riduce il
            # residuo tramite market_absorption. Gli xG passati a questa funzione
            # (mu_h, mu_a) sono già corretti da time_decay.
            # Il Markov propaga i rate puri, senza duplicare l'aggiustamento tattico.
            lh = base_rate_h
            la = base_rate_a

            # Probabilità di transizione base (cap per evitare > 1)
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
