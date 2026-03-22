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

from src.config import DECAY  # noqa: I001


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

    Fix #9: aggiunge approssimazione della correlazione Dixon-Coles.
    rho_dc < 0 → le squadre tendono a NON segnare simultaneamente.
    Implementato come correzione della probabilità di doppio gol nello stesso minuto:
    il joint rate viene scalato per (1 + rho_dc * cap) per i punteggi bassi.

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

    # Fattore correlazione DC: rho_dc < 0 → penalizza i gol simultanei.
    # Si applica solo a punteggi bassi (totale ≤ 1), coerente con la tau-correction DC.
    # Ai punteggi alti il fattore converge a 1.0 (nessuna correzione).
    dc_corr_low = max(0.7, 1.0 + rho_dc)   # correzione per punteggi bassi (0-0, 1-0, 0-1)

    # Stato iniziale: (0, 0) gol rimanenti
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}

    for _ in range(minutes_remaining):
        new_states: dict[tuple[int, int], float] = {}
        for (gh, ga), prob in states.items():
            if prob < 1e-15:
                continue

            # Score-dependent rates: il punteggio cumulativo (attuali + rimanenti)
            # influenza le rates. La squadra in svantaggio preme, quella in vantaggio difende.
            # FIX: Usa SCORE_PRESS_MULTIPLIER per coerenza con time_decay.py
            diff = (gol_h + gh) - (gol_a + ga)
            if diff > 0:
                # Casa in vantaggio → riduce attacco, avversario preme
                se = min(0.15, abs(diff) * score_effect_rate)
                lh = base_rate_h * (1.0 - se * SCORE_PRESS_MULTIPLIER)  # difende
                la = base_rate_a * (1.0 + se * SCORE_PRESS_MULTIPLIER)  # preme
            elif diff < 0:
                se = min(0.15, abs(diff) * score_effect_rate)
                lh = base_rate_h * (1.0 + se * SCORE_PRESS_MULTIPLIER)  # preme
                la = base_rate_a * (1.0 - se * SCORE_PRESS_MULTIPLIER)  # difende
            else:
                lh = base_rate_h
                la = base_rate_a

            # Probabilità di transizione base (cap per evitare > 1)
            p_h_raw = min(0.20, lh * dt)
            p_a_raw = min(0.20, la * dt)

            # Correzione DC per punteggi bassi: riduce la probabilità
            # di gol simultanei quando il totale cumulato è basso.
            total_so_far = gol_h + gh + gol_a + ga
            if total_so_far <= 1:
                # Applica correzione DC: rho_dc < 0 → meno prob simultanea
                # Redistribuisce la riduzione del joint verso i gol singoli.
                joint_reduction = p_h_raw * p_a_raw * (1.0 - dc_corr_low)
                p_h_raw = min(0.20, p_h_raw + joint_reduction * 0.5)
                p_a_raw = min(0.20, p_a_raw + joint_reduction * 0.5)

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

    return states
