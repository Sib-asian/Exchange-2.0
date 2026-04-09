"""
live_recalibration.py — Ricalibrazione intra-partita basata su osservazioni live.

Confronta le previsioni attese del modello con la realtà osservata durante
la partita per correggere bias in tempo reale.

Se a minuto 45 il modello si aspettava ~1.25 gol totali (50% di 2.5)
ma il punteggio è 0-0, il modello era troppo ottimista → ridurre xG residui.

L'effetto è leggero e conservativo per evitare overreaction a varianza normale.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Minuto minimo per attivare la ricalibrazione (serve abbastanza tempo per il segnale)
MIN_MINUTE_FOR_RECAL: int = 20

# Massimo aggiustamento xG dalla ricalibrazione live
MAX_LIVE_ADJUSTMENT: float = 0.12  # ±12% massimo

# Sensibilità: quanto peso dare alla deviazione osservata
LIVE_RECAL_SENSITIVITY: float = 0.30


# ---------------------------------------------------------------------------
# Calcolo fattore di ricalibrazione
# ---------------------------------------------------------------------------

def compute_live_recalibration_factor(
    xg_total_prematch: float,
    gol_attuali: int,
    minuto: int,
) -> float:
    """
    Calcola il fattore di ricalibrazione live basato su osservazioni.

    Confronta i gol attesi al minuto corrente (proporzione lineare del totale
    prematch) con i gol effettivamente segnati.

    Args:
        xg_total_prematch: Total atteso a inizio partita (tot_op).
        gol_attuali: Gol totali segnati finora.
        minuto: Minuto attuale [0, 90].

    Returns:
        Fattore moltiplicativo per xG residui in [1 - MAX, 1 + MAX].
        1.0 = nessun aggiustamento.
        < 1.0 = modello sovrastimava gol → ridurre xG residui.
        > 1.0 = modello sottostimava gol → aumentare xG residui.
    """
    if minuto < MIN_MINUTE_FOR_RECAL or xg_total_prematch <= 0:
        return 1.0

    frac_played = min(minuto / 90.0, 1.0)

    # Gol attesi entro questo minuto (distribuzione uniforme nel tempo, approssimazione)
    expected_goals_by_now = xg_total_prematch * frac_played

    if expected_goals_by_now < 0.3:
        return 1.0

    # Deviazione: positiva = più gol del previsto, negativa = meno gol del previsto
    deviation = gol_attuali - expected_goals_by_now

    # Normalizzazione: quanto devia in proporzione ai gol attesi
    normalized_deviation = deviation / expected_goals_by_now

    # Aggiustamento: proporzionale alla deviazione, pesato per il tempo giocato
    # Più tempo giocato = più il segnale è affidabile
    time_reliability = min(1.0, frac_played / 0.50)  # piena affidabilità dopo 50%

    adjustment = normalized_deviation * LIVE_RECAL_SENSITIVITY * time_reliability

    # Clamp
    adjustment = max(-MAX_LIVE_ADJUSTMENT, min(MAX_LIVE_ADJUSTMENT, adjustment))

    return 1.0 + adjustment


__all__ = ["compute_live_recalibration_factor"]
