"""
live_recalibration.py — Ricalibrazione intra-partita basata su osservazioni live.

Confronta le previsioni attese del modello con la realtà osservata durante
la partita per correggere bias in tempo reale.

Se a minuto 45 il modello si aspettava ~1.25 gol totali (50% di 2.5)
ma il punteggio è 0-0, il modello era troppo ottimista → ridurre xG residui.

L'effetto è leggero e conservativo per evitare overreaction a varianza normale.

FIX PRECISION #2: Recalibration ora è ASIMMETRICA — fattori separati per casa e
trasferta, basati sulla deviazione per-squadra tra gol osservati e xG attesi.
Questo corregge il bias quando una squadra sovraperforma e l'altra sottoperforma.
"""

from __future__ import annotations

from src.config import BAYES, ENGINE

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Minuto minimo per attivare la ricalibrazione (serve abbastanza tempo per il segnale)
MIN_MINUTE_FOR_RECAL: int = 30

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
    *,
    tot_cur_remaining: float | None = None,
) -> float:
    """
    Calcola il fattore di ricalibrazione live basato su osservazioni.

    Versione LEGACY (simmetrica) — mantenuta per retrocompatibilità.
    Usa il fattore totale come media tra casa e trasferta.

    Args:
        xg_total_prematch: Total atteso a inizio partita (tot_op).
        gol_attuali: Gol totali segnati finora.
        minuto: Minuto attuale [0, 90].
        tot_cur_remaining: Linea Total corrente (gol rimanenti).

    Returns:
        Fattore moltiplicativo per xG residui in [1 - MAX, 1 + MAX].
    """
    factor_h, factor_a = compute_live_recalibration_factors(
        xg_total_prematch, gol_attuali, minuto,
        gol_casa=gol_attuali,  # fallback: assegna tutto alla casa
        gol_trasf=0,
        xg_h_share=0.50,
        tot_cur_remaining=tot_cur_remaining,
    )
    return 0.5 * (factor_h + factor_a)


def compute_live_recalibration_factors(
    xg_total_prematch: float,
    gol_attuali: int,
    minuto: int,
    *,
    gol_casa: int = 0,
    gol_trasf: int = 0,
    xg_h_share: float = 0.50,
    tot_cur_remaining: float | None = None,
) -> tuple[float, float]:
    """
    Calcola fattori di ricalibrazione SEPARATI per casa e trasferta.

    Confronta i gol segnati da ciascuna squadra con il suo xG atteso al minuto
    corrente. Se una squadra sovraperforma (più gol del previsto) il suo xG
    residuo viene ridotto (regressione alla media). Se sottoforma, aumenta.

    Args:
        xg_total_prematch: Total atteso a inizio partita (tot_op).
        gol_attuali: Gol totali segnati finora (per compatibilità).
        gol_casa: Gol segnati dalla squadra di casa.
        gol_trasf: Gol segnati dalla squadra in trasferta.
        minuto: Minuto attuale [0, 90].
        xg_h_share: Quota xG casa sul totale (xg_h / (xg_h + xg_a)).
        tot_cur_remaining: Linea Total corrente (gol rimanenti).

    Returns:
        (factor_h, factor_a): Fattori moltiplicativi per xG residui casa e trasferta.
        Ciascuno in [1 - MAX, 1 + MAX]. 1.0 = nessun aggiustamento.
    """
    if minuto < MIN_MINUTE_FOR_RECAL or xg_total_prematch <= 0:
        return 1.0, 1.0

    frac_played = min(minuto / 90.0, 1.0)

    # Prior: gol attesi entro ora se il ritmo fosse uniforme sul tot prematch
    expected_uniform = xg_total_prematch * (1.0 - (1.0 - frac_played) ** 1.3)

    # Mercato: totale partita implicito ≈ segnati + rimanenti (linea live)
    expected_market_timeline = expected_uniform
    if tot_cur_remaining is not None and float(tot_cur_remaining) > BAYES.TOT_BAYES_MIN:
        implied_full = max(
            float(gol_attuali) + float(tot_cur_remaining),
            xg_total_prematch * 0.35,
        )
        expected_market_timeline = implied_full * frac_played

    w_m = ENGINE.LIVE_RECAL_MARKET_BLEND
    if tot_cur_remaining is None or float(tot_cur_remaining) <= BAYES.TOT_BAYES_MIN:
        w_m = 0.0
    expected_goals_by_now = w_m * expected_market_timeline + (1.0 - w_m) * expected_uniform

    if expected_goals_by_now < 0.3:
        return 1.0, 1.0

    # --- FIX PRECISION #2: Recalibration ASIMMETRICA ---
    # Dividi il totale atteso tra casa e trasferta usando la quota xG.
    # Se xg_h_share = 0.55, la casa dovrebbe aver segnato il 55% dei gol attesi.
    expected_h_by_now = expected_goals_by_now * xg_h_share
    expected_a_by_now = expected_goals_by_now * (1.0 - xg_h_share)

    # Evita divisione per zero
    expected_h_by_now = max(0.15, expected_h_by_now)
    expected_a_by_now = max(0.15, expected_a_by_now)

    # Deviazione per squadra
    deviation_h = gol_casa - expected_h_by_now
    deviation_a = gol_trasf - expected_a_by_now

    # Normalizzazione: quanto devia in proporzione ai gol attesi
    norm_dev_h = deviation_h / expected_h_by_now
    norm_dev_a = deviation_a / expected_a_by_now

    # Affidabilità temporale: cresce con il tempo giocato
    time_reliability = min(1.0, frac_played / 0.50)

    # Fattori individuali
    adj_h = norm_dev_h * LIVE_RECAL_SENSITIVITY * time_reliability
    adj_a = norm_dev_a * LIVE_RECAL_SENSITIVITY * time_reliability

    # Clamp
    factor_h = 1.0 + max(-MAX_LIVE_ADJUSTMENT, min(MAX_LIVE_ADJUSTMENT, adj_h))
    factor_a = 1.0 + max(-MAX_LIVE_ADJUSTMENT, min(MAX_LIVE_ADJUSTMENT, adj_a))

    return factor_h, factor_a


__all__ = [
    "compute_live_recalibration_factor",
    "compute_live_recalibration_factors",
]
