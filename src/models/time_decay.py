"""
time_decay.py — Aggiustamenti tattico-comportamentali sugli xG live.

Implementa:
  - Score effect residuale (l'AH live cattura ~80%, qui il residuo ~5-8%)
  - Impatto dei cartellini rossi con effetto marginale decrescente
  - Calcolo del momentum di mercato

Letteratura:
  Brechot & Flepp (2020): effetti cartellini, score effects
  Robberechts et al. (2021): pressing e qualità tiri in svantaggio
"""

from __future__ import annotations

import math

from src.config import DECAY, MOMENTUM


def calcola_momentum_mercato(
    delta_ah: float,
    delta_tot: float,
    minuto: int,
) -> float:
    """
    Indice [0, 6] di intensità del movimento di mercato relativo al tempo giocato.

    Un grande delta early-game vale più di uno late-game:
    - Early: il mercato ha ricevuto informazione non ancora scontata.
    - Late: gran parte dell'informazione è già incorporata nelle quote.

    Usa sqrt invece di scala lineare per evitare overshoot early-game:
    - A minuto 10: sqrt=0.33 (amplifica ×3), lineare=0.11 (×9) → troppo.

    A minuto=0 restituisce 0 — la differenza tra apertura e corrente in
    prematch non è un movimento intrapartita, ma una differenza tra
    due rilevazioni statiche.

    Soglie interpretative:
    - 0.0–1.0  Mercato stabile
    - 1.0–2.5  Movimento moderato
    - 2.5–4.0  Movimento significativo
    - 4.0+     Movimento estremo (infortuni, rossi non registrati, info asimmetrica)

    Args:
        delta_ah: Variazione AH (ah_cur - ah_op).
        delta_tot: Variazione Total (tot_cur - tot_op).
        minuto: Minuto attuale [0, 90].

    Returns:
        Indice momentum in [0.0, 6.0].
    """
    if minuto == 0:
        return 0.0
    frac = max(MOMENTUM.FRAC_FLOOR, math.sqrt(minuto / 90.0))
    return min((abs(delta_ah) + abs(delta_tot) * MOMENTUM.TOT_WEIGHT) / frac, MOMENTUM.MOMENTUM_CAP)


def time_decay_dinamico(
    xg_casa: float,
    xg_trasf: float,
    minuto: int,
    gol_casa: int,
    gol_trasf: int,
    rossi_casa: int,
    rossi_trasf: int,
    momentum: float = 0.0,
) -> tuple[float, float]:
    """
    Aggiustamenti tattico-comportamentali sugli xG proiettati al tempo rimanente.

    ── Cosa NON fa ───────────────────────────────────────────────────────────
    Il decadimento Weibull (t_remaining/90)^0.85 è stato rimosso.
    `tot_cur` è la linea live dei GOL RIMANENTI: `calcola_xg_bayesiani` restituisce
    già stime in "remaining goals space". Applicare un ulteriore fattore temporale
    dimezzava le stime rispetto a ciò che il mercato prezzava → bias sistematico.

    ── Cosa fa ───────────────────────────────────────────────────────────────
    1. Score effect RESIDUALE (cap 8%):
       L'AH corrente già incorpora ~80% dell'effetto punteggio.
       Il residuo cattura il comportamento tattico non ancora pienamente prezzato:
       - Pressing disperato (team che perde): xG ↑ ma qualità ↓ (tiri da lontano)
       - Parking the bus (team in vantaggio): xG ↓ ma contropiedismo ↑
       Il residuo scala inversamente col minuto: early-game AH ha latency maggiore.

    2. Cartellini rossi — effetto marginale decrescente (Brechot & Flepp 2020):
       -32% xG per il team ridotto, +28% per l'avversario al primo rosso.
       Effetto marginale decresce con rossi successivi (tabella precalcolata).
       Asimmetria home/away: la trasferta perde il supporto del pubblico (+5% penalità).

    Args:
        xg_casa, xg_trasf: xG blendati (da calcola_xg_bayesiani + blend_xg_shots).
        minuto: Minuto attuale [0, 90].
        gol_casa, gol_trasf: Gol attuali.
        rossi_casa, rossi_trasf: Cartellini rossi attuali.

    Returns:
        (xg_casa_adj, xg_trasf_adj): xG aggiustati, con floor XG_FLOOR.
    """
    if minuto >= 90:
        return DECAY.XG_FLOOR, DECAY.XG_FLOOR

    xg_c = float(xg_casa)
    xg_t = float(xg_trasf)

    # 1. Score effect residuale ASIMMETRICO
    # La squadra in svantaggio preme con tiri disperati (volume ↑, qualità ↓):
    #   il boost xG è ridotto (SCORE_DOWN_MULTIPLIER = 0.65).
    # La squadra in vantaggio controlla possesso e difende in modo organizzato:
    #   il suo vantaggio difensivo è amplificato (SCORE_UP_MULTIPLIER = 1.15).
    # In partite ad alto punteggio (3-2, 4-3), l'AH live ha già incorporato
    #   la volatilità → il residuo va scalato in base ai gol totali.
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat = abs(diff) / (DECAY.SCORE_SATURATION + abs(diff))
        minute_scale = max(
            DECAY.SCORE_MINUTE_SCALE_FLOOR,
            DECAY.SCORE_MINUTE_SCALE_A - DECAY.SCORE_MINUTE_SCALE_B * (minuto / 90.0),
        )
        # Goal intensity: residuo cala con gol totali (AH già prezza la volatilità)
        gol_tot = gol_casa + gol_trasf
        goal_intensity = 1.0 - min(0.60, gol_tot * DECAY.SCORE_GOAL_INTENSITY_SCALE)
        residual_base = min(DECAY.SCORE_EFFECT_MAX, DECAY.SCORE_EFFECT_BASE * sat * minute_scale * goal_intensity)

        if diff < 0:  # casa in svantaggio
            xg_c *= (1.0 + residual_base * DECAY.SCORE_DOWN_MULTIPLIER)
            xg_t *= (1.0 - residual_base * DECAY.SCORE_UP_MULTIPLIER)
        else:  # casa in vantaggio
            xg_t *= (1.0 + residual_base * DECAY.SCORE_DOWN_MULTIPLIER)
            xg_c *= (1.0 - residual_base * DECAY.SCORE_UP_MULTIPLIER)

    # 2. Cartellini rossi — effetto marginale decrescente × tempo rimanente
    # Un rosso al 10' (80 minuti restanti) ha effetto pieno: la squadra gioca
    # in inferiorità per quasi tutta la partita.
    # Un rosso all'80' (10 minuti restanti) ha effetto dimezzato: pochi minuti
    # restanti riducono l'impatto tattico.
    time_remaining_pct = max(0.0, (90.0 - minuto) / 90.0)
    red_time_mult = DECAY.RED_TIME_FLOOR + (1.0 - DECAY.RED_TIME_FLOOR) * time_remaining_pct

    if rossi_casa > 0:
        idx = min(rossi_casa, len(DECAY.RED_DECAY) - 1)
        decay_eff = 1.0 + (DECAY.RED_DECAY[idx] - 1.0) * red_time_mult
        boost_eff = 1.0 + (DECAY.RED_BOOST[idx] - 1.0) * red_time_mult
        xg_c *= decay_eff
        xg_t *= boost_eff

    if rossi_trasf > 0:
        idx = min(rossi_trasf, len(DECAY.RED_DECAY) - 1)
        decay_eff = 1.0 + (DECAY.RED_DECAY[idx] - 1.0) * red_time_mult
        boost_eff = 1.0 + (DECAY.RED_BOOST[idx] - 1.0) * red_time_mult
        xg_t *= decay_eff * DECAY.RED_AWAY_PENALTY
        xg_c *= boost_eff * DECAY.RED_HOME_BOOST

    # 3. Smorzamento xG per momentum estremo
    # Mercati molto volatili (momentum > 2.5) segnalano informazione non catturata
    # dal modello → smorzare xG per evitare falsi edge in mercati instabili.
    if momentum > DECAY.MOMENTUM_XG_THRESHOLD:
        excess = momentum - DECAY.MOMENTUM_XG_THRESHOLD
        damp = 1.0 - min(DECAY.MOMENTUM_XG_DAMP_MAX, excess * DECAY.MOMENTUM_XG_DAMP_RATE)
        xg_c *= damp
        xg_t *= damp

    return max(DECAY.XG_FLOOR, xg_c), max(DECAY.XG_FLOOR, xg_t)
