"""
btts.py — Calcolo probabilità BTTS (Both Teams To Score).

La probabilità BTTS è calcolata condizionalmente sullo stato attuale:
- Se entrambe hanno già segnato: BTTS Sì = 1.0 (mercato chiuso)
- Se solo una ha segnato: BTTS Sì = P(l'altra segna almeno 1 nel rimasto)
- Se nessuna ha segnato: BTTS Sì = P(entrambe segnano almeno 1 nel rimasto)

Usa la matrice full che include la correlazione bivariate + DC.

Calibrazione BTTS:
- Corregge bias del modello Poisson (sovrastima in partite difensive)
- Considera forza offensiva/defensiva delle squadre
- Integra dati H2H storici
- Considera forma recente (streak gol / clean sheet)
"""

from __future__ import annotations


def calcola_btts(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
) -> float:
    """
    Calcola P(BTTS Sì) tenendo conto del punteggio attuale.

    La matrice full contiene P(X_rem=a, Y_rem=b) per i GOL RIMANENTI.
    BTTS richiede che entrambe le squadre abbiano almeno 1 gol nel
    punteggio FINALE: quindi combina gol già segnati con gol rimanenti.

    Args:
        full: Matrice bivariata completa normalizzata (gol rimanenti).
        gol_casa: Gol attuali della casa.
        gol_trasf: Gol attuali della trasferta.

    Returns:
        P(BTTS Sì) in [0, 1].
    """
    if gol_casa > 0 and gol_trasf > 0:
        # Entrambe hanno già segnato: BTTS Sì garantito
        return 1.0
    elif gol_casa > 0:
        # Casa ha già segnato; BTTS iff trasferta segna almeno 1 nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if b > 0)
    elif gol_trasf > 0:
        # Trasferta ha già segnato; BTTS iff casa segna almeno 1 nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if a > 0)
    else:
        # Nessuna ha segnato; BTTS iff entrambe segnano nel rimasto
        p_btts = sum(p for (a, b), p in full.items() if a > 0 and b > 0)

    return min(1.0, max(0.0, p_btts))


def calibra_btts(
    p_btts_raw: float,
    tot_cur: float,
    xg_h: float,
    xg_a: float,
    h2h_btts_pct: float = 0.0,
    last6_gf_h: float = 0.0,
    last6_ga_h: float = 0.0,
    last6_gf_a: float = 0.0,
    last6_ga_a: float = 0.0,
    scoring_streak_h: int = 0,
    scoring_streak_a: int = 0,
    clean_sheet_streak_h: int = 0,
    clean_sheet_streak_a: int = 0,
    minuto: int = 0,
) -> float:
    """
    Calibra la probabilità BTTS grezza con fattori contestuali.

    Il modello Poisson tende a:
    - SOVRASTIMARE BTTS Sì in partite difensive (total basso)
    - SOTTOSTIMARE BTTS Sì in partite aperte (total alto)

    Questa funzione corregge il bias usando:
    1. Total atteso: total basso → riduci BTTS, total alto → aumenta BTTS
    2. Forza offensiva/defensiva: mismatch aumenta BTTS
    3. H2H storico: partite storiche con molti gol aumentano BTTS
    4. Forma recente: streak gol aumenta BTTS, clean sheet lo riduce

    Args:
        p_btts_raw: Probabilità BTTS grezza dal modello Poisson.
        tot_cur: Total atteso corrente (gol rimanenti o full-game).
        xg_h, xg_a: xG attesi per casa e trasferta.
        h2h_btts_pct: % storica di partite H2H con BTTS (0-100).
        last6_gf_h, last6_ga_h: Gol fatti/subiti casa nelle ultime 6.
        last6_gf_a, last6_ga_a: Gol fatti/subiti trasferta nelle ultime 6.
        scoring_streak_h, scoring_streak_a: Partite consecutive con gol segnato.
        clean_sheet_streak_h, clean_sheet_streak_a: Partite consecutive senza subire.
        minuto: Minuto attuale (la calibrazione è più forte in prematch).

    Returns:
        Probabilità BTTS calibrata in [BTTS_MIN, BTTS_MAX].
    """
    from src.config import BTTS_CALIBRATION

    p_btts = p_btts_raw
    adjustment = 0.0

    # === 1. Calibrazione basata su Total ===
    if tot_cur > 0:
        if tot_cur < BTTS_CALIBRATION.TOTAL_LOW_THRESHOLD:
            # Partita difensiva: riduci BTTS
            adjustment += BTTS_CALIBRATION.BTTS_LOW_TOTAL_ADJUST
        elif tot_cur > BTTS_CALIBRATION.TOTAL_HIGH_THRESHOLD:
            # Partita aperta: aumenta BTTS
            adjustment += BTTS_CALIBRATION.BTTS_HIGH_TOTAL_ADJUST

    # === 2. Calibrazione basata su forza offensiva/defensiva ===
    if xg_h > 0 and xg_a > 0:
        # Calcola "forza attacco" vs "debolezza difesa"
        # Una squadra con xG alto (attacco forte) vs avversario con xG alto subendo (difesa debole)
        # → più probabile che entrambe segnino

        # Attacco casa forte AND Difesa trasferta debole (xg_a alto = trasferta segna molto)
        attack_strong_h = xg_h >= BTTS_CALIBRATION.STRONG_ATTACK_THRESHOLD
        defense_weak_a = xg_a >= BTTS_CALIBRATION.WEAK_DEFENSE_THRESHOLD

        # Attacco trasferta forte AND Difesa casa debole
        attack_strong_a = xg_a >= BTTS_CALIBRATION.STRONG_ATTACK_THRESHOLD
        defense_weak_h = xg_h >= BTTS_CALIBRATION.WEAK_DEFENSE_THRESHOLD

        # Mismatch: se entrambe hanno attacco forte e difesa debole
        if (attack_strong_h and defense_weak_a) and (attack_strong_a and defense_weak_h):
            adjustment += BTTS_CALIBRATION.ATTACK_DEFENSE_MISMATCH_BONUS

        # Entrambe difese forti (xg basso per entrambe = poche xG concesse)
        defense_strong_h = xg_h < 1.0  # Difesa casa forte = concede poco
        defense_strong_a = xg_a < 1.0  # Difesa trasferta forte
        if defense_strong_h and defense_strong_a:
            adjustment += BTTS_CALIBRATION.BOTH_STRONG_DEFENSE_PENALTY

    # === 3. Calibrazione basata su H2H ===
    if h2h_btts_pct > 0:
        if h2h_btts_pct >= BTTS_CALIBRATION.H2H_BTTS_HIGH_THRESHOLD * 100:
            # H2H storico con alto BTTS
            adjustment += BTTS_CALIBRATION.H2H_BTTS_BONUS

    # === 4. Calibrazione basata su forma recente (Last 6) ===
    if last6_gf_h > 0 or last6_gf_a > 0:
        # Calcola media gol fatti per partita nelle ultime 6
        avg_gf_h = last6_gf_h / 6.0 if last6_gf_h > 0 else 0
        avg_gf_a = last6_gf_a / 6.0 if last6_gf_a > 0 else 0

        # Se entrambe le squadre fanno più di 1.3 gol/partita → +BTTS
        if avg_gf_h > 1.3 and avg_gf_a > 1.3:
            adjustment += 0.02

        # Se entrambe subiscono più di 1.3 gol/partita → +BTTS
        avg_ga_h = last6_ga_h / 6.0 if last6_ga_h > 0 else 0
        avg_ga_a = last6_ga_a / 6.0 if last6_ga_a > 0 else 0
        if avg_ga_h > 1.3 and avg_ga_a > 1.3:
            adjustment += 0.02

    # === 5. Calibrazione basata su streak ===
    if scoring_streak_h >= 3:
        adjustment += BTTS_CALIBRATION.RECENT_SCORING_STREAK_BONUS
    if scoring_streak_a >= 3:
        adjustment += BTTS_CALIBRATION.RECENT_SCORING_STREAK_BONUS

    if clean_sheet_streak_h >= 2:
        adjustment += BTTS_CALIBRATION.RECENT_CLEAN_SHEET_PENALTY
    if clean_sheet_streak_a >= 2:
        adjustment += BTTS_CALIBRATION.RECENT_CLEAN_SHEET_PENALTY

    # === 6. Applica calibrazione ===
    # La calibrazione è più forte in prematch (minuto=0) e si riduce col tempo
    # perché la partita stessa fornisce evidenza empirica
    time_factor = 1.0 if minuto == 0 else max(0.3, 1.0 - minuto / 90.0 * 0.7)

    p_btts_calibrated = p_btts + adjustment * time_factor

    # Clamp finale
    p_btts_calibrated = max(BTTS_CALIBRATION.BTTS_MIN,
                           min(BTTS_CALIBRATION.BTTS_MAX, p_btts_calibrated))

    return p_btts_calibrated

