"""
kelly.py — Kelly Criterion per il dimensionamento delle stake su exchange.

Implementa:
  - Kelly frazionato per BACK con aggiustamento commissione
  - Kelly frazionato per LAY con formula commission-adjusted corretta
  - Calcolo edge netto e break-even probability per il lay

Riferimenti:
  Kelly (1956), "A New Interpretation of Information Rate"
  Thorp (1962), "Beat the Dealer"
  Hausch & Ziemba (2008), "Handbook of Sports and Lottery Markets"
"""

from __future__ import annotations

from src.config import KELLY


def calcola_kelly_fraction(
    minuto: int,
    n_shots_tot: int,
) -> float:
    """
    Calcola la frazione Kelly dinamica in base al contesto di gioco.

    Riduce la fraction quando:
    - Late-game (>75'): spread ampi, modello meno affidabile.
    - Partita in corso senza dati tiri: informazione incompleta.

    Args:
        minuto: Minuto attuale [0, 90].
        n_shots_tot: Numero totale di tiri inseriti.

    Returns:
        Frazione Kelly in [KELLY_MIN_FRACTION, KELLY_BASE_FRACTION].
    """
    fraction = KELLY.KELLY_BASE_FRACTION
    if minuto > KELLY.KELLY_LATE_START:
        # Riduzione graduale: cresce linearmente da 0 a KELLY_LATE_GAME_REDUCTION
        # tra KELLY_LATE_START e 90'. Evita cliff-edge a un singolo minuto.
        progress = (minuto - KELLY.KELLY_LATE_START) / max(1, 90 - KELLY.KELLY_LATE_START)
        fraction -= KELLY.KELLY_LATE_GAME_REDUCTION * min(1.0, progress)
    if minuto > 0 and n_shots_tot == 0:
        fraction -= KELLY.KELLY_NO_SHOTS_REDUCTION
    return max(KELLY.KELLY_MIN_FRACTION, fraction)


def calcola_stake_kelly(
    prob_modello: float,
    quota_netta: float,
    bankroll: float,
    frazione: float = 0.5,
    edge_back: float | None = None,
) -> float:
    """
    Kelly frazionato per BACK su exchange.

    Formula:
        f* = (p * Q_netto - 1) / (Q_netto - 1)
        stake = bankroll * f* * frazione

    dove Q_netto = quota già al netto della commissione exchange.

    Cap adattivo per edge:
    - edge < 5%  → cap stake a 2.5% * fraction (edge piccolo, incertezza alta)
    - edge < 10% → cap stake a 4.0% * fraction
    - edge >= 10%→ cap Kelly standard al 5%

    Args:
        prob_modello: Probabilità stimata dal modello.
        quota_netta: Quota exchange al netto della commissione.
        bankroll: Capitale disponibile.
        frazione: Frazione Kelly (default 0.5 = half-Kelly).
        edge_back: Edge netto precalcolato (opzionale, per cap adattivo).

    Returns:
        Stake in euro, 0.0 se EV <= 0.
    """
    if quota_netta <= KELLY.MIN_QUOTA_NETTA or prob_modello * quota_netta <= 1.0:
        return 0.0

    kelly_pct = (prob_modello * quota_netta - 1.0) / (quota_netta - 1.0)
    kelly_pct = max(0.0, min(kelly_pct, KELLY.KELLY_MAX_PCT))

    stake_raw = bankroll * kelly_pct * frazione

    # Cap adattivo per edge
    if edge_back is not None:
        if edge_back < KELLY.KELLY_SMALL_EDGE_THRESHOLD:
            stake_raw = min(stake_raw, bankroll * KELLY.KELLY_SMALL_EDGE_CAP_PCT * frazione)
        elif edge_back < KELLY.KELLY_MEDIUM_EDGE_THRESHOLD:
            stake_raw = min(stake_raw, bankroll * KELLY.KELLY_MEDIUM_EDGE_CAP_PCT * frazione)

    return stake_raw


def calcola_stake_lay(
    prob_modello: float,
    quota_exc: float,
    bankroll: float,
    frazione: float = 0.5,
    comm_rate: float = 0.0,
) -> tuple[float, float] | None:
    """
    Kelly frazionato per LAY su exchange, con aggiustamento commissione.

    Per un lay a quota Q con commissione c:
    - Vinci: S*(1-c)  se l'evento NON accade  (prob = 1 - p)
    - Perdi: S*(Q-1)  se l'evento accade       (prob = p)

    Break-even probability per il layer:
        p_BE = (1 - c) / (Q - c)

    Kelly fraction della liability:
        f_liability = p_BE - p = (1-c)/(Q-c) - p

    Questa formula è la derivazione corretta per il lay con commissione.
    La formula alternativa f = 1 - p*(Q-c)/(1-c) è equivalente.

    Cap: liability massima al 5% del bankroll (stesso cap del back).
    Sanity: stake non può superare il bankroll.

    Args:
        prob_modello: Probabilità che l'evento ACCADA (prob della selezione laiata).
        quota_exc: Quota exchange (odds decimali, lordi).
        bankroll: Capitale disponibile.
        frazione: Frazione Kelly.
        comm_rate: Commissione exchange in [0, 1).

    Returns:
        (stake_visibile, liability) o None se non c'è valore.
    """
    if quota_exc <= KELLY.LAY_MIN_ODDS:
        return None
    if comm_rate >= 1.0:
        return None

    denom = quota_exc - comm_rate
    if denom <= 0:
        return None

    # Kelly fraction della liability (derivazione corretta).
    #
    # Il layer vince (1-c) per unità di stake se l'evento NON accade (prob 1-p),
    # e perde (Q-1) per unità di stake se l'evento accade (prob p).
    #
    # Massimizzando E[log(W)] rispetto alla liability L = stake*(Q-1):
    #   f_L = (1-p) - p*(Q-1)/(1-c)
    #       = 1 - p*(Q-c)/(1-c)
    #       = 1 - p * denom / (1-comm_rate)
    #
    # NOTA: la formula p_BE - p = (1-c)/(Q-c) - p NON è equivalente:
    # differisce di un fattore Q/(Q-1) nel regime non cappato.
    f_liability = 1.0 - prob_modello * denom / (1.0 - comm_rate)

    if f_liability <= 0:
        return None

    f_liability = min(f_liability, KELLY.KELLY_MAX_PCT)
    liability = bankroll * f_liability * frazione
    stake = liability / (quota_exc - 1.0)

    if stake > bankroll:
        return None

    return stake, liability


def quota_netta(quota_exc: float, comm_rate: float) -> float:
    """
    Quota effettiva per il BACK dopo la commissione exchange.

    Q_netto = 1 + (Q_exc - 1) * (1 - comm)

    Args:
        quota_exc: Quota nominale sull'exchange.
        comm_rate: Commissione in [0, 1).

    Returns:
        Quota netta (sempre > 1.0 se quota_exc > 1.0).
    """
    return 1.0 + (quota_exc - 1.0) * (1.0 - comm_rate)


def calcola_edge_back(prob_modello: float, q_netto: float) -> float:
    """
    Edge netto per il BACK (dopo commissione).

    edge = prob_modello - 1/Q_netto

    Un edge positivo significa che il modello vede più valore di quanto
    il mercato prezza, al netto dei costi di commissione.

    Args:
        prob_modello: Probabilità del modello.
        q_netto: Quota netta dopo commissione.

    Returns:
        Edge in [-1, 1]. Positivo = valore.
    """
    if q_netto <= 1.0:
        return -1.0
    return prob_modello - (1.0 / q_netto)


def calcola_edge_lay(
    prob_modello: float,
    quota_exc: float,
    comm_rate: float,
) -> float:
    """
    Edge netto per il LAY (commission-adjusted).

    edge_lay = p_BE - prob_modello
    dove p_BE = (1 - comm) / (Q - comm)

    Un edge positivo significa che il mercato sopravvaluta l'evento:
    il layer ha vantaggio rispetto al suo break-even commission-adjusted.

    Args:
        prob_modello: Probabilità del modello.
        quota_exc: Quota exchange nominale.
        comm_rate: Commissione in [0, 1).

    Returns:
        Edge lay in [-1, 1]. Positivo = valore per il layer.
    """
    denom = quota_exc - comm_rate
    if denom <= 0:
        return -1.0
    p_be = (1.0 - comm_rate) / denom
    return p_be - prob_modello


def calcola_ev_back(stake: float, prob_modello: float, q_netto: float) -> float:
    """EV atteso in euro per un back stake."""
    return stake * (prob_modello * q_netto - 1.0)


def calcola_ev_lay(
    stake_lay: float,
    prob_modello: float,
    quota_exc: float,
    comm_rate: float,
) -> float:
    """EV atteso in euro per un lay stake."""
    return stake_lay * ((1.0 - prob_modello) * (1.0 - comm_rate) - prob_modello * (quota_exc - 1.0))
