"""
signals.py — Generazione dei segnali di betting (back/lay).

Implementa la logica di valutazione dei segnali sia rapidi (senza quote exchange)
che avanzati (con quote exchange e Kelly criterion).

Tutto il flusso è ora senza stato globale: le funzioni restituiscono
liste di Signal invece di usare flag globali.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config import KELLY, SIGNALS
from src.engine import ExchangeQuotes  # noqa: F401 — used in type annotations
from src.models.kelly import (
    calcola_edge_back,
    calcola_edge_lay,
    calcola_ev_back,
    calcola_ev_lay,
    calcola_kelly_fraction,
    calcola_stake_kelly,
    calcola_stake_lay,
    quota_netta,
)

# ---------------------------------------------------------------------------
# Dataclass per un singolo segnale
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Rappresenta un singolo segnale di betting."""

    tipo: str            # "BACK" | "LAY" | "INFO_BACK" | "INFO_LAY"
    mercato: str         # Etichetta del mercato (es. "1 CASA", "Over 2.5")
    prob_mod: float      # Probabilità del modello
    quota_fair: float    # Quota fair del modello

    # Campi opzionali (presenti solo con quote exchange)
    quota_exc: float = 0.0
    quota_netta: float = 0.0
    edge: float = 0.0
    stake: float = 0.0
    liability: float = 0.0   # Solo per LAY
    ev_euro: float = 0.0
    riduzioni: list[str] = field(default_factory=list)

    @property
    def prob_implicita(self) -> float:
        return 1.0 / self.quota_exc if self.quota_exc > 0 else 0.0


# ---------------------------------------------------------------------------
# Calcolo soglie dinamiche
# ---------------------------------------------------------------------------

def calcola_soglie(
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
) -> dict[str, float]:
    """
    Calcola le soglie di probabilità dinamiche per tutti i mercati.

    Le soglie crescono col tempo per compensare la riduzione di incertezza
    e i costi di spread più elevati nelle fasi avanzate della partita.

    Args:
        minuto: Minuto attuale [0, 90].
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.

    Returns:
        Dict con le soglie per ogni mercato.
    """
    frac = minuto / 90.0
    # Sqrt scaling: l'informazione si accumula velocemente nei primi 45' (tattiche chiare,
    # dominio visibile) e lentamente dopo il 70' (conferma di quanto già osservato).
    # sqrt(0.5) = 0.71 vs 0.50 lineare → soglie più alte a metà partita (più selettivi),
    # sqrt(0.83) = 0.91 vs 0.83 lineare → quasi pieni a fine partita.
    frac_sqrt = math.sqrt(frac) if frac > 0 else 0.0
    base_1x2 = max(SIGNALS.SOGLIA_BACK_MIN, SIGNALS.SOGLIA_BACK_BASE + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt)
    base_ou = max(SIGNALS.OVER_BASE_MIN, SIGNALS.OVER_BASE_THRESHOLD + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt)

    gol_mancanti = max(0.0, linea_ou - gol_attuali)
    ou_gol_bonus = min(SIGNALS.OVER_GOL_BONUS_CAP, max(0.0, (gol_mancanti - 1.0) * SIGNALS.OVER_GOL_BONUS_RATE))

    return {
        "1x2": base_1x2,
        "btts_si": base_1x2,
        "btts_no": max(SIGNALS.SOGLIA_BTTS_NO_MIN, SIGNALS.SOGLIA_BTTS_NO_BASE + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt),
        "ou_over": base_ou + ou_gol_bonus,
        "ou_under": base_ou,
        "gol_mancanti": gol_mancanti,
    }


# ---------------------------------------------------------------------------
# Segnali rapidi (senza quote exchange)
# ---------------------------------------------------------------------------

def genera_segnali_rapidi(
    prob_1: float,
    prob_x: float,
    prob_2: float,
    prob_over: float,
    prob_under: float,
    prob_btts: float,
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
) -> list[Signal]:
    """
    Genera segnali rapidi basati sulle probabilità del modello, senza quote exchange.

    Richiede solo la probabilità del modello e la confronta con le soglie dinamiche.
    Utile per utenti senza accesso diretto alle quote dell'exchange.

    Args:
        prob_*: Probabilità del modello per ogni mercato.
        minuto: Minuto attuale.
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.

    Returns:
        Lista di Signal di tipo INFO_BACK o INFO_LAY.
    """
    soglie = calcola_soglie(minuto, linea_ou, gol_attuali)
    segnali: list[Signal] = []

    mercati = [
        ("1 Casa", prob_1, soglie["1x2"]),
        ("X Pareggio", prob_x, soglie["1x2"]),
        ("2 Trasf.", prob_2, soglie["1x2"]),
        (f"Over {linea_ou}", prob_over, soglie["ou_over"]),
        (f"Under {linea_ou}", prob_under, soglie["ou_under"]),
        ("BTTS Sì", prob_btts, soglie["btts_si"]),
        ("BTTS No", 1.0 - prob_btts, soglie["btts_no"]),
    ]

    for etichetta, prob, soglia_back in mercati:
        q_fair = 1.0 / prob if prob > SIGNALS.MIN_PROB_FOR_QUOTE else SIGNALS.MAX_QUOTE_FALLBACK

        # Skip eventi quasi certi
        if q_fair < SIGNALS.QUICK_SIGNAL_MIN_FAIR_Q:
            continue

        q_min_back = q_fair * (1.0 + SIGNALS.MARGINE_RAPIDO)
        q_max_lay = q_fair / (1.0 + SIGNALS.MARGINE_RAPIDO)

        if prob >= soglia_back:
            segnali.append(Signal(
                tipo="INFO_BACK",
                mercato=etichetta,
                prob_mod=prob,
                quota_fair=q_fair,
                quota_exc=q_min_back,  # Quota minima cercata
            ))
        elif prob <= SIGNALS.SOGLIA_LAY_MAX and q_fair >= SIGNALS.LAY_MIN_FAIR_Q:
            segnali.append(Signal(
                tipo="INFO_LAY",
                mercato=etichetta,
                prob_mod=prob,
                quota_fair=q_fair,
                quota_exc=q_max_lay,  # Quota massima cercata
            ))

    return segnali


# ---------------------------------------------------------------------------
# Segnali avanzati (con quote exchange)
# ---------------------------------------------------------------------------

def valuta_mercato(
    etichetta: str,
    prob_mod: float,
    q_exc: float,
    soglia_back: float,
    bankroll: float,
    comm_rate: float,
    kelly_frac: float,
    momentum_factor: float,
    back_only: bool = False,
    minuto: int = 0,
) -> Signal | None:
    """
    Valuta un singolo mercato con quota exchange.

    Calcola l'edge netto (dopo commissione) e genera un segnale
    BACK o LAY se l'edge supera la soglia minima.

    Args:
        etichetta: Nome del mercato.
        prob_mod: Probabilità del modello.
        q_exc: Quota sull'exchange (0 = non inserita).
        soglia_back: Soglia minima di probabilità per il back.
        bankroll: Capitale disponibile.
        comm_rate: Commissione exchange in [0, 1).
        kelly_frac: Frazione Kelly.
        momentum_factor: Fattore riduzione stake per momentum.
        back_only: Se True, non valuta il LAY.

    Returns:
        Signal se trovato valore, None altrimenti.
    """
    q_fair = 1.0 / prob_mod if prob_mod > SIGNALS.MIN_PROB_FOR_QUOTE else SIGNALS.MAX_QUOTE_FALLBACK

    # Skip eventi quasi certi
    if q_fair < SIGNALS.MIN_FAIR_Q:
        return None

    # Senza quota exchange: indicazione qualitativa con soglia adattiva al tempo
    if q_exc <= 1.0:
        frac_giocata = minuto / 90.0
        frac_sqrt = math.sqrt(frac_giocata) if frac_giocata > 0 else 0.0
        soglia_q = max(SIGNALS.SOGLIA_QUALITATIVA_MIN,
                       SIGNALS.SOGLIA_QUALITATIVA_BASE + SIGNALS.SOGLIA_QUALITATIVA_SLOPE * frac_sqrt)
        if prob_mod >= soglia_q:
            return Signal(
                tipo="INFO_BACK",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
            )
        if prob_mod <= SIGNALS.SOGLIA_LAY_MAX and not back_only:
            return Signal(
                tipo="INFO_LAY",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
            )
        return None

    # Con quota exchange: calcolo edge preciso
    q_net = quota_netta(q_exc, comm_rate)
    edge_back = calcola_edge_back(prob_mod, q_net)
    edge_lay = calcola_edge_lay(prob_mod, q_exc, comm_rate)

    # BACK
    if edge_back >= SIGNALS.MIN_EDGE_BACK and prob_mod >= soglia_back:
        stake_raw = calcola_stake_kelly(prob_mod, q_net, bankroll, kelly_frac, edge_back)
        stake = stake_raw * momentum_factor

        if stake > 0:
            ev = calcola_ev_back(stake, prob_mod, q_net)
            riduzioni = _build_riduzioni(comm_rate, momentum_factor, kelly_frac, q_net)
            return Signal(
                tipo="BACK",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
                quota_exc=q_exc,
                quota_netta=q_net,
                edge=edge_back,
                stake=stake,
                ev_euro=ev,
                riduzioni=riduzioni,
            )

    # LAY
    if not back_only and edge_lay >= SIGNALS.MIN_EDGE_LAY and q_exc >= KELLY.LAY_MIN_ODDS:
        result = calcola_stake_lay(prob_mod, q_exc, bankroll, kelly_frac, comm_rate)
        if result is not None:
            stake_lay, liab_lay = result
            stake_lay *= momentum_factor
            liab_lay *= momentum_factor
            ev = calcola_ev_lay(stake_lay, prob_mod, q_exc, comm_rate)
            riduzioni = _build_riduzioni(0.0, momentum_factor, kelly_frac, None)
            return Signal(
                tipo="LAY",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
                quota_exc=q_exc,
                edge=edge_lay,
                stake=stake_lay,
                liability=liab_lay,
                ev_euro=ev,
                riduzioni=riduzioni,
            )

    return None


def _build_riduzioni(
    comm_rate: float,
    momentum_factor: float,
    kelly_frac: float,
    q_net: float | None,
) -> list[str]:
    """Costruisce la lista testuale delle riduzioni applicate alla stake."""
    riduzioni = []
    if comm_rate > 0 and q_net is not None:
        riduzioni.append(f"comm {comm_rate*100:.1f}% → @{q_net:.3f} netto")
    if momentum_factor < 1.0:
        riduzioni.append(f"momentum ×{momentum_factor:.2f}")
    if kelly_frac < KELLY.KELLY_BASE_FRACTION:
        riduzioni.append(f"Kelly ×{kelly_frac:.2f}")
    return riduzioni


def genera_segnali_avanzati(
    prob_1: float,
    prob_x: float,
    prob_2: float,
    prob_over: float,
    prob_under: float,
    prob_btts: float,
    quotes: ExchangeQuotes,
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
    bankroll: float,
    comm_rate: float,
    n_shots_tot: int,
    momentum: float,
) -> list[Signal]:
    """
    Genera segnali avanzati con quote exchange, Kelly criterion e EV.

    Args:
        prob_*: Probabilità del modello.
        quotes: Quote dall'exchange.
        minuto: Minuto attuale.
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.
        bankroll: Capitale disponibile.
        comm_rate: Commissione exchange.
        n_shots_tot: Numero tiri totali inseriti.
        momentum: Indice momentum di mercato.

    Returns:
        Lista di Signal con calcoli Kelly/EV completi.
    """
    soglie = calcola_soglie(minuto, linea_ou, gol_attuali)
    kelly_frac = calcola_kelly_fraction(minuto, n_shots_tot)
    momentum_factor = max(
        SIGNALS.MOMENTUM_STAKE_FLOOR,
        1.0 - SIGNALS.MOMENTUM_STAKE_REDUCTION_RATE * max(0.0, momentum - SIGNALS.MOMENTUM_STAKE_THRESHOLD),
    )
    segnali: list[Signal] = []

    def _valuta(etichetta: str, prob: float, q_exc: float, soglia: float, back_only: bool = False) -> None:
        s = valuta_mercato(
            etichetta, prob, q_exc, soglia,
            bankroll, comm_rate, kelly_frac, momentum_factor, back_only,
            minuto=minuto,
        )
        if s is not None:
            segnali.append(s)

    # 1X2
    _valuta("1 Casa", prob_1, quotes.q_1, soglie["1x2"])
    _valuta("2 Trasf.", prob_2, quotes.q_2, soglie["1x2"])
    if quotes.q_x > 1.0:
        _valuta("X Pareggio", prob_x, quotes.q_x, soglie["1x2"])

    # Over/Under
    # LAY Over = scommettere che non arriveranno abbastanza gol = equivalente a BACK Under.
    # Per evitare segnali doppi (LAY Over + BACK Under per lo stesso scenario), il LAY
    # Over è disabilitato in condizioni normali. Eccezione: late game (>75') con ≥2 gol
    # mancanti, dove la liquidità del mercato Over è superiore e il LAY è più efficiente.
    gol_mancanti = soglie["gol_mancanti"]
    if minuto >= SIGNALS.LATE_GAME_LAY_OVER_MINUTE and gol_mancanti >= SIGNALS.LATE_GAME_LAY_OVER_GOALS:
        _valuta(f"Over {linea_ou}", prob_over, quotes.q_over, soglie["ou_over"], back_only=False)
    else:
        _valuta(f"Over {linea_ou}", prob_over, quotes.q_over, soglie["ou_over"], back_only=True)
    _valuta(f"Under {linea_ou}", prob_under, quotes.q_under, soglie["ou_under"])

    # BTTS
    _valuta("BTTS Sì", prob_btts, quotes.q_btts_si, soglie["btts_si"])
    _valuta("BTTS No", 1.0 - prob_btts, quotes.q_btts_no, soglie["btts_no"])

    return segnali
