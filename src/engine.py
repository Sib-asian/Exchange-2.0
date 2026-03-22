"""
engine.py — Orchestratore centrale del motore di analisi probabilistica.

Coordina l'intera pipeline:
  1. Calibrazione xG (Bayesiana + blend tiri)
  2. Aggiustamenti time-decay / score effect / cartellini
  3. Costruzione matrice bivariata
  4. Calcolo probabilità per tutti i mercati

Tutti i dati di input e output sono tipizzati con dataclass per
garantire validazione, leggibilità e testabilità.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclass di Input
# ---------------------------------------------------------------------------

@dataclass
class MatchState:
    """Stato attuale della partita."""

    minuto: int          # Minuto [0, 90]
    gol_casa: int        # Gol attuali casa [0, +inf)
    gol_trasf: int       # Gol attuali trasferta
    rossi_casa: int      # Cartellini rossi casa [0, 4]
    rossi_trasf: int     # Cartellini rossi trasferta

    # Linee asiatiche
    ah_op: float         # AH apertura (full 90')
    tot_op: float        # Total apertura
    ah_cur: float        # AH corrente (gol rimanenti)
    tot_cur: float       # Total corrente (gol rimanenti)

    # Linea Over/Under da analizzare
    linea_ou: float

    # Tiri live (0 = non disponibili)
    sot_h: int = 0       # Tiri in porta casa
    soff_h: int = 0      # Tiri fuori casa
    sot_a: int = 0       # Tiri in porta trasferta
    soff_a: int = 0      # Tiri fuori trasferta

    # Bankroll e commissione
    bankroll: float = 1000.0
    comm_rate: float = 0.025   # 2.5%

    def __post_init__(self) -> None:
        """Validazione dei campi."""
        assert 0 <= self.minuto <= 90, f"Minuto fuori range: {self.minuto}"
        assert self.gol_casa >= 0, f"Gol casa negativo: {self.gol_casa}"
        assert self.gol_trasf >= 0, f"Gol trasf negativo: {self.gol_trasf}"
        assert 0 <= self.rossi_casa <= 4, f"Rossi casa fuori range: {self.rossi_casa}"
        assert 0 <= self.rossi_trasf <= 4, f"Rossi trasf fuori range: {self.rossi_trasf}"
        assert self.tot_op > 0, f"Total apertura non valido: {self.tot_op}"
        assert self.tot_cur > 0, f"Total corrente non valido: {self.tot_cur}"
        assert self.bankroll > 0, f"Bankroll non valido: {self.bankroll}"
        assert 0.0 <= self.comm_rate < 1.0, f"Commissione non valida: {self.comm_rate}"


@dataclass
class ExchangeQuotes:
    """Quote opzionali dall'exchange per l'analisi avanzata."""

    q_1: float = 0.0
    q_x: float = 0.0
    q_2: float = 0.0
    q_over: float = 0.0
    q_under: float = 0.0
    q_btts_si: float = 0.0
    q_btts_no: float = 0.0

    @property
    def any_active(self) -> bool:
        """True se almeno una quota è stata inserita (> 1.0)."""
        return any(
            q > 1.0
            for q in [self.q_1, self.q_x, self.q_2, self.q_over, self.q_under,
                       self.q_btts_si, self.q_btts_no]
        )


# ---------------------------------------------------------------------------
# Dataclass di Output
# ---------------------------------------------------------------------------

@dataclass
class ProbabilitaModello:
    """Probabilità calcolate dal modello per tutti i mercati."""

    # 1X2
    p1: float
    px: float
    p2: float

    # Over/Under
    p_under: float
    p_over: float

    # BTTS
    p_btts: float

    # Correct Score (top N)
    top_cs: list[tuple[tuple[int, int], float]]

    # Distribuzione gol totali
    gol_tot_dist: dict[int, float]

    # Parametri interni
    rho: float
    xg_h_final: float      # xG casa dopo tutti gli aggiustamenti
    xg_a_final: float      # xG trasferta dopo tutti gli aggiustamenti
    xg_h_base: float       # xG casa da linee (pre-blend)
    xg_a_base: float       # xG trasferta da linee
    xg_h_blend: float      # xG casa dopo blend tiri
    xg_a_blend: float      # xG trasferta dopo blend tiri
    xg_h_accum: float      # xG accumulato dai tiri (debug)
    xg_a_accum: float
    alpha_t: float         # Peso T nel blend
    alpha_d: float         # Peso D nel blend
    shot_dom: float        # Indice dominio tiri
    momentum: float        # Indice momentum mercato
    flat_lines: bool       # Linee di mercato invariate
    model_confidence: float = 0.0  # Score [0, 1] di fiducia del modello

    # Matrici
    joint_ind: dict[tuple[int, int], float] = field(default_factory=dict)
    full_matrix: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine principale
# ---------------------------------------------------------------------------

def analizza(
    state: MatchState,
) -> ProbabilitaModello:
    """
    Esegue l'intera pipeline di analisi.

    Pipeline:
    1. Calibrazione xG Bayesiana dalle linee AH + Total
    2. Blend xG con dati tiri (se disponibili)
    3. Aggiustamenti time-decay / score effect / cartellini rossi
    4. Costruzione matrice bivariata (Poisson + DC + Z)
    5. Calcolo probabilità 1X2, Over/Under, BTTS, Correct Score, AH

    Args:
        state: MatchState validato con tutti i dati di input.

    Returns:
        ProbabilitaModello con tutte le probabilità e i parametri interni.
    """
    from src.config import BAYES, UI
    from src.markets.btts import calcola_btts
    from src.markets.over_under import calcola_over_under
    from src.markets.result import calcola_1x2, calcola_correct_score
    from src.models.calibration import blend_xg_shots, calcola_xg_bayesiani
    from src.models.poisson import build_bivariate_matrix
    from src.models.time_decay import calcola_momentum_mercato, time_decay_dinamico

    # 1. xG da linee (prior bayesiano)
    xg_h_base, xg_a_base = calcola_xg_bayesiani(
        state.ah_op, state.tot_op,
        state.ah_cur, state.tot_cur,
        state.minuto,
        gol_diff=state.gol_casa - state.gol_trasf,
        gol_tot=state.gol_casa + state.gol_trasf,
    )

    # 2. Blend tiri + linee (solo se ci sono tiri inseriti)
    n_shots_tot = state.sot_h + state.soff_h + state.sot_a + state.soff_a
    if n_shots_tot > 0 and state.minuto > 0:
        xg_h_blend, xg_a_blend, xg_h_accum, xg_a_accum, alpha_t, alpha_d, shot_dom = blend_xg_shots(
            xg_h_base, xg_a_base,
            state.sot_h, state.soff_h,
            state.sot_a, state.soff_a,
            state.gol_casa, state.gol_trasf,
            state.minuto,
        )
    else:
        xg_h_blend = xg_h_base
        xg_a_blend = xg_a_base
        xg_h_accum = xg_a_accum = 0.0
        alpha_t = alpha_d = shot_dom = 0.0

    # 3. Momentum mercato (calcolato PRIMA del time-decay per alimentare lo smorzamento xG)
    # ah_cur è in "gol rimanenti" (conversione full-game già applicata in inputs.py):
    #   ah_cur = ah_cur_full + (gol_casa - gol_trasf)
    # Per il momentum vogliamo il movimento PURO del mercato, eliminando l'effetto
    # meccanico del punteggio. Riconvertiamo entrambe le quantità in full-game:
    gol_diff = state.gol_casa - state.gol_trasf
    gol_tot_scored = state.gol_casa + state.gol_trasf
    ah_cur_full = state.ah_cur - gol_diff          # rimuove l'offset del punteggio
    tot_cur_full = state.tot_cur + gol_tot_scored   # ripristina il totale full-game
    delta_ah = ah_cur_full - state.ah_op            # variazione pura del mercato AH
    delta_tot = tot_cur_full - state.tot_op         # variazione pura del mercato Total
    momentum = calcola_momentum_mercato(delta_ah, delta_tot, state.minuto)
    flat_lines = abs(delta_ah) < BAYES.FLAT_LINE_THRESHOLD and abs(delta_tot) < BAYES.FLAT_LINE_THRESHOLD

    # 4. Time decay + score effect + rossi + momentum dampening
    xg_h_final, xg_a_final = time_decay_dinamico(
        xg_h_blend, xg_a_blend,
        state.minuto,
        state.gol_casa, state.gol_trasf,
        state.rossi_casa, state.rossi_trasf,
        momentum=momentum,
    )

    # 5. Matrice bivariata
    joint_ind, full_matrix, rho = build_bivariate_matrix(
        xg_h_final, xg_a_final,
        state.minuto,
        state.tot_cur,
        shot_dom=shot_dom,
        gol_totali=state.gol_casa + state.gol_trasf,
    )

    # 6. Probabilità mercati
    p1, px, p2 = calcola_1x2(joint_ind, state.gol_casa, state.gol_trasf)
    p_under, p_over = calcola_over_under(full_matrix, state.gol_casa + state.gol_trasf, state.linea_ou)
    p_btts = calcola_btts(full_matrix, state.gol_casa, state.gol_trasf)
    top_cs, gol_tot_dist = calcola_correct_score(full_matrix, state.gol_casa, state.gol_trasf, UI.TOP_CS_COUNT)

    # 7. Model confidence score [0, 1]
    # Combina: qualità dati tiri, movimento linee, blend weights, tempo giocato.
    # Score basso = modello opera su prior, alta incertezza.
    # Score alto = dati abbondanti, linee attive, blend maturo.
    import math as _math
    _shots_conf = min(1.0, n_shots_tot / 15.0)
    _line_conf = 0.0 if flat_lines else 1.0
    _blend_conf = (alpha_t + alpha_d) / 2.0 if n_shots_tot > 0 else 0.0
    _time_conf = _math.sqrt(state.minuto / 90.0) if state.minuto > 0 else 0.0
    # Media geometrica: se un componente è zero, il confidence è basso
    _product = max(1e-9, _shots_conf * max(0.01, _line_conf) * max(0.01, _blend_conf) * max(0.01, _time_conf))
    model_confidence = min(1.0, _product ** 0.25)

    return ProbabilitaModello(
        p1=p1, px=px, p2=p2,
        p_under=p_under, p_over=p_over,
        p_btts=p_btts,
        top_cs=top_cs,
        gol_tot_dist=gol_tot_dist,
        rho=rho,
        xg_h_final=xg_h_final, xg_a_final=xg_a_final,
        xg_h_base=xg_h_base, xg_a_base=xg_a_base,
        xg_h_blend=xg_h_blend, xg_a_blend=xg_a_blend,
        xg_h_accum=xg_h_accum, xg_a_accum=xg_a_accum,
        alpha_t=alpha_t, alpha_d=alpha_d,
        shot_dom=shot_dom,
        momentum=momentum,
        flat_lines=flat_lines,
        model_confidence=model_confidence,
        joint_ind=joint_ind,
        full_matrix=full_matrix,
    )
