"""
engine.py — Orchestratore centrale del motore di analisi probabilistica.

Coordina l'intera pipeline:
  1. Calibrazione xG (Bayesiana + blend tiri)
  2. Aggiustamenti time-decay / score effect / cartellini
  3. Costruzione matrice bivariata
  4. Calcolo probabilità per tutti i mercati

Tutti i dati di input e output sono tipizzati con dataclass per
garantire validazione, leggibilità e testabilità.

Performance:
  - I 3 modelli (Poisson, Copula, Markov) sono eseguiti in parallelo
  - I risultati sono cached per evitare ricalcoli inutili
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.config import BAYES, CACHE, ENGINE

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

    # Statistiche live avanzate da screenshot (0 = non disponibili)
    corner_h: int = 0
    corner_a: int = 0
    possesso_h: float = 0.0   # Percentuale (0-100)
    possesso_a: float = 0.0
    att_pericolosi_h: int = 0  # Dangerous attacks
    att_pericolosi_a: int = 0
    att_totali_h: int = 0      # Total attacks (att. normali + pericolosi)
    att_totali_a: int = 0
    tiri_bloccati_h: int = 0   # #2: Blocked shots
    tiri_bloccati_a: int = 0
    gialli_casa: int = 0       # #4/#9: Yellow cards
    gialli_trasf: int = 0
    falli_casa: int = 0        # #5: Fouls
    falli_trasf: int = 0

    # Probabilità implicite OCR prematch (0.0 = non disponibili)
    # #1 #6 #12: Quote lette dallo screenshot prematch
    ocr_p1: float = 0.0
    ocr_px: float = 0.0
    ocr_p2: float = 0.0
    ocr_p_over: float = 0.0
    ocr_p_under: float = 0.0
    ocr_p_btts: float = 0.0
    ocr_confidence: str = ""   # "high", "medium", "low", "" = non disponibile

    # Bankroll e commissione
    bankroll: float = 1000.0
    comm_rate: float = 0.025   # 2.5%

    def __post_init__(self) -> None:
        """Validazione dei campi con eccezioni proper (Fix #3.6)."""
        # NOTA: In produzione con -O gli assert vengono disabilitati,
        # quindi usiamo ValueError per validazione critica
        if not (0 <= self.minuto <= 90):
            raise ValueError(f"Minuto fuori range: {self.minuto}")
        if self.gol_casa < 0:
            raise ValueError(f"Gol casa negativo: {self.gol_casa}")
        if self.gol_trasf < 0:
            raise ValueError(f"Gol trasf negativo: {self.gol_trasf}")
        if not (0 <= self.rossi_casa <= 4):
            raise ValueError(f"Rossi casa fuori range: {self.rossi_casa}")
        if not (0 <= self.rossi_trasf <= 4):
            raise ValueError(f"Rossi trasf fuori range: {self.rossi_trasf}")
        if self.tot_op <= 0:
            raise ValueError(f"Total apertura non valido: {self.tot_op}")
        if self.tot_cur <= 0:
            raise ValueError(f"Total corrente non valido: {self.tot_cur}")
        if self.bankroll <= 0:
            raise ValueError(f"Bankroll non valido: {self.bankroll}")
        if not (0.0 <= self.comm_rate < 1.0):
            raise ValueError(f"Commissione non valida: {self.comm_rate}")


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
    stale_line: bool = False       # True se la linea corrente è stantia
    model_agreement: float = 1.0   # [0,1] accordo tra i 3 modelli del consensus
    market_divergence: float = 0.0 # Divergenza modello-mercato (proxy Brier)
    delta_ah: float = 0.0          # Variazione pura AH (market movement)
    delta_tot: float = 0.0         # Variazione pura Total (market movement)

    # FIX: Campo per indicare linee probabilmente non aggiornate
    # True se ci sono gol ma le linee sembrano ancora quelle d'apertura
    lines_need_update: bool = False

    # #12: Divergenza quote OCR vs modello (mercato → modello)
    # Dict {mercato: (p_model, p_ocr, divergenza)} per mercati con divergenza > soglia
    ocr_divergence: dict = field(default_factory=dict)

    # Intervalli di credibilità multi-modello
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Matrici
    joint_ind: dict[tuple[int, int], float] = field(default_factory=dict)
    full_matrix: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions per parallelizzazione
# ---------------------------------------------------------------------------

def _compute_bivariate_model(
    xg_h: float,
    xg_a: float,
    minuto: int,
    tot_cur: float,
    shot_dom: float,
    gol_totali: int,
    rho_dc: float,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], float]:
    """Wrapper per calcolo modello bivariate Poisson (per parallelizzazione)."""
    from src.models.poisson import build_bivariate_matrix
    return build_bivariate_matrix(
        xg_h, xg_a, minuto, tot_cur,
        shot_dom=shot_dom, gol_totali=gol_totali, rho_dc_preset=rho_dc
    )


def _compute_copula_model(
    xg_h: float,
    xg_a: float,
    copula_theta: float,
    nu: float,
) -> dict[tuple[int, int], float]:
    """Wrapper per calcolo modello CMP + Copula (per parallelizzazione)."""
    from src.models.copula import build_copula_matrix
    return build_copula_matrix(xg_h, xg_a, copula_theta, nu=nu)


def _compute_markov_model(
    xg_h: float,
    xg_a: float,
    minuto: int,
    gol_h: int,
    gol_a: int,
    rho_dc: float,
) -> dict[tuple[int, int], float]:
    """Wrapper per calcolo modello Markov chain (per parallelizzazione)."""
    from src.models.markov import markov_score_distribution
    return markov_score_distribution(
        xg_h, xg_a, minuto, gol_h, gol_a, rho_dc=rho_dc
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_consensus_matrix(
    full_bp: dict[tuple[int, int], float],
    full_copula: dict[tuple[int, int], float],
    full_markov: dict[tuple[int, int], float],
    w_bp: float,
    w_cop: float,
    w_mk: float,
) -> dict[tuple[int, int], float]:
    """
    Costruisce la matrice consensus come media pesata delle 3 matrici.

    Fix #2.7: Funzione helper per evitare duplicazione di codice.
    Usata per coerenza con le probabilità 1X2/OU/BTTS del consensus.

    Args:
        full_bp: Matrice dal modello bivariate Poisson + DC.
        full_copula: Matrice dal modello CMP + Frank copula.
        full_markov: Matrice dal Markov chain score-state.
        w_bp, w_cop, w_mk: Pesi dei modelli (somma = 1).

    Returns:
        Matrice consensus normalizzata.
    """
    all_keys = set(full_bp.keys()) | set(full_copula.keys()) | set(full_markov.keys())
    consensus: dict[tuple[int, int], float] = {}

    for key in all_keys:
        p = (w_bp * full_bp.get(key, 0.0)
             + w_cop * full_copula.get(key, 0.0)
             + w_mk * full_markov.get(key, 0.0))
        if p > 0:
            consensus[key] = p

    total = sum(consensus.values())
    if total > 0:
        consensus = {k: v / total for k, v in consensus.items()}

    return consensus


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
    from src.config import CMP, CONSENSUS, COPULA, MOMENTUM, STALE, UI
    from src.markets.result import calcola_correct_score
    from src.models.calibration import blend_xg_shots, calcola_xg_bayesiani
    from src.models.consensus import (
        calibrate_probabilities,
        compute_consensus,
        compute_model_credible_intervals,
    )
    from src.models.time_decay import calcola_momentum_mercato, time_decay_dinamico

    # Cap temporale: i gol rimanenti non possono superare ~(90-minuto)/90 * 4.0.
    # Protegge dal caso frequente in cui l'utente cambia il punteggio/minuto ma
    # dimentica di aggiornare le linee live → tot_cur rimane il valore full-game
    # d'apertura (es. 2.75) che diventa insensato come "gol rimanenti" al 80'.
    # Esempio: 0-0 al 80', default tot_cur=2.75 → cap a 0.44 (= 4.0 × 10/90).
    _mins_rem = max(1, 90 - state.minuto)
    _tot_cap = max(BAYES.TOT_BAYES_MIN, _mins_rem / 90.0 * BAYES.TOT_TEMPORAL_MAX)
    tot_cur_eff = min(state.tot_cur, _tot_cap)

    # 1. xG da linee (prior bayesiano)
    xg_h_base, xg_a_base = calcola_xg_bayesiani(
        state.ah_op, state.tot_op,
        state.ah_cur, tot_cur_eff,
        state.minuto,
        gol_diff=state.gol_casa - state.gol_trasf,
        gol_tot=state.gol_casa + state.gol_trasf,
    )

    # 1b. #1 — OCR calibratore prematch: blenda xG base con probabilità implicite OCR.
    # Solo a minuto=0 con OCR disponibile: le 1X2 OCR forniscono un secondo prior
    # indipendente dalle linee AH/Total, utile quando le linee asiatiche sono illiquide.
    # A minuto>0 le linee live sono molto più informative → OCR ignorato.
    _ocr_available = (state.minuto == 0
                      and state.ocr_p1 > 0 and state.ocr_p2 > 0
                      and state.ocr_p1 + state.ocr_px + state.ocr_p2 > 0.5)
    if _ocr_available:
        from src.config import OCR_CALIB
        # Le prob OCR sono già normalizzate (tolto overround nell'inputs.py).
        # Stima implicita xG da prob 1X2: usa solo il ratio p1/(p1+p2) per estrarre
        # il differenziale relativo, mantenendo il totale dal modello AH/Total.
        _sum_12 = state.ocr_p1 + state.ocr_p2
        if _sum_12 > 0.1:
            # Rapporto forza implicita: p1/(p1+p2) → indica quanto xG_h deve superare xG_a
            _ocr_strength_ratio = state.ocr_p1 / _sum_12  # [0,1], 0.5=equilibrio
            # Converti in differenziale: -0.5 a +0.5 normalizzato
            _ocr_delta_norm = _ocr_strength_ratio - 0.5   # positivo = casa favorita
            # xG impliciti dal totale AH (già calcolato) con differenziale OCR
            _tot_base = xg_h_base + xg_a_base
            # Differenziale OCR: scala rispetto al totale (un totale di 2.5 con ratio 0.6/0.4
            # → delta ≈ 0.1 × tot = 0.25 extra per la casa)
            _ocr_delta = _ocr_delta_norm * _tot_base * 1.0  # 1.0 = scale factor
            _xg_h_ocr = max(BAYES.TOT_BAYES_MIN, 0.5 * (_tot_base + _ocr_delta))
            _xg_a_ocr = max(BAYES.TOT_BAYES_MIN, 0.5 * (_tot_base - _ocr_delta))
            # Blend: OCR_CALIB.XG_OCR_BLEND_WEIGHT verso OCR, resto dal modello
            _w = OCR_CALIB.XG_OCR_BLEND_WEIGHT
            xg_h_base = (1.0 - _w) * xg_h_base + _w * _xg_h_ocr
            xg_a_base = (1.0 - _w) * xg_a_base + _w * _xg_a_ocr

    # 2. Blend tiri + linee (solo se ci sono tiri inseriti)
    n_shots_tot = (state.sot_h + state.soff_h + state.sot_a + state.soff_a
                   + state.tiri_bloccati_h + state.tiri_bloccati_a)
    if n_shots_tot > 0 and state.minuto > 0:
        xg_h_blend, xg_a_blend, xg_h_accum, xg_a_accum, alpha_t, alpha_d, shot_dom = blend_xg_shots(
            xg_h_base, xg_a_base,
            state.sot_h, state.soff_h,
            state.sot_a, state.soff_a,
            state.gol_casa, state.gol_trasf,
            state.minuto,
            corner_h=state.corner_h,
            corner_a=state.corner_a,
            possesso_h=state.possesso_h,
            possesso_a=state.possesso_a,
            att_pericolosi_h=state.att_pericolosi_h,
            att_pericolosi_a=state.att_pericolosi_a,
            att_totali_h=state.att_totali_h,
            att_totali_a=state.att_totali_a,
            tiri_bloccati_h=state.tiri_bloccati_h,
            tiri_bloccati_a=state.tiri_bloccati_a,
            falli_h=state.falli_casa,
            falli_a=state.falli_trasf,
            tot_op=state.tot_op,
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

    # FIX: Il momentum deve riflettere ANCHE l'attività della partita!
    # Se ci sono gol ma le linee non si sono mosse, c'è comunque "momentum"
    # perché il mercato sta reagendo (o dovrebbe reagire).
    # Aggiungiamo un "momentum da gol" che aumenta con i gol segnati.
    momentum_from_goals = min(1.5, gol_tot_scored * 0.3) if gol_tot_scored > 0 else 0.0

    momentum_market = calcola_momentum_mercato(delta_ah, delta_tot, state.minuto) + momentum_from_goals

    # #10 — Momentum statistico da attacchi/tiri blendato con momentum mercato.
    # Se un team domina nettamente in attacchi/tiri, il momentum statistico
    # amplifica (o smorza) il segnale di mercato.
    from src.config import STAT_MOM
    _stat_momentum = 0.0
    if state.minuto >= STAT_MOM.MIN_MINUTE:
        _att_per_tot = state.att_pericolosi_h + state.att_pericolosi_a
        _sot_tot_live = state.sot_h + state.sot_a
        # Usa att_pericolosi se disponibili, altrimenti SOT
        if _att_per_tot > 0:
            _stat_dom = (state.att_pericolosi_h - state.att_pericolosi_a) / _att_per_tot
        elif _sot_tot_live > 0:
            _stat_dom = (state.sot_h - state.sot_a) / _sot_tot_live
        else:
            _stat_dom = 0.0
        # Momentum statistico: valore assoluto (intensità di dominanza, non direzione)
        _stat_momentum = abs(_stat_dom) * STAT_MOM.STAT_SCALE

    momentum = min(
        MOMENTUM.MOMENTUM_CAP,
        (1.0 - STAT_MOM.STAT_WEIGHT) * momentum_market + STAT_MOM.STAT_WEIGHT * _stat_momentum,
    )

    flat_lines = abs(delta_ah) < BAYES.FLAT_LINE_THRESHOLD and abs(delta_tot) < BAYES.FLAT_LINE_THRESHOLD

    # 4. Time decay + score effect + rossi + momentum dampening
    # #8: Bypass score effect a minuto=0 — a prematch diff=0 quindi già bypassato,
    # ma lo passiamo come gol_casa=gol_trasf=0 esplicito per sicurezza.
    _td_gol_casa = state.gol_casa if state.minuto > 0 else 0
    _td_gol_trasf = state.gol_trasf if state.minuto > 0 else 0
    xg_h_final, xg_a_final = time_decay_dinamico(
        xg_h_blend, xg_a_blend,
        state.minuto,
        _td_gol_casa, _td_gol_trasf,
        state.rossi_casa, state.rossi_trasf,
        momentum=momentum,
        delta_ah=delta_ah,
    )

    # #4 — Gialli → fattore rischio rosso + rallentamento ritmo.
    # Applicato DOPO il time-decay per non interferire con score effect / rossi.
    from src.config import YELLOWS
    _gialli_tot = state.gialli_casa + state.gialli_trasf
    if _gialli_tot > 0 and state.minuto > 0:
        # Riduzione ritmo: molti gialli = gioco più tattico/spezzettato
        _tempo_red = min(YELLOWS.TEMPO_CAP,
                         max(0.0, _gialli_tot - YELLOWS.TEMPO_THRESHOLD) * YELLOWS.TEMPO_RATE)
        if _tempo_red > 0:
            xg_h_final *= (1.0 - _tempo_red)
            xg_a_final *= (1.0 - _tempo_red)
        # Rischio rosso futuro: squadra con più gialli rischia l'espulsione →
        # leggero abbassamento xG offensivo (più prudente in attacco)
        _gialli_h = state.gialli_casa
        _gialli_a = state.gialli_trasf
        if _gialli_h > 1:
            _red_risk_h = min(YELLOWS.RED_RISK_CAP, (_gialli_h - 1) * YELLOWS.RED_RISK_PER_YELLOW)
            xg_h_final *= (1.0 - _red_risk_h)
        if _gialli_a > 1:
            _red_risk_a = min(YELLOWS.RED_RISK_CAP, (_gialli_a - 1) * YELLOWS.RED_RISK_PER_YELLOW)
            xg_a_final *= (1.0 - _red_risk_a)

    # 4b. Stale line detection: se le linee non si sono mosse dopo >15 minuti
    # di partita, il mercato è potenzialmente illiquido o sospeso.
    stale_line = flat_lines and state.minuto >= STALE.THRESHOLD_MINUTES

    # 5. Calcola parametri condivisi tra i modelli
    from src.models.poisson import rho_dc_dinamico as _calc_rho_dc
    # #9: Passa gialli_tot a rho_dc_dinamico per aumentare la correlazione negativa
    _rho_dc_shared = _calc_rho_dc(
        tot_cur_eff, state.minuto,
        state.gol_casa + state.gol_trasf,
        gialli_tot=state.gialli_casa + state.gialli_trasf,
    )

    # Parametri per il modello Copula
    frac_giocata = state.minuto / 90.0
    copula_theta = max(0.1, COPULA.THETA_BASE
                       + COPULA.THETA_TOT_SCALE * max(0.0, tot_cur_eff - COPULA.THETA_TOT_REF)
                       + COPULA.THETA_TIME_SCALE * frac_giocata)
    nu_dynamic = max(CMP.NU_MIN, min(CMP.NU_MAX,
                                     CMP.NU + CMP.NU_TOT_SCALE * (tot_cur_eff - CMP.NU_TOT_REF)))

    # 6. Esegui i 3 modelli in parallelo per performance
    # Ogni modello è computazionalmente indipendente → parallelizzazione efficace
    full_bp: dict[tuple[int, int], float] = {}
    full_copula: dict[tuple[int, int], float] = {}
    full_markov: dict[tuple[int, int], float] = {}
    joint_ind: dict[tuple[int, int], float] = {}
    rho = 0.0

    if CACHE.ENABLED:
        # Usa cache se abilitato
        from src.models.cache import get_matrix_cache
        cache = get_matrix_cache()

        # FIX: Aggiungi gol_totali alla chiave della cache per invalidare
        # quando il punteggio cambia
        gol_totali = state.gol_casa + state.gol_trasf

        joint_ind, full_bp, rho = cache.get_or_compute(
            lambda: _compute_bivariate_model(
                xg_h_final, xg_a_final, state.minuto, tot_cur_eff,
                shot_dom, gol_totali, _rho_dc_shared
            ),
            "bivariate", xg_h_final, xg_a_final, state.minuto, tot_cur_eff, gol_totali
        )
        full_copula = cache.get_or_compute(
            lambda: _compute_copula_model(xg_h_final, xg_a_final, copula_theta, nu_dynamic),
            "copula", xg_h_final, xg_a_final, copula_theta, nu_dynamic
        )
        full_markov = cache.get_or_compute(
            lambda: _compute_markov_model(
                xg_h_final, xg_a_final, state.minuto,
                state.gol_casa, state.gol_trasf, _rho_dc_shared
            ),
            "markov", xg_h_final, xg_a_final, state.minuto, state.gol_casa, state.gol_trasf
        )
    else:
        # Esecuzione parallela senza cache
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _compute_bivariate_model,
                    xg_h_final, xg_a_final, state.minuto, tot_cur_eff,
                    shot_dom, state.gol_casa + state.gol_trasf, _rho_dc_shared
                ): "bivariate",
                executor.submit(
                    _compute_copula_model,
                    xg_h_final, xg_a_final, copula_theta, nu_dynamic
                ): "copula",
                executor.submit(
                    _compute_markov_model,
                    xg_h_final, xg_a_final, state.minuto,
                    state.gol_casa, state.gol_trasf, _rho_dc_shared
                ): "markov",
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    if model_name == "bivariate":
                        joint_ind, full_bp, rho = result
                    elif model_name == "copula":
                        full_copula = result
                    else:  # markov
                        full_markov = result
                except Exception as e:
                    # Fallback: se un modello fallisce, usa gli altri
                    import logging
                    logging.getLogger("exchange.engine").warning(
                        f"Model {model_name} failed: {e}"
                    )

    # 8. Consensus multi-modello: media pesata
    consensus_probs = compute_consensus(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, state.linea_ou,
        weights=(CONSENSUS.W_BIVARIATE, CONSENSUS.W_COPULA, CONSENSUS.W_MARKOV),
    )

    # 9. Calibrazione isotonica
    p1, px, p2, p_over, p_under, p_btts = calibrate_probabilities(
        consensus_probs["p1"], consensus_probs["px"], consensus_probs["p2"],
        consensus_probs["p_over"], consensus_probs["p_under"], consensus_probs["p_btts"],
        draw_shrinkage=CONSENSUS.DRAW_SHRINKAGE,
    )

    # 10. Correct score e distribuzione gol dal consensus (Fix #5).
    # Fix #2.7: Usa funzione helper per costruire la matrice consensus
    full_consensus = _build_consensus_matrix(full_bp, full_copula, full_markov,
                                              CONSENSUS.W_BIVARIATE, CONSENSUS.W_COPULA, CONSENSUS.W_MARKOV)
    full_matrix = full_consensus if full_consensus else full_bp
    top_cs, gol_tot_dist = calcola_correct_score(full_matrix, state.gol_casa, state.gol_trasf, UI.TOP_CS_COUNT)

    # 11. Intervalli di credibilità multi-modello
    credible_intervals = compute_model_credible_intervals(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, state.linea_ou,
    )

    # 12. Accordo tra modelli: se i 3 modelli concordano, agreement → 1.0
    import math as _math
    _spreads = [hi - lo for lo, hi in credible_intervals.values() if hi > lo]
    model_agreement = max(0.0, 1.0 - (sum(_spreads) / max(len(_spreads), 1)) * 5.0) if _spreads else 1.0

    # 13. Model confidence score [0, 1] (Fix #3, Fix #4.1, Fix #4.5, Fix #6.4)
    # Quando non ci sono tiri, usa la qualità delle linee come proxy del blend.
    # Questo evita la penalizzazione artificiale a minuto 0 con linee fresche.
    if n_shots_tot > 0:
        _shots_conf = min(1.0, n_shots_tot / 15.0)
        _blend_conf = (alpha_t + alpha_d) / 2.0
    else:
        # Baseline line-only: le linee di mercato sono l'unica fonte di informazione.
        # Fix #4.1/#4.5: Usa parametri dal config invece di hardcoded
        _shots_conf = ENGINE.PREMATCH_SHOTS_CONF
        _blend_conf = max(ENGINE.BLEND_CONF_STALE,
                          (ENGINE.BLEND_CONF_STALE if stale_line
                           else (ENGINE.BLEND_CONF_FLAT if flat_lines
                                else ENGINE.BLEND_CONF_NORMAL)))
    _line_conf = ENGINE.LINE_CONF_STALE if stale_line else (ENGINE.LINE_CONF_FLAT if flat_lines else ENGINE.LINE_CONF_NORMAL)
    _time_conf = _math.sqrt(state.minuto / 90.0) if state.minuto > 0 else ENGINE.PREMATCH_TIME_CONF
    _agreement_conf = model_agreement
    # _agreement_conf può essere 0.0 → il suo guard è necessario.
    _product = max(1e-9, _shots_conf * _line_conf * _blend_conf
                   * _time_conf * max(0.01, _agreement_conf))
    # Fix #6.4: Usa parametro dal config per la radice
    model_confidence = min(1.0, _product ** ENGINE.CONFIDENCE_ROOT_POWER)

    # #6 — Confidence prematch adattiva con OCR.
    # Se le quote OCR sono disponibili con buona affidabilità, la fiducia
    # nel modello prematch aumenta (secondo prior indipendente).
    if _ocr_available and state.minuto == 0:
        from src.config import OCR_CALIB
        if state.ocr_confidence == "high":
            model_confidence = min(1.0, model_confidence + OCR_CALIB.HIGH_CONF_BOOST)
        elif state.ocr_confidence == "medium":
            model_confidence = min(1.0, model_confidence + OCR_CALIB.MEDIUM_CONF_BOOST)

    # FIX: Rileva se le linee sembrano non aggiornate dopo i gol
    # Se ci sono gol segnati ma le linee sono ancora "flat" (uguale all'apertura),
    # l'utente probabilmente ha dimenticato di aggiornarle.
    lines_need_update = flat_lines and gol_tot_scored > 0 and state.minuto >= 15

    # #12 — Warning divergenza se quote OCR discordano >15% dal modello.
    # Confronta le probabilità implicite OCR (normalizzate, senza overround)
    # con quelle del modello. Se divergono >15%, segnala possibile errore.
    ocr_divergence: dict = {}
    if (state.ocr_p1 > 0 or state.ocr_p_over > 0 or state.ocr_p_btts > 0):
        from src.config import OCR_CALIB
        _ocr_checks = [
            ("1X2 Casa",      p1,     state.ocr_p1),
            ("1X2 Pareggio",  px,     state.ocr_px),
            ("1X2 Trasferta", p2,     state.ocr_p2),
            ("Over",          p_over, state.ocr_p_over),
            ("Under",         p_under, state.ocr_p_under),
            ("BTTS",          p_btts, state.ocr_p_btts),
        ]
        for _label, _p_model, _p_ocr in _ocr_checks:
            if _p_ocr > 0.01 and _p_model > 0.01:
                _div = abs(_p_model - _p_ocr)
                if _div > OCR_CALIB.DIVERGENCE_THRESHOLD:
                    ocr_divergence[_label] = (_p_model, _p_ocr, _div)

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
        stale_line=stale_line,
        model_agreement=model_agreement,
        credible_intervals=credible_intervals,
        joint_ind=joint_ind,
        full_matrix=full_matrix,
        delta_ah=delta_ah,
        delta_tot=delta_tot,
        lines_need_update=lines_need_update,
        ocr_divergence=ocr_divergence,
    )
