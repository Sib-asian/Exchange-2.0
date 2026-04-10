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

from src.config import BAYES, CACHE, DECAY, ENGINE

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

    # Statistiche aggiuntive da OCR (0 = non disponibili)
    gialli_casa: int = 0       # Cartellini gialli casa
    gialli_trasf: int = 0      # Cartellini gialli trasferta
    blk_h: int = 0             # Tiri bloccati casa
    blk_a: int = 0             # Tiri bloccati trasferta
    att_h: int = 0             # Attacchi totali casa
    att_a: int = 0             # Attacchi totali trasferta
    falli_casa: int = 0        # Falli casa
    falli_trasf: int = 0       # Falli trasferta

    # Linea O/U estratta da OCR (0 = non disponibile).
    # Usata come soft prior per la calibrazione Bayesiana in prematch.
    ocr_imp_total: float = 0.0

    # Prior storico H2H (media gol partite precedenti tra queste due squadre).
    # Calcolato da Gemini + Google Search. 0.0 = non disponibile.
    # Blendato con tot_op al 10% solo in prematch (minuto=0).
    fixture_historical_total: float = 0.0

    # Qualità del movimento linee (da Gemini). Range [0.80, 1.30], default 1.0.
    # > 1.0 = movimento affidabile (sharp/notizie) → w_cur aumentato.
    # < 1.0 = movimento rumoroso (pubblico/liquidità) → w_cur diminuito.
    movement_quality: float = 1.0

    # Movimento linee raw estratto da Nowgoal (full-game, apertura→chiusura).
    # Usato in prematch come segnale secondario (peso basso) per evitare
    # dipendenza totale dal solo movement_quality sintetico.
    line_movement_ah_raw: float = 0.0
    line_movement_total_raw: float = 0.0

    # Copertura estrazione prematch [0,1] per gating dei micro-prior.
    extraction_coverage: float = 0.0

    # Team stats prematch (ultimi 10): usati con peso conservativo come micro-prior.
    team_stats_home_shots: float = 0.0
    team_stats_away_shots: float = 0.0
    team_stats_home_corners: float = 0.0
    team_stats_away_corners: float = 0.0
    team_stats_home_possession: float = 0.0
    team_stats_away_possession: float = 0.0

    # Team stats prematch aggiuntivi (da Nowgoal, ultimi 10)
    team_stats_home_yellows: float = 0.0
    team_stats_away_yellows: float = 0.0
    team_stats_home_fouls: float = 0.0
    team_stats_away_fouls: float = 0.0

    # Previous scores Over% per squadra (0-100, da Nowgoal)
    prev_over_pct_h: float = 0.0
    prev_over_pct_a: float = 0.0

    # HT/FT transition counts (da Nowgoal, ultimi ~10 match)
    # 9 cells per squadra: HTW/HTD/HTL → FTW/FTD/FTL
    htft_home_htw_ftw: int = 0
    htft_home_htw_ftd: int = 0
    htft_home_htw_ftl: int = 0
    htft_home_htd_ftw: int = 0
    htft_home_htd_ftd: int = 0
    htft_home_htd_ftl: int = 0
    htft_home_htl_ftw: int = 0
    htft_home_htl_ftd: int = 0
    htft_home_htl_ftl: int = 0
    htft_away_htw_ftw: int = 0
    htft_away_htw_ftd: int = 0
    htft_away_htw_ftl: int = 0
    htft_away_htd_ftw: int = 0
    htft_away_htd_ftd: int = 0
    htft_away_htd_ftl: int = 0
    htft_away_htl_ftw: int = 0
    htft_away_htl_ftd: int = 0
    htft_away_htl_ftl: int = 0

    # Scala di confidenza delle quote OCR (da validatore Gemini). Range [0.70, 1.0].
    # 1.0 = quote verificate concordi col mercato; < 1.0 = quote sospette/anomale.
    ocr_confidence_scale: float = 1.0

    # Quote bookmaker prematch estratte da OCR (0.0 = non disponibili).
    # Usate come prior addizionali per la calibrazione Bayesiana in prematch.
    ocr_quota_1: float = 0.0
    ocr_quota_x: float = 0.0
    ocr_quota_2: float = 0.0
    ocr_quota_over: float = 0.0
    ocr_quota_under: float = 0.0
    ocr_quota_gg: float = 0.0
    ocr_quota_ng: float = 0.0

    # Moltiplicatori xG da dati AI (da Gemini Ricerca Pre-Partita).
    # Calcolati da assenze squadre e forma recente, scalati per affidabilità.
    # 1.0 = nessun effetto (default quando la ricerca non è disponibile).
    # < 1.0 = riduzione xG (proprie assenze); > 1.0 = aumento xG (GK avversario assente).
    absence_mult_h: float = 1.0   # moltiplicatore xG casa (assenze + GK trasf. assente)
    absence_mult_a: float = 1.0   # moltiplicatore xG trasferta (assenze + GK casa assente)
    forma_mult_h: float = 1.0     # moltiplicatore xG casa dalla forma recente
    forma_mult_a: float = 1.0     # moltiplicatore xG trasferta dalla forma recente

    # Dati prematch per post-consensus correction (solo prematch, 0 = non disponibile)
    mkt_init_1: float = 0.0       # quota 1X2 iniziale casa (media bookmaker)
    mkt_init_x: float = 0.0       # quota 1X2 iniziale pareggio
    mkt_init_2: float = 0.0       # quota 1X2 iniziale trasferta
    h2h_home_win_pct: float = 0.0 # % vittorie casa nei precedenti H2H (0-100)
    h2h_draw_pct: float = 0.0     # % pareggi nei precedenti H2H
    h2h_away_win_pct: float = 0.0 # % vittorie trasferta nei precedenti H2H
    h2h_over_pct: float = 0.0     # % partite H2H andate Over (0-100)

    # Strength da Nowgoal (0-100)
    strength_home: int = 0        # punteggio forza casa
    strength_away: int = 0        # punteggio forza trasferta

    # Previous scores (ultime 10 partite) per blend xG
    prev_avg_scored_h: float = 0.0    # media gol segnati casa
    prev_avg_conceded_h: float = 0.0  # media gol subiti casa
    prev_avg_scored_a: float = 0.0    # media gol segnati trasferta
    prev_avg_conceded_a: float = 0.0  # media gol subiti trasferta

    # === FORM ANALYSIS - Dati estratti da Nowgoal ===

    # Standings (Classifica) - per calcolo motivazione
    standings_rank_h: int = 0          # posizione in classifica casa (1 = prima)
    standings_rank_a: int = 0          # posizione in classifica trasferta
    standings_points_h: int = 0        # punti in classifica casa
    standings_points_a: int = 0        # punti in classifica trasferta
    standings_played_h: int = 0        # partite giocate casa
    standings_played_a: int = 0        # partite giocate trasferta
    standings_total_teams: int = 20    # numero totale squadre in lega

    # Last 6 games (Forma recente specifica)
    last6_points_h: float = 0.0        # punti conquistati nelle ultime 6 casa
    last6_points_a: float = 0.0        # punti conquistati nelle ultime 6 trasferta
    last6_gf_h: float = 0.0            # gol fatti nelle ultime 6 casa
    last6_ga_h: float = 0.0            # gol subiti nelle ultime 6 casa
    last6_gf_a: float = 0.0            # gol fatti nelle ultime 6 trasferta
    last6_ga_a: float = 0.0            # gol subiti nelle ultime 6 trasferta

    # Home/Away Performance (Rendimento specifico)
    home_ppg_h: float = 0.0            # punti per partita in casa della squadra di casa
    away_ppg_a: float = 0.0            # punti per partita in trasferta della squadra trasferta
    home_gf_h: float = 0.0             # gol fatti medi in casa
    home_ga_h: float = 0.0             # gol subiti medi in casa
    away_gf_a: float = 0.0             # gol fatti medi in trasferta
    away_ga_a: float = 0.0             # gol subiti medi in trasferta

    # Goal Timing (Quando segnano - opzionale)
    late_goals_pct_h: float = 0.0      # % gol segnati nei minuti 75-90 casa
    late_goals_pct_a: float = 0.0      # % gol segnati nei minuti 75-90 trasferta
    early_conceded_pct_h: float = 0.0  # % gol subiti nei minuti 0-30 casa
    early_conceded_pct_a: float = 0.0  # % gol subiti nei minuti 0-30 trasferta

    # === H2H Half-Time (primo tempo negli scontri diretti) ===
    h2h_ht_home_win_pct: float = 0.0   # % volte casa in vantaggio a HT nei H2H (0-100)
    h2h_ht_draw_pct: float = 0.0       # % pareggi a HT nei H2H (0-100)
    h2h_ht_away_win_pct: float = 0.0   # % volte trasferta in vantaggio a HT nei H2H (0-100)

    # === BTTS Calibration - Dati aggiuntivi ===
    h2h_btts_pct: float = 0.0          # % storica partite H2H con BTTS (0-100)
    h2h_matches_count: int = 0         # numero di H2H usati per stimare h2h_btts_pct
    scoring_streak_h: int = 0          # partite consecutive con gol segnato casa
    scoring_streak_a: int = 0          # partite consecutive con gol segnato trasferta
    clean_sheet_streak_h: int = 0      # partite consecutive senza subire casa
    clean_sheet_streak_a: int = 0      # partite consecutive senza subire trasferta

    # Weather impact on xG (da OpenWeather API o da Gemini Vision)
    weather_xg_impact: float = 0.0     # impatto meteo su xG (-0.15 a +0.02)

    # Contradiction Detection: warning quando live stats contraddicono pre-match expectations
    contradiction_warning: str = ""    # messaggio di warning per l'utente
    has_contradiction: bool = False    # True se rilevata contraddizione significativa

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
    # Quality firewall score [0,1]: misura sintetica di affidabilità operativa.
    quality_score: float = 1.0
    # Se True, blocca segnali operativi (No-Bet engine / quality firewall).
    signals_blocked: bool = False
    # Motivo sintetico del blocco operativo (utile in UI e tracking).
    signals_block_reason: str = ""

    # Market shock: True se il momentum supera la soglia di "movimento anomalo".
    # Indica che le linee si sono mosse molto rispetto al tempo giocato →
    # potrebbe esserci informazione asimmetrica (infortuni, formazioni, notizie).
    market_shock: bool = False

    # Pesi consensus effettivi e marginali 1X2 per modello (tracking / ensemble adattivo)
    consensus_w_bp: float = 0.5
    consensus_w_cop: float = 0.3
    consensus_w_mk: float = 0.2
    p1_bp: float = 0.0
    px_bp: float = 0.0
    p2_bp: float = 0.0
    p1_cop: float = 0.0
    px_cop: float = 0.0
    p2_cop: float = 0.0
    p1_mk: float = 0.0
    px_mk: float = 0.0
    p2_mk: float = 0.0

    # Intervalli di credibilità multi-modello
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Matrici
    joint_ind: dict[tuple[int, int], float] = field(default_factory=dict)
    full_matrix: dict[tuple[int, int], float] = field(default_factory=dict)

    # Over/Under linea 1.5 (marginali; stessa pipeline shrink/calibrazione prematch della linea principale)
    p_over_15: float = 0.0
    p_under_15: float = 0.0
    # Over/Under canonico 2.5 (sempre calcolato dalla stessa distribuzione, indipendente da linea_ou scelta)
    p_over_25_ref: float = 0.0
    p_under_25_ref: float = 0.0

    # Upgrade 8-1: Score di coerenza cross-mercato [0, 1].
    # 1.0 = probabilità perfettamente coerenti tra mercati.
    # <0.5 = divergenza significativa tra consensus e matrice punteggio.
    probability_coherence: float = 1.0


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
        agreement_1x2_from_per_raw,
        calibrate_probabilities,
        compute_consensus,
        compute_model_credible_intervals,
        compute_model_market_divergence,
        logistic_sharpen_over,
        per_model_market_probs,
    )
    from src.models.ensemble_adaptive import blend_consensus_weights_with_history
    from src.models.time_decay import calcola_momentum_mercato, time_decay_dinamico

    # Cap temporale: i gol rimanenti non possono superare ~(90-minuto)/90 * 4.0.
    # Protegge dal caso frequente in cui l'utente cambia il punteggio/minuto ma
    # dimentica di aggiornare le linee live → tot_cur rimane il valore full-game
    # d'apertura (es. 2.75) che diventa insensato come "gol rimanenti" al 80'.
    # Esempio: 0-0 al 80', default tot_cur=2.75 → cap a 0.44 (= 4.0 × 10/90).
    _mins_rem = max(1, 90 - state.minuto)
    _tot_cap = max(BAYES.TOT_BAYES_MIN, _mins_rem / 90.0 * BAYES.TOT_TEMPORAL_MAX)
    tot_cur_eff = min(state.tot_cur, _tot_cap)

    # 0. Segnali OCR da quote bookmaker (solo prematch, minuto == 0)
    from src.models.calibration import estrai_segnali_ocr_da_quote
    _ocr_total_quotes = 0.0
    _ocr_delta_quotes = 0.0
    _ocr_overround_ou = 0.0
    _ocr_overround_1x2 = 0.0
    if state.minuto == 0:
        # Se disponibile, usa la linea O/U estratta dalla stessa fonte delle quote OCR
        # (Nowgoal open line) per evitare mismatch con la linea analisi selezionata.
        _ou_line_for_ocr = state.ocr_imp_total if state.ocr_imp_total > 0.0 else state.linea_ou
        _ocr_total_quotes, _ocr_delta_quotes = estrai_segnali_ocr_da_quote(
            state.ocr_quota_1, state.ocr_quota_x, state.ocr_quota_2,
            state.ocr_quota_over, state.ocr_quota_under,
            _ou_line_for_ocr,
        )
        # #3: Calcola overround per quality-aware blending in calcola_xg_bayesiani()
        if state.ocr_quota_over > 1.0 and state.ocr_quota_under > 1.0:
            _ocr_overround_ou = 1.0 / state.ocr_quota_over + 1.0 / state.ocr_quota_under
        if state.ocr_quota_1 > 1.0 and state.ocr_quota_x > 1.0 and state.ocr_quota_2 > 1.0:
            _ocr_overround_1x2 = 1.0 / state.ocr_quota_1 + 1.0 / state.ocr_quota_x + 1.0 / state.ocr_quota_2

    # 1. xG da linee (prior bayesiano)
    xg_h_base, xg_a_base = calcola_xg_bayesiani(
        state.ah_op, state.tot_op,
        state.ah_cur, tot_cur_eff,
        state.minuto,
        gol_diff=state.gol_casa - state.gol_trasf,
        gol_tot=state.gol_casa + state.gol_trasf,
        ocr_imp_total=state.ocr_imp_total,
        ocr_total_quotes=_ocr_total_quotes,
        ocr_delta_quotes=_ocr_delta_quotes,
        ocr_overround_ou=_ocr_overround_ou,
        ocr_overround_1x2=_ocr_overround_1x2,
        fixture_historical_total=state.fixture_historical_total,
        movement_quality=state.movement_quality,
        ocr_confidence_scale=state.ocr_confidence_scale,
        line_movement_ah_raw=state.line_movement_ah_raw,
        line_movement_total_raw=state.line_movement_total_raw,
        extraction_coverage=state.extraction_coverage,
        team_stats_home_shots=state.team_stats_home_shots,
        team_stats_away_shots=state.team_stats_away_shots,
        team_stats_home_corners=state.team_stats_home_corners,
        team_stats_away_corners=state.team_stats_away_corners,
        team_stats_home_possession=state.team_stats_home_possession,
        team_stats_away_possession=state.team_stats_away_possession,
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
            corner_h=state.corner_h,
            corner_a=state.corner_a,
            possesso_h=state.possesso_h,
            possesso_a=state.possesso_a,
            att_pericolosi_h=state.att_pericolosi_h,
            att_pericolosi_a=state.att_pericolosi_a,
            blk_h=state.blk_h,
            blk_a=state.blk_a,
            att_h=state.att_h,
            att_a=state.att_a,
        )
    else:
        xg_h_blend = xg_h_base
        xg_a_blend = xg_a_base
        xg_h_accum = xg_a_accum = 0.0
        alpha_t = alpha_d = shot_dom = 0.0

    # 2b. Aggiustamenti AI: assenze + forma (miglioramento #1, #3, #7).
    # Applicati DOPO il blend per modular sia l'xG da linee che l'xG da tiri.
    # ABSENCE_MARKET_ALPHA=0.40 già applicato in calcola_assenze_mult → nessun double-count.
    # Condizione: skip se entrambi i moltiplicatori sono 1.0 per evitare ricompute inutile.
    if state.absence_mult_h != 1.0 or state.forma_mult_h != 1.0:
        xg_h_blend = max(DECAY.XG_FLOOR, xg_h_blend * state.absence_mult_h * state.forma_mult_h)
    if state.absence_mult_a != 1.0 or state.forma_mult_a != 1.0:
        xg_a_blend = max(DECAY.XG_FLOOR, xg_a_blend * state.absence_mult_a * state.forma_mult_a)

    # Meteo: applica un micro-adjustment simmetrico se disponibile.
    if state.weather_xg_impact != 0.0:
        _wx_mult = max(0.85, min(1.05, 1.0 + state.weather_xg_impact))
        xg_h_blend = max(DECAY.XG_FLOOR, xg_h_blend * _wx_mult)
        xg_a_blend = max(DECAY.XG_FLOOR, xg_a_blend * _wx_mult)

    # 2c. Previous scores blend (Miglioramento #2) - solo prematch.
    # Se abbiamo dati sulle ultime 10 partite, li usiamo per calibrare gli xG.
    # Formula: xG stimato = (gol segnati + gol subiti avversario) / 2
    # Blend conservativo: 85% xG da linee + 15% xG da previous scores.
    if state.minuto == 0:
        _prev_alpha = 0.15
        # xG casa: media tra gol che la casa segna e gol che la trasferta subisce
        if state.prev_avg_scored_h > 0 and state.prev_avg_conceded_a > 0:
            _xg_h_from_prev = (state.prev_avg_scored_h + state.prev_avg_conceded_a) / 2.0
            xg_h_blend = (1.0 - _prev_alpha) * xg_h_blend + _prev_alpha * _xg_h_from_prev
        # xG trasferta: media tra gol che la trasferta segna e gol che la casa subisce
        if state.prev_avg_scored_a > 0 and state.prev_avg_conceded_h > 0:
            _xg_a_from_prev = (state.prev_avg_scored_a + state.prev_avg_conceded_h) / 2.0
            xg_a_blend = (1.0 - _prev_alpha) * xg_a_blend + _prev_alpha * _xg_a_from_prev
        # Floor di sicurezza
        xg_h_blend = max(DECAY.XG_FLOOR, xg_h_blend)
        xg_a_blend = max(DECAY.XG_FLOOR, xg_a_blend)

        # 2c-bis. Strength model blend (Upgrade 3) — xG indipendente dal mercato.
        # Usa solo dati storici (gol segnati/subiti, casa/trasferta, forma)
        # per rompere la dipendenza circolare xG←mercato.
        from src.models.strength_model import blend_strength_with_market, compute_strength_xg
        _strength_xg = compute_strength_xg(
            state.prev_avg_scored_h, state.prev_avg_conceded_h,
            state.prev_avg_scored_a, state.prev_avg_conceded_a,
            home_gf_h=state.home_gf_h, home_ga_h=state.home_ga_h,
            away_gf_a=state.away_gf_a, away_ga_a=state.away_ga_a,
            last6_gf_h=state.last6_gf_h, last6_ga_h=state.last6_ga_h,
            last6_gf_a=state.last6_gf_a, last6_ga_a=state.last6_ga_a,
        )
        if _strength_xg is not None:
            xg_h_blend, xg_a_blend = blend_strength_with_market(
                xg_h_blend, xg_a_blend,
                _strength_xg[0], _strength_xg[1],
            )

    # 2d. Form Analysis - Standings, Last6, Home/Away Performance (solo prematch).
    # RIDUCE la dipendenza dalle linee manuali utilizzando dati estratti da Nowgoal.
    # Questi aggiustamenti sono applicati DOPO previous scores per non interferire.
    if state.minuto == 0:
        from src.config import FORM_ANALYSIS

        # === 2d.1 STANDINGS - Fattore motivazione ===
        # Squadre in zona critica (retrocessione/titolo/europa) sono più motivate.
        # Squadre senza obiettivi (posizione centrale) possono essere meno motivate.
        _motivation_mult_h = 1.0
        _motivation_mult_a = 1.0

        if state.standings_total_teams > 0 and state.standings_rank_h > 0 and state.standings_rank_a > 0:
            # Zona retrocessione (ultime N posizioni)
            if state.standings_rank_h > state.standings_total_teams - FORM_ANALYSIS.RELEGATION_ZONE:
                _motivation_mult_h += FORM_ANALYSIS.RELEGATION_MOTIVATION_BONUS
            if state.standings_rank_a > state.standings_total_teams - FORM_ANALYSIS.RELEGATION_ZONE:
                _motivation_mult_a += FORM_ANALYSIS.RELEGATION_MOTIVATION_BONUS

            # Zona titolo (prime N posizioni)
            if state.standings_rank_h <= FORM_ANALYSIS.TITLE_ZONE:
                _motivation_mult_h += FORM_ANALYSIS.TITLE_MOTIVATION_BONUS
            if state.standings_rank_a <= FORM_ANALYSIS.TITLE_ZONE:
                _motivation_mult_a += FORM_ANALYSIS.TITLE_MOTIVATION_BONUS

            # Zona europa (prime N posizioni per qualificazione europea)
            elif state.standings_rank_h <= FORM_ANALYSIS.EUROPE_ZONE:
                _motivation_mult_h += FORM_ANALYSIS.EUROPE_MOTIVATION_BONUS
            elif state.standings_rank_a <= FORM_ANALYSIS.EUROPE_ZONE:
                _motivation_mult_a += FORM_ANALYSIS.EUROPE_MOTIVATION_BONUS

            # Nessun obiettivo (posizione centrale) - penalità solo se le posizioni sono veramente centrali
            # (non in zona critica né vicino)
            _mid_start = FORM_ANALYSIS.EUROPE_ZONE + 1
            _mid_end = state.standings_total_teams - FORM_ANALYSIS.RELEGATION_ZONE - 1
            if _mid_start < _mid_end:
                if _mid_start <= state.standings_rank_h <= _mid_end:
                    _motivation_mult_h += FORM_ANALYSIS.NO_STAKES_PENALTY
                if _mid_start <= state.standings_rank_a <= _mid_end:
                    _motivation_mult_a += FORM_ANALYSIS.NO_STAKES_PENALTY

            # Retrocessione + netto svantaggio PPG: urgenza oltre il semplice bonus zona
            _ph = state.standings_played_h
            _pa = state.standings_played_a
            if _ph > 0 and _pa > 0:
                _pph = state.standings_points_h / _ph
                _ppa = state.standings_points_a / _pa
                _gap = _pph - _ppa
                _rel_z = state.standings_total_teams - FORM_ANALYSIS.RELEGATION_ZONE
                if state.standings_rank_h > _rel_z and _gap < -FORM_ANALYSIS.RELEGATION_PPG_GAP_THRESHOLD:
                    _motivation_mult_h += FORM_ANALYSIS.RELEGATION_UNDERDOG_BONUS
                if state.standings_rank_a > _rel_z and _gap > FORM_ANALYSIS.RELEGATION_PPG_GAP_THRESHOLD:
                    _motivation_mult_a += FORM_ANALYSIS.RELEGATION_UNDERDOG_BONUS

        # === 2d.2 LAST 6 GAMES - Forma recente specifica ===
        # Calcola punti per partita (PPG) nelle ultime 6 e applica boost/penalità.
        _last6_mult_h = 1.0
        _last6_mult_a = 1.0

        if state.last6_points_h > 0 or state.last6_points_a > 0:
            # PPG = punti / 6 partite (max 18 punti)
            _ppg_h = state.last6_points_h / 6.0 if state.last6_points_h > 0 else 1.5  # default neutro
            _ppg_a = state.last6_points_a / 6.0 if state.last6_points_a > 0 else 1.5

            # Forma eccellente (PPG alto)
            if _ppg_h >= FORM_ANALYSIS.LAST6_POINTS_EXCELLENT:
                _last6_mult_h += FORM_ANALYSIS.LAST6_MAX_BOOST
            elif _ppg_h <= FORM_ANALYSIS.LAST6_POINTS_POOR:
                _last6_mult_h += FORM_ANALYSIS.LAST6_MAX_PENALTY
            else:
                # Interpolazione lineare tra povero e eccellente
                _range = FORM_ANALYSIS.LAST6_POINTS_EXCELLENT - FORM_ANALYSIS.LAST6_POINTS_POOR
                _pos = (_ppg_h - FORM_ANALYSIS.LAST6_POINTS_POOR) / max(_range, 0.1)
                _effect = FORM_ANALYSIS.LAST6_MAX_PENALTY + _pos * (FORM_ANALYSIS.LAST6_MAX_BOOST - FORM_ANALYSIS.LAST6_MAX_PENALTY)
                _last6_mult_h += _effect * FORM_ANALYSIS.LAST6_WEIGHT

            if _ppg_a >= FORM_ANALYSIS.LAST6_POINTS_EXCELLENT:
                _last6_mult_a += FORM_ANALYSIS.LAST6_MAX_BOOST
            elif _ppg_a <= FORM_ANALYSIS.LAST6_POINTS_POOR:
                _last6_mult_a += FORM_ANALYSIS.LAST6_MAX_PENALTY
            else:
                _range = FORM_ANALYSIS.LAST6_POINTS_EXCELLENT - FORM_ANALYSIS.LAST6_POINTS_POOR
                _pos = (_ppg_a - FORM_ANALYSIS.LAST6_POINTS_POOR) / max(_range, 0.1)
                _effect = FORM_ANALYSIS.LAST6_MAX_PENALTY + _pos * (FORM_ANALYSIS.LAST6_MAX_BOOST - FORM_ANALYSIS.LAST6_MAX_PENALTY)
                _last6_mult_a += _effect * FORM_ANALYSIS.LAST6_WEIGHT

        # === 2d.3 HOME/AWAY PERFORMANCE ===
        # Rendimento specifico casa/trasferta per le due squadre.
        _home_away_mult_h = 1.0
        _home_away_mult_a = 1.0

        if state.home_ppg_h > 0 or state.away_ppg_a > 0:
            # Squadra di casa: bonus se forte in casa
            if state.home_ppg_h >= FORM_ANALYSIS.HOME_STRONG_THRESHOLD:
                _home_away_mult_h += FORM_ANALYSIS.HOME_STRONG_BONUS

            # Squadra trasferta: penalità se debole fuori casa
            if state.away_ppg_a > 0 and state.away_ppg_a <= FORM_ANALYSIS.AWAY_WEAK_THRESHOLD:
                _home_away_mult_a += FORM_ANALYSIS.AWAY_WEAK_PENALTY

            # Effetto gol fatti/subiti casa/trasferta
            # Se la casa fa molti gol in casa, aumenta xG casa
            if state.home_gf_h > 1.5:
                _home_away_mult_h += 0.02
            # Se la trasferta subisce molti gol fuori, aumenta xG casa
            if state.away_ga_a > 1.3:
                _home_away_mult_h += 0.015
            # Se la trasferta fa molti gol fuori, aumenta xG trasferta
            if state.away_gf_a > 1.3:
                _home_away_mult_a += 0.02
            # Se la casa subisce molti gol in casa, aumenta xG trasferta
            if state.home_ga_h > 1.3:
                _home_away_mult_a += 0.015

        # === 2d.4 GOAL TIMING (opzionale) ===
        # Squadre che segnano a fine partita possono essere più pericolose.
        _timing_mult_h = 1.0
        _timing_mult_a = 1.0

        if state.late_goals_pct_h > 35:
            _timing_mult_h += FORM_ANALYSIS.LATE_SCORER_BONUS
        if state.late_goals_pct_a > 35:
            _timing_mult_a += FORM_ANALYSIS.LATE_SCORER_BONUS
        if state.early_conceded_pct_h > 40:
            _timing_mult_h += FORM_ANALYSIS.EARLY_CONCEDER_PENALTY
        if state.early_conceded_pct_a > 40:
            _timing_mult_a += FORM_ANALYSIS.EARLY_CONCEDER_PENALTY

        # === APPLICA TUTTI GLI AGGIUSTAMENTI FORM ANALYSIS ===
        # Peso complessivo ridotto per evitare double-counting con previous scores e assenze
        _form_weight = 0.40  # Gli aggiustamenti form analysis pesano al 40% del totale
        _combined_mult_h = 1.0 + _form_weight * (_motivation_mult_h + _last6_mult_h + _home_away_mult_h + _timing_mult_h - 4.0)
        _combined_mult_a = 1.0 + _form_weight * (_motivation_mult_a + _last6_mult_a + _home_away_mult_a + _timing_mult_a - 4.0)

        # Applica con floor di sicurezza
        xg_h_blend = max(DECAY.XG_FLOOR, xg_h_blend * _combined_mult_h)
        xg_a_blend = max(DECAY.XG_FLOOR, xg_a_blend * _combined_mult_a)

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

    momentum = calcola_momentum_mercato(delta_ah, delta_tot, state.minuto) + momentum_from_goals

    # #10: Momentum statistico — aggiunge componente direzionale da tiri/attacchi.
    # Attiva solo se i dati sono sufficienti (>4 tiri) e la dominanza è significativa (>40%).
    # Max contributo: STAT_MOMENTUM_MAX = 0.5. Conservativo: il mercato domina (80%+).
    if n_shots_tot > 4 and state.minuto > 0:
        _sot_tot = state.sot_h + state.sot_a
        _shot_dom_abs = abs(state.sot_h - state.sot_a) / _sot_tot if _sot_tot > 0 else 0.0
        if _shot_dom_abs > 0.40:
            _stat_contrib = (_shot_dom_abs - 0.40) / 0.60 * MOMENTUM.STAT_MOMENTUM_MAX
            momentum = momentum + _stat_contrib
    elif (state.att_pericolosi_h + state.att_pericolosi_a) > 10 and state.minuto > 0:
        _att_tot = state.att_pericolosi_h + state.att_pericolosi_a
        _att_dom = abs(state.att_pericolosi_h - state.att_pericolosi_a) / _att_tot
        if _att_dom > 0.40:
            _stat_contrib = (_att_dom - 0.40) / 0.60 * MOMENTUM.STAT_MOMENTUM_MAX * 0.70
            momentum = momentum + _stat_contrib

    momentum = min(momentum, MOMENTUM.MOMENTUM_CAP)  # Rispetta il cap

    flat_lines = abs(delta_ah) < BAYES.FLAT_LINE_THRESHOLD and abs(delta_tot) < BAYES.FLAT_LINE_THRESHOLD

    # 4. Time decay + score effect + rossi + momentum dampening
    # FIX: Passa delta_ah per evitare doppio conteggio score effect
    xg_h_final, xg_a_final = time_decay_dinamico(
        xg_h_blend, xg_a_blend,
        state.minuto,
        state.gol_casa, state.gol_trasf,
        state.rossi_casa, state.rossi_trasf,
        momentum=momentum,
        delta_ah=delta_ah,
        gialli_casa=state.gialli_casa,
        gialli_trasf=state.gialli_trasf,
        falli_casa=state.falli_casa,
        falli_trasf=state.falli_trasf,
        late_goals_pct_h=state.late_goals_pct_h,
        late_goals_pct_a=state.late_goals_pct_a,
    )

    # 4a. Live recalibration (Upgrade 5): correggi xG residui se il modello
    # sovra/sotto-stima i gol osservati finora.
    if state.minuto > 0:
        from src.models.live_recalibration import compute_live_recalibration_factor
        _live_recal = compute_live_recalibration_factor(
            state.tot_op,
            state.gol_casa + state.gol_trasf,
            state.minuto,
            tot_cur_remaining=state.tot_cur,
        )
        if _live_recal != 1.0:
            xg_h_final = max(DECAY.XG_FLOOR, xg_h_final * _live_recal)
            xg_a_final = max(DECAY.XG_FLOOR, xg_a_final * _live_recal)

    # 4b. Stale line detection: se le linee non si sono mosse dopo >15 minuti
    # di partita, il mercato è potenzialmente illiquido o sospeso.
    stale_line = flat_lines and state.minuto >= STALE.THRESHOLD_MINUTES

    # 5. Calcola parametri condivisi tra i modelli
    from src.models.poisson import rho_dc_dinamico as _calc_rho_dc
    # Upgrade 1: In prematch, usa le statistiche gialli/falli da team_stats come proxy.
    # In live, usa i gialli reali della partita.
    _gialli_for_rho = state.gialli_casa + state.gialli_trasf
    if state.minuto == 0 and _gialli_for_rho == 0:
        # Proxy prematch: media gialli per partita delle due squadre (arrotondato)
        _prematch_yellows = state.team_stats_home_yellows + state.team_stats_away_yellows
        if _prematch_yellows > 0:
            _gialli_for_rho = int(round(_prematch_yellows))
    _rho_dc_shared = _calc_rho_dc(
        tot_cur_eff, state.minuto, state.gol_casa + state.gol_trasf,
        gialli_totali=_gialli_for_rho,
    )

    # Parametri per il modello Copula
    frac_giocata = state.minuto / 90.0
    _theta_shot = (
        COPULA.THETA_SHOT_DOM_SCALE
        * shot_dom
        * (1.0 - COPULA.THETA_SHOT_DOM_TIME_DAMP * frac_giocata)
    )
    copula_theta = max(
        0.1,
        min(
            COPULA.THETA_MAX,
            COPULA.THETA_BASE
            + COPULA.THETA_TOT_SCALE * max(0.0, tot_cur_eff - COPULA.THETA_TOT_REF)
            + COPULA.THETA_TIME_SCALE * frac_giocata
            + _theta_shot,
        ),
    )
    nu_dynamic = CMP.NU + CMP.NU_TOT_SCALE * (tot_cur_eff - CMP.NU_TOT_REF)
    if state.minuto == 0 and state.strength_home > 0 and state.strength_away > 0:
        _sd = abs(float(state.strength_home) - float(state.strength_away))
        _dom = min(1.0, _sd / max(CMP.NU_STRENGTH_REF, 1e-6))
        nu_dynamic -= CMP.NU_STRENGTH_SCALE * _dom
    nu_dynamic = max(CMP.NU_MIN, min(CMP.NU_MAX, nu_dynamic))

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
            "bivariate", xg_h_final, xg_a_final, state.minuto, tot_cur_eff, gol_totali,
            shot_dom, _rho_dc_shared,
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
            "markov", xg_h_final, xg_a_final, state.minuto, state.gol_casa, state.gol_trasf,
            _rho_dc_shared,
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

    # 7b. Fallback modello: se una matrice è vuota (modello fallito),
    # redistribuisci il suo peso ai modelli sopravvissuti.
    _bp_ok = len(full_bp) > 0
    _cop_ok = len(full_copula) > 0
    _mk_ok = len(full_markov) > 0
    _n_ok = int(_bp_ok) + int(_cop_ok) + int(_mk_ok)

    if _n_ok == 0:
        raise RuntimeError("Tutti e 3 i modelli hanno fallito — impossibile calcolare consensus")

    # 8. Consensus multi-modello: media pesata con pesi dinamici per 4 fasi di gioco.
    if state.minuto == 0:
        _w_bp, _w_cop, _w_mk = CONSENSUS.W_BP_PREMATCH, CONSENSUS.W_COP_PREMATCH, CONSENSUS.W_MK_PREMATCH
    elif state.minuto <= CONSENSUS.EARLY_GAME_MINUTE:
        _w_bp, _w_cop, _w_mk = CONSENSUS.W_BP_EARLY, CONSENSUS.W_COP_EARLY, CONSENSUS.W_MK_EARLY
    elif state.minuto <= CONSENSUS.MID_GAME_MINUTE:
        _w_bp, _w_cop, _w_mk = CONSENSUS.W_BP_MID, CONSENSUS.W_COP_MID, CONSENSUS.W_MK_MID
    else:
        _w_bp, _w_cop, _w_mk = CONSENSUS.W_BP_LATE, CONSENSUS.W_COP_LATE, CONSENSUS.W_MK_LATE

    if _n_ok < 3:
        import logging as _log_mod
        _log_mod.getLogger("exchange.engine").warning(
            "Solo %d/3 modelli disponibili — redistribuzione pesi consensus", _n_ok
        )
        if not _bp_ok:
            _w_bp = 0.0
        if not _cop_ok:
            _w_cop = 0.0
        if not _mk_ok:
            _w_mk = 0.0
        _w_sum = _w_bp + _w_cop + _w_mk
        if _w_sum > 0:
            _w_bp /= _w_sum
            _w_cop /= _w_sum
            _w_mk /= _w_sum

    _w_bp, _w_cop, _w_mk = blend_consensus_weights_with_history(
        state.minuto, _w_bp, _w_cop, _w_mk
    )

    # Upgrade 8-2: Correlation-aware ensemble — aggiusta pesi per correlazione.
    # Se 2 modelli danno previsioni quasi identiche, riduce il peso combinato
    # e redistribuisce al terzo (meno correlato → più informativo).
    try:
        from src.models.correlation_ensemble import adjust_weights_for_correlation
        _per_raw_pre = per_model_market_probs(
            full_bp, full_copula, full_markov,
            state.gol_casa, state.gol_trasf, state.linea_ou,
        )
        _w_bp, _w_cop, _w_mk = adjust_weights_for_correlation(
            _w_bp, _w_cop, _w_mk,
            _per_raw_pre["bp"], _per_raw_pre["copula"], _per_raw_pre["markov"],
        )
    except Exception:
        pass  # Best-effort: non rompere il pipeline

    _per_raw = per_model_market_probs(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, state.linea_ou,
    )
    _agree_1x2_pre = agreement_1x2_from_per_raw(_per_raw)
    consensus_probs = compute_consensus(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, state.linea_ou,
        weights=(_w_bp, _w_cop, _w_mk),
    )
    # O/U 1.5: stesso consensus (gol totali ≥2 vs ≤1). H2H "Over %" Nowgoal è tipicamente su 2.5 → non blendare qui.
    consensus_probs_15 = compute_consensus(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, 1.5,
        weights=(_w_bp, _w_cop, _w_mk),
    )
    _raw_o15 = consensus_probs_15["p_over"]
    p_over_15 = (
        logistic_sharpen_over(_raw_o15)
        if _raw_o15 not in (0.0, 1.0)
        else _raw_o15
    )
    p_under_15 = 1.0 - p_over_15
    # O/U 2.5 canonico per workflow book europei (sempre disponibile in output UI).
    consensus_probs_25 = compute_consensus(
        full_bp, full_copula, full_markov,
        state.gol_casa, state.gol_trasf, 2.5,
        weights=(_w_bp, _w_cop, _w_mk),
    )
    _raw_o25 = consensus_probs_25["p_over"]
    p_over_25_ref = (
        logistic_sharpen_over(_raw_o25)
        if _raw_o25 not in (0.0, 1.0)
        else _raw_o25
    )
    p_under_25_ref = 1.0 - p_over_25_ref

    # 9. Calibrazione isotonica con draw shrinkage dinamico (#4).
    # Partite difensive (tot basso) → meno correzione sul pareggio.
    # Partite aperte (tot alto) → più correzione (la Poisson sovrastima di più il draw).
    _draw_factor = CONSENSUS.DRAW_SHRINKAGE_BASE * (tot_cur_eff / CONSENSUS.DRAW_SHRINKAGE_TOT_REF)
    _draw_factor = max(CONSENSUS.DRAW_SHRINKAGE_MIN_FACTOR, min(CONSENSUS.DRAW_SHRINKAGE_MAX_FACTOR, _draw_factor))
    _draw_shrinkage_dyn = 1.0 - _draw_factor
    _draw_shrinkage_dyn *= 1.0 - CONSENSUS.DRAW_AGREEMENT_SHRINK_BONUS * (1.0 - _agree_1x2_pre)
    _draw_shrinkage_dyn = max(
        CONSENSUS.DRAW_SHRINKAGE_ABS_FLOOR,
        min(1.0, _draw_shrinkage_dyn),
    )

    p1, px, p2, p_over, p_under, p_btts = calibrate_probabilities(
        consensus_probs["p1"], consensus_probs["px"], consensus_probs["p2"],
        consensus_probs["p_over"], consensus_probs["p_under"], consensus_probs["p_btts"],
        draw_shrinkage=_draw_shrinkage_dyn,
    )

    # 9a. BTTS Calibration contestuale (solo prematch).
    # Corregge il bias del modello Poisson basandosi su:
    # - Total atteso (partite difensive vs aperte)
    # - Forza offensiva/defensiva (mismatch)
    # - H2H storico BTTS
    # - Forma recente (streak gol/clean sheet)
    if state.minuto == 0:
        from src.markets.btts import calibra_btts
        p_btts = calibra_btts(
            p_btts_raw=p_btts,
            tot_cur=tot_cur_eff,
            xg_h=xg_h_final,
            xg_a=xg_a_final,
            h2h_btts_pct=state.h2h_btts_pct,
            h2h_btts_n=state.h2h_matches_count,
            last6_gf_h=state.last6_gf_h,
            last6_ga_h=state.last6_ga_h,
            last6_gf_a=state.last6_gf_a,
            last6_ga_a=state.last6_ga_a,
            scoring_streak_h=state.scoring_streak_h,
            scoring_streak_a=state.scoring_streak_a,
            clean_sheet_streak_h=state.clean_sheet_streak_h,
            clean_sheet_streak_a=state.clean_sheet_streak_a,
            minuto=state.minuto,
        )

    # 9b. Prior BTTS da quote OCR (solo prematch): nudge conservativo.
    # Le quote GG/NG contengono informazione su formazioni e stile di gioco
    # non catturata dalle linee AH/Total.
    if state.minuto == 0 and state.ocr_quota_gg > 1.0 and state.ocr_quota_ng > 1.0:
        from src.config import OCR_QUOTES
        from src.models.calibration import _devig_two_way
        _overround_btts = 1.0 / state.ocr_quota_gg + 1.0 / state.ocr_quota_ng
        if _overround_btts <= OCR_QUOTES.MAX_OVERROUND_2WAY:
            _p_gg_ocr = _devig_two_way(state.ocr_quota_gg, state.ocr_quota_ng)
            # #3: Peso scalato per qualità dell'overround BTTS
            _quality_btts = 1.0 - min(OCR_QUOTES.BTTS_QUALITY_MAX_PENALTY,
                                       max(0.0, (_overround_btts - 1.0) * OCR_QUOTES.BTTS_QUALITY_OVERROUND_RATE))
            _w_btts = OCR_QUOTES.BTTS_PRIOR_WEIGHT * _quality_btts
            p_btts = (1.0 - _w_btts) * p_btts + _w_btts * _p_gg_ocr

    # 9c. Post-consensus 1X2 correction da market initial odds e H2H prior (solo prematch).
    # Due segnali indipendenti dal modello Poisson/Copula/Markov vengono blended
    # con peso conservativo per ancorare la 1X2 a informazioni esterne.
    # Pesi: 8% market-implied 1X2 + 5% H2H storico → max 13% totale.
    if state.minuto == 0:
        _alpha_mkt  = 0.08  # peso quote 1X2 iniziali bookmaker
        _alpha_h2h  = 0.05  # peso H2H storico 1X2
        _p1_adj, _px_adj, _p2_adj = p1, px, p2  # partenza dal consensus calibrato

        # Market-implied 1X2 (rimuovi vig e normalizza)
        if state.mkt_init_1 > 1.0 and state.mkt_init_x > 1.0 and state.mkt_init_2 > 1.0:
            _raw1 = 1.0 / state.mkt_init_1
            _rawx = 1.0 / state.mkt_init_x
            _raw2 = 1.0 / state.mkt_init_2
            _tot_raw = _raw1 + _rawx + _raw2
            if _tot_raw > 0:
                _p1_mkt = _raw1 / _tot_raw
                _px_mkt = _rawx / _tot_raw
                _p2_mkt = _raw2 / _tot_raw
                _p1_adj = (1.0 - _alpha_mkt) * _p1_adj + _alpha_mkt * _p1_mkt
                _px_adj = (1.0 - _alpha_mkt) * _px_adj + _alpha_mkt * _px_mkt
                _p2_adj = (1.0 - _alpha_mkt) * _p2_adj + _alpha_mkt * _p2_mkt

        # H2H storico 1X2 (normalizza le percentuali)
        _h2h_sum = state.h2h_home_win_pct + state.h2h_draw_pct + state.h2h_away_win_pct
        if _h2h_sum > 0:
            _p1_h2h = state.h2h_home_win_pct / _h2h_sum
            _px_h2h = state.h2h_draw_pct / _h2h_sum
            _p2_h2h = state.h2h_away_win_pct / _h2h_sum
            _p1_adj = (1.0 - _alpha_h2h) * _p1_adj + _alpha_h2h * _p1_h2h
            _px_adj = (1.0 - _alpha_h2h) * _px_adj + _alpha_h2h * _px_h2h
            _p2_adj = (1.0 - _alpha_h2h) * _p2_adj + _alpha_h2h * _p2_h2h

        # Rinormalizza per garantire somma = 1
        _sum_1x2 = _p1_adj + _px_adj + _p2_adj
        if _sum_1x2 > 0:
            p1 = _p1_adj / _sum_1x2
            px = _px_adj / _sum_1x2
            p2 = _p2_adj / _sum_1x2

    # 9c-bis. Upgrade 8-4: HT/FT predictive model.
    # Usa i pattern di transizione HT→FT storici delle squadre per calibrare 1X2.
    # In prematch: distribuzione aggregata. In live (dopo HT): condiziona su stato HT.
    try:
        from src.models.htft_model import compute_htft_adjustment
        _ht_result = ""
        if state.minuto >= 45:
            if state.gol_casa > state.gol_trasf:
                _ht_result = "W"
            elif state.gol_casa < state.gol_trasf:
                _ht_result = "L"
            else:
                _ht_result = "D"
        p1, px, p2 = compute_htft_adjustment(
            p1, px, p2,
            htft_home_htw_ftw=state.htft_home_htw_ftw,
            htft_home_htw_ftd=state.htft_home_htw_ftd,
            htft_home_htw_ftl=state.htft_home_htw_ftl,
            htft_home_htd_ftw=state.htft_home_htd_ftw,
            htft_home_htd_ftd=state.htft_home_htd_ftd,
            htft_home_htd_ftl=state.htft_home_htd_ftl,
            htft_home_htl_ftw=state.htft_home_htl_ftw,
            htft_home_htl_ftd=state.htft_home_htl_ftd,
            htft_home_htl_ftl=state.htft_home_htl_ftl,
            htft_away_htw_ftw=state.htft_away_htw_ftw,
            htft_away_htw_ftd=state.htft_away_htw_ftd,
            htft_away_htw_ftl=state.htft_away_htw_ftl,
            htft_away_htd_ftw=state.htft_away_htd_ftw,
            htft_away_htd_ftd=state.htft_away_htd_ftd,
            htft_away_htd_ftl=state.htft_away_htd_ftl,
            htft_away_htl_ftw=state.htft_away_htl_ftw,
            htft_away_htl_ftd=state.htft_away_htl_ftd,
            htft_away_htl_ftl=state.htft_away_htl_ftl,
            minuto=state.minuto,
            ht_result=_ht_result,
        )
    except Exception:
        pass  # Best-effort

    # 9d. H2H Over % blend (solo prematch) - Miglioramento #3.
    # Il dato H2H Over% è in genere riferito alla soglia 2.5: applichiamo il blend
    # solo quando la linea analizzata è 2.5 per evitare mismatch di mercato.
    if state.minuto == 0 and state.h2h_over_pct > 0 and abs(state.linea_ou - 2.5) < 1e-6:
        _alpha_over_h2h = 0.15
        _p_over_h2h = state.h2h_over_pct / 100.0  # Converti da % a proporzione
        p_over = (1.0 - _alpha_over_h2h) * p_over + _alpha_over_h2h * _p_over_h2h
        p_under = 1.0 - p_over  # Rinormalizza
    # Applica lo stesso micro-blend H2H anche al canale canonico O/U 2.5.
    if state.minuto == 0 and state.h2h_over_pct > 0:
        _alpha_over_h2h = 0.15
        _p_over_h2h = state.h2h_over_pct / 100.0
        p_over_25_ref = (1.0 - _alpha_over_h2h) * p_over_25_ref + _alpha_over_h2h * _p_over_h2h
        p_under_25_ref = 1.0 - p_over_25_ref

    # 9e. Previous Scores Over% blend (solo prematch) — Upgrade 1.
    # Il dato prev_over_pct indica la % di Over nelle ultime 10 partite per squadra.
    # È un segnale indipendente dall'H2H: cattura la tendenza recente della squadra.
    if state.minuto == 0:
        _prev_over_avg = 0.0
        _prev_over_count = 0
        if state.prev_over_pct_h > 0:
            _prev_over_avg += state.prev_over_pct_h / 100.0
            _prev_over_count += 1
        if state.prev_over_pct_a > 0:
            _prev_over_avg += state.prev_over_pct_a / 100.0
            _prev_over_count += 1
        if _prev_over_count > 0:
            _prev_over_avg /= _prev_over_count
            _alpha_prev_over = 0.08  # conservativo: 8% peso
            p_over = (1.0 - _alpha_prev_over) * p_over + _alpha_prev_over * _prev_over_avg
            p_under = 1.0 - p_over
            if abs(state.linea_ou - 2.5) < 1e-6:
                p_over_25_ref = (1.0 - _alpha_prev_over) * p_over_25_ref + _alpha_prev_over * _prev_over_avg
                p_under_25_ref = 1.0 - p_over_25_ref

    # 10. Correct score e distribuzione gol dal consensus (Fix #5).
    # Fix #2.7: Usa funzione helper per costruire la matrice consensus (pesi dinamici)
    full_consensus = _build_consensus_matrix(full_bp, full_copula, full_markov,
                                              _w_bp, _w_cop, _w_mk)
    full_matrix = full_consensus if full_consensus else full_bp
    top_cs, gol_tot_dist = calcola_correct_score(full_matrix, state.gol_casa, state.gol_trasf, UI.TOP_CS_COUNT)

    # 10b. Upgrade 8-1: Riconciliazione cross-mercato via score matrix.
    # Bilancia le probabilità "libere" del consensus con quelle "vincolate"
    # derivate dalla matrice di punteggio (coerenti per costruzione).
    _coherence_score = 1.0
    try:
        from src.models.probability_reconciliation import reconcile_probabilities
        p1, px, p2, p_over, p_under, p_btts, _coherence_score = reconcile_probabilities(
            p1, px, p2, p_over, p_under, p_btts,
            full_matrix, state.gol_casa, state.gol_trasf, state.linea_ou,
            p_over_15=p_over_15, p_under_15=p_under_15,
        )
    except Exception:
        pass  # Best-effort

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
        # #6: Se le quote OCR sono disponibili in prematch, la confidenza sale:
        # le quote del bookmaker hanno già digerito info su formazioni, meteo, ecc.
        _ocr_prematch = state.ocr_imp_total > 0 and state.minuto == 0
        _shots_conf = (ENGINE.OCR_PREMATCH_SHOTS_CONF if _ocr_prematch
                       else ENGINE.PREMATCH_SHOTS_CONF)
        _blend_conf = max(ENGINE.BLEND_CONF_STALE,
                          (ENGINE.BLEND_CONF_STALE if stale_line
                           else (ENGINE.BLEND_CONF_FLAT if flat_lines
                                else ENGINE.BLEND_CONF_NORMAL)))
    # FIX: Al prematch (minuto=0), flat_lines è sempre True perché apertura=corrente
    # è la normalità prima della partita. Usare LINE_CONF_FLAT/BLEND_CONF_FLAT
    # penalizza artificialmente il prematch riducendo la confidenza sotto soglia.
    # In prematch trattiamo le linee come NORMALI (linee fresche d'apertura).
    if state.minuto == 0:
        _line_conf = ENGINE.LINE_CONF_NORMAL
        if n_shots_tot == 0:
            _blend_conf = ENGINE.BLEND_CONF_NORMAL
    else:
        # Quando ci sono dati live reali (tiri, possesso, ecc.) i dati statistici
        # sono la fonte primaria di informazione. Le linee "flat" sono attese perché
        # l'utente non aggiorna le linee live: non penalizzare LINE_CONF in questo caso.
        if n_shots_tot > 0:
            _line_conf = ENGINE.LINE_CONF_STALE if stale_line else ENGINE.LINE_CONF_NORMAL
        else:
            _line_conf = ENGINE.LINE_CONF_STALE if stale_line else (ENGINE.LINE_CONF_FLAT if flat_lines else ENGINE.LINE_CONF_NORMAL)
    _time_conf = _math.sqrt(state.minuto / 90.0) if state.minuto > 0 else ENGINE.PREMATCH_TIME_CONF
    _agreement_conf = model_agreement
    # _agreement_conf può essere 0.0 → il suo guard è necessario.
    _product = max(1e-9, _shots_conf * _line_conf * _blend_conf
                   * _time_conf * max(0.01, _agreement_conf))
    # Fix #6.4: Usa parametro dal config per la radice
    model_confidence = min(1.0, _product ** ENGINE.CONFIDENCE_ROOT_POWER)

    # 13. Strength boost (Miglioramento #4) - solo prematch.
    # Se c'è una grande disparità di forza tra le squadre, aumenta la confidenza.
    # strength_diff alto → pronostico più affidabile.
    if state.minuto == 0 and state.strength_home > 0 and state.strength_away > 0:
        _strength_diff = abs(state.strength_home - state.strength_away) / 100.0
        _cov_sc = max(0.0, min(1.0, state.extraction_coverage))
        _str_w = 0.10 * (0.72 + 0.38 * (1.0 - _cov_sc))
        _strength_boost = 1.0 + _strength_diff * _str_w
        model_confidence = min(1.0, model_confidence * _strength_boost)

    # Upgrade 8-5: Calibrazione confidence da dati storici.
    # Sostituisce la formula euristica con una curva calibrata isotonicamente.
    if state.minuto == 0:
        try:
            from src.models.confidence_calibration import (
                apply_confidence_calibration,
                build_confidence_calibration_map,
            )
            _conf_cal_map = build_confidence_calibration_map()
            if _conf_cal_map:
                model_confidence = apply_confidence_calibration(
                    model_confidence, _conf_cal_map,
                )
        except Exception:
            pass  # Best-effort

    # L'utente usa sempre le linee di apertura/chiusura manualmente inserite
    # e non aggiorna le linee live — l'avviso non ha senso in questo workflow.
    lines_need_update = False

    # Market shock: movimento anomalo delle linee rispetto al tempo giocato.
    # Segnala possibile informazione asimmetrica (infortuni, formazioni, notizie).
    from src.config import MOMENTUM as _MOM
    market_shock = (state.minuto > 0 and momentum >= _MOM.MOMENTUM_SHOCK_THRESHOLD)

    # 14. Divergenza modello-mercato (proxy Brier) dalle quote OCR.
    # Confronta le probabilità del modello con quelle implicite nelle quote bookmaker.
    # Se le quote non sono disponibili, la divergenza resta 0.0 (nessuna penalità).
    _market_probs: dict[str, float] = {}
    if state.ocr_quota_1 > 1.0 and state.ocr_quota_x > 1.0 and state.ocr_quota_2 > 1.0:
        _sum_inv = 1.0 / state.ocr_quota_1 + 1.0 / state.ocr_quota_x + 1.0 / state.ocr_quota_2
        _market_probs["p1"] = (1.0 / state.ocr_quota_1) / _sum_inv
        _market_probs["px"] = (1.0 / state.ocr_quota_x) / _sum_inv
        _market_probs["p2"] = (1.0 / state.ocr_quota_2) / _sum_inv
    if state.ocr_quota_over > 1.0 and state.ocr_quota_under > 1.0:
        _sum_inv_ou = 1.0 / state.ocr_quota_over + 1.0 / state.ocr_quota_under
        _market_probs["p_over"] = (1.0 / state.ocr_quota_over) / _sum_inv_ou
        _market_probs["p_under"] = (1.0 / state.ocr_quota_under) / _sum_inv_ou
    _model_probs_for_div: dict[str, float] = {
        "p1": p1, "px": px, "p2": p2,
        "p_over": p_over, "p_under": p_under,
    }
    _market_divergence = compute_model_market_divergence(_model_probs_for_div, _market_probs)

    return ProbabilitaModello(
        p1=p1, px=px, p2=p2,
        p_under=p_under, p_over=p_over,
        p_btts=p_btts,
        p_over_15=p_over_15,
        p_under_15=p_under_15,
        p_over_25_ref=p_over_25_ref,
        p_under_25_ref=p_under_25_ref,
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
        market_divergence=_market_divergence,
        market_shock=market_shock,
        consensus_w_bp=_w_bp,
        consensus_w_cop=_w_cop,
        consensus_w_mk=_w_mk,
        p1_bp=_per_raw["bp"]["p1"],
        px_bp=_per_raw["bp"]["px"],
        p2_bp=_per_raw["bp"]["p2"],
        p1_cop=_per_raw["copula"]["p1"],
        px_cop=_per_raw["copula"]["px"],
        p2_cop=_per_raw["copula"]["p2"],
        p1_mk=_per_raw["markov"]["p1"],
        px_mk=_per_raw["markov"]["px"],
        p2_mk=_per_raw["markov"]["p2"],
        probability_coherence=_coherence_score,
    )
