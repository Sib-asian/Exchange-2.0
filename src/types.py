"""
types.py — Tipi centralizzati del motore di analisi.

MatchState (input), ExchangeQuotes (quote opzionali), ProbabilitaModello (output).
Estratti da engine.py per evitare dipendenze circolari e migliorare la modularità.
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

    # H2H: gol medi per squadra negli scontri diretti (URL Nowgoal, 0 = assente).
    h2h_avg_goals_home: float = 0.0
    h2h_avg_goals_away: float = 0.0
    # Numero match H2H con HT valido (affidabilità HT/FT).
    h2h_ht_matches_count: int = 0
    # Segnale sharp da movimento linee (Nowgoal), 0 = assente.
    odds_sharp_signal: float = 0.0
    # Peso sezione H2H (extraction_section_scores), [0,1]. Riduce blend gol medi H2H se basso.
    h2h_core_weight: float = 1.0

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
    # Fiducia sintetica [0.55,1]: penalità se extraction_notes segnala buchi critici.
    extraction_trust_factor: float = 1.0

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

    # H2H: % partite in cui la casa ha coperto l'Asian Handicap (0–100, da URL)
    h2h_ah_home_cover_pct: float = 0.0
    # % vittorie ultime partite (previous scores, 0–100)
    prev_win_pct_h: float = 0.0
    prev_win_pct_a: float = 0.0
    # xG/media gol implicita da partite recente (post OCR, 0 = assente)
    recent_xg_prior_h: float = 0.0
    recent_xg_prior_a: float = 0.0
    # Motivazione qualitativa estratta (high | normal | low)
    motivation_home: str = "normal"
    motivation_away: str = "normal"
    # Trend forma URL ([-1,1], positivo = forma in miglioramento).
    url_form_trend_h: float = 0.0
    url_form_trend_a: float = 0.0
    # Qualità linee manuali prematch [0,1] da validazioni input.
    line_quality_factor: float = 1.0

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
    p_over_bp: float = 0.0
    p_over_cop: float = 0.0
    p_over_mk: float = 0.0
    # Over 2.5 per modello (marginali da matrice a linea 2.5) — ensemble storico EU.
    p_over_bp_eu: float = 0.0
    p_over_cop_eu: float = 0.0
    p_over_mk_eu: float = 0.0
    p_btts_bp: float = 0.0
    p_btts_cop: float = 0.0
    p_btts_mk: float = 0.0
    consensus_w_ou_bp: float = 0.0
    consensus_w_ou_cop: float = 0.0
    consensus_w_ou_mk: float = 0.0
    consensus_w_btts_bp: float = 0.0
    consensus_w_btts_cop: float = 0.0
    consensus_w_btts_mk: float = 0.0
    xg_h_pre_prev_blend: float = 0.0
    xg_a_pre_prev_blend: float = 0.0
    prev_lambda_h: float = 0.0
    prev_lambda_a: float = 0.0

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
    # Dopo pipeline prematch: tightness CI [0,1] per Kelly (1 = modelli concordi).
    pipeline_ci_tightness: float = 0.55

    # Upgrade 8-1: Score di coerenza cross-mercato [0, 1].
    # 1.0 = probabilità perfettamente coerenti tra mercati.
    # <0.5 = divergenza significativa tra consensus e matrice punteggio.
    probability_coherence: float = 1.0

    # Correct score: diagnostica sulla distribuzione completa (non solo top-N).
    cs_entropy_nats: float = 0.0
    cs_top3_mass: float = 0.0
