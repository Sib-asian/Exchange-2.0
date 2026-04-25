"""
config.py — Costanti e parametri centralizzati del motore Radar Pro Live.

Tutti i magic numbers del motore sono qui documentati con fonte e motivazione.
Modificare solo qui: nessuna costante hardcodata nei moduli.
"""

from dataclasses import dataclass

# Versione unica app / motore (UI.VERSION e ENGINE.MODEL_REVISION devono coincidere).
MODEL_APP_VERSION: str = "2.0-phase3-seven-upgrades"


@dataclass(frozen=True)
class PoissonConfig:
    """Parametri per la distribuzione di Poisson e la matrice bivariata."""

    # Coda troncata: P(X > k) < tail_mass prima di fermarsi
    TAIL_MASS: float = 1e-12

    # Soglia di probabilità sotto cui saltare i termini nel prodotto
    PROB_SKIP_THRESHOLD: float = 1e-16

    # Iterazioni bisection per trovare delta* (2^52 ≈ 4e15: convergenza float garantita)
    BISECTION_ITERS: int = 52

    # Tolleranza convergenza bisection (EV residuo)
    BISECTION_TOL: float = 1e-9

    # Cap su lambda0 / min(mu_c, mu_t): 75% (Karlis & Ntzoufras 2003)
    LAMBDA0_CAP_RATIO: float = 0.75

    # Cap assoluto per lambda_h/lambda_a: prevenire valori insensati
    LAMBDA_MAX: float = 8.0

    # Soglia minima per xG (evita divisioni per zero)
    EPS: float = 1e-9

    # PMF truncation: parametri per calcolo max_k
    # max_k = max(PMF_MIN_K, int(mu + PMF_SIGMA * sqrt(mu) + PMF_EXTRA_BUFFER))
    # PMF_MIN_K=20: garantisce copertura minima per mu bassi
    # PMF_SIGMA=6: 6 deviazioni standard coprono >99.9999998% della distribuzione
    # PMF_EXTRA_BUFFER=10: margine extra per overdispersion e casi limite
    PMF_MIN_K: int = 20
    PMF_SIGMA: float = 6.0
    PMF_EXTRA_BUFFER: int = 10


@dataclass(frozen=True)
class DixonColesConfig:
    """Parametri correzione Dixon-Coles per punteggi bassi."""

    # Rho DC statico (fallback / default per test)
    RHO_DC: float = -0.13

    # Clamp tau: [0.05, 3.0] — 0.05 evita l'azzeramento, 3.0 evita amplificazioni eccessive
    TAU_MIN: float = 0.05
    TAU_MAX: float = 3.0

    # rho_DC dinamico: dipende da total, tempo, gol segnati.
    # Partite difensive (basso total) → più negativo (struttura difensiva).
    # Late game con punteggio chiuso → più negativo (parking the bus).
    # Partite aperte (alto gol) → meno negativo (difese aperte).
    RHO_DC_BASE: float = -0.08        # base per partite aperte
    RHO_DC_TOT_SCALE: float = -0.04   # effetto total basso → più negativo
    RHO_DC_TOT_REF: float = 3.0       # total di riferimento (sopra = meno negativo)
    RHO_DC_TIME_SCALE: float = -0.04  # effetto tempo → più negativo a fine partita
    RHO_DC_GOAL_DAMPEN: float = 0.15  # riduzione per gol segnati (partita aperta)
    RHO_DC_MIN: float = -0.25         # floor (max correlazione negativa)
    RHO_DC_MAX: float = -0.03         # ceiling (quasi indipendenza)

    # Markov chain DC correction floor
    # dc_corr_low = max(DC_CORR_FLOOR, 1.0 + rho_dc) per punteggi bassi
    # DC_CORR_FLOOR=0.70: evita correzioni troppo aggressive che rompono la normalizzazione
    DC_CORR_FLOOR: float = 0.70

    # Cartellini gialli: partita tesa → struttura difensiva → rho_DC più negativo
    # Shift aggiuntivo di -0.010 per ogni giallo sopra soglia (6), cap -0.04.
    RHO_DC_YELLOW_THRESHOLD: int = 6
    RHO_DC_YELLOW_SCALE: float = 0.010
    RHO_DC_YELLOW_MAX: float = 0.040


@dataclass(frozen=True)
class RhoConfig:
    """Parametri per il coefficiente di correlazione dinamico rho."""

    # Correlazione base (equilibrio, inizio partita)
    BASE_MAX: float = 0.14

    # Riduzione per gol attesi alti: 0.018 per unità di tot_cur
    BASE_DECAY_RATE: float = 0.018

    # Limite inferiore di tot_cur applicato al calcolo base
    BASE_TOT_CAP: float = 4.5

    # Decadimento temporale: -40% a fine partita
    TIME_DECAY_FACTOR: float = 0.40

    # Decadimento per gol segnati: -10% per gol, floor 50%
    GOAL_DECAY_RATE: float = 0.10
    GOAL_DECAY_FLOOR: float = 0.50

    # Riduzione per dominanza tiri: -45% con dominio totale
    SHOT_DOM_REDUCTION: float = 0.45

    # Interazione shot_dom × tempo: early game il dominio tiri è instabile,
    # late game è predittivo. Floor 0.40 = al minuto 0 solo il 40% dell'effetto.
    SHOT_DOM_TIME_FLOOR: float = 0.40

    # Interazione shot_dom × totale: in partite aperte (alto totale) il dominio
    # tiri è meno informativo (entrambe le squadre tirano molto).
    SHOT_DOM_TOT_CAP: float = 4.5     # tot_cur a cui si raggiunge la riduzione max
    SHOT_DOM_TOT_REDUCTION: float = 0.50  # riduzione massima del shot_dom

    # Floor assoluto di rho
    RHO_MIN: float = 0.02


@dataclass(frozen=True)
class ShotConfig:
    """Parametri per la stima xG dai tiri."""

    # xG per tiro in porta senza dati posizionali (top-5 leagues, open play)
    # Letteratura: 0.10-0.35 a seconda del contesto; 0.30 è stima conservativa
    # per tiri in porta in assenza di heatmap (Caley 2015, Statsbomb open data)
    XG_SOT: float = 0.30

    # xG per tiro fuori porta (StatsBomb open data senza dati posizionali: 0.06-0.10)
    XG_SOFF: float = 0.07

    # xG per tiro bloccato: valore < soff perché il tiro è stato intercettato prima
    # dello specchio → qualità media inferiore. Letteratura: ~0.03-0.05.
    XG_BLK: float = 0.04

    # Numero di tiri totali per considerare il campione "sufficiente"
    SHOT_INFO_THRESHOLD: int = 15

    # Peso massimo dei tiri sul Totale (il Total-line è molto efficiente)
    ALPHA_T_MAX: float = 0.25

    # Tasso di crescita del peso T con la frazione giocata (calibrato su top-5 leagues)
    # Crescita lenta: il Total-line è molto efficiente e i tiri lo correggono poco
    ALPHA_T_RATE: float = 0.30

    # Peso massimo dei tiri sul Differenziale (più informativo del mercato)
    ALPHA_D_MAX: float = 0.70

    # Limite superiore alpha_D quando la qualità degli attacchi è alta
    # (rapporto att.pericolosi/att.totali elevato → i tiri sono più informativi)
    ALPHA_D_MAX_QUALITY: float = 0.80

    # #2: Late-game dampening di alpha_D per riequilibrare T vs D.
    # Dopo il 60' (frac=0.67), alpha_D cresce più veloce di alpha_T (sqrt vs lineare).
    # Applichiamo una riduzione lineare del 25% sull'eccesso dopo frac=0.67,
    # con floor a 0.75 del valore calcolato per non azzerare mai l'effetto D.
    ALPHA_D_LATE_GAME_FRAC: float = 0.667   # dal 60' in poi
    ALPHA_D_LATE_GAME_DAMP: float = 0.25    # riduzione massima del 25% a fine partita
    ALPHA_D_LATE_GAME_MIN: float = 0.75     # floor: mantiene almeno 75% del valore

    # Smorzamento proiezione tiri: curva esponenziale (regressione alla media)
    # 0.75 a inizio partita → 1.0 a fine (campione ≈ universo).
    # Decay rate 2.0: converge rapidamente dopo il 30' (campione affidabile).
    RATE_DAMP_FLOOR: float = 0.75
    RATE_DAMP_DECAY: float = 2.0

    # Shrinkage Bayesiano verso la media di lega (Stein shrinkage).
    # Riduce la varianza delle previsioni shot-based verso il prior di lega.
    # 10% verso la media con campione perfetto; cresce con pochi tiri.
    SHRINKAGE_WEIGHT: float = 0.10
    LEAGUE_MEAN_RATE: float = 2.7  # gol/90' media top-5 leagues (fallback)

    # Moltiplicatori qualità tiri basati su rapporto att.pericolosi/att.totali
    # q_ratio alto (es. 0.40) → attacchi di alta qualità → XG_SOT effettivo più alto
    # Range: [ATT_QUALITY_MIN_MULT, ATT_QUALITY_MAX_MULT] × XG_SOT base
    ATT_QUALITY_MIN_MULT: float = 0.85   # qualità minima (pochi pericol. su molti att.)
    ATT_QUALITY_MAX_MULT: float = 1.15   # qualità massima (quasi tutti pericolosi)

    # #5: Scala temporale adv_shift modulata dalla magnitudine del dominio.
    # Un dominio 70-30 è significativo anche al 10', mentre 52-48 è rumore.
    # time_scale = min(1, frac * dom_factor) dove:
    #   dom_factor ∈ [ADV_DOM_SCALE_MIN, ADV_DOM_SCALE_MAX]
    # Magnitudine bassa → fattore minimo (conservativo), magnitudine alta → fattore max.
    ADV_DOM_SCALE_MIN: float = 1.0   # dom=0 (bilanciato) → time_scale = frac × 1.0
    ADV_DOM_SCALE_MAX: float = 2.0   # dom=1 (totale) → time_scale = frac × 2.0

    # Correzione game-state differenziata per tipo di tiro
    # SOT (in porta): +10% per gol di vantaggio (contropiede di qualità)
    # SOFF (fuori porta): +3% (pressing disperato, bassa qualità)
    GAME_STATE_RATE_SOT: float = 0.10
    GAME_STATE_RATE_SOFF: float = 0.03
    GAME_STATE_CAP: float = 0.20


@dataclass(frozen=True)
class TimeDecayConfig:
    """Parametri per l'effetto temporale e score effect."""

    # Score effect residuale: cap assoluto (AH live già copre ~80% dell'effetto)
    SCORE_EFFECT_MAX: float = 0.08
    SCORE_EFFECT_BASE: float = 0.07

    # Ritmo di gioco: cartellini gialli sopra soglia → partita spezzata → meno gol
    # Effetto leggero: max -10% con 12+ gialli totali. Soglia=6 (3 per squadra).
    YELLOW_RHYTHM_THRESHOLD: int = 6     # gialli totali sopra cui scatta la correzione
    YELLOW_RHYTHM_RATE: float = 0.008    # riduzione per giallo oltre soglia
    YELLOW_RHYTHM_MIN: float = 0.90      # floor: max -10%

    # Ritmo di gioco: falli sopra soglia → gioco spezzettato → meno gol
    # Max -5% con 30+ falli totali. Soglia=20 (10 per squadra).
    FOUL_RHYTHM_THRESHOLD: int = 20      # falli totali sopra cui scatta la correzione
    FOUL_RHYTHM_RATE: float = 0.003      # riduzione per fallo oltre soglia
    FOUL_RHYTHM_MIN: float = 0.95        # floor: max -5%

    # Saturazione score effect: 1.5 (più rapida di 2.0 per pressing tardivo)
    SCORE_SATURATION: float = 1.5

    # Scale temporale per lo score effect:
    # 0' → 1.20 (max boost ~8.4%), 45' → 0.65, 75' → 0.38, 85' → 0.30
    SCORE_MINUTE_SCALE_A: float = 1.20
    SCORE_MINUTE_SCALE_B: float = 1.10

    # Floor scale temporale (fine partita)
    SCORE_MINUTE_SCALE_FLOOR: float = 0.30

    # Asimmetria score effect (Brechot & Flepp 2020, Robberechts 2021):
    # la squadra in svantaggio preme con tiri di bassa qualità (pressing disperato),
    # mentre la squadra in vantaggio mantiene qualità ma riduce volume.
    SCORE_DOWN_MULTIPLIER: float = 0.65   # squadra in svantaggio: boost (pressing disperato)
    # Asimmetria: la squadra che difende riduce il volume meno di quanto quella che preme aumenta.
    # Pressing team (losing): +0.65*residual (urgency boost)
    # Defending team (winning): -0.40*residual (small reduction, parking the bus = controlled)
    SCORE_DEFENSE_MULT: float = 0.40      # squadra in vantaggio: riduzione difensiva (< SCORE_DOWN)

    # Goal intensity: il residuo cala in partite ad alto punteggio
    # (l'AH live ha già incorporato la volatilità, il residuo cattura solo il delta)
    SCORE_GOAL_INTENSITY_SCALE: float = 0.25  # riduzione per ogni gol segnato (cap a 4)

    # Cartellini rossi — effetto marginale decrescente (Brechot & Flepp 2020)
    # Tabella precalcolata: indice = numero di rossi (0-4)
    RED_DECAY: tuple = (1.000, 0.680, 0.578, 0.532, 0.500)
    RED_BOOST: tuple = (1.000, 1.280, 1.434, 1.520, 1.566)

    # Asimmetria home/away per cartellini rossi (~5% più grave in trasferta)
    RED_AWAY_PENALTY: float = 0.95   # moltiplicatore decay extra per la trasferta
    RED_HOME_BOOST: float = 1.04     # boost aggiuntivo per la casa

    # Interazione rosso × tempo rimanente: rosso early → effetto pieno, rosso late → dimezzato
    RED_TIME_FLOOR: float = 0.50     # floor del moltiplicatore temporale (rosso a min 90)

    # Smorzamento xG per momentum estremo: mercati molto volatili segnalano
    # informazione non catturata → smorzare xG offensivo per evitare falsi edge.
    MOMENTUM_XG_THRESHOLD: float = 2.5   # sotto questa soglia nessun effetto
    MOMENTUM_XG_DAMP_RATE: float = 0.05  # riduzione xG per unità di momentum sopra soglia
    MOMENTUM_XG_DAMP_MAX: float = 0.15   # riduzione massima (15%)

    # Floor per xG dopo tutti gli aggiustamenti
    XG_FLOOR: float = 0.001


@dataclass(frozen=True)
class BayesianConfig:
    """Parametri per il blend bayesiano linee apertura/corrente."""

    # Peso minimo della linea corrente (inizio partita)
    W_CUR_MIN: float = 0.65

    # Peso massimo della linea corrente (fine partita)
    W_CUR_MAX: float = 0.90

    # Decadimento esponenziale del peso apertura: le quote d'apertura perdono
    # valore più rapidamente early game (eventi compound) ma mantengono un residuo.
    # w_op = 0.35 * exp(-2.5 * frac²) + 0.10 * (1 - frac)
    W_OP_EXP_RATE: float = 2.5

    # Floor della frazione rimanente (evita "ghost goals" in zona recupero)
    FRAC_RIMASTA_FLOOR: float = 0.005

    # Totale minimo bayesiano (evita tot_bayes troppo basso)
    TOT_BAYES_MIN: float = 0.20

    # Soglia delta per considerare le linee "flat" (non applicare il blend).
    # Le linee si muovono in step di ±0.25 o ±0.5; sotto 0.10 il mercato non si è mosso.
    FLAT_LINE_THRESHOLD: float = 0.10

    # Cap rapporto xG: se max(xg_h,xg_a)/min(xg_h,xg_a) > XG_RATIO_CAP,
    # blend verso approssimazione lineare per evitare split estremi a fine partita.
    # 5:1 scelto come compromesso: permette split plausibili (70%-30% del totale)
    # ma evita rapporti >10:1 irrealistici quando tot_bayes < 0.5.
    XG_RATIO_CAP: float = 5.0

    # Cap temporale per i gol rimanenti: max = (90-minuto)/90 * TOT_TEMPORAL_MAX.
    # Protegge dal caso in cui l'utente non aggiorna le linee live al progredire
    # della partita. Senza questo cap, al 80' con linee invariate da inizio partita
    # il modello usava tot_cur=2.75 remaining (invece di ~0.4) → xG insensati.
    # 4.0 = massimo realistico per una partita con gol/90 elevatissimo:
    #   minuto=0: max=4.0 | minuto=45: max=2.0 | minuto=80: max=0.44
    TOT_TEMPORAL_MAX: float = 4.0

    # Interpolazione AH EV: correzione Hermite per quarter lines
    # La correzione curvatura riduce l'errore di interpolazione da ±2.5% a ±0.2%
    # HERMITE_CORRECTION_COEF = 0.05: coefficiente conservativo calibrato empiricamente
    HERMITE_CORRECTION_COEF: float = 0.05

    # Newton-Raphson per bisection AH
    NEWTON_MAX_ITER: int = 15
    NEWTON_H: float = 1e-7  # step per derivata numerica

    # Peso della linea O/U estratta da OCR nel blend del prior Bayesiano.
    # Solo in prematch (minuto=0): blenda tot_op con la linea OCR al 15%.
    # Conservativo: l'OCR può avere margine o leggere un mercato leggermente diverso.
    OCR_PRIOR_WEIGHT: float = 0.15

    # Peso del prior storico H2H (da Gemini) nel blend del totale atteso.
    # Solo in prematch (minuto=0): contributo conservativo 10%.
    # I dati H2H storici sono informativi ma possono riflettere formazioni passate.
    FIXTURE_PRIOR_WEIGHT: float = 0.10

    # Clamp del moltiplicatore qualità movimento linee (da Gemini).
    # < 1.0 = movimento non affidabile (rumore/pubblico) → meno fiducia sulla linea corrente.
    # > 1.0 = movimento affidabile (sharp/notizie) → più fiducia sulla linea corrente.
    MOVEMENT_QUALITY_MIN: float = 0.80
    MOVEMENT_QUALITY_MAX: float = 1.30

    # --- Potenziamento uso movimento apertura → corrente (linee manuali / live) ---
    # Dopo i pesi tempo+movement_quality, se il mercato si è mosso abbastanza in
    # spazio full-game, aumenta w_cur (fino a W_CUR_MAX) così la bisection AH
    # ancori gli xG alla linea corrente, non solo al blend statico.
    LINE_MOVE_TOT_SCALE: float = 0.50   # stessa scala del Total nel momentum (½ peso vs AH)
    LINE_MOVE_BOOST_THRESHOLD: float = 0.10  # sotto ~un quarto di linea: nessun boost extra
    LINE_MOVE_W_CUR_BOOST_RATE: float = 0.22  # incremento w_cur per unità di movimento oltre soglia
    LINE_MOVE_W_CUR_BOOST_MAX: float = 0.14   # tetto incremento (evita di ignorare del tutto l'apertura)

    # GG/NG OCR (prematch): aggiustamento conservativo del totale implicito verso P(both score) fair.
    BTTS_OCR_TOT_ANCHOR: float = 0.52
    BTTS_OCR_TOT_ADJ_SCALE: float = 0.48  # × (p_gg - anchor), poi × ocr_cs e coverage
    BTTS_OCR_TOT_ADJ_CAP: float = 0.22    # max |Δ| sul tot_op prima della bisection

    # Sharp signal (Nowgoal): rafforza il contributo del movimento raw sul boost w_cur.
    SHARP_LINE_MOVE_EXTRA_RATE: float = 0.14  # × min(1, sharp) × coverage

    # Fallback bisection (stesso segno EV agli estremi): delta ≈ -ah_bayes.
    # Se P(0,0) implicita è alta (tot_bayes basso), smorza verso split più centrato.
    FALLBACK_ZERO_ZERO_WEIGHT: float = 0.72
    FALLBACK_DELTA_MIN_MULT: float = 0.38


@dataclass(frozen=True)
class MomentumConfig:
    """Parametri per il calcolo del momentum di mercato."""

    # Sqrt invece di lineare: evita overshoot early-game
    # A minuto 10: sqrt=0.33 (amplif. ×3) vs lineare=0.11 (amplif. ×9)
    FRAC_FLOOR: float = 0.15

    # Peso del Total nel calcolo del momentum (meno informativo dell'AH)
    TOT_WEIGHT: float = 0.5

    # Cap del momentum
    MOMENTUM_CAP: float = 6.0

    # Soglia "market shock": momentum superiore a questa soglia indica
    # un movimento anomalo delle linee (informazione asimmetrica, infortuni,
    # notizie non ancora riflesse nelle statistiche live).
    # > 4.0 = movimento estremo, molto raro in partite normali.
    MOMENTUM_SHOCK_THRESHOLD: float = 4.0

    # Contributo massimo del momentum statistico (dominio tiri/attacchi).
    # Aggiunto al momentum di mercato quando la dominanza è > 40%.
    # Conservativo: max +0.5 quando tutta la pressione è da un lato.
    STAT_MOMENTUM_MAX: float = 0.50

    # Soglie interpretative
    STABLE_THRESHOLD: float = 1.0
    MODERATE_THRESHOLD: float = 2.5
    SIGNIFICANT_THRESHOLD: float = 4.0


@dataclass(frozen=True)
class KellyConfig:
    """Parametri per il Kelly criterion e il dimensionamento delle stake."""

    # Frazione Kelly base (50% = half-Kelly)
    KELLY_BASE_FRACTION: float = 0.50

    # Riduzione late-game: graduale da KELLY_LATE_START a 90'.
    # Lo spread exchange esplode nel late game (4-8x) → riduzione 0.22.
    KELLY_LATE_GAME_REDUCTION: float = 0.22
    KELLY_LATE_START: int = 65   # inizio riduzione graduale

    # Scaling della confidenza sul Kelly: conf=1.0 → nessuna riduzione,
    # conf=0.0 → riduzione massima del 40%.
    KELLY_CONFIDENCE_SCALE: float = 0.60

    # Riduzione senza dati tiri live
    KELLY_NO_SHOTS_REDUCTION: float = 0.05

    # Intervalli di credibilità stretti (modelli concordi) → meno riduzione Kelly.
    # tightness ∈ [0,1] da credible_intervals; blend verso 1.0 = nessuna penalità CI.
    KELLY_CI_BLEND: float = 0.24

    # Floor assoluto della frazione Kelly
    KELLY_MIN_FRACTION: float = 0.20

    # Cap Kelly: max 5% del bankroll in un singolo bet
    KELLY_MAX_PCT: float = 0.05

    # Cap adattivo per edge piccolo: edge < 5% → cap 2.5% * fraction
    KELLY_SMALL_EDGE_THRESHOLD: float = 0.05
    KELLY_SMALL_EDGE_CAP_PCT: float = 0.025

    # Cap adattivo per edge medio: edge < 10% → cap 4% * fraction
    KELLY_MEDIUM_EDGE_THRESHOLD: float = 0.10
    KELLY_MEDIUM_EDGE_CAP_PCT: float = 0.040

    # Quota netta minima per calcolo Kelly BACK (sotto → stake 0)
    MIN_QUOTA_NETTA: float = 1.01

    # Quota minima per il LAY (sotto questa quota non ha senso)
    LAY_MIN_ODDS: float = 1.30


@dataclass(frozen=True)
class SignalConfig:
    """Parametri per la generazione dei segnali di betting."""

    # Edge netto minimo per BACK (dopo commissione) — baseline con confidenza piena.
    # Con confidenza bassa viene scalato dinamicamente: vedi MIN_EDGE_CONF_BOOST.
    # Aumentato da 0.030 a 0.040 per ridurre falsi positivi e segnali quasi identici.
    MIN_EDGE_BACK: float = 0.040

    # Edge netto minimo per LAY (rischio asimmetrico)
    # Aumentato da 0.040 a 0.050 per maggiore cautela sui lay.
    MIN_EDGE_LAY: float = 0.050

    # Boost sull'edge minimo quando il modello ha bassa confidenza.
    # Sotto MIN_CONFIDENCE_FOR_SIGNALS (0.45) gli avanzati vengono soppressi.
    # Tra 0.45 e CONF_EDGE_BOOST_HIGH il requisito di edge cresce linearmente
    # fino a MIN_EDGE_BACK + CONF_EDGE_BOOST_MAX.
    # Razionale: con linee stantie o modelli in disaccordo serve un edge molto
    # più netto per giustificare l'operazione (il modello potrebbe essere fuori).
    CONF_EDGE_BOOST_HIGH: float = 0.70  # sopra questa confidenza: edge normale
    CONF_EDGE_BOOST_MAX: float = 0.040  # max boost aggiuntivo (+4%) a conf minima

    # Probabilità minima per calcolo quota fair (sotto → fallback)
    MIN_PROB_FOR_QUOTE: float = 0.001

    # Quota fallback per probabilità quasi-zero
    MAX_QUOTE_FALLBACK: float = 999.0

    # Quota fair minima: eventi >80% → skip (già nel prezzo)
    MIN_FAIR_Q: float = 1.25

    # Margine lordo minimo per i segnali rapidi (senza quota exchange)
    MARGINE_RAPIDO: float = 0.06

    # Soglia prob minima per segnali rapidi BACK (cresce col tempo)
    # Aumentate per ridurre segnali quasi identici tra mercati diversi.
    SOGLIA_BACK_MIN: float = 0.58      # Aumentato da 0.55
    SOGLIA_BACK_BASE: float = 0.52     # Aumentato da 0.50
    SOGLIA_BACK_SLOPE: float = 0.08    # Ridotto da 0.10 per crescita più graduale

    # Soglia minima LIVE (minuto > 0): floor più alto per evitare segnali generici.
    # In live il mercato ha già reagito agli eventi → servono probabilità più nette.
    # Esempio: 1-0 al 30' → p1=68% è "ovvio", non vale un segnale (floor=0.63).
    SOGLIA_LIVE_BACK_MIN: float = 0.63   # +5% rispetto al prematch (0.58) — per 1 e 2
    SOGLIA_LIVE_DRAW_MIN: float = 0.58   # X (draw): nessun "lead penalty" → floor più basso
    SOGLIA_LIVE_OU_MIN: float = 0.65     # O/U live richiedono ancora più certezza

    # Penalità "vantaggio ovvio": alzare soglia per il BACK sulla squadra già vincente.
    # +8% per ogni gol di vantaggio (cap 12%), solo tra il 1' e il 69'.
    # Razionale: 1-0 al 30' → p1=68% è in gran parte l'effetto-punteggio, non edge.
    # La soglia sale a 0.71 (0.63+0.08) → il segnale scatta solo con vera convinzione.
    LEAD_SOGLIA_PENALTY_RATE: float = 0.08   # per gol di vantaggio
    LEAD_SOGLIA_PENALTY_CAP: float = 0.12    # cap assoluto
    LEAD_SOGLIA_MINUTE_CUTOFF: int = 70      # non applicare dopo il 69'

    # Double Chance: copre 2/3 esiti → threshold più alta del 1x2
    SOGLIA_DC: float = 0.73
    SOGLIA_DC_LIVE_MIN: float = 0.76

    # Correct Score: minimo per mostrare come segnale informativo
    SOGLIA_CS_MIN: float = 0.12

    # Soglia prob massima per segnali rapidi LAY
    # Ridotta da 0.35 a 0.30 per essere più selettivi.
    SOGLIA_LAY_MAX: float = 0.30
    LAY_MIN_FAIR_Q: float = 1.35       # Aumentato da 1.30

    # Soglia per BTTS No (strutturalmente più probabile di Sì).
    # Con 0.45/0.50, il segnale scattava anche con BTTS quasi 50/50 → troppo basso.
    # Con 0.55/0.60, richiede che BTTS No sia chiaramente dominante.
    SOGLIA_BTTS_NO_BASE: float = 0.55
    SOGLIA_BTTS_NO_MIN: float = 0.60

    # FIX: Soglie differenziate per mercato 1X2 vs O/U vs BTTS
    # Ogni mercato ha caratteristiche diverse e merita soglie dedicate.
    # 1X2: tre esiti, soglia base
    # Over/Under: due esiti, richiede probabilità più alta
    # BTTS Sì: richiede che ENTRAMBE le squadre segnino → più incerto → soglia più alta
    SOGLIA_1X2_OFFSET: float = 0.00      # Nessun offset per 1X2 (usa base)
    SOGLIA_OU_OFFSET: float = 0.03       # +3% per Over/Under (due esiti)
    SOGLIA_BTTS_OFFSET: float = 0.05     # +5% per BTTS (più volatile)

    # FIX: Distanza minima tra quote fair di segnali diversi
    # Evita di raccomandare segnali con quote quasi identiche.
    # Esempio: se BACK 1 ha fair @1.80 e BACK Over ha fair @1.82, sono troppo simili.
    MIN_QUOTE_DISTANCE: float = 0.15     # Differenza minima tra quote fair

    # FIX: Numero massimo di segnali rapidi da mostrare
    # Ridotto da illimitato a 3 per evitare confusione.
    MAX_SEGNALI_RAPIDI: int = 3
    MAX_SEGNALI_AVANZATI: int = 3

    # Quota fair minima per mostrare un segnale rapido.
    # Portato a 1.25 (da 1.15) per coerenza con la soglia degli avanzati:
    # eventi con P > 80% (q_fair < 1.25) sono già troppo prezzati per trovare value.
    QUICK_SIGNAL_MIN_FAIR_Q: float = 1.25

    # Confidenza minima del modello per mostrare qualsiasi segnale.
    # Sotto questa soglia la calibrazione non è affidabile (linee stantie, assenza dati tiri,
    # modelli in forte disaccordo) → nessun segnale, solo avviso.
    MIN_CONFIDENCE_FOR_SIGNALS: float = 0.45

    # Penalità sulle soglie quando i modelli sono in disaccordo.
    # model_agreement < MODEL_AGREEMENT_LOW → le soglie salgono fino a +PENALTY_MAX.
    # Esempio: agreement=0.40, low=0.60 → penalty = (0.60-0.40)/0.60 × 0.08 = 2.7%
    MODEL_AGREEMENT_LOW: float = 0.60
    MODEL_AGREEMENT_PENALTY_MAX: float = 0.08

    # Soglia qualitativa senza quota exchange
    SOGLIA_QUALITATIVA_BASE: float = 0.60
    SOGLIA_QUALITATIVA_SLOPE: float = 0.10
    SOGLIA_QUALITATIVA_MIN: float = 0.62

    # Over bonus per gol mancanti (cap 6%)
    OVER_GOL_BONUS_RATE: float = 0.02
    OVER_GOL_BONUS_CAP: float = 0.06
    OVER_BASE_THRESHOLD: float = 0.58
    OVER_BASE_MIN: float = 0.55

    # Riduzione stake per momentum anomalo (modulata dall'edge)
    MOMENTUM_STAKE_REDUCTION_RATE: float = 0.10
    MOMENTUM_STAKE_THRESHOLD: float = 2.5
    MOMENTUM_STAKE_FLOOR: float = 0.40

    # #8: Moltiplicatori momentum per mercato.
    # 1X2: meno sensibile al momentum (forma pre-partita domina)
    # BTTS Sì: molto sensibile (gol multipli = alta volatilità)
    # BTTS No: moderatamente sensibile
    # Over: neutro
    # Under: meno sensibile (il momentum Over non implica Under automatico)
    MOMENTUM_MKT_1X2: float = 0.60
    MOMENTUM_MKT_BTTS_SI: float = 1.20
    MOMENTUM_MKT_BTTS_NO: float = 1.10
    MOMENTUM_MKT_OVER: float = 1.00
    MOMENTUM_MKT_UNDER: float = 0.80

    # #4: Scala la riduzione da momentum per model_agreement.
    # Quando i modelli divergono, la stima xG è già incerta → non amplificare con momentum.
    # effective_momentum = momentum * max(MOMENTUM_AGREE_FLOOR, model_agreement)
    MOMENTUM_AGREE_FLOOR: float = 0.50   # anche con agreement=0 usa 50% del momentum

    # Modulazione edge sulla riduzione momentum:
    # edge forte (>5%) → meno riduzione (il momentum è rumore)
    # edge debole (<2%) → più riduzione (il momentum è segnale)
    MOMENTUM_EDGE_STRONG: float = 0.05   # soglia edge "forte"
    MOMENTUM_EDGE_AMPLIFY: float = 0.50  # amplificazione per edge debole
    MOMENTUM_EDGE_DAMPEN: float = 0.30   # smorzamento per edge forte

    # Late-game LAY Over: abilitato solo dopo questo minuto con gol mancanti sufficienti.
    # Il LAY Over è normalmente disabilitato per evitare duplicati con BACK Under,
    # ma in late game la liquidità Over è superiore.
    LATE_GAME_LAY_OVER_MINUTE: int = 75
    LATE_GAME_LAY_OVER_GOALS: int = 2

    # Soglia fine partita: nessun segnale oltre questo minuto
    GAME_END_THRESHOLD: int = 85

    # BTTS No settled: partita a 88'+, mc_btts < 10%, almeno una squadra a 0 gol
    BTTS_NO_SETTLED_MINUTE: int = 88
    BTTS_NO_SETTLED_PROB_THRESHOLD: float = 0.10

    # Incoerenza BTTS Sì + Under bassa linea: soglie
    BTTS_UNDER_INCOHERENCE_BTTS: float = 0.50
    BTTS_UNDER_INCOHERENCE_UNDER: float = 0.55
    BTTS_UNDER_MAX_LINE: float = 1.5

    # Incoerenza Over + BTTS No: soglie
    OVER_BTTS_INCOHERENCE_OVER: float = 0.50
    OVER_BTTS_INCOHERENCE_BTTS: float = 0.35

    # FIX: Boost BTTS quando una squadra ha già segnato
    # Se una squadra ha segnato, BTTS Sì diventa più probabile.
    # Il boost è applicato solo se gol_casa > 0 XOR gol_trasf > 0.
    BTTS_ONE_GOAL_BOOST: float = 0.08     # +8% se esattamente una squadra ha segnato
    BTTS_TWO_GOAL_BOOST: float = 0.05     # +5% extra se entrambe hanno segnato (BTTS già vinto)

    # FIX: Gestione recovery time (minuti > 90)
    # Il modello continua a funzionare ma con warning
    RECOVERY_TIME_WARNING: int = 90       # Mostra warning dopo questo minuto
    RECOVERY_TIME_HARD_CAP: int = 120     # Blocca l'analisi dopo questo minuto


@dataclass(frozen=True)
class UIConfig:
    """Parametri di configurazione dell'interfaccia Streamlit."""

    PAGE_TITLE: str = "Radar Pro Live"
    PAGE_ICON: str = "⚡"
    VERSION: str = MODEL_APP_VERSION
    LAYOUT: str = "centered"

    # Linee U/O disponibili nel selectbox (step 0.25 per coprire Asian quarter lines).
    # Include quindi X.00, X.25, X.50, X.75 (es. 2.25 / 2.75) fino a 5.5.
    LINEE_OU: tuple = tuple(round(0.5 + 0.25 * i, 2) for i in range(21))

    # Tiri attesi per minuto (entrambe le squadre) — soglia warning input
    TIRI_PER_MINUTO: float = 0.65
    TIRI_WARNING_BUFFER: int = 4
    TIRI_MIN_BASE: int = 6

    # Numero massimo di correct score da mostrare
    TOP_CS_COUNT: int = 5

    # Massimo gol totali nella distribuzione
    MAX_GOL_DIST: int = 10

    # Correzione overdispersion per Correct Score con molti gol.
    # Il modello Poisson sottostima punteggi ad alto totale (i+j ≥ 3) perché
    # la varianza reale dei gol in una partita supera la media (overdispersion).
    # Overdispersion correct score: legge continua su gol futuri (monotona, no scalini).
    # mult = 1 + CS_OVERDISP_ALPHA * max(0, future_goals - CS_OVERDISP_K0)^CS_OVERDISP_EXP,
    # poi cap CS_OVERDISP_MAX. (Vecchi scalini 3/4/5 restano come commento di calibrazione.)
    CS_OVERDISP_K0: float = 2.5
    CS_OVERDISP_ALPHA: float = 0.102
    CS_OVERDISP_EXP: float = 1.12
    CS_OVERDISP_MAX: float = 1.24
    # Legacy step-function multipliers — non più usati, mantenuti solo come riferimento.
    # CS_OVERDISP_3: float = 1.05  ≈ mult(future_goals=3)  ~ 1 + 0.102 * 0.5^1.12 ≈ 1.046
    # CS_OVERDISP_4: float = 1.12  ≈ mult(future_goals=4)  ~ 1 + 0.102 * 1.5^1.12 ≈ 1.163
    # CS_OVERDISP_5: float = 1.20  ≈ mult(future_goals=5)  ~ 1 + 0.102 * 2.5^1.12 ≈ 1.240

    # Livelli AH da mostrare nell'expander
    AH_LEVELS: tuple = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0, +2.5)

    # Bankroll default e step
    BANKROLL_DEFAULT: float = 1000.0
    BANKROLL_STEP: float = 100.0

    # Commissione default
    COMM_DEFAULT: float = 2.5
    COMM_MAX: float = 10.0
    COMM_STEP: float = 0.5


@dataclass(frozen=True)
class CMPConfig:
    """Parametri per Conway-Maxwell-Poisson."""

    # Parametro di dispersione base: ν < 1 = overdispersion.
    # 0.92 calibrato su Premier League + La Liga (varianza/media ≈ 1.08-1.12)
    NU: float = 0.92

    # nu dinamico: partite difensive (basso total) → meno overdispersion (nu → 1.0)
    # partite aperte (alto total) → più overdispersion (nu → 0.85)
    NU_MIN: float = 0.85      # floor per partite molto aperte
    NU_MAX: float = 0.98      # ceiling per partite molto difensive
    NU_TOT_REF: float = 2.5   # total di riferimento (nu base)
    NU_TOT_SCALE: float = -0.04  # effetto total: basso → nu ↑, alto → nu ↓
    # Prematch: forte mismatch strength Nowgoal → più overdispersion (nu più basso)
    NU_STRENGTH_SCALE: float = 0.024
    NU_STRENGTH_REF: float = 55.0  # |sh_h - sh_a| su scala strength tipica


@dataclass(frozen=True)
class CopulaConfig:
    """Parametri per la copula Frank."""

    # θ base: dipendenza positiva leggera (gol di una squadra → più probabili dell'altra)
    THETA_BASE: float = 1.2
    # Riduzione θ per total alto (partite aperte → meno correlazione)
    THETA_TOT_SCALE: float = -0.25
    THETA_TOT_REF: float = 2.5
    # Riduzione θ per tempo avanzato
    THETA_TIME_SCALE: float = -0.20
    # θ risponde al dominio tiri (partite sbilanciate → leggera coda congiunta)
    THETA_SHOT_DOM_SCALE: float = 0.32
    THETA_SHOT_DOM_TIME_DAMP: float = 0.35  # early game: meno peso al shot_dom
    THETA_MAX: float = 3.45

    # Soglie numeriche per stabilità della copula Frank
    # Se |θ| < THETA_NEAR_ZERO, la copula degenera in indipendenza (C(u,v) = u*v)
    THETA_NEAR_ZERO: float = 1e-8
    # Se |exp(-θ) - 1| < INNER_NEAR_ZERO, il denominatore della formula è ~0
    INNER_NEAR_ZERO: float = 1e-15


@dataclass(frozen=True)
class HawkesConfig:
    """Parametri per il processo di Hawkes (self-exciting goals)."""

    # Boost rate per gol sopra atteso: max 3%, 1% per unità di eccesso
    # Valori conservativi per evitare che il Hawkes sovrasti lo score effect
    ALPHA: float = 0.01
    MAX_BOOST: float = 0.03
    # Tasso gol di riferimento (top-5 leagues: ~2.7/90')
    RATE_REF_PER_90: float = 2.7
    # Prior sulla media gol/90′ (Gamma–Poisson): λ_post = (k + λ_ref·w) / (u + w), u=minuto/90
    BAYES_PRIOR_GAMES: float = 0.32
    # Squadre con molti gol tardi (profilo “volatile”) → Hawkes meno aggressivo
    LATE_GOALS_HAWKES_THRESHOLD: float = 36.0
    LATE_GOALS_HAWKES_DAMP: float = 0.38  # riduzione max su ALPHA efficace

    # Minuto minimo per attivare Hawkes boost
    # Sotto questo minuto il campione è troppo piccolo (rumore).
    # MIN_MINUTE=15: sotto i 15' anche 2 gol sono rumore (rate = 2/15*90 = 12 gol/90')
    # Il floor a 15 è calibrato empiricamente: 2 gol al 6' → rate = 30 gol/90' = rumore puro
    MIN_MINUTE: int = 15


@dataclass(frozen=True)
class SubstitutionConfig:
    """Parametri per l'effetto sostituzione."""

    # Finestra sostituzione: 55'-70' = crescita, 70'-90' = plateau + decadimento
    BOOST_START: int = 55
    BOOST_PEAK: int = 70
    # Boost massimo: +6% sul rate di gol (Brechot & Flepp 2020)
    BOOST_MAX: float = 0.06
    # Asimmetria score: la squadra in svantaggio sostituisce più aggressivamente
    # ma con qualità inferiore (pressing disperato). Riduzione boost 40%.
    BOOST_SCORE_ASYMMETRY: float = 0.40


@dataclass(frozen=True)
class ConsensusConfig:
    """Parametri per il consenso multi-modello."""

    # Pesi dei 3 modelli nel consensus (default: prematch/early game)
    W_BIVARIATE: float = 0.50   # bivariate Poisson + DC (modello principale)
    W_COPULA: float = 0.30      # CMP + Frank copula (overdispersion)
    W_MARKOV: float = 0.20      # Markov chain (score-dependent rates)

    # #1/#6: Pesi dinamici per 4 fasi di gioco.
    # Prematch (0'): Poisson domina, nessuno stato partita.
    # Early (1-20'): transizione, Markov cresce.
    # Mid (20-60'): equilibrio, Markov contribuisce moderatamente.
    # Late (>60'): Markov eccelle con punteggio fissato come contesto.
    W_BP_PREMATCH: float = 0.55   # Bivariate Poisson prematch (minuto=0)
    W_COP_PREMATCH: float = 0.30  # Copula prematch
    W_MK_PREMATCH: float = 0.15   # Markov prematch (nessuno stato)
    W_BP_EARLY: float = 0.50      # Bivariate Poisson early (1-20')
    W_COP_EARLY: float = 0.25     # Copula early
    W_MK_EARLY: float = 0.25      # Markov early (pochi eventi, moderato)
    W_BP_MID: float = 0.45        # Bivariate Poisson mid (20-60')
    W_COP_MID: float = 0.25       # Copula mid
    W_MK_MID: float = 0.30        # Markov mid (stato parziale, cresce)
    W_BP_LATE: float = 0.40       # Bivariate Poisson late (>60')
    W_COP_LATE: float = 0.20      # Copula late
    W_MK_LATE: float = 0.40       # Markov late (punteggio fissato → molto informativo)
    EARLY_GAME_MINUTE: int = 20   # Sotto questo minuto (escl. 0) → pesi early
    MID_GAME_MINUTE: int = 60     # Sotto questo minuto (da EARLY a MID) → pesi mid
    LATE_GAME_MINUTE: int = 60    # Sopra questo minuto → pesi late (alias compat.)

    # Calibrazione isotonica — draw shrinkage dinamico
    # Il valore fisso 0.97 è ora usato come FALLBACK solo nei test;
    # in produzione si usa DRAW_SHRINKAGE_BASE per calcolo dinamico.
    DRAW_SHRINKAGE: float = 0.97  # fallback / compatibilità test

    # Draw shrinkage dinamico: riduzione proporzionale al totale atteso.
    # Partite difensive (tot basso) → minor correzione; aperte (tot alto) → maggiore.
    # Formula: draw_factor = DRAW_SHRINKAGE_BASE × (tot_cur / DRAW_SHRINKAGE_TOT_REF)
    # Clipped in [DRAW_SHRINKAGE_MIN_FACTOR, DRAW_SHRINKAGE_MAX_FACTOR].
    # draw_shrinkage = 1.0 - draw_factor
    DRAW_SHRINKAGE_BASE: float = 0.030        # fattore base a tot=2.5 → -3% draw
    DRAW_SHRINKAGE_TOT_REF: float = 2.5       # totale di riferimento
    DRAW_SHRINKAGE_MIN_FACTOR: float = 0.010  # min: shrinkage=0.990 (tot basso)
    DRAW_SHRINKAGE_MAX_FACTOR: float = 0.055  # max: shrinkage=0.945 (tot alto)
    # Spread 1X2 tra i 3 modelli (prima dell’isotonica): basso accordo → draw shrinkage più forte
    AGREEMENT_1X2_SPREAD_SCALE: float = 4.5
    DRAW_AGREEMENT_SHRINK_BONUS: float = 0.042
    DRAW_SHRINKAGE_ABS_FLOOR: float = 0.882
    # Uncertainty shrink: tirare il pareggio verso 1/3 più dei risultati se accordo basso
    DRAW_UNCERTAINTY_EXTRA: float = 0.032

    # Logistic sharpening: α > 1 rende le probabilità estreme più estreme
    # calibrate su dati Poisson vs reali: modello sottostima certezza agli estremi
    LOGISTIC_ALPHA_OVER: float = 1.03   # sharpening per Over/Under
    LOGISTIC_ALPHA_BTTS: float = 1.02   # sharpening per BTTS (più conservativo)
    # Code alte: più sharpening; code basse: meno (spesso già conservative)
    LOGISTIC_ALPHA_OVER_HIGH: float = 1.048
    LOGISTIC_ALPHA_OVER_LOW: float = 1.014
    LOGISTIC_ALPHA_BTTS_HIGH: float = 1.036
    LOGISTIC_ALPHA_BTTS_LOW: float = 1.012
    LOGISTIC_EXTREME_HIGH: float = 0.85
    LOGISTIC_EXTREME_LOW: float = 0.15

    # BTTS clamp epsilon: probabilità entro questa distanza da 0 o 1 sono clampate
    BTTS_CLAMP_EPSILON: float = 1e-12


@dataclass(frozen=True)
class StaleLineConfig:
    """Parametri per la rilevazione linee stantie."""

    # Minuti di inattività per considerare la linea stale
    THRESHOLD_MINUTES: int = 15
    # Degradazione del peso della linea corrente quando stale
    WEIGHT_DEGRADATION: float = 0.15


@dataclass(frozen=True)
class CacheConfig:
    """Parametri per il sistema di caching."""

    # Numero massimo di entry nel cache LRU
    MAX_SIZE: int = 100
    # Time-to-live in secondi (5 minuti)
    TTL_SECONDS: float = 300.0
    # Abilita/disabilita cache (utile per debug)
    ENABLED: bool = True


@dataclass(frozen=True)
class CleanSheetConfig:
    """Parametri per il calcolo Clean Sheet."""

    # Boost per clean sheet quando la squadra è in vantaggio (parking the bus)
    LEAD_BOOST: float = 0.05  # +5% per gol di vantaggio
    LEAD_BOOST_MAX: float = 0.15  # Cap a +15%
    # Penalità per partite con alto total atteso
    HIGH_TOTAL_PENALTY: float = 0.03  # -3% per ogni gol sopra 2.5


@dataclass(frozen=True)
class EngineConfig:
    """Parametri per l'engine di calcolo."""

    # Confidenza prematch (minuto=0, senza tiri)
    # PREMATCH_SHOTS_CONF=0.35: le linee di mercato sono l'unica fonte, confidenza media
    # PREMATCH_TIME_CONF=0.35: nessun tempo giocato, ma linee fresche
    PREMATCH_SHOTS_CONF: float = 0.35
    PREMATCH_TIME_CONF: float = 0.35

    # Boost confidenza prematch quando le quote OCR sono disponibili.
    # Le quote di bookmaker portano info aggiuntiva (formazioni, meteo, ecc.)
    # non contenuta nelle linee AH/Total → confidenza sale a 0.50.
    OCR_PREMATCH_SHOTS_CONF: float = 0.50

    # Model confidence: prodotto di 5 componenti, radice 5ª per normalizzare
    # CONFIDENCE_ROOT_POWER=0.20 (= 1/5): radice quinta
    CONFIDENCE_ROOT_POWER: float = 0.20

    # Confidenza base per linee: varia in base a stato
    # LINE_CONF_STALE=0.30: linee stantie (non aggiornate da >15')
    # LINE_CONF_FLAT=0.50: linee piatte (nessun movimento)
    # LINE_CONF_NORMAL=1.00: linee normali (movimenti rilevati)
    LINE_CONF_STALE: float = 0.30
    LINE_CONF_FLAT: float = 0.50
    LINE_CONF_NORMAL: float = 1.00

    # Blend confidence per line-only analysis
    BLEND_CONF_STALE: float = 0.20
    BLEND_CONF_FLAT: float = 0.15
    BLEND_CONF_NORMAL: float = 0.50

    # Revisione motore / pipeline (tracking, champion–challenger, audit log).
    MODEL_REVISION: str = MODEL_APP_VERSION
    # Live recalibration: peso sul totale implicito (gol fatti + tot_cur rimanente) vs prior tot_op lineare.
    LIVE_RECAL_MARKET_BLEND: float = 0.58


@dataclass(frozen=True)
class InputValidationConfig:
    """Parametri per la validazione degli input utente."""

    # Validazione tot_cur: se tot_cur > tot_cap * TOT_VALIDATION_MULTIPLIER → ERRORE
    # TOT_VALIDATION_MULTIPLIER=1.5: permette un 50% di tolleranza prima di bloccare
    # TOT_VALIDATION_WARNING=1.2: warning al 20% sopra il cap
    TOT_VALIDATION_MULTIPLIER: float = 1.5
    TOT_VALIDATION_WARNING: float = 1.2

    # Validazione AH: se |ah_cur| > tot_cur + AH_VALIDATION_BUFFER → ERRORE
    # AH_VALIDATION_BUFFER=0.5: permette piccolo buffer per errori di arrotondamento
    AH_VALIDATION_BUFFER: float = 0.5

    # FIX: Validazione AH vs punteggio attuale
    # Se il punteggio è cambiato ma AH è rimasto identico all'apertura, probabile errore.
    AH_SCORE_MOVE_THRESHOLD: float = 0.10  # Movimento minimo AH dopo gol
    AH_SCORE_MOVE_MINUTE: int = 10          # Minuto minimo per check

    # FIX: Errori bloccanti vs warning
    # I dati palesemente sbagliati bloccano l'analisi invece di mostrare solo warning.
    BLOCK_ON_CRITICAL_ERRORS: bool = True


@dataclass(frozen=True)
class OcrQuotesConfig:
    """Parametri per l'integrazione delle quote bookmaker estratte da OCR prematch."""

    # Peso del total implicito dalle quote O/U nel blend di tot_op.
    # Le quote O/U sono efficienti: il bookmaker ha già digerito formazioni/meteo.
    # Leggermente superiore a OCR_PRIOR_WEIGHT (linea grezza) perché le probabilità
    # fair da devigging sono più precise della linea tabellare.
    TOTAL_WEIGHT: float = 0.20

    # Peso del delta implicito dalle quote 1X2 nel blend di ah_op.
    # Conservativo: il 1X2 include vig elevato e la conversione delta→AH
    # introduce approssimazioni (Dixon-Coles non applicato qui).
    DELTA_WEIGHT: float = 0.12

    # Peso prior BTTS dalla quota GG/NG (solo prematch, mercato due vie).
    # BTTS è efficiente ma ha vig 5-8%: peso conservativo.
    BTTS_PRIOR_WEIGHT: float = 0.24

    # Massimo overround accettabile per considerare le quote valide.
    # >1.30 su 1X2 (30% di vig) → OCR ha probabilmente letto male le quote.
    # >1.12 su mercati a due vie → vig eccessivo.
    MAX_OVERROUND_3WAY: float = 1.30
    MAX_OVERROUND_2WAY: float = 1.12

    # #3: Scaling qualità basato sull'overround effettivo.
    # Overround basso (1-2%) → quote molto efficienti → peso pieno.
    # Overround alto (vicino al cap) → peso ridotto proporzionalmente.
    # Formula: quality = 1 - min(MAX_PENALTY, (overround-1) * RATE)
    # Per O/U a due vie: overround 1.02 → quality=0.97, overround 1.10 → quality=0.80
    TOTAL_QUALITY_OVERROUND_RATE: float = 2.0    # scala per O/U (2 vie)
    TOTAL_QUALITY_MAX_PENALTY: float = 0.20      # max riduzione peso O/U (-20%)
    DELTA_QUALITY_OVERROUND_RATE: float = 0.67   # scala per 1X2 (3 vie, overround più alto)
    DELTA_QUALITY_MAX_PENALTY: float = 0.20      # max riduzione peso 1X2 (-20%)
    BTTS_QUALITY_OVERROUND_RATE: float = 3.0     # scala per GG/NG (2 vie, più sensibile)
    BTTS_QUALITY_MAX_PENALTY: float = 0.20       # max riduzione peso BTTS (-20%)

    # Usa il prior 1X2 estratto nel motore prematch (con quality-gate nel bridge).
    USE_EXTRACTED_1X2_PRIOR: bool = True


@dataclass(frozen=True)
class AIAdjConfig:
    """
    Parametri per gli aggiustamenti xG da dati AI (assenze + forma).

    ABSENCE_MARKET_ALPHA: il mercato ha già prezzato ~60% dell'impatto delle assenze
    nelle linee di apertura. Applichiamo solo il 40% residuo non catturato.
    Questo evita il double-counting tra il prior bayesiano delle linee e i dati AI.

    Riferimento: Frick & Simmons (2008), Goddard & Asimakopoulos (2004):
    il mercato tipicamente incorpora ~50-70% dell'impatto delle notizie pubbliche.
    """

    # Fattore di attenuazione per evitare double-counting con le linee di mercato.
    # 0.40 = applichiamo solo il 40% dell'impatto calcolato (il mercato ha già il 60%).
    ABSENCE_MARKET_ALPHA: float = 0.40

    # Moltiplicatori per ruolo × status (applicati PRIMA di ABSENCE_MARKET_ALPHA)
    STRIKER_CONFIRMED_MULT: float = 0.88   # striker confermato assente → -12% xG
    STRIKER_PROBABLE_MULT: float = 0.94    # striker probabile assente → -6% xG
    GK_OPP_CONFIRMED_MULT: float = 1.08   # portiere avversario confermato assente → +8% xG avversario
    GK_OPP_PROBABLE_MULT: float = 1.04    # portiere avversario probabile assente → +4%
    MID_CONFIRMED_MULT: float = 0.97       # centrocampista confermato assente → -3% xG
    MID_PROBABLE_MULT: float = 0.985       # centrocampista probabile assente → -1.5%
    DEF_CONFIRMED_MULT: float = 0.98       # difensore confermato assente → -2% xG
    DEF_PROBABLE_MULT: float = 0.99        # difensore probabile assente → -1%

    # Clamp moltiplicatori assenze
    ABSENCE_MULT_MIN: float = 0.82         # floor: max -18% (es. 3 striker confermati)
    ABSENCE_MULT_MAX_GK: float = 1.12      # cap: max +12% per GK avversario assente

    # Forma recente: pesi decrescenti, il risultato più recente (primo) pesa di più
    # Somma = 1.0 = [W1, W2, W3, W4, W5] dal più recente al più antico
    FORMA_WEIGHTS: tuple = (0.35, 0.25, 0.20, 0.12, 0.08)

    # Effetto massimo della forma sul xG: ±8%
    # Forma perfetta WWWWW → +8%, disastrosa LLLLL → -8%
    FORMA_MAX_EFFECT: float = 0.08

    # Scala per affidabilità Gemini:
    # alta=1.0 (usa tutto), media=0.65 (taglia 35%), bassa=0.35 (quasi inutile)
    AFFIDABILITA_ALTA: float = 1.0
    AFFIDABILITA_MEDIA: float = 0.65
    AFFIDABILITA_BASSA: float = 0.35


@dataclass(frozen=True)
class FormAnalysisConfig:
    """
    Parametri per l'analisi della forma squadre (standings, last6, home/away performance).

    RIDUCE la dipendenza dalle linee manuali dal 70-75% al 50-60% utilizzando
    dati estratti da Nowgoal che erano precedentemente ignorati.
    """

    # === STANDINGS (Classifica) ===
    # Fattore motivazione basato sulla posizione in classifica.
    # Squadre in zona retrocessione (ultime 3) o titolo (prime 3) sono più motivate.
    STANDINGS_MOTIVATION_WEIGHT: float = 0.06  # max ±6% su xG per motivazione

    # Zone di classifica che aumentano motivazione
    RELEGATION_ZONE: int = 3      # ultime N posizioni = zona retrocessione
    TITLE_ZONE: int = 3           # prime N posizioni = zona titolo
    EUROPE_ZONE: int = 6          # prime N posizioni per qualificazione europea

    # Bonus motivazione per posizione in zona critica
    RELEGATION_MOTIVATION_BONUS: float = 0.04   # +4% xG per squadra in zona retrocessione
    TITLE_MOTIVATION_BONUS: float = 0.02        # +2% xG per squadra in zona titolo
    EUROPE_MOTIVATION_BONUS: float = 0.015      # +1.5% xG per squadra in zona europea

    # Penalità per squadra "senza obiettivi" (posizione centrale in classifica)
    NO_STAKES_PENALTY: float = -0.02            # -2% xG per squadra senza motivazione
    # Retrocessione + distacco PPG netto rispetto all’avversario → urgenza tattica extra
    RELEGATION_PPG_GAP_THRESHOLD: float = 0.28
    RELEGATION_UNDERDOG_BONUS: float = 0.018

    # === LAST 6 GAMES (Forma recente specifica) ===
    # Dalle ultime 6 partite, calcoliamo punti, gol fatti, gol subiti.
    # Questo è più granulare del simple W/D/L string.

    LAST6_WEIGHT: float = 0.12    # peso della forma last6 nel blend xG

    # Punti attesi per partita: 3.0 = perfetto, 0.0 = disastroso
    LAST6_POINTS_EXCELLENT: float = 2.3   # > 2.3 PPG = forma eccellente
    LAST6_POINTS_POOR: float = 0.8        # < 0.8 PPG = forma pessima

    # Effetto forma last6 su xG
    LAST6_MAX_BOOST: float = 0.08         # max +8% xG per forma eccellente
    LAST6_MAX_PENALTY: float = -0.08      # max -8% xG per forma pessima

    # === HOME/AWAY PERFORMANCE ===
    # Rendimento separato casa vs trasferta.
    # Alcune squadre sono molto più forti in casa (fattore casa).

    HOME_AWAY_WEIGHT: float = 0.10        # peso del rendimento casa/trasferta

    # Fattore casa medio per top-5 leagues: circa +10% xG per la casa
    HOME_ADVANTAGE_BASE: float = 0.10

    # Squadre con forte rendimento casa (es. >2.0 PPG in casa) ottengono bonus
    HOME_STRONG_THRESHOLD: float = 2.0    # PPG in casa > questo = forte in casa
    HOME_STRONG_BONUS: float = 0.03       # +3% xG per forte rendimento casa

    # Squadre con debole rendimento trasferta (es. <0.8 PPG fuori) ottengono penalità
    AWAY_WEAK_THRESHOLD: float = 0.8      # PPG trasferta < questo = debole fuori
    AWAY_WEAK_PENALTY: float = -0.03      # -3% xG per debole rendimento trasferta

    # === Segnali aggiuntivi da estrazione URL (Nowgoal) ===
    # Motivazione testuale (high/normal/low) dall’analisi pagina: affianca la motivazione da classifica.
    OCR_MOTIVATION_HIGH_BONUS: float = 0.022
    OCR_MOTIVATION_LOW_ADJ: float = -0.014
    # Ancoraggio xG da media gol nelle partite recenti derivata in post-processing OCR.
    RECENT_XG_PRIOR_ALPHA_MAX: float = 0.052
    # Tilt asimmetrico casa/trasferta da % storica copertura AH casa negli H2H.
    H2H_AH_COVER_TILT_MAX: float = 0.026
    # Rafforzo forma da % vittorie “previous scores” (ultime 10) se coerente con estrazione.
    PREV_WIN_PCT_TILT_MAX: float = 0.018

    # Media gol H2H per squadra (URL): blend asimmetrico su λ dopo Bayes (prematch).
    H2H_AVG_GOALS_XG_BLEND_MAX: float = 0.078

    # Somma λ vs tot_op (mercato): coerenza soft post-blend prematch (rapporto H/A invariato).
    LAMBDA_TOT_COHERE_TRIGGER_REL: float = 0.085
    LAMBDA_TOT_MARKET_COHERE_MAX: float = 0.11
    LAMBDA_TOT_COHERE_K: float = 0.42
    LAMBDA_TOT_COHERE_LOW_COV_BOOST: float = 0.35

    # H2H: % HT casa / X / trasferta quando manca la matrice HT→FT (fallback prematch).
    H2H_HT_MARGINAL_PREMATCH_BLEND_MAX: float = 0.058
    # H2H Over% è spesso vs 2.5: traslazione euristica verso altra linea O/U analizzata.
    H2H_OVER_LINE_SLOPE_PER_HALF: float = 0.092
    H2H_OVER_BLEND_BASE_ALPHA: float = 0.175
    # O/U europeo canonico 2.5: integra informazione dalla linea selezionata (es. 3.0).
    O25_FROM_SELECTED_LINE_SHIFT_PER_HALF: float = 0.105
    O25_FROM_SELECTED_LINE_BLEND_ALPHA: float = 0.30
    # Reconciliazione finale O/U 2.5 con distribuzione gol (coerenza probabilistica).
    O25_DIST_RECON_ALPHA_BASE: float = 0.24
    # Coerenza O/U ladder (Over 1.5 e Over 2.5) vs distribuzione gol del consensus.
    OU_DIST_RECON_ALPHA_BASE: float = 0.26
    # 1X2 prematch: alphas base adattivi (market + H2H), poi scalati da trust/coverage/campione.
    PREMATCH_1X2_MKT_ALPHA_BASE: float = 0.08
    PREMATCH_1X2_H2H_ALPHA_BASE: float = 0.05
    # Cap del peso totale dei prior esterni 1X2 (market+H2H) in prematch.
    PREMATCH_1X2_EXTERNAL_ALPHA_CAP: float = 0.16
    # Coerenza soft tra ladder O/U e BTTS in prematch (evita combinazioni incoerenti).
    PREMATCH_OU_BTTS_COHERENCE_ALPHA: float = 0.36
    # Scenario-mixture prematch: total λ come mix low/mid/high tempo.
    PREMATCH_TEMPO_MIX_AMPLITUDE: float = 0.12
    PREMATCH_TEMPO_MIX_BLEND_MAX: float = 0.34
    # Moltiplicatore affidabilità per nota parser grave (bridge → extraction_trust_factor).
    EXTRACTION_NOTE_TRUST_PENALTY: float = 0.88
    EXTRACTION_TRUST_FLOOR: float = 0.55

    # === GOAL TIMING (Quando segnano) ===
    # Squadre che segnano a fine partita (ultimi 15') possono essere più pericolose
    # in partite aperte, mentre squadre che subiscono a fine partita sono vulnerabili.

    GOAL_TIMING_WEIGHT: float = 0.03      # peso del timing dei gol (basso, informativo)

    # Bonus per squadra che segna spesso nell'ultimo quarto d'ora
    LATE_SCORER_BONUS: float = 0.02       # +2% se >35% gol nei minuti 75-90
    EARLY_CONCEDER_PENALTY: float = -0.02 # -2% se >40% gol subiti nei minuti 0-30


@dataclass(frozen=True)
class BTTSCalibrationConfig:
    """
    Parametri per la calibrazione BTTS (Both Teams To Score).

    Il modello Poisson tende a sovrastimare BTTS Sì in partite difensive
    e sottostimarlo in partite aperte. Questi parametri correggono il bias.
    """

    # === Calibrazione basata su Total atteso ===
    # Total basso (< 2.0) → partita difensiva → BTTS meno probabile
    # Total alto (> 3.0) → partita aperta → BTTS più probabile

    # Soglie di total per calibrazione
    TOTAL_LOW_THRESHOLD: float = 2.0      # sotto questo: partita difensiva
    TOTAL_HIGH_THRESHOLD: float = 3.0     # sopra questo: partita aperta

    # Correzioni BTTS per total
    BTTS_LOW_TOTAL_ADJUST: float = -0.04  # -4% BTTS per total < 2.0
    BTTS_HIGH_TOTAL_ADJUST: float = 0.03  # +3% BTTS per total > 3.0

    # === Calibrazione basata su forza offensiva/defensiva ===
    # Se una squadra ha attacco molto forte E difesa debole → BTTS più probabile
    # Se entrambe le squadre hanno difesa forte → BTTS meno probabile

    # Soglia per "attacco forte" (gol/segni per partita)
    STRONG_ATTACK_THRESHOLD: float = 1.5
    # Soglia per "difesa debole" (gol/subiti per partita)
    WEAK_DEFENSE_THRESHOLD: float = 1.3

    # Bonus BTTS per mismatch attacco-difesa
    ATTACK_DEFENSE_MISMATCH_BONUS: float = 0.03  # +3% se attacco forte vs difesa debole

    # Penalità per doppia difesa forte
    BOTH_STRONG_DEFENSE_PENALTY: float = -0.04    # -4% se entrambe difese forti

    # === Calibrazione basata su H2H ===
    # Se storico H2H mostra molti gol, aumentiamo BTTS
    H2H_BTTS_WEIGHT: float = 0.08         # peso dell'H2H su BTTS
    H2H_BTTS_HIGH_THRESHOLD: float = 0.60 # % partite H2H con BTTS sopra questa
    H2H_BTTS_BONUS: float = 0.04          # +4% se H2H storico alto

    # === Calibrazione forma recente ===
    # Squadre che hanno segnato nelle ultime 3 partite → più probabile che segnino
    RECENT_SCORING_STREAK_BONUS: float = 0.02  # +2% se segnato in 3+ partite consecutive
    RECENT_CLEAN_SHEET_PENALTY: float = -0.02  # -2% se 2+ clean sheet consecutivi

    # === Clamp finali ===
    BTTS_MIN: float = 0.15   # BTTS non può scendere sotto 15%
    BTTS_MAX: float = 0.85   # BTTS non può salire sopra 85%


@dataclass(frozen=True)
class PrecisionConfig:
    """
    Guardrail di precisione operativa (No-Bet + Quality Firewall + promotion).
    """

    # === Fase 4/5: No-Bet + Data Quality Firewall ===
    # Score minimo complessivo per consentire segnali operativi.
    QUALITY_SCORE_MIN: float = 0.50
    # Soglia dura sotto la quale si blocca completamente (hard block).
    # Tra HARD_BLOCK e QUALITY_SCORE_MIN: degradazione graduata (riduce confidence).
    QUALITY_HARD_BLOCK_MIN: float = 0.20
    # Confidenza minima del modello (oltre al gate già presente nei segnali).
    MODEL_CONFIDENCE_MIN: float = 0.45
    # Accordo minimo tra modelli consensus.
    MODEL_AGREEMENT_MIN: float = 0.45
    # Divergenza modello-mercato oltre la quale conviene sospendere.
    MARKET_DIVERGENCE_MAX: float = 0.26
    # In prematch con estrazione scarsa i segnali diventano rumorosi.
    PREMATCH_COVERAGE_MIN: float = 0.55
    # In live linee stantie + scoreline non aggiornato = blocco operativo.
    STALE_LINE_MINUTE_BLOCK: int = 20
    # Se True, nessun segnale operativo quando il firewall blocca.
    HARD_BLOCK_ON_FIREWALL: bool = True

    # === Fase 6: Champion/Challenger gates ===
    # Campione minimo per valutazione promozione.
    CHAMPION_MIN_SAMPLES: int = 30
    # Miglioramenti minimi richiesti (valori negativi = riduzione metrica).
    CHAMPION_MAX_DELTA_BRIER: float = -0.003
    CHAMPION_MAX_DELTA_LOGLOSS: float = -0.004
    CHAMPION_MAX_DELTA_ECE: float = -0.010
    # CLV proxy non deve peggiorare oltre questa tolleranza.
    CHAMPION_MIN_DELTA_CLV: float = -0.010

    # ECE: numero bin per report/valutazione
    ECE_BINS: int = 10

    # Micro-correzione draw da learn_draw_shrinkage: i record sono già post-engine shrink;
    # scala la delta rispetto al baseline CONSENSUS.DRAW_SHRINKAGE per evitare doppio intervento forte.
    PARAMETER_LEARNING_DRAW_MICRO_SCALE: float = 0.42

    # Riduce forza Platt quando la calibrazione per lega ha già peso alto (anti doppia correzione).
    PLATT_STRENGTH_DAMP_PER_LEAGUE_WEIGHT: float = 0.38
    PLATT_STRENGTH_FLOOR: float = 0.34

    # Contributo firewall qualità da intervalli credibilità stretti (prematch).
    CI_QUALITY_FIREWALL_WEIGHT: float = 0.088

    # Platt con holdout temporale (ordine timestamp): tieni mappa solo se migliora log-loss sul test.
    CALIBRATION_TEMPORAL_TRAIN_FRAC: float = 0.72
    CALIBRATION_TEMPORAL_MIN_TEST: int = 10
    CALIBRATION_TEMPORAL_MIN_TOTAL: int = 40

    # Correct score: micro-blend verso frequenze empiriche (lega + fascia tot_op).
    CORRECT_SCORE_HISTORY_BLEND_MAX: float = 0.14
    CORRECT_SCORE_HISTORY_MIN_SAMPLES: int = 14
    CORRECT_SCORE_HISTORY_DIRICHLET_ALPHA: float = 0.12


# Istanze globali immutabili — importare da qui
POISSON   = PoissonConfig()
DC        = DixonColesConfig()
RHO       = RhoConfig()
SHOTS     = ShotConfig()
DECAY     = TimeDecayConfig()
BAYES     = BayesianConfig()
MOMENTUM  = MomentumConfig()
KELLY     = KellyConfig()
SIGNALS   = SignalConfig()
UI        = UIConfig()
CMP       = CMPConfig()
COPULA    = CopulaConfig()
HAWKES    = HawkesConfig()
SUBST     = SubstitutionConfig()
CONSENSUS = ConsensusConfig()
STALE     = StaleLineConfig()
CACHE     = CacheConfig()
CLEAN_SHEET = CleanSheetConfig()
ENGINE = EngineConfig()
INPUT_VALIDATION = InputValidationConfig()
OCR_QUOTES = OcrQuotesConfig()
AI_ADJ = AIAdjConfig()
FORM_ANALYSIS = FormAnalysisConfig()
BTTS_CALIBRATION = BTTSCalibrationConfig()
PRECISION = PrecisionConfig()
