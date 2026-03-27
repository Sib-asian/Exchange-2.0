"""
config.py — Costanti e parametri centralizzati del motore Radar Pro Live.

Tutti i magic numbers del motore sono qui documentati con fonte e motivazione.
Modificare solo qui: nessuna costante hardcodata nei moduli.
"""

from dataclasses import dataclass


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

    # Numero di tiri totali per considerare il campione "sufficiente"
    SHOT_INFO_THRESHOLD: int = 15

    # Peso massimo dei tiri sul Totale (il Total-line è molto efficiente)
    ALPHA_T_MAX: float = 0.25

    # Tasso di crescita del peso T con la frazione giocata (calibrato su top-5 leagues)
    # Crescita lenta: il Total-line è molto efficiente e i tiri lo correggono poco
    ALPHA_T_RATE: float = 0.30

    # Peso massimo dei tiri sul Differenziale (più informativo del mercato)
    ALPHA_D_MAX: float = 0.70

    # Smorzamento proiezione tiri: curva esponenziale (regressione alla media)
    # 0.75 a inizio partita → 1.0 a fine (campione ≈ universo).
    # Decay rate 2.0: converge rapidamente dopo il 30' (campione affidabile).
    RATE_DAMP_FLOOR: float = 0.75
    RATE_DAMP_DECAY: float = 2.0

    # Shrinkage Bayesiano verso la media di lega (Stein shrinkage).
    # Riduce la varianza delle previsioni shot-based verso il prior di lega.
    # 10% verso la media con campione perfetto; cresce con pochi tiri.
    SHRINKAGE_WEIGHT: float = 0.10
    LEAGUE_MEAN_RATE: float = 2.7  # gol/90' media top-5 leagues

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
    SCORE_DOWN_MULTIPLIER: float = 0.65   # squadra in svantaggio: meno boost (pressing bassa qualità)
    SCORE_UP_MULTIPLIER: float = 1.15     # squadra in vantaggio: mantiene qualità difensiva

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
    VERSION: str = "2.0.0"
    LAYOUT: str = "centered"

    # Linee U/O disponibili nel selectbox
    LINEE_OU: tuple = (0.5, 1.5, 1.75, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.5, 5.5)

    # Tiri attesi per minuto (entrambe le squadre) — soglia warning input
    TIRI_PER_MINUTO: float = 0.65
    TIRI_WARNING_BUFFER: int = 4
    TIRI_MIN_BASE: int = 6

    # Numero massimo di correct score da mostrare
    TOP_CS_COUNT: int = 5

    # Massimo gol totali nella distribuzione
    MAX_GOL_DIST: int = 10

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

    # Pesi dei 3 modelli nel consensus
    W_BIVARIATE: float = 0.50   # bivariate Poisson + DC (modello principale)
    W_COPULA: float = 0.30      # CMP + Frank copula (overdispersion)
    W_MARKOV: float = 0.20      # Markov chain (score-dependent rates)

    # Calibrazione isotonica
    DRAW_SHRINKAGE: float = 0.97  # riduzione draw (-3%)

    # Logistic sharpening: α > 1 rende le probabilità estreme più estreme
    # calibrate su dati Poisson vs reali: modello sottostima certezza agli estremi
    LOGISTIC_ALPHA_OVER: float = 1.03   # sharpening per Over/Under
    LOGISTIC_ALPHA_BTTS: float = 1.02   # sharpening per BTTS (più conservativo)

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
class YellowCardConfig:
    """Parametri per l'effetto dei cartellini gialli.

    #4: Gialli → fattore rischio rosso + rallentamento ritmo.
    #9: rho_dc più negativo con gialli alti.
    """

    # Soglia gialli per attivare il ritmo ridotto (somma casa + trasf)
    TEMPO_THRESHOLD: int = 4          # sotto: nessun effetto ritmo
    # Riduzione gol attesi per ogni giallo sopra la soglia: -1.5% per giallo
    TEMPO_RATE: float = 0.015
    # Cap riduzione ritmo per i gialli (max -8%)
    TEMPO_CAP: float = 0.08

    # Rischio rosso futuro: probabilità incrementale per giallo (moltiplicatore xG)
    # Un giallo ≈ +12% chance di prossimo rosso → leggero abbassamento xG attaccante
    RED_RISK_PER_YELLOW: float = 0.04   # -4% xG per ogni giallo sopra 1
    RED_RISK_CAP: float = 0.12          # cap -12%

    # Effetto su rho_dc: più gialli → gioco più duro → struttura difensiva
    # Applica un moltiplicatore su RHO_DC_BASE: -0.005 per giallo, cap -0.03
    RHO_DC_YELLOW_RATE: float = 0.005
    RHO_DC_YELLOW_CAP: float = 0.03


@dataclass(frozen=True)
class FoulConfig:
    """Parametri per l'effetto dei falli sul tot_bayes.

    #5: Falli → correzione leggera tot_bayes.
    """

    # Soglia falli totali (casa + trasf) per attivare l'effetto
    THRESHOLD: int = 20               # sotto: nessun effetto
    # Riduzione tot_bayes per ogni fallo sopra la soglia: -0.3%
    RATE: float = 0.003
    # Cap totale riduzione sul tot_bayes (max -5%)
    CAP: float = 0.05


@dataclass(frozen=True)
class BlockedShotConfig:
    """Parametri per i tiri bloccati nel calcolo xG.

    #2: Tiri bloccati nel xG accum.
    """

    # xG per tiro bloccato: minore di SOT (bloccato prima del portiere)
    # ma maggiore di SOFF (era diretto, non fuori)
    # Letteratura: ~0.04-0.06 (StatsBomb: blocked shots ≈ 0.05)
    XG_BLOCKED: float = 0.05


@dataclass(frozen=True)
class OcrCalibrationConfig:
    """Parametri per la calibrazione OCR prematch e warning divergenza.

    #1: Quote OCR come calibratore Bayesiano prematch.
    #6: Confidence prematch adattiva con OCR.
    #12: Warning divergenza quote OCR vs modello.
    """

    # Peso delle probabilità implicite OCR nel blend prematch xG
    # 0 = solo linee AH/Total, 1 = solo OCR quote
    # 0.20: le linee AH sono più affidabili, ma le 1X2 OCR aggiungono un segnale utile
    XG_OCR_BLEND_WEIGHT: float = 0.20

    # Boost alla confidence prematch quando OCR disponibile con alta affidabilità
    HIGH_CONF_BOOST: float = 0.15    # +15% confidence con OCR high
    MEDIUM_CONF_BOOST: float = 0.08  # +8% confidence con OCR medium

    # Soglia divergenza per warning: se |p_model - p_ocr| > threshold → avvisa
    DIVERGENCE_THRESHOLD: float = 0.15   # 15%

    # Margine bookmaker: prob implicite normalizzate sopra questa soglia
    # (rimuove l'overround prima del confronto)
    MIN_OVERROUND_TO_NORMALIZE: float = 0.01


@dataclass(frozen=True)
class StatMomentumConfig:
    """Parametri per il momentum statistico (attacchi/tiri) blendato con momentum mercato.

    #10: Momentum statistico da attacchi/tiri blendato con momentum mercato.
    """

    # Peso del momentum statistico vs momentum mercato nel blend finale
    # 0 = solo mercato, 1 = solo statistiche
    STAT_WEIGHT: float = 0.25

    # Scala il momentum stat: max contributo = STAT_SCALE (es. 0.5 = fino a +0.5 di momentum)
    STAT_SCALE: float = 1.0

    # Minuto minimo per attivare il momentum statistico (campione troppo piccolo prima)
    MIN_MINUTE: int = 15


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
YELLOWS = YellowCardConfig()
FOULS = FoulConfig()
BLOCKED = BlockedShotConfig()
OCR_CALIB = OcrCalibrationConfig()
STAT_MOM = StatMomentumConfig()
