import streamlit as st
import math

# ==========================================
# ⚙️ FUNZIONI DI SUPPORTO
# ==========================================

def rho_dinamico(tot_cur, minuto, shot_dom=0.0, gol_totali=0):
    """
    Correlazione comune bivariata non più fissa a 0.12.

    Logica base:
    - tot_cur alto → partita aperta → le squadre segnano più indipendentemente → rho scende
    - Avanzare del minuto → i gol già osservati dominano l'informazione → rho scende
    - gol_totali alti → la correlazione residua tra i gol futuri è minore: le squadre
      hanno già "consumato" gran parte della struttura correlata → rho scende ulteriormente

    Correzione shot-dominance (shot_dom in [0,1]):
    - Quando una squadra domina nettamente i tiri la partita è essenzialmente
      unidirezionale: una Poisson preme, l'altra difende/contropiede.
      Le due componenti diventano quasi indipendenti → rho scende fino al 25%.
    - shot_dom = |sot_h - sot_a| / (sot_h + sot_a)  (0 = equilibrio, 1 = dominio totale)

    Range risultante: ~0.02 (dominio totale late, gol alti) → ~0.14 (equilibrio inizio gara)
    """
    base  = max(0.02, 0.14 - 0.018 * min(tot_cur, 4.5))
    decay = 1.0 - 0.40 * max(0.0, min(minuto / 90.0, 1.0))
    # Gol già segnati: ogni gol riduce la correlazione residua del ~10%.
    # Con 4+ gol la partita è già strutturalmente "decisa" per correlazione → floor 50%.
    gol_decay = max(0.50, 1.0 - 0.10 * gol_totali)
    rho_base = base * decay * gol_decay
    # Riduzione fino al 45% con dominio tiri totale: partita unidirezionale
    # → le due Poisson diventano quasi indipendenti → rho → 0.
    return max(0.02, rho_base * (1.0 - 0.45 * shot_dom))


def dixon_coles_tau(i, j, mu_h, mu_a, rho_dc=-0.13):
    """
    Fattore correttivo Dixon-Coles per punteggi bassi rimanenti.

    Poisson indipendente sovrastima P(0-0) e sottostima P(1-0)/P(0-1).
    tau(i,j) corregge solo per i 4 punteggi con i+j <= 2 (e i,j <= 1):
      tau(0,0) = 1 - lambda_h * lambda_a * rho_dc
      tau(1,0) = 1 + lambda_a * rho_dc
      tau(0,1) = 1 + lambda_h * rho_dc
      tau(1,1) = 1 - rho_dc
      tau(i,j) = 1  per tutti gli altri punteggi (i+j > 2 o i>1 o j>1)

    rho_dc empiricamente negativo (-0.13): le due squadre tendono
    a non segnare simultaneamente.
    """
    # Clamp [0.05, 3.0]: 1e-6 azzerava la probabilità invece di limitarla.
    # 0.05 = floor minimo coerente; 3.0 = cap per evitare amplificazioni eccessive.
    if   i == 0 and j == 0: tau = 1.0 - mu_h * mu_a * rho_dc
    elif i == 1 and j == 0: tau = 1.0 + mu_a * rho_dc
    elif i == 0 and j == 1: tau = 1.0 + mu_h * rho_dc
    elif i == 1 and j == 1: tau = 1.0 - rho_dc
    else:                   return 1.0
    return max(0.05, min(tau, 3.0))


def _poisson_pmf_norm(mu, tail_mass=1e-12):
    if mu <= 0:
        return [1.0]
    # mu + 6*sqrt(mu) copre la tail P(X > k) < 1e-9 per qualsiasi mu realistico
    max_k = max(20, int(mu + 6.0 * math.sqrt(max(mu, 1.0)) + 10))
    p0 = math.exp(-mu)
    pmfs, cumsum, k, p = [p0], p0, 0, p0
    while cumsum < (1.0 - tail_mass) and k < max_k:
        k += 1
        p = p * mu / k
        pmfs.append(p)
        cumsum += p
    return [x / cumsum for x in pmfs]


def calcola_momentum_mercato(delta_ah, delta_tot, minuto):
    """
    Indice 0-6 di intensita del movimento di mercato relativo al tempo giocato.

    Un grande delta early-game vale di piu di uno late-game:
    - early: il mercato ha ricevuto informazione non ancora scontata
    - late: gran parte dell'informazione e gia incorporata nelle quote

    Soglie interpretative:
    - 0.0-1.0  Mercato stabile
    - 1.0-2.5  Movimento moderato
    - 2.5-4.0  Movimento significativo
    - 4.0+     Movimento estremo (infortuni, rossi non registrati, info asimmetrica)

    Nota v45: a minuto=0 restituisce 0 — la differenza tra linea apertura e corrente
    in prematch non è un "movimento intrapartita" ma semplicemente la distanza tra
    due rilevazioni statiche. Dividere per frac=0.10 amplificava artificialmente.
    """
    if minuto == 0:
        return 0.0
    # sqrt invece di lineare: evita overshoot early-game.
    # A minuto 10 la frac lineare è 0.11 (amplifica ×9), sqrt è 0.33 (amplifica ×3).
    frac = max(0.15, math.sqrt(minuto / 90.0))
    return min((abs(delta_ah) + abs(delta_tot) * 0.5) / frac, 6.0)


# ==========================================
# CALIBRAZIONE BAYESIANA
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto):
    """
    Blend bayesiano a pesi dinamici con normalizzazione temporale.

    Pesi time-varying (v45):
      w_cur = min(0.90, 0.65 + 0.20 * frac_giocata)   → 65% prematch → 85% al 90'
      w_op  = 1 - w_cur

    Motivazione: a partita inoltrata la linea live incorpora molte più informazioni
    (eventi accaduti, condizioni di gioco, mercato liquido) rispetto all'apertura.
    Lasciare w_op fisso a 35% tutta la partita sovrappesava il prior di pre-partita.

    L'apertura viene scalata al tempo rimanente prima del blend: entrambi gli input
    diventano omogenei (gol rimanenti).
    """
    frac_giocata = minuto / 90.0
    # Floor 0.005 (non 0.05): a 90' vogliamo ~zero tempo rimanente.
    # Il vecchio floor 5% gonfiava artificialmente le xG in zona recupero,
    # producendo "ghost goals" che distorcevano le probabilità da 85' in poi.
    frac_rimasta = max(0.005, 1.0 - frac_giocata)

    # Pesi time-varying: linea live diventa sempre più affidabile col tempo
    w_cur = min(0.90, 0.65 + 0.20 * frac_giocata)
    w_op  = 1.0 - w_cur

    # Se non c'e' movimento di linea, usa direttamente i valori correnti
    # senza blend: il blend con frac_rimasta creerebbe deriva artificiale
    # su linee piatte (le quote cambierebbero col solo passare del minuto
    # anche senza nessuna informazione nuova dal mercato).
    delta_ah_inner  = abs(ah_cur  - ah_op)
    delta_tot_inner = abs(tot_cur - tot_op)
    if delta_ah_inner < 1e-6 and delta_tot_inner < 1e-6:
        ah_bayes  = float(ah_cur)
        tot_bayes = max(0.2, float(tot_cur))
    else:
        ah_bayes  = (ah_op  * frac_rimasta) * w_op + ah_cur  * w_cur
        tot_bayes = max(0.2, (tot_op * frac_rimasta) * w_op + tot_cur * w_cur)
    eps = 1e-6

    # _ah_ev_half con correzione Dixon-Coles (v55):
    # Prima usava Poisson indipendente → le xG estratte dalla bisection non erano
    # coerenti col modello finale (bivariate Poisson + DC). DC ridistribuisce ~2-5%
    # di probabilità tra i punteggi bassi (0-0, 1-0, 0-1, 1-1), che influenza
    # l'EV dell'AH in modo non trascurabile. Z si cancella nella differenza (i-j)
    # quindi basta aggiungere DC. Pass unico accumula ev e dc_norm insieme.
    def _ah_ev_half(mh, ma, h):
        ev = dc_norm = 0.0
        pmf_h = _poisson_pmf_norm(mh)
        pmf_a = _poisson_pmf_norm(ma)
        for i, pi in enumerate(pmf_h):
            if pi < 1e-18: continue
            for j, pj in enumerate(pmf_a):
                if pj < 1e-18: continue
                w = pi * pj * dixon_coles_tau(i, j, mh, ma)
                dc_norm += w
                s = (i - j) + h
                if   s > 0: ev += w
                elif s < 0: ev -= w
        return ev / dc_norm if dc_norm > 0 else 0.0

    def _ah_ev(mh, ma, ah):
        ah2 = float(ah) * 2.0
        if abs(ah2 - round(ah2)) < 1e-9:
            return _ah_ev_half(mh, ma, float(ah))
        h1 = math.floor(ah2) / 2.0
        return 0.5 * _ah_ev_half(mh, ma, h1) + 0.5 * _ah_ev_half(mh, ma, h1 + 0.5)

    def _ev(delta):
        lh = max(eps, 0.5 * (tot_bayes + delta))
        la = max(eps, 0.5 * (tot_bayes - delta))
        return _ah_ev(lh, la, ah_bayes)

    lo, hi       = -tot_bayes + eps, tot_bayes - eps
    ev_lo, ev_hi = _ev(lo), _ev(hi)

    if ev_lo == 0.0:
        delta_star = lo
    elif ev_hi == 0.0:
        delta_star = hi
    elif ev_lo * ev_hi > 0:
        delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
    else:
        inc = ev_hi > ev_lo
        dl, dr = lo, hi
        for _ in range(52):   # 2^52 ≈ 4e15: convergenza numerica garantita
            m  = 0.5 * (dl + dr)
            em = _ev(m)
            if abs(em) < 1e-9 or (dr - dl) < 1e-12:
                break          # converged: EV≈0 o intervallo sotto float resolution
            if inc:
                if em > 0: dr = m
                else:      dl = m
            else:
                if em > 0: dl = m
                else:      dr = m
        delta_star = 0.5 * (dl + dr)

    return max(eps, 0.5*(tot_bayes+delta_star)), max(eps, 0.5*(tot_bayes-delta_star))


# ==========================================
# TIME DECAY + SCORE EFFECTS + RED CARDS
# ==========================================

def time_decay_dinamico(xg_casa, xg_trasf, minuto,
                         gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    """
    Aggiustamenti tattico-comportamentali sugli xG già proiettati al tempo rimanente.

    ── Cosa NON fa più (v45) ──────────────────────────────────────────────────
    Il decadimento Weibull (90-min/90)^0.85 è stato rimosso.
    `tot_cur` è la linea live dei GOL RIMANENTI: `calcola_xg_bayesiani` restituisce
    già stime in "remaining goals space". Applicare un ulteriore fattore temporale
    dimezzava le stime rispetto a ciò che il mercato prezzava → bias sistematico.

    ── Cosa fa ────────────────────────────────────────────────────────────────
    1. Score effect RESIDUALE max 4%
       L'AH corrente già incorpora ~80% dell'effetto del punteggio.
       Il residuo cattura il comportamento tattico non ancora pienamente prezzato:
       pressing disperato (team che perde), parking the bus (team in vantaggio).
       Cap a 4% per evitare il double-counting con ah_cur.

    2. Cartellini rossi: effetto costante per carta
       -32% xG per il team ridotto, +28% per l'avversario.
       (da letteratura su ~15.000 partite con espulsioni, Brechot & Flepp 2020)
    """
    if minuto >= 90:
        return 0.001, 0.001

    xg_c = float(xg_casa)
    xg_t = float(xg_trasf)

    # 1. Score effect residuale (max 7% — il grosso è già nell'AH live)
    # La saturazione più rapida (1.5 invece di 2.0) cattura meglio il pressing
    # disperato late-game: una squadra sotto di 2 gol al 75' preme molto più
    # intensamente di quanto l'AH live riesca a prezzare in tempo reale.
    # Letteratura (Brechot & Flepp 2020, Robberechts et al. 2021):
    # ~10-20% incremento xG in svantaggio; residuo dopo AH live ≈ 5-8%.
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat = abs(diff) / (1.5 + abs(diff))
        # Il residuo scala inversamente col minuto: early-game il mercato AH ha latency
        # (poco aggiornato dopo il gol) → il residuo cattura più informazione non prezzata.
        # Late-game l'AH ha già aggiornato su 60-70' di nuove info → residuo minore.
        # Scale: 0' → 1.20 (boost max 8.4%), 45' → 0.65, 75' → 0.38, 85' → 0.30
        minute_scale = max(0.30, 1.20 - 1.10 * (minuto / 90.0))
        residual = min(0.08, 0.07 * sat) * minute_scale
        if diff < 0:                        # casa in svantaggio → preme di più
            xg_c *= (1.0 + residual)
            xg_t *= (1.0 - residual)
        else:                               # casa in vantaggio → si abbassa
            xg_t *= (1.0 + residual)
            xg_c *= (1.0 - residual)

    # 2. Cartellini rossi — effetto marginale decrescente.
    # Modello precedente: pow(0.68, n) → 2 rossi = -54%, 3 rossi = -68% (esagerato).
    # Letteratura (Brechot & Flepp 2020): il 2° rosso porta beneficio aggiuntivo
    # minore perché la squadra è già in difficoltà strutturale (linea difensiva
    # abbassata, pressing ridotto comunque). Tabella precalcolata:
    #   n=0: ×1.000  |  boost avversario ×1.000
    #   n=1: ×0.680  |  boost avversario ×1.280  (−32% / +28%)
    #   n=2: ×0.578  |  boost avversario ×1.434  (−42% / +43%)
    #   n=3: ×0.532  |  boost avversario ×1.520  (−47% / +52%)
    #   n=4: ×0.500  |  boost avversario ×1.566  (−50% / +57%)
    _RED_DECAY = [1.000, 0.680, 0.578, 0.532, 0.500]
    _RED_BOOST = [1.000, 1.280, 1.434, 1.520, 1.566]
    if rossi_casa > 0:
        idx = min(rossi_casa, 4)
        xg_c *= _RED_DECAY[idx]
        xg_t *= _RED_BOOST[idx]
    if rossi_trasf > 0:
        idx = min(rossi_trasf, 4)
        # Asimmetria home advantage: la trasferta perde il supporto del pubblico
        # e la familiarità col campo → il rosso alla trasferta è ~5% più grave
        # rispetto al rosso in casa (modesto ma consistente in letteratura).
        xg_t *= _RED_DECAY[idx] * 0.95
        xg_c *= _RED_BOOST[idx] * 1.04

    return max(0.001, xg_c), max(0.001, xg_t)


# ==========================================
# CALCOLO PROBABILITA — UN UNICO PASSAGGIO
# ==========================================

def calcola_tutto(mu_c_rem, mu_t_rem, gol_casa, gol_trasf, linea_ou, tot_cur, minuto,
                  shot_dom=0.0):
    """
    Costruisce la matrice bivariata completa una volta sola e ne ricava
    in un unico passaggio: 1X2, U/O, BTTS, Correct Score top-5.

    Modello bivariate Poisson:
      X_rem = X_ind + Z    (gol rimanenti casa)
      Y_rem = Y_ind + Z    (gol rimanenti trasf.)
      X_ind ~ Poisson(mu_c_ind)
      Y_ind ~ Poisson(mu_t_ind)
      Z     ~ Poisson(lambda0)   con lambda0 = rho_dinamico * min(mu_c, mu_t)

    shot_dom: indice di dominio tiri [0,1] — riduce rho quando il gioco è unidirezionale.
    """
    mu_c_rem = max(1e-9, float(mu_c_rem))
    mu_t_rem = max(1e-9, float(mu_t_rem))

    rho     = rho_dinamico(tot_cur, minuto, shot_dom, gol_casa + gol_trasf)
    # Media geometrica invece di min_mu (v45):
    # lambda0 = rho * min(mu_c, mu_t) sottostimava la correlazione quando una squadra
    # era nettamente più forte (es. mu_c=2.0, mu_a=0.3 → min=0.3, geom=0.775).
    # La media geometrica sqrt(mu_c*mu_a) è proporzionale ad entrambi i tassi,
    # più robusta e coerente con la struttura della bivariate Poisson (Karlis 2003).
    geom_mu = math.sqrt(mu_c_rem * mu_t_rem)
    # Cap abbassato da 0.90 a 0.75: in partite sbilanciate (es. mu_c=2.5, mu_a=0.3)
    # lambda0 al 90% della squadra più debole (0.27) sopravvalutava la correlazione
    # e gonfiava le probabilità di pareggio in blowout. 75% è più conservativo
    # e allineato con la letteratura (Karlis & Ntzoufras 2003).
    lambda0 = max(0.0, min(rho * geom_mu, 0.75 * min(mu_c_rem, mu_t_rem)))

    mu_c_ind = max(1e-9, mu_c_rem - lambda0)
    mu_t_ind = max(1e-9, mu_t_rem - lambda0)

    pmf_c = _poisson_pmf_norm(mu_c_ind)
    pmf_t = _poisson_pmf_norm(mu_t_ind)
    pmf_z = _poisson_pmf_norm(lambda0)

    # 1. Matrice indipendente con correzione Dixon-Coles
    joint_ind = {}
    dc_sum    = 0.0
    for i, pi in enumerate(pmf_c):
        if pi < 1e-16: continue
        for j, pj in enumerate(pmf_t):
            if pj < 1e-16: continue
            tau = dixon_coles_tau(i, j, mu_c_ind, mu_t_ind)
            val = pi * pj * tau
            joint_ind[(i, j)] = val
            dc_sum += val

    if dc_sum > 0:
        joint_ind = {k: v / dc_sum for k, v in joint_ind.items()}

    # 2. Matrice full: convoluzione con Z
    # P(X_rem=a, Y_rem=b) = sum_z P(X_ind=a-z, Y_ind=b-z) * P(Z=z)
    full = {}
    for (i, j), pij in joint_ind.items():
        for z, pz in enumerate(pmf_z):
            if pz < 1e-16: continue
            a, b = i + z, j + z
            full[(a, b)] = full.get((a, b), 0.0) + pij * pz

    fj_sum = sum(full.values())
    if fj_sum > 0:
        full = {k: v / fj_sum for k, v in full.items()}

    # 3. 1X2 — Z si cancella nella differenza, usiamo joint_ind direttamente
    p1 = px = p2 = 0.0
    for (i, j), pij in joint_ind.items():
        diff = (gol_casa + i) - (gol_trasf + j)
        if   diff > 0: p1 += pij
        elif diff < 0: p2 += pij
        else:          px += pij

    s12x = p1 + px + p2
    if s12x > 0:
        p1 /= s12x; px /= s12x; p2 /= s12x

    # 4. Under / Over — gestione quarter lines (2.25, 2.75, 3.25, 3.75)
    # Quarter line AH: metà stake su floor-half, metà su ceil-half.
    # Under 2.75 = ½×Under 2.5 + ½×Under 3.0
    # → se totale=3: mezzo win (U3.0 push) + mezza perdita (U2.5 lose) → P eff = P(≤2) + ½×P(=3)
    S     = gol_casa + gol_trasf
    line4 = round(linea_ou * 4)
    if line4 % 2 != 0:    # quarter line: 2.25, 2.75, 3.25 …
        h_low  = (line4 - 1) / 4.0    # es. 2.75 → 2.5
        h_high = (line4 + 1) / 4.0    # es. 2.75 → 3.0
        p_u_low  = sum(p for (a, b), p in full.items() if S + a + b < h_low)
        p_u_high = sum(p for (a, b), p in full.items() if S + a + b < h_high)
        p_under  = 0.5 * (p_u_low + p_u_high)
    else:                              # half line (2.5, 3.5 …) o intera (2.0, 3.0 …)
        p_under = sum(p for (a, b), p in full.items() if S + a + b < linea_ou)
    p_under = min(max(p_under, 0.0), 1.0)
    p_over  = 1.0 - p_under

    # 5. BTTS — usa la matrice full (include DC + correlazione bivariate)
    # La matrice full contiene P(X_rem=a, Y_rem=b) per tutti (a,b).
    # BTTS richiede che entrambe le squadre abbiano segnato nel totale partita.
    if gol_casa > 0 and gol_trasf > 0:
        p_btts = 1.0
    elif gol_casa > 0:
        # Casa ha già segnato; BTTS iff trasf segna almeno 1 nel rimasto
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if b > 0))
    elif gol_trasf > 0:
        # Trasf ha già segnato; BTTS iff casa segna almeno 1 nel rimasto
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if a > 0))
    else:
        # Nessuno ha ancora segnato; BTTS iff entrambi segnano nel rimasto
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if a > 0 and b > 0))
    p_btts = min(1.0, p_btts)

    # 6. Correct Score (punteggio finale)
    cs_final = {}
    for (a, b), p in full.items():
        key = (gol_casa + a, gol_trasf + b)
        cs_final[key] = cs_final.get(key, 0.0) + p

    top_cs = sorted(cs_final.items(), key=lambda x: x[1], reverse=True)[:5]

    # Distribuzione gol totali finali (usata per CS cumulativi nel UI)
    gol_tot_dist: dict[int, float] = {}
    for (fc, ft), p in cs_final.items():
        tot = fc + ft
        gol_tot_dist[tot] = gol_tot_dist.get(tot, 0.0) + p

    return p1, px, p2, p_under, p_over, p_btts, top_cs, rho, gol_tot_dist, full


# ==========================================
# UTILITY QUOTE E STAKING
# ==========================================

def calcola_quota_reale(prob):
    return 1.0 / prob if prob > 0.001 else 999.0


def calcola_stake_kelly(prob_modello, quota_target, bankroll, frazione=0.5):
    """
    Kelly frazionato: stake = bankroll * f* * frazione

    Formula standard: f* = (p*Q - 1) / (Q - 1)
      dove p = prob_modello, Q = quota_target (decimal odds, già NETTA di commissione)
    Cap al 5% del bankroll. Restituisce 0 se edge <= 0.

    La `frazione` è dinamica in v45: tipicamente 0.50, ridotta a 0.40 late-game
    o senza dati tiri, calcolata esternamente e passata qui.
    """
    if quota_target <= 1.01 or prob_modello * quota_target <= 1.0:
        return 0.0
    kelly_pct = (prob_modello * quota_target - 1.0) / (quota_target - 1.0)
    kelly_pct = max(0.0, min(kelly_pct, 0.05))   # [0, 5%] — max(0) era mancante
    return bankroll * kelly_pct * frazione


def calcola_stake_lay(prob_modello, quota_exc, bankroll, frazione=0.5, comm_rate=0.0):
    """
    Kelly frazionato per LAY su exchange, con aggiustamento commissione.

    Quando banci a quota Q con commissione c:
      - Vinci la stake*(1-c)  se l'evento NON accade  (prob = 1 - prob_modello)
      - Perdi la liability = stake * (Q - 1)           se l'evento accade

    Kelly fraction della liability (commission-adjusted):
      f_liability* = 1 - prob_modello * (Q - c) / (1 - c)

    Senza commissione (c=0) si riduce alla formula standard: 1 - p*Q.
    Con commissione (es. 2.5%) overcalcolare di ~2.6% usando la formula non aggiustata.

    Restituisce (stake_visibile, liability) oppure None se non c'è valore.
    Cap al 5% del bankroll come liability.
    """
    if quota_exc <= 1.30:   # lay non ha senso con quote molto basse
        return None
    if comm_rate >= 1.0:    # sanity: commissione impossibile
        return None
    # Commission-adjusted Kelly lay fraction
    f_liability = 1.0 - prob_modello * (quota_exc - comm_rate) / (1.0 - comm_rate)
    if f_liability <= 0:
        return None
    f_liability = min(f_liability, 0.05)
    liability   = bankroll * f_liability * frazione
    stake       = liability / (quota_exc - 1.0)
    # Sanity: stake non può superare il bankroll (scenario quota quasi 1.0)
    if stake > bankroll:
        return None
    return stake, liability


# ==========================================
# BLEND xG LINEE + TIRI
# ==========================================

def blend_xg_shots(mu_h_line, mu_a_line,
                   sot_h, soff_h, sot_a, soff_a,
                   gol_h, gol_a, minuto):
    """
    Integra le stime xG da linee di mercato con le evidenze empiriche dei tiri.

    ── xG PER TIRO (valori calibrati su top-5 leagues) ─────────────────────────
      XG_SOT  = 0.33  (tiro in porta medio)
      XG_SOFF = 0.05  (tiro fuori porta)

      Correzione game-state ±10%:
        Chi è in svantaggio pressa in modo disperato → qualità tiri inferiore (×0.90)
        Chi è in vantaggio segna spesso in contropiede → qualità superiore (×1.10)

    ── PROIEZIONE AL TEMPO RIMANENTE ────────────────────────────────────────────
      rate_h  = xg_h_accum / frac_giocata   (xG per 90' implicato dai tiri)
      mu_h_shots = rate_h × frac_rimasta

    ── BLEND IN SPAZIO T+D ──────────────────────────────────────────────────────
      Separare Totale e Differenziale permette pesi diversi:
        α_T  (Totale):      max 0.25  — il Total line è molto efficiente, muoviti poco
        α_D  (Differenziale): max 0.70 — i tiri rivelano chi domina meglio del mercato

      Entrambi i pesi crescono con:
        • shot_info = min(1, n_shots/15)  — campione sufficiente (15 tiri totali)
        • frac_giocata (o sua radice) — i dati live sono più affidabili col tempo

      T_blend = (1−α_T)·T_line + α_T·T_shots
      D_blend = (1−α_D)·D_line + α_D·D_shots
      mu_h = max(ε, (T_blend + D_blend) / 2)
      mu_a = max(ε, (T_blend − D_blend) / 2)

    Restituisce:
      mu_h_final, mu_a_final  — xG rimanenti blendati
      xg_h_accum, xg_a_accum  — xG accumulato dai tiri (per debug)
      alpha_T, alpha_D         — pesi effettivi del blend (per debug)
      shot_dom                 — indice dominio tiri [0,1] (per rho e debug)
    """
    # 0.30 SOT (era 0.33): valore conservativo senza dati di posizione.
    # 0.33 è la media globale includendo rigori/tap-in; open-play medio = 0.24-0.30.
    XG_SOT  = 0.30
    XG_SOFF = 0.05
    # MU_MAX dinamico: in gare già ad alto punteggio il cap fisso 3.5 sottostimava
    # la proiezione residua (es. 5-0 al 45' → ritmo reale ≈ 10 gol/90', cap a 3.5
    # produce previsioni di soli ~1.75 gol nel 2° tempo, chiaramente troppo basse).
    # Nuovo: max(3.5, gol_totali * 1.2) — si adatta alle gare eccezionali.
    gol_totali_now = float(gol_h + gol_a)
    MU_MAX  = max(3.5, gol_totali_now * 1.2)

    # Correzione game-state sulla qualità dei tiri in porta — progressiva per margine.
    # +7% per gol di vantaggio (max 15%): 1 gol → ±7%, 2+ gol → ±14-15%.
    # La squadra in svantaggio pressa in modo sempre più disperato → qualità tiri
    # cala (tiri da lontano/angolati); la squadra in vantaggio segna in contropiede
    # su difese scoperte → qualità migliore.
    diff_score = gol_h - gol_a
    gs_adj = min(0.15, abs(diff_score) * 0.07)
    if diff_score > 0:
        k_h, k_a = 1.0 + gs_adj, 1.0 - gs_adj   # casa in vantaggio → contropiede
    elif diff_score < 0:
        k_h, k_a = 1.0 - gs_adj, 1.0 + gs_adj   # trasf in vantaggio → contropiede
    else:
        k_h, k_a = 1.00, 1.00

    xg_h_accum = sot_h * XG_SOT * k_h + soff_h * XG_SOFF
    xg_a_accum = sot_a * XG_SOT * k_a + soff_a * XG_SOFF

    frac_giocata = max(minuto, 1) / 90.0
    frac_rimasta = max(0.0, (90.0 - minuto) / 90.0)

    # Proiezione rate → mu rimanenti (shots-based)
    rate_h     = xg_h_accum / frac_giocata
    rate_a     = xg_a_accum / frac_giocata
    mu_h_shots = rate_h * frac_rimasta
    mu_a_shots = rate_a * frac_rimasta

    # Indice di dominio (solo tiri in porta, più informativi)
    sot_tot = sot_h + sot_a
    shot_dom = abs(sot_h - sot_a) / sot_tot if sot_tot > 0 else 0.0

    # Pesi blend dinamici
    n_shots   = sot_h + soff_h + sot_a + soff_a
    shot_info = min(1.0, n_shots / 15.0)   # 15 tiri totali = campione sufficiente
    alpha_T   = min(0.25, shot_info * frac_giocata * 0.30)
    alpha_D   = min(0.70, shot_info * math.sqrt(frac_giocata))

    # Blend in spazio T+D
    T_line  = mu_h_line  + mu_a_line
    D_line  = mu_h_line  - mu_a_line
    T_shots = mu_h_shots + mu_a_shots
    D_shots = mu_h_shots - mu_a_shots

    T_blend = (1.0 - alpha_T) * T_line + alpha_T * T_shots
    D_blend = (1.0 - alpha_D) * D_line + alpha_D * D_shots

    eps = 1e-9
    mu_h_final = max(eps, min((T_blend + D_blend) / 2.0, MU_MAX))
    mu_a_final = max(eps, min((T_blend - D_blend) / 2.0, MU_MAX))

    return mu_h_final, mu_a_final, xg_h_accum, xg_a_accum, alpha_T, alpha_D, shot_dom


# ==========================================
# UI STREAMLIT
# ==========================================

st.set_page_config(page_title="Radar Pro Live", page_icon="⚡", layout="centered")
st.title("⚡ Radar Pro Live")
st.caption("v56 · Soglie Over/Under separate: bonus gol_mancanti solo sull'Over, Under invariato")

# INPUT
st.header("1. Stato Partita")
minuto_gioco = st.slider("Minuto Attuale", 0, 90, 0, 1)

col_g1, col_g2 = st.columns(2)
with col_g1:
    gol_casa  = st.number_input("Gol CASA",   value=0, min_value=0)
with col_g2:
    gol_trasf = st.number_input("Gol TRASF.", value=0, min_value=0)

col_r1, col_r2 = st.columns(2)
with col_r1:
    rossi_casa  = st.number_input("Rossi CASA",   value=0, min_value=0, max_value=4)
with col_r2:
    rossi_trasf = st.number_input("Rossi TRASF.", value=0, min_value=0, max_value=4)

st.divider()
st.header("2. Linee Asiatiche")

col_a1, col_a2 = st.columns(2)
with col_a1:
    st.markdown("**Apertura — full 90'**")
    ah_op  = st.number_input("AH Apertura",     value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50,  step=0.25)
with col_a2:
    st.markdown("**Corrente — gol rimanenti**")
    ah_cur  = st.number_input("AH Corrente",     value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75,  step=0.25)

_linee_ou = [0.5, 1.5, 1.75, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.5, 5.5]
# Default: seleziona la linea più vicina a tot_cur per coerenza col mercato
_idx_ou_default = min(range(len(_linee_ou)), key=lambda i: abs(_linee_ou[i] - tot_cur))
linea_target_ou = st.selectbox(
    "Linea U/O da analizzare:", _linee_ou, index=_idx_ou_default,
    help="Selezionata automaticamente in base al Totale Corrente. Le linee X.75/X.25 "
         "sono Asian quarter lines (mezzo stake su ogni semi-linea)."
)

col_bk, col_cm = st.columns(2)
with col_bk:
    cassa = st.number_input("Bankroll (€)", value=1000.0, step=100.0)
with col_cm:
    comm_pct = st.number_input(
        "Commissione exchange (%)", value=2.5, min_value=0.0, max_value=10.0, step=0.5,
        help="Betfair Standard: 2-5%. Usata per calcolare l'edge netto."
    )
comm_rate = comm_pct / 100.0

st.divider()
st.header("3. Tiri (Live)")
st.caption("Lascia a 0 per analisi prematch. Live: inserisci i tiri totali da inizio gara.")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    sot_h  = st.number_input("In porta CASA",   min_value=0, value=0, step=1)
with col_t2:
    soff_h = st.number_input("Fuori CASA",      min_value=0, value=0, step=1)
with col_t3:
    sot_a  = st.number_input("In porta TRASF.", min_value=0, value=0, step=1)
with col_t4:
    soff_a = st.number_input("Fuori TRASF.",    min_value=0, value=0, step=1)

# Validazione coerenza tiri / minuto
_n_tiri_totali = sot_h + soff_h + sot_a + soff_a
if _n_tiri_totali > 0 and minuto_gioco == 0:
    st.warning("⚠️ Tiri inseriti ma minuto = 0: per analisi prematch lascia tutti i tiri a 0, "
               "altrimenti il modello proietta il rate su 90' interi.")
elif _n_tiri_totali > 0 and minuto_gioco > 0:
    # Ceiling generoso: ~0.65 tiri totali/minuto in media europea (entrambe le squadre)
    _tiri_max_attesi = max(6, int(minuto_gioco * 0.65) + 4)
    if _n_tiri_totali > _tiri_max_attesi:
        st.warning(
            f"⚠️ {_n_tiri_totali} tiri totali al {minuto_gioco}' sembra elevato "
            f"(atteso ≤ ~{_tiri_max_attesi}) — verifica che siano tiri totali da inizio gara, "
            "non dell'ultimo periodo."
        )

st.divider()

if st.button("ANALIZZA", use_container_width=True, type="primary"):

    with st.spinner("Calcolo matrice bivariata..."):
        # 1. xG da linee (prior bayesiano)
        xg1_base, xg2_base = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, minuto_gioco
        )

        # 2. Blend tiri + linee (solo se ci sono tiri inseriti)
        n_shots_tot = sot_h + soff_h + sot_a + soff_a
        if n_shots_tot > 0 and minuto_gioco > 0:
            xg1_blend, xg2_blend, \
            xg_h_accum, xg_a_accum, \
            alpha_T, alpha_D, shot_dom = blend_xg_shots(
                xg1_base, xg2_base,
                sot_h, soff_h, sot_a, soff_a,
                gol_casa, gol_trasf, minuto_gioco
            )
        else:
            xg1_blend, xg2_blend = xg1_base, xg2_base
            xg_h_accum = xg_a_accum = 0.0
            alpha_T = alpha_D = shot_dom = 0.0

        # 3. Time decay + score effect + rossi (opera sullo xG blendato)
        xg1_live, xg2_live = time_decay_dinamico(
            xg1_blend, xg2_blend, minuto_gioco,
            gol_casa, gol_trasf, rossi_casa, rossi_trasf
        )

        # 4. Probabilità complete (rho aggiustato per dominio tiri)
        mc_1, mc_x, mc_2, mc_u, mc_o, mc_btts, top_cs, rho_used, gol_tot_dist, full_matrix = calcola_tutto(
            xg1_live, xg2_live,
            gol_casa, gol_trasf,
            linea_target_ou, tot_cur, minuto_gioco,
            shot_dom=shot_dom
        )
        delta_ah  = ah_cur  - ah_op
        delta_tot = tot_cur - tot_op
        momentum  = calcola_momentum_mercato(delta_ah, delta_tot, minuto_gioco)
        flat_lines = abs(ah_cur - ah_op) < 1e-6 and abs(tot_cur - tot_op) < 1e-6

    # ── QUOTE FAIR (sempre visibili, sempre informative) ─────────────────────
    st.header(f"Quote Fair  —  {minuto_gioco}' | {gol_casa}–{gol_trasf}")
    st.caption("Confronta queste quote con quelle sull'exchange: se vedi di meglio, c'è valore.")

    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa",     f"@{calcola_quota_reale(mc_1):.2f}",  f"{mc_1:.1%}")
    cx.metric("X — Pareggio", f"@{calcola_quota_reale(mc_x):.2f}", f"{mc_x:.1%}")
    c2.metric("2 — Trasf.",   f"@{calcola_quota_reale(mc_2):.2f}", f"{mc_2:.1%}")

    cu, co, cb = st.columns(3)
    cu.metric(f"Under {linea_target_ou}", f"@{calcola_quota_reale(mc_u):.2f}", f"{mc_u:.1%}")
    co.metric(f"Over  {linea_target_ou}", f"@{calcola_quota_reale(mc_o):.2f}", f"{mc_o:.1%}")
    cb.metric("BTTS — Si",               f"@{calcola_quota_reale(mc_btts):.2f}", f"{mc_btts:.1%}")

    with st.expander("Correct Score — Top 5 + Totali"):
        cs_cols = st.columns(5)
        for idx, ((fc, ft), prob) in enumerate(top_cs):
            cs_cols[idx].metric(
                label=f"{fc}–{ft}",
                value=f"@{calcola_quota_reale(prob):.2f}",
                delta=f"{prob:.2%}"
            )
        # Distribuzione gol totali cumulativi
        st.caption("**Distribuzione gol totali partita**")
        max_tot = max(gol_tot_dist.keys()) if gol_tot_dist else 6
        tot_rows = []
        cum = 0.0
        for t in range(min(max_tot + 1, 10)):
            p = gol_tot_dist.get(t, 0.0)
            cum += p
            tot_rows.append(f"**{t} gol**: {p:.1%} · cumul. ≤{t}: {cum:.1%}")
        st.markdown("  \n".join(tot_rows))

    with st.expander("Asian Handicap — Probabilità a diversi livelli"):
        st.caption(
            "Handicap sui **gol rimanenti** — coerente con la linea AH Corrente inserita sopra. "
            "Usa la matrice bivariate completa (Dixon-Coles + correlazione)."
        )
        ah_levels = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0, +2.5]
        ah_rows = []
        for level in ah_levels:
            win = push = lose = 0.0
            for (a, b), p in full_matrix.items():
                # Gol RIMANENTI + handicap (coerente con calcola_xg_bayesiani che usa i-j+h)
                diff = (a - b) + level
                if   diff > 1e-9:  win  += p
                elif diff < -1e-9: lose += p
                else:              push += p
            # P efficace Asian: push restituisce la stake → conta come 0.5 win
            p_eff   = win + 0.5 * push
            q_cover = calcola_quota_reale(p_eff) if p_eff > 1e-9 else 999.0
            sign     = "+" if level >= 0 else ""
            push_txt = f" · push {push:.1%}" if push > 0.005 else ""
            ah_rows.append(
                f"AH {sign}{level:.1f} → Casa copre: **{p_eff:.1%}** · fair @{q_cover:.2f}{push_txt}"
            )
        st.markdown("  \n".join(ah_rows))

    # ── MOMENTUM ─────────────────────────────────────────────────────────────
    st.divider()
    if momentum < 1.0:
        mom_label = f"Mercato stabile [{momentum:.2f}/6.0]"
    elif momentum < 2.5:
        mom_label = f"Movimento moderato [{momentum:.2f}/6.0]"
    elif momentum < 4.0:
        mom_label = f"Movimento significativo [{momentum:.2f}/6.0]"
    else:
        mom_label = f"Movimento estremo — verifica eventi non registrati [{momentum:.2f}/6.0]"
    st.progress(min(momentum / 6.0, 1.0), text=mom_label)

    # ── SEGNALI RAPIDI (senza quote exchange) ────────────────────────────────
    st.divider()
    st.header("Segnali rapidi")
    st.caption("Nessuna quota da inserire — confronta a occhio con l'exchange.")

    if minuto_gioco >= 85:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()

    frac_giocata = minuto_gioco / 90.0
    # Floor 55%/58%: evitare segnali su eventi quasi-equi (p≈51%) a inizio partita.
    # La soglia cresce col tempo per compensare la riduzione di incertezza.
    soglia_min_1x2  = max(0.55, 0.50 + 0.10 * frac_giocata)
    soglia_min_btts = max(0.55, 0.50 + 0.10 * frac_giocata)
    gol_attuali     = gol_casa + gol_trasf
    gol_mancanti    = max(0.0, linea_target_ou - gol_attuali)
    _base_ou = max(0.58, 0.55 + 0.10 * frac_giocata)
    # Soglia Over: cresce con i gol mancanti — più gol servono, più alto deve essere
    # il conviction del modello. 1 gol: +0%, 2 gol: +2%, 3+: +4-6% (cap 6%).
    # mc_o già scende all'aumentare dei gol mancanti, ma la soglia contestuale evita
    # che un P=60% con 3 gol mancanti sembri lo stesso segnale di P=60% con 1 gol.
    _ou_gol_bonus   = min(0.06, max(0.0, (gol_mancanti - 1.0) * 0.02))
    soglia_min_ou_over  = _base_ou + _ou_gol_bonus
    # Soglia Under: nessun bonus gol_mancanti — per Under la "difficoltà" cresce con
    # i gol già segnati (più gol ci sono, più è difficile che rimanga Under). Questo
    # è già catturato direttamente da mc_u che scende al crescere di gol_attuali.
    # Alzare la soglia Under in base a gol_mancanti la penalizzerebbe due volte.
    soglia_min_ou_under = _base_ou
    soglia_min_ou       = soglia_min_ou_over  # alias legacy per compatibilità interna

    # Soglia minima con margine 6%: la quota sull'exchange deve battere fair * 1.06
    MARGINE_RAPIDO = 0.06

    def _q_target_back(prob):
        """Quota minima cercata sull'exchange per avere ~6% di edge lordo."""
        fair = calcola_quota_reale(prob)
        return fair * (1.0 + MARGINE_RAPIDO)

    def _q_target_lay(prob):
        """Quota massima cercata per il lay: fair / (1 + margine)."""
        fair = calcola_quota_reale(prob)
        return fair / (1.0 + MARGINE_RAPIDO)

    quick_signals = False

    def _segnale_rapido(etichetta, prob, soglia_back, tipo_ou=None):
        """
        Mostra il segnale rapido senza richiedere la quota exchange.
        tipo_ou: 'over' | 'under' | None
        """
        global quick_signals
        fair = calcola_quota_reale(prob)
        if fair < 1.15:   # evento quasi certo, skip
            return
        q_min = _q_target_back(prob)
        q_max = _q_target_lay(prob)

        if prob >= soglia_back:
            st.success(
                f"**BACK candidato — {etichetta}** · Modello {prob:.1%} · Fair @{fair:.2f}\n\n"
                f"✅ Cerca sull'exchange **almeno @{q_min:.2f}** per avere edge"
            )
            quick_signals = True
        elif prob <= 0.35 and fair >= 1.30:
            st.warning(
                f"**LAY candidato — {etichetta}** · Modello {prob:.1%} · Fair @{fair:.2f}\n\n"
                f"✅ Banca se la quota sull'exchange è **al massimo @{q_max:.2f}**"
            )
            quick_signals = True

    _segnale_rapido("1 Casa",      mc_1, soglia_min_1x2)
    _segnale_rapido("2 Trasf.",    mc_2, soglia_min_1x2)
    if mc_x >= soglia_min_1x2:
        _segnale_rapido("X Pareggio", mc_x, soglia_min_1x2)

    if gol_attuali >= linea_target_ou:
        st.success(f"✅ Over {linea_target_ou} già **VINTO** — {gol_attuali:.0f} gol totali. Mercato chiuso.")
        st.error(f"❌ Under {linea_target_ou} già **PERSO**. Mercato chiuso.")
    else:
        _segnale_rapido(f"Over {linea_target_ou}",  mc_o, soglia_min_ou_over)
        _segnale_rapido(f"Under {linea_target_ou}", mc_u, soglia_min_ou_under)

    btts_yes_settled = gol_casa > 0 and gol_trasf > 0
    if not btts_yes_settled:
        _segnale_rapido("BTTS Sì", mc_btts, soglia_min_btts)
        # BTTS No ha una soglia più bassa: "No" si realizza se ALMENO UNA squadra
        # non segna — evento strutturalmente più probabile di "Sì" in molte gare.
        # Usare la stessa soglia 55% sovrasopprimeva segnali legittimi su squadre
        # difensive. Soglia separata: max 50% a inizio → cresce col tempo.
        soglia_min_btts_no = max(0.50, 0.45 + 0.10 * frac_giocata)
        _segnale_rapido("BTTS No", 1.0 - mc_btts, soglia_min_btts_no)

    if not quick_signals:
        st.info("Nessun candidato forte al momento — il modello non vede probabilità dominanti.")

    # ── WARNING AFFIDABILITÀ ─────────────────────────────────────────────────
    if flat_lines and n_shots_tot == 0 and minuto_gioco >= 30:
        st.warning(
            "⚠️ **Affidabilità ridotta**: linee di mercato invariate e nessun tiro live. "
            "Il modello opera su prior di prematch — inserisci i tiri per aumentare la precisione."
        )
    elif n_shots_tot == 0 and minuto_gioco >= 45:
        st.warning(
            "⚠️ **Nessun dato tiri live**: oltre il 45' senza tiri, "
            "il modello ha affidabilità ridotta. Inserisci i tiri per sbloccare l'alpha."
        )

    # ── ANALISI AVANZATA CON QUOTE EXCHANGE (opzionale) ──────────────────────
    st.divider()
    with st.expander("⚙️ Analisi avanzata con quote exchange (opzionale)"):
        st.caption(
            "Inserisci le quote che vedi sull'exchange per ottenere edge preciso, "
            "stake Kelly in € e EV atteso. Lascia a 0 i mercati che non ti interessano."
        )
        eq1, eqx, eq2 = st.columns(3)
        q_exc_1 = eq1.number_input("Quota 1 (Casa)",     min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_exc_x = eqx.number_input("Quota X (Pareggio)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_exc_2 = eq2.number_input("Quota 2 (Trasf.)",   min_value=0.0, value=0.0, step=0.05, format="%.2f")

        equ, eqo, eqb = st.columns(3)
        q_exc_u    = equ.number_input(f"Quota Under {linea_target_ou}", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_exc_o    = eqo.number_input(f"Quota Over {linea_target_ou}",  min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_exc_btts = eqb.number_input("Quota BTTS Sì",                  min_value=0.0, value=0.0, step=0.05, format="%.2f")

        eqbn_col, _, _ = st.columns(3)
        q_exc_btts_no = eqbn_col.number_input("Quota BTTS No", min_value=0.0, value=0.0, step=0.05, format="%.2f")

    # ── SEGNALI EXCHANGE (con quote) ──────────────────────────────────────────
    segnali      = False

    # Riduzione stake proporzionale al momentum (eventi non registrati)
    momentum_stake_factor = max(0.4, 1.0 - 0.10 * max(0.0, momentum - 2.5))

    # Kelly fraction dinamica (v45):
    # Diminuisce quando il gioco è avanzato (più rumore) o mancano dati sui tiri live.
    kelly_base = 0.50
    if minuto_gioco > 75:
        kelly_base -= 0.10              # late game: spread ampi, più rumore
    if minuto_gioco > 0 and n_shots_tot == 0:
        kelly_base -= 0.05              # partita in corso senza dati tiri
    kelly_frac = max(0.20, kelly_base)

    any_exc_quote = any(
        q > 1.0 for q in [q_exc_1, q_exc_x, q_exc_2, q_exc_u, q_exc_o, q_exc_btts, q_exc_btts_no]
    )

    # Edge netto di commissione:
    # La quota effettiva per il back è q_netto = 1 + (q_exc-1)*(1-comm)
    # Un edge lordo del 3% con commissione 5% può essere EV negativo.
    # MIN_EDGE_BACK e MIN_EDGE_LAY sono già sui valori netti.
    MIN_EDGE_BACK = 0.030   # 3% edge netto minimo per back
    MIN_EDGE_LAY  = 0.040   # 4% edge netto minimo per lay (rischio asimmetrico)
    # Alzato da 1.15 (>87%) a 1.25 (>80%): 87% non è una certezza, può contenere
    # edge reale se il modello vede 92%+. Con 1.25 tagliamo solo eventi dove la
    # quota exchange è così bassa da non offrire mai leva sufficiente.
    MIN_FAIR_Q    = 1.25    # eventi quasi certi (>80%) → skip, già nel prezzo

    def _q_netto(q_exc):
        """Quota effettiva back dopo commissione exchange."""
        return 1.0 + (q_exc - 1.0) * (1.0 - comm_rate)

    def _valuta(etichetta, prob_mod, q_exc, soglia_back, back_only=False):
        """
        Logica unificata back/lay per ogni mercato.

        Con quota exchange:
          PUNTA  se  prob_mod - 1/q_netto >= MIN_EDGE_BACK  AND  prob_mod >= soglia_back
                     (edge calcolato sulla quota NETTA di commissione)
          BANCA  se  1/q_exc - prob_mod >= MIN_EDGE_LAY   AND  q_exc >= 1.30
                     (il lay non paga commissione sulla perdita, solo sulla vincita)

        back_only=True: valuta solo il BACK (usato per Over a fine gara dove il back
        non ha senso ma il LAY sull'Over è comunque valutabile via UNDER).

        Senza quota exchange (q_exc <= 1.0):
          Indicazione qualitativa solo se il modello è fortemente convinto.
          Nessun Kelly senza prezzo reale.

        Skip sempre se l'evento è quasi certo (fair < MIN_FAIR_Q = @1.15).
        """
        global segnali

        q_fair = calcola_quota_reale(prob_mod)

        # Evento quasi certo: già nel prezzo, nessun edge reale
        if q_fair < MIN_FAIR_Q:
            return False

        # ── SENZA QUOTA EXCHANGE ──────────────────────────────────────────────
        if q_exc <= 1.0:
            # Soglia qualitativa cresce col tempo: a partita inoltrata il modello
            # deve essere più convinto per giustificare un'indicazione senza prezzo.
            soglia_qualitativa = max(0.62, 0.60 + 0.10 * frac_giocata)
            if prob_mod >= soglia_qualitativa:
                st.info(
                    f"📈 **{etichetta}**: modello {prob_mod:.1%} (fair @{q_fair:.2f}) "
                    f"— potenziale back · inserisci quota per edge preciso"
                )
                segnali = True
                return True
            if prob_mod <= 0.30 and not back_only:
                st.info(
                    f"📉 **{etichetta}**: modello {prob_mod:.1%} (fair @{q_fair:.2f}) "
                    f"— mercato probabilmente sopravvaluta · considera lay · inserisci quota per conferma"
                )
                segnali = True
                return True
            return False

        # ── CON QUOTA EXCHANGE ────────────────────────────────────────────────
        q_net     = _q_netto(q_exc)          # quota netta per back
        prob_imp  = 1.0 / q_exc              # probabilità implicita del mercato
        edge_back = prob_mod - (1.0 / q_net) # edge netto commissione per back
        # Edge lay con commissione corretta:
        # Break-even lay: p_be = (1 - comm) / (Q - comm)
        # edge_lay = p_be - prob_mod > 0 → lay ha valore
        _denom    = q_exc - comm_rate
        p_be_lay  = (1.0 - comm_rate) / _denom if _denom > 0 else 1.0
        edge_lay  = p_be_lay - prob_mod
        segnalato = False

        # BACK (con edge netto)
        if edge_back >= MIN_EDGE_BACK and prob_mod >= soglia_back:
            stake_raw = calcola_stake_kelly(prob_mod, q_net, cassa, kelly_frac)
            # Cap adattivo per edge: edge piccolo → stake massima più conservativa.
            # Kelly formula già riduce naturalmente con edge bassi, ma il cap fisso
            # al 5% si attivava uguale per edge 3% e edge 10%.
            if edge_back < 0.05:
                stake_raw = min(stake_raw, cassa * 0.025 * kelly_frac)
            elif edge_back < 0.10:
                stake_raw = min(stake_raw, cassa * 0.040 * kelly_frac)
            # edge >= 10%: cap 5% già applicato da calcola_stake_kelly
            stake = stake_raw * momentum_stake_factor
            if stake > 0:
                riduzioni = []
                if comm_rate > 0:
                    riduzioni.append(f"comm {comm_pct:.1f}% → @{q_net:.3f} netto")
                if momentum_stake_factor < 1.0:
                    riduzioni.append(f"momentum ×{momentum_stake_factor:.2f}")
                if kelly_frac < 0.50:
                    riduzioni.append(f"Kelly ×{kelly_frac:.2f}")
                rid_txt = f" _({', '.join(riduzioni)})_" if riduzioni else ""
                ev_euro_back = stake * (prob_mod * q_net - 1.0)
                st.success(
                    f"**PUNTA {etichetta}** — "
                    f"Modello {prob_mod:.1%} · Mercato @{q_exc:.2f} ({prob_imp:.1%}) · "
                    f"Edge netto **+{edge_back*100:.1f}%** · EV **€{ev_euro_back:+.2f}**"
                )
                st.write(f"Stake Kelly: **€{stake:.2f}**{rid_txt}")
                segnali   = True
                segnalato = True

        # LAY — quota minima 1.30. Non valutato se back_only=True.
        if not segnalato and not back_only and edge_lay >= MIN_EDGE_LAY and q_exc >= 1.30:
            result = calcola_stake_lay(prob_mod, q_exc, cassa, kelly_frac, comm_rate)
            if result is not None:
                stake_lay, liab_lay = result
                stake_lay *= momentum_stake_factor
                liab_lay  *= momentum_stake_factor
                riduzioni = []
                if momentum_stake_factor < 1.0:
                    riduzioni.append(f"momentum ×{momentum_stake_factor:.2f}")
                if kelly_frac < 0.50:
                    riduzioni.append(f"Kelly ×{kelly_frac:.2f}")
                rid_txt = f" _({', '.join(riduzioni)})_" if riduzioni else ""
                ev_euro_lay = stake_lay * ((1.0 - prob_mod) * (1.0 - comm_rate) - prob_mod * (q_exc - 1.0))
                st.warning(
                    f"**BANCA {etichetta}** — "
                    f"Modello {prob_mod:.1%} · Mercato @{q_exc:.2f} ({prob_imp:.1%}) · "
                    f"Edge lay netto **+{edge_lay*100:.1f}%** · EV **€{ev_euro_lay:+.2f}**"
                )
                st.write(f"Stake lay: **€{stake_lay:.2f}** · Liability: **€{liab_lay:.2f}**{rid_txt}")
                segnali   = True
                segnalato = True

        return segnalato

    # ── VALUTAZIONE MERCATI ───────────────────────────────────────────────────

    # 1X2
    _valuta("1 CASA",      mc_1, q_exc_1, soglia_min_1x2)
    _valuta("2 TRASFERTA", mc_2, q_exc_2, soglia_min_1x2)
    if q_exc_x > 1.0:  # pareggio: troppo incerto senza quota di mercato
        _valuta("X PAREGGIO", mc_x, q_exc_x, soglia_min_1x2)

    # Over/Under
    if gol_attuali >= linea_target_ou:
        st.success(f"✅ Over {linea_target_ou} già VINTO — {gol_attuali:.0f} gol. Mercato chiuso.")
        st.error(f"❌ Under {linea_target_ou} già PERSO — {gol_attuali:.0f} gol. Mercato chiuso.")
    else:
        if minuto_gioco >= 75 and gol_mancanti >= 2:
            # Back Over: troppo tardi. Ma LAY Over (= back Under) rimane valido.
            st.info(
                f"Over {linea_target_ou}: al {minuto_gioco}' mancano {gol_mancanti:.0f} gol ({mc_o:.1%}) "
                f"— BACK sconsigliato. Valuto solo LAY sull'Over se hai la quota."
            )
            _valuta(f"OVER {linea_target_ou}", mc_o, q_exc_o, soglia_min_ou_over, back_only=False)
        else:
            _valuta(f"OVER {linea_target_ou}", mc_o, q_exc_o, soglia_min_ou_over)
        _valuta(f"UNDER {linea_target_ou}", mc_u, q_exc_u, soglia_min_ou_under)

    # BTTS — gestione mercato già chiuso
    btts_yes_settled = gol_casa > 0 and gol_trasf > 0
    btts_no_settled  = minuto_gioco >= 88 and (gol_casa == 0 or gol_trasf == 0) and mc_btts < 0.10

    if btts_yes_settled:
        st.success("✅ BTTS SÌ già VINTO — entrambe le squadre hanno segnato. Mercato chiuso.")
        st.error("❌ BTTS NO già PERSO. Mercato chiuso.")
    elif btts_no_settled:
        st.error("❌ BTTS SÌ quasi impossibile a questo punto. Mercato chiuso praticamente.")
        st.success("✅ BTTS NO quasi VINTO — mercato sul filo, non entrare ora.")
    else:
        _valuta("BTTS SÌ", mc_btts,       q_exc_btts,    soglia_min_btts)
        _valuta("BTTS NO", 1.0 - mc_btts, q_exc_btts_no, soglia_min_btts)

    # Incoerenza BTTS Sì + Under bassa linea
    # BTTS Sì richiede ≥2 gol totali → non può coesistere con Under 1.5 ad alta prob.
    if mc_btts > 0.50 and mc_u > 0.55 and linea_target_ou <= 1.5 and not btts_yes_settled:
        st.warning(
            f"⚠️ Incoerenza interna: BTTS Sì ({mc_btts:.0%}) e Under {linea_target_ou} "
            f"({mc_u:.0%}) sono incompatibili — BTTS Sì richiede ≥2 gol totali. "
            "Verifica le linee inserite."
        )

    # Incoerenza Over + BTTS No
    if mc_o > 0.50 and mc_btts < 0.35 and gol_attuali < linea_target_ou:
        st.warning(
            f"⚠️ Incoerenza: Over {linea_target_ou} probabile ({mc_o:.0%}) "
            f"ma BTTS solo {mc_btts:.0%}. "
            f"Over senza BTTS richiede gol multipli dalla stessa squadra."
        )

    if not segnali:
        if any_exc_quote:
            st.error("NO BET — Quote exchange allineate al modello, nessun edge sufficiente.")
        else:
            st.info("Nessuna indicazione forte. Il modello non vede probabilità dominanti al momento.")

    # DEBUG
    st.divider()
    with st.expander("Parametri interni del motore"):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Probabilità modello**")
            st.write(f"P(1)       = {mc_1:.4f}")
            st.write(f"P(X)       = {mc_x:.4f}")
            st.write(f"P(2)       = {mc_2:.4f}")
            sum_1x2 = mc_1 + mc_x + mc_2
            ok_1x2  = "✅" if abs(sum_1x2 - 1.0) < 0.001 else "⚠️"
            st.write(f"Σ(1+X+2)   = {sum_1x2:.4f} {ok_1x2}")
            st.divider()
            st.write(f"P(U{linea_target_ou})  = {mc_u:.4f}")
            st.write(f"P(O{linea_target_ou})  = {mc_o:.4f}")
            sum_ou  = mc_u + mc_o
            ok_ou   = "✅" if abs(sum_ou - 1.0) < 0.001 else "⚠️"
            st.write(f"Σ(U+O)     = {sum_ou:.4f} {ok_ou}")
            st.divider()
            st.write(f"P(BTTS)    = {mc_btts:.4f}")
            st.write(f"P(BTTSno)  = {1.0-mc_btts:.4f}")
        with col_r:
            st.markdown("**Parametri motore**")
            st.write(f"rho dinamico  = {rho_used:.4f}")
            st.write(f"lambda casa   = {xg1_live:.4f}")
            st.write(f"lambda trasf  = {xg2_live:.4f}")
            st.write(f"xG base casa  = {xg1_base:.4f}")
            st.write(f"xG base trasf = {xg2_base:.4f}")
            st.divider()
            if n_shots_tot > 0:
                st.markdown("**Tiri & blend**")
                st.write(f"xG accum casa  = {xg_h_accum:.3f}")
                st.write(f"xG accum trasf = {xg_a_accum:.3f}")
                st.write(f"xG blend casa  = {xg1_blend:.4f}")
                st.write(f"xG blend trasf = {xg2_blend:.4f}")
                st.write(f"α_T (Total)    = {alpha_T:.3f}")
                st.write(f"α_D (Diff)     = {alpha_D:.3f}")
                st.write(f"Shot dominance = {shot_dom:.3f}")
                st.divider()
            st.write(f"Soglia 1X2     = {soglia_min_1x2:.3f}")
            st.write(f"Soglia Over    = {soglia_min_ou_over:.3f}  (+{_ou_gol_bonus:.2f} gol-bonus)")
            st.write(f"Soglia Under   = {soglia_min_ou_under:.3f}")
            st.write(f"Soglia BTTS    = {soglia_min_btts:.3f}")
            st.divider()
            st.write(f"Delta AH     = {delta_ah:+.2f}")
            st.write(f"Delta Tot    = {delta_tot:+.2f}")
            st.write(f"Momentum     = {momentum:.2f}/6.0")
            st.write(f"Stake factor = ×{momentum_stake_factor:.2f}")
