import streamlit as st
import math

# ==========================================
# ⚙️ FUNZIONI DI SUPPORTO
# ==========================================

def rho_dinamico(tot_cur, minuto):
    """
    Correlazione comune bivariata non più fissa a 0.12.

    Logica:
    - tot_cur alto → partita aperta → le squadre segnano più indipendentemente → rho scende
    - Avanzare del minuto → i gol già osservati dominano l'informazione → rho scende
      (il modello dipende meno dalla struttura latente comune)

    Range risultante: ~0.03 (partita open late) → ~0.14 (partita chiusa inizio gara)
    Nota: base = 0.14 - 0.018*tot_cur → massimo teorico 0.14 (a tot_cur=0, minuto=0).
    """
    base  = 0.14 - 0.018 * min(tot_cur, 4.5)
    decay = 1.0 - 0.40 * (minuto / 90.0)
    return max(0.03, min(base * decay, 0.15))


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
    if   i == 0 and j == 0: return max(1e-6, 1.0 - mu_h * mu_a * rho_dc)
    elif i == 1 and j == 0: return max(1e-6, 1.0 + mu_a * rho_dc)
    elif i == 0 and j == 1: return max(1e-6, 1.0 + mu_h * rho_dc)
    elif i == 1 and j == 1: return max(1e-6, 1.0 - rho_dc)
    return 1.0


def _poisson_pmf_norm(mu, tail_mass=1e-12):
    if mu <= 0:
        return [1.0]
    max_k = min(int(max(20, mu * 12.0 + 60)), 500)
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
    """
    frac = max(0.10, minuto / 90.0)
    return min((abs(delta_ah) + abs(delta_tot) * 0.5) / frac, 6.0)


# ==========================================
# CALIBRAZIONE BAYESIANA
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto):
    """
    Blend bayesiano 35/65 con normalizzazione temporale.
    Scala la linea di apertura (full 90') al tempo rimanente
    prima del blend: entrambi gli input diventano omogenei (gol rimanenti).
    """
    frac_rimasta = max(0.05, (90.0 - minuto) / 90.0)

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
        ah_bayes  = (ah_op  * frac_rimasta) * 0.35 + ah_cur  * 0.65
        tot_bayes = max(0.2, (tot_op * frac_rimasta) * 0.35 + tot_cur * 0.65)
    eps = 1e-6

    def _pmf_ev(mu):
        if mu <= 0: return [1.0]
        max_k = min(int(max(20, mu * 8 + 40)), 300)
        p = math.exp(-mu); pmfs = [p]; cum = p; k = 0
        while cum < 1 - 1e-12 and k < max_k:
            k += 1; p = p * mu / k; pmfs.append(p); cum += p
        return [x / cum for x in pmfs]

    def _ah_ev_half(mh, ma, h):
        ev = 0.0
        for i, pi in enumerate(_pmf_ev(mh)):
            if pi < 1e-18: continue
            for j, pj in enumerate(_pmf_ev(ma)):
                if pj < 1e-18: continue
                s = (i - j) + h
                if s > 0: ev += pi * pj
                elif s < 0: ev -= pi * pj
        return ev

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
        for _ in range(18):
            m  = 0.5 * (dl + dr)
            em = _ev(m)
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
    - Decadimento Weibull sul tempo rimanente
    - Score effect residuale max 12% (il grosso e gia nell'AH live)
    - Red cards con effetto costante (senza moltiplicatore frac)
    """
    if minuto >= 90:
        return 0.001, 0.001

    fattore_tempo = math.pow((90.0 - minuto) / 90.0, 0.85)
    xg_c = xg_casa  * fattore_tempo
    xg_t = xg_trasf * fattore_tempo

    diff = gol_casa - gol_trasf
    if diff != 0:
        sat      = abs(diff) / (2.0 + abs(diff))
        residual = 0.12 * sat
        if diff < 0:
            xg_c *= (1.0 + residual)
            xg_t *= (1.0 - residual)
        else:
            xg_t *= (1.0 + residual)
            xg_c *= (1.0 - residual)

    if rossi_casa > 0:
        xg_c *= math.pow(0.68, rossi_casa)
        xg_t *= math.pow(1.28, rossi_casa)
    if rossi_trasf > 0:
        xg_t *= math.pow(0.68, rossi_trasf)
        xg_c *= math.pow(1.28, rossi_trasf)

    return max(0.001, xg_c), max(0.001, xg_t)


# ==========================================
# CALCOLO PROBABILITA — UN UNICO PASSAGGIO
# ==========================================

def calcola_tutto(mu_c_rem, mu_t_rem, gol_casa, gol_trasf, linea_ou, tot_cur, minuto):
    """
    Costruisce la matrice bivariata completa una volta sola e ne ricava
    in un unico passaggio: 1X2, U/O, BTTS, Correct Score top-5.

    Modello bivariate Poisson:
      X_rem = X_ind + Z    (gol rimanenti casa)
      Y_rem = Y_ind + Z    (gol rimanenti trasf.)
      X_ind ~ Poisson(mu_c_ind)
      Y_ind ~ Poisson(mu_t_ind)
      Z     ~ Poisson(lambda0)   con lambda0 = rho_dinamico * min(mu_c, mu_t)

    Novita v42:
      - rho dinamico (non piu fisso 0.12)
      - correzione Dixon-Coles sulle componenti indipendenti per i punteggi
        bassi (0-0, 1-0, 0-1, 1-1): maggiore precisione a bassa intensita di gol
    """
    mu_c_rem = max(1e-9, float(mu_c_rem))
    mu_t_rem = max(1e-9, float(mu_t_rem))

    rho     = rho_dinamico(tot_cur, minuto)
    min_mu  = min(mu_c_rem, mu_t_rem)
    lambda0 = max(0.0, min(rho * min_mu, 0.95 * min_mu))

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

    # 4. Under / Over
    S       = gol_casa + gol_trasf
    p_under = sum(p for (a, b), p in full.items() if S + a + b < linea_ou)
    p_under = min(max(p_under, 0.0), 1.0)
    p_over  = 1.0 - p_under

    # 5. BTTS
    if gol_casa > 0 and gol_trasf > 0:
        p_btts = 1.0
    elif gol_casa > 0:
        p_btts = max(0.0, 1.0 - math.exp(-mu_t_rem))
    elif gol_trasf > 0:
        p_btts = max(0.0, 1.0 - math.exp(-mu_c_rem))
    else:
        # P(BTTS) = 1 - P(X_rem=0) - P(Y_rem=0) + P(X_rem=0 AND Y_rem=0)
        # Con bivariate Poisson: P(X=0,Y=0) = exp(-(mu_c_ind + mu_t_ind + lambda0))
        p_x0   = math.exp(-mu_c_rem)
        p_y0   = math.exp(-mu_t_rem)
        p_xy0  = math.exp(-(mu_c_ind + mu_t_ind + lambda0))
        p_btts = max(0.0, min(1.0, 1.0 - p_x0 - p_y0 + p_xy0))

    # 6. Correct Score (punteggio finale)
    cs_final = {}
    for (a, b), p in full.items():
        key = (gol_casa + a, gol_trasf + b)
        cs_final[key] = cs_final.get(key, 0.0) + p

    top_cs = sorted(cs_final.items(), key=lambda x: x[1], reverse=True)[:5]

    return p1, px, p2, p_under, p_over, p_btts, top_cs, rho


# ==========================================
# UTILITY QUOTE E STAKING
# ==========================================

def calcola_quota_reale(prob):
    return 1.0 / prob if prob > 0.001 else 999.0


def calcola_stake_kelly(prob_modello, quota_target, bankroll, frazione=0.5):
    """
    Kelly frazionato: stake = bankroll * f* * frazione
    Formula standard: f* = (p*Q - 1) / (Q - 1)
      dove p = prob_modello, Q = quota_target (decimal odds)
    Equivalentemente: f* = (b*p - (1-p)) / b  con b = Q-1 (net odds)
    Cap al 5% del bankroll. Restituisce 0 se edge <= 0.
    """
    edge = prob_modello - (1.0 / quota_target)
    if edge <= 0 or quota_target <= 1.0:
        return 0.0
    # Standard Kelly: (p*Q - 1) / (Q - 1)
    kelly_pct = min((prob_modello * quota_target - 1.0) / (quota_target - 1.0), 0.05)
    return bankroll * kelly_pct * frazione


def calcola_stake_lay(prob_modello, quota_exc, bankroll, frazione=0.5):
    """
    Kelly frazionato per LAY su exchange.

    Quando banci a quota Q:
      - Vinci la stake se l'evento NON accade  (prob = 1 - prob_modello)
      - Perdi la liability = stake * (Q - 1)   se l'evento accade

    Viewing it as "risking L (liability) to gain L/(Q-1)":
      f_liability* = 1 - prob_modello * Q
    Si ha valore nel lay quando prob_modello * Q < 1,
    ovvero quando il mercato sopravvaluta l'evento rispetto al modello.

    Restituisce (stake_visibile, liability) oppure None se non c'è valore.
    La stake visibile è quanto inserisci sull'exchange (il backer paga quella).
    La liability è quanto rischi tu (= stake * (Q-1)).
    Cap al 5% del bankroll come liability.
    """
    if quota_exc <= 1.01:
        return None
    f_liability = 1.0 - prob_modello * quota_exc
    if f_liability <= 0:
        return None
    f_liability = min(f_liability, 0.05)
    liability   = bankroll * f_liability * frazione
    stake       = liability / (quota_exc - 1.0)
    return stake, liability


# ==========================================
# UI STREAMLIT
# ==========================================

st.set_page_config(page_title="Radar Pro Live", page_icon="⚡", layout="centered")
st.title("⚡ Radar Pro Live")
st.caption("v43 · Dixon-Coles · Rho dinamico · Momentum stake · Kelly fix · Settled markets · BTTS No")

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

linea_target_ou = st.selectbox(
    "Linea U/O da analizzare:", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], index=2
)
cassa = st.number_input("Bankroll (€)", value=1000.0, step=100.0)

st.divider()

if st.button("ANALIZZA", use_container_width=True, type="primary"):

    with st.spinner("Calcolo matrice bivariata..."):
        xg1_base, xg2_base = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, minuto_gioco
        )
        xg1_live, xg2_live = time_decay_dinamico(
            xg1_base, xg2_base, minuto_gioco,
            gol_casa, gol_trasf, rossi_casa, rossi_trasf
        )
        mc_1, mc_x, mc_2, mc_u, mc_o, mc_btts, top_cs, rho_used = calcola_tutto(
            xg1_live, xg2_live,
            gol_casa, gol_trasf,
            linea_target_ou, tot_cur, minuto_gioco
        )
        delta_ah  = ah_cur  - ah_op
        delta_tot = tot_cur - tot_op
        momentum  = calcola_momentum_mercato(delta_ah, delta_tot, minuto_gioco)

    # ── QUOTE FAIR (sempre visibili, sempre informative) ─────────────────────
    st.header(f"Quote Fair  —  {minuto_gioco}' | {gol_casa}–{gol_trasf}")
    st.caption("Riferimento del modello. Confrontale con quello che vedi sull'exchange.")

    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa",     f"@{calcola_quota_reale(mc_1):.2f}",  f"{mc_1:.1%}")
    cx.metric("X — Pareggio", f"@{calcola_quota_reale(mc_x):.2f}", f"{mc_x:.1%}")
    c2.metric("2 — Trasf.",   f"@{calcola_quota_reale(mc_2):.2f}", f"{mc_2:.1%}")

    cu, co, cb = st.columns(3)
    cu.metric(f"Under {linea_target_ou}", f"@{calcola_quota_reale(mc_u):.2f}", f"{mc_u:.1%}")
    co.metric(f"Over  {linea_target_ou}", f"@{calcola_quota_reale(mc_o):.2f}", f"{mc_o:.1%}")
    cb.metric("BTTS — Si",               f"@{calcola_quota_reale(mc_btts):.2f}", f"{mc_btts:.1%}")

    with st.expander("Correct Score — Top 5"):
        cs_cols = st.columns(5)
        for idx, ((fc, ft), prob) in enumerate(top_cs):
            cs_cols[idx].metric(
                label=f"{fc}–{ft}",
                value=f"@{calcola_quota_reale(prob):.1f}",
                delta=f"{prob:.1%}"
            )

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

    # ── QUOTE EXCHANGE (input utente) ─────────────────────────────────────────
    st.divider()
    st.header("Quote Exchange Attuali")
    st.caption(
        "Inserisci le quote che vedi ora sull'exchange. "
        "Il motore calcola l'edge rispetto alla sua stima e segnala solo dove c'è valore reale. "
        "Lascia a 0 i mercati che non ti interessano."
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
    q_exc_btts_no = eqbn_col.number_input("Quota BTTS No",          min_value=0.0, value=0.0, step=0.05, format="%.2f")

    # ── SEGNALI ───────────────────────────────────────────────────────────────
    st.divider()
    st.header("Segnali Exchange")

    if minuto_gioco >= 85:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()

    frac_giocata = minuto_gioco / 90.0
    # Back: soglie probabilità minime. Crescono col tempo (più tardi = più certezza).
    # Non servono per il lay (l'edge parla da solo).
    soglia_min_1x2  = 0.50 + 0.10 * frac_giocata   # 50% → 60%
    soglia_min_ou   = 0.55 + 0.10 * frac_giocata   # 55% → 65%
    soglia_min_btts = 0.50 + 0.10 * frac_giocata   # 50% → 60%

    gol_attuali  = gol_casa + gol_trasf
    gol_mancanti = linea_target_ou - gol_attuali
    segnali      = False

    # Riduzione stake proporzionale al momentum (eventi non registrati)
    momentum_stake_factor = max(0.4, 1.0 - 0.10 * max(0.0, momentum - 2.5))

    any_exc_quote = any(
        q > 1.0 for q in [q_exc_1, q_exc_x, q_exc_2, q_exc_u, q_exc_o, q_exc_btts, q_exc_btts_no]
    )

    MIN_EDGE_BACK = 0.030   # 3% edge minimo per back
    MIN_EDGE_LAY  = 0.040   # 4% edge minimo per lay (rischio asimmetrico)
    MIN_FAIR_Q    = 1.15    # eventi quasi certi (>87%) → skip, già nel prezzo

    def _valuta(etichetta, prob_mod, q_exc, soglia_back):
        """
        Logica unificata back/lay per ogni mercato.

        Con quota exchange:
          PUNTA  se  prob_mod - 1/q_exc >= MIN_EDGE_BACK  AND  prob_mod >= soglia_back
          BANCA  se  1/q_exc - prob_mod >= MIN_EDGE_LAY   AND  q_exc >= 1.30
          (lay a quota < 1.30 ha rischio/reward strutturalmente sfavorevole)

        Senza quota exchange (q_exc <= 1.0):
          Indicazione qualitativa solo se il modello è fortemente convinto:
          >62% → potenziale back · <30% → mercato probabilmente sopravvaluta (lay)
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
            if prob_mod >= 0.62:
                st.info(
                    f"📈 **{etichetta}**: modello {prob_mod:.1%} (fair @{q_fair:.2f}) "
                    f"— potenziale back · inserisci quota per edge preciso"
                )
                segnali = True
                return True
            if prob_mod <= 0.30:
                st.info(
                    f"📉 **{etichetta}**: modello {prob_mod:.1%} (fair @{q_fair:.2f}) "
                    f"— mercato probabilmente sopravvaluta · considera lay · inserisci quota per conferma"
                )
                segnali = True
                return True
            return False

        # ── CON QUOTA EXCHANGE ────────────────────────────────────────────────
        prob_imp  = 1.0 / q_exc
        edge_back = prob_mod - prob_imp
        edge_lay  = prob_imp - prob_mod
        segnalato = False

        # BACK
        if edge_back >= MIN_EDGE_BACK and prob_mod >= soglia_back:
            stake_raw = calcola_stake_kelly(prob_mod, q_exc, cassa)
            stake     = stake_raw * momentum_stake_factor
            if stake > 0:
                st.success(
                    f"**PUNTA {etichetta}** — "
                    f"Modello {prob_mod:.1%} · Mercato @{q_exc:.2f} ({prob_imp:.1%}) · "
                    f"Edge **+{edge_back*100:.1f}%**"
                )
                if momentum_stake_factor < 1.0:
                    st.write(
                        f"Stake ½-Kelly: **€{stake:.2f}** "
                        f"_(ridotta da €{stake_raw:.2f} · momentum {momentum:.1f}/6.0 · ×{momentum_stake_factor:.2f})_"
                    )
                else:
                    st.write(f"Stake ½-Kelly: **€{stake:.2f}**")
                segnali   = True
                segnalato = True

        # LAY — quota minima 1.30 (sotto quel valore il lay non ha senso economico)
        if not segnalato and edge_lay >= MIN_EDGE_LAY and q_exc >= 1.30:
            result = calcola_stake_lay(prob_mod, q_exc, cassa)
            if result is not None:
                stake_lay, liab_lay = result
                stake_lay *= momentum_stake_factor
                liab_lay  *= momentum_stake_factor
                st.warning(
                    f"**BANCA {etichetta}** — "
                    f"Modello {prob_mod:.1%} · Mercato @{q_exc:.2f} ({prob_imp:.1%}) · "
                    f"Edge lay **+{edge_lay*100:.1f}%**"
                )
                if momentum_stake_factor < 1.0:
                    st.write(
                        f"Stake lay: **€{stake_lay:.2f}** · Liability: **€{liab_lay:.2f}** "
                        f"_(ridotta · momentum ×{momentum_stake_factor:.2f})_"
                    )
                else:
                    st.write(f"Stake lay: **€{stake_lay:.2f}** · Liability: **€{liab_lay:.2f}**")
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
            st.info(f"Over {linea_target_ou}: al {minuto_gioco}' mancano {gol_mancanti:.0f} gol ({mc_o:.1%}) — troppo tardi per entrare.")
        else:
            _valuta(f"OVER {linea_target_ou}",  mc_o, q_exc_o, soglia_min_ou)
        _valuta(f"UNDER {linea_target_ou}", mc_u, q_exc_u, soglia_min_ou)

    # BTTS
    _valuta("BTTS SÌ", mc_btts,       q_exc_btts,    soglia_min_btts)
    _valuta("BTTS NO", 1.0 - mc_btts, q_exc_btts_no, soglia_min_btts)

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
            st.write(f"Soglia 1X2   = {soglia_min_1x2:.3f}")
            st.write(f"Soglia U/O   = {soglia_min_ou:.3f}")
            st.write(f"Soglia BTTS  = {soglia_min_btts:.3f}")
            st.divider()
            st.write(f"Delta AH     = {delta_ah:+.2f}")
            st.write(f"Delta Tot    = {delta_tot:+.2f}")
            st.write(f"Momentum     = {momentum:.2f}/6.0")
            st.write(f"Stake factor = ×{momentum_stake_factor:.2f}")
