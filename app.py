import streamlit as st
import math

# ==========================================
# ⚙️ MOTORE MATEMATICO: SCORE EFFECTS & RED CARDS
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur):
    """
    Calibrazione di Δ da AH (senza SCALE):
    1) fondiamo AH e Tot (apertura vs corrente) via media pesata
    2) imposto lambda_casa + lambda_trasf = tot_bayes
    3) scelgo Δ (quindi la ripartizione tra lambda_casa e lambda_trasf) in modo che
       il valore atteso della scommessa AH sia ~0 (linea "fair" a margine 0).

    Nota: qui usiamo una modellazione indipendente per la differenza reti
    (la componente comune bivariata non cambia la differenza).
    """
    peso_prior = 0.35
    peso_evidenza = 0.65

    ah_bayes = (ah_op * peso_prior) + (ah_cur * peso_evidenza)
    tot_bayes = (tot_op * peso_prior) + (tot_cur * peso_evidenza)

    tot_bayes = max(0.2, float(tot_bayes))  # stabilità numerica

    eps = 1e-6

    def _poisson_pmf_for_ev(mu):
        # PMF limitata e "pulita" per calcolo EV: accuracy alta ma runtime contenuto.
        # Qui mu è tipicamente nell'intorno 0.5..6.
        tail_mass = 1e-12
        if mu <= 0:
            return [1.0]
        max_k = int(max(20.0, mu * 8.0 + 40.0))
        max_k = min(max_k, 300)
        p = math.exp(-mu)
        pmfs = [p]
        cumsum = p
        k = 0
        while cumsum < (1.0 - tail_mass) and k < max_k:
            k += 1
            p = p * mu / k
            pmfs.append(p)
            cumsum += p
        pmfs = [x / cumsum for x in pmfs]
        return pmfs

    def _asian_ev_half(mu_home, mu_away, h_half):
        """
        EV "fair" per una linea AH a passi di 0.5:
        payoff: win => +1, lose => -1, push => 0.
        """
        pmf_h = _poisson_pmf_for_ev(mu_home)
        pmf_a = _poisson_pmf_for_ev(mu_away)
        ev = 0.0
        # doppio ciclo con skip per ridurre lavoro
        for i, pi in enumerate(pmf_h):
            if pi < 1e-18:
                continue
            for j, pj in enumerate(pmf_a):
                if pj < 1e-18:
                    continue
                d = i - j
                s = d + h_half  # differenza effettiva "home + handicap"
                prob = pi * pj
                if s > 0:
                    ev += prob
                elif s < 0:
                    ev -= prob
                # se s == 0 => push => contributo 0
        return ev

    def _asian_ev(mu_home, mu_away, ah):
        """
        EV per AH possibilmente a 0.25 (come nel tuo input). Se AH non è multiplo di 0.5,
        applichiamo la convenzione di split tra due mezze-linee vicine.
        """
        ah2 = float(ah) * 2.0
        if abs(ah2 - round(ah2)) < 1e-9:
            # multiplo di 0.5 => nessuno split
            return _asian_ev_half(mu_home, mu_away, float(ah))

        # split su due mezze-linee (distanza 0.5): peso 0.5 ciascuna
        h1 = math.floor(ah2) / 2.0
        h2 = h1 + 0.5
        return 0.5 * _asian_ev_half(mu_home, mu_away, h1) + 0.5 * _asian_ev_half(mu_home, mu_away, h2)

    def _ev_for_delta(delta):
        lam_h = 0.5 * (tot_bayes + delta)
        lam_a = 0.5 * (tot_bayes - delta)
        lam_h = max(eps, lam_h)
        lam_a = max(eps, lam_a)
        return _asian_ev(lam_h, lam_a, ah_bayes)

    # d = lambda_home - lambda_away in [-tot, +tot]
    lo = -tot_bayes + eps
    hi = tot_bayes - eps
    ev_lo = _ev_for_delta(lo)
    ev_hi = _ev_for_delta(hi)

    # Se non c'è attraversamento, scegliamo la soluzione che rende |EV| minimo.
    if ev_lo == 0.0:
        delta_star = lo
    elif ev_hi == 0.0:
        delta_star = hi
    elif ev_lo * ev_hi > 0:
        delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
    else:
        # Determiniamo la monotonicità empirica (in pratica dovrebbe essere crescente).
        increasing = ev_hi > ev_lo
        delta_l = lo
        delta_r = hi
        ev_l = ev_lo
        ev_r = ev_hi

        for _ in range(18):
            mid = 0.5 * (delta_l + delta_r)
            ev_mid = _ev_for_delta(mid)
            # target: ev = 0. Se ev_mid è positivo, vuol dire "home troppo forte".
            if increasing:
                if ev_mid > 0:
                    delta_r = mid
                    ev_r = ev_mid
                else:
                    delta_l = mid
                    ev_l = ev_mid
            else:
                if ev_mid > 0:
                    delta_l = mid
                    ev_l = ev_mid
                else:
                    delta_r = mid
                    ev_r = ev_mid
        delta_star = 0.5 * (delta_l + delta_r)

    lambda_casa = 0.5 * (tot_bayes + delta_star)
    lambda_trasf = 0.5 * (tot_bayes - delta_star)
    return max(eps, lambda_casa), max(eps, lambda_trasf)


def time_decay_dinamico(xg_casa, xg_trasf, minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    """
    - Weibull/potenza sul tempo rimanente
    - Score effects: chi è sotto aumenta l'attacco, chi è sopra riduce l'attacco avversario
      (con cap/min per evitare moltiplicatori esplosivi)
    - Red cards: multipli come nel tuo originale
    """
    if minuto >= 90:
        return 0.001, 0.001

    # 1. Decadimento Weibull Base
    tempo_rimanente = 90 - minuto
    fattore_tempo = math.pow((tempo_rimanente / 90.0), 0.85)
    xg_c_live = xg_casa * fattore_tempo
    xg_t_live = xg_trasf * fattore_tempo

    # 2. Score Effects (bounded + liscio)
    # Mappiamo il "who loses attacks more" con funzioni saturanti:
    # - att (saturazione) tende a `max_attack_mult`
    # - de  (saturazione) tende a `min_def_mult`
    differenza_reti = gol_casa - gol_trasf
    abs_diff = abs(differenza_reti)

    max_attack_mult = 1.8
    min_def_mult = 0.6
    A = max_attack_mult - 1.0  # ampiezza boost attacco
    B = 1.0 - min_def_mult    # ampiezza riduzione difesa/attacco avversario

    sat = (abs_diff / (1.0 + abs_diff)) if abs_diff > 0 else 0.0
    att = 1.0 + A * sat
    de = 1.0 - B * sat

    # Rate-based: quando resta poco tempo, gli effetti su rate devono pesare meno
    frac = max(0.0, min(1.0, float(90 - minuto) / 90.0))  # frazione di tempo rimanente

    if differenza_reti < 0:  # Casa perde => casa attacca di più
        xg_c_live *= math.pow(att, frac)
        xg_t_live *= math.pow(de, frac)
    elif differenza_reti > 0:  # Trasferta perde => trasferta attacca di più
        xg_t_live *= math.pow(att, frac)
        xg_c_live *= math.pow(de, frac)

    # 3. Red Card Multipliers time-weighted (più affidabile)
    if rossi_casa > 0:
        xg_c_live *= math.pow(0.65, rossi_casa * frac)  # casa perde potenziale offensivo
        xg_t_live *= math.pow(1.35, rossi_casa * frac)  # avversario guadagna potenziale
    if rossi_trasf > 0:
        xg_t_live *= math.pow(0.65, rossi_trasf * frac)
        xg_c_live *= math.pow(1.35, rossi_trasf * frac)

    return max(0.001, xg_c_live), max(0.001, xg_t_live)


# ==========================================
# 🎲 PROBABILITA' ESATTE: Poisson (niente Monte Carlo)
# ==========================================

def _poisson_pmf_normalized(mu, tail_mass=1e-12, max_k=None):
    """
    Calcola P(K=k) per k=0..kmax con taglio coda controllato e normalizzazione.
    Serve per evitare instabilità numeriche e garantire massa ~1.
    """
    if mu <= 0:
        return [1.0]

    # Se max_k non è fissato, scegliamo un upper bound ragionevole in funzione della media.
    # (Migliora precisione quando le λ sono alte, senza esplodere in runtime.)
    if max_k is None:
        max_k = int(max(20, mu * 12.0 + 60))
        max_k = min(max_k, 500)

    p0 = math.exp(-mu)
    pmfs = [p0]
    cumsum = p0

    k = 0
    p = p0
    while cumsum < (1.0 - tail_mass) and k < max_k:
        k += 1
        p = p * mu / k
        pmfs.append(p)
        cumsum += p

    # normalizzazione (entro il taglio di coda)
    pmfs = [x / cumsum for x in pmfs]
    return pmfs


def probabilita_poisson_esatta(mu_casa_rem, mu_trasf_rem, gol_casa, gol_trasf, linea_ou):
    """
    mu_*_rem: gol rimanenti attesi (medie marginali).
    gol_*    : gol già fatti.
    linea_ou : es. 2.5, 3.5, ...

    Upgrade 3 (correlazione): uso un bivariate Poisson con componente comune Z.
    - Differenza reti (1X2) dipende da X-Y, quindi si ottiene usando le componenti "indipendenti"
      mu_home_ind = mu_home_rem - lambda0, mu_away_ind = mu_away_rem - lambda0
    - Totale (Under/Over) dipende anche da Z (perché compare come 2Z).
    """
    mu_casa_rem = max(1e-9, float(mu_casa_rem))
    mu_trasf_rem = max(1e-9, float(mu_trasf_rem))

    # Forza della correlazione tra goals: componente comune Z
    # Valore conservativo per evitare instabilità.
    rho_corr = 0.12
    min_mu = min(mu_casa_rem, mu_trasf_rem)
    lambda0 = rho_corr * min_mu
    lambda0 = max(0.0, min(lambda0, 0.95 * min_mu))  # garantisce mu_ind >= 0

    mu_c_ind = max(1e-9, mu_casa_rem - lambda0)
    mu_t_ind = max(1e-9, mu_trasf_rem - lambda0)

    # --- 1X2 via componenti indipendenti X-Y (Z cancella la differenza) ---
    pmf_c = _poisson_pmf_normalized(mu_c_ind)
    pmf_t = _poisson_pmf_normalized(mu_t_ind)

    p1 = px = p2 = 0.0
    for i, pi in enumerate(pmf_c):  # aggiuntivi casa (X)
        if pi < 1e-16:
            continue
        for j, pj in enumerate(pmf_t):  # aggiuntivi trasferta (Y)
            if pj < 1e-16:
                continue
            tot_c = gol_casa + i
            tot_t = gol_trasf + j
            if tot_c > tot_t:
                p1 += pi * pj
            elif tot_c == tot_t:
                px += pi * pj
            else:
                p2 += pi * pj

    # Normalizza 1X2 (errore da truncation)
    psum = p1 + px + p2
    if psum > 0:
        p1 /= psum
        px /= psum
        p2 /= psum

    # --- Under/Over: totale = X + Y + 2Z ---
    S = gol_casa + gol_trasf

    pmf_z = _poisson_pmf_normalized(lambda0)  # distribuzione del common component
    mu_w = mu_c_ind + mu_t_ind
    pmf_w = _poisson_pmf_normalized(mu_w)  # W = X + Y, Poisson(mu_w)

    # prefix sums per P(W <= k)
    prefix = [0.0]
    c = 0.0
    for p in pmf_w:
        c += p
        prefix.append(c)

    p_under = 0.0
    for z, pz in enumerate(pmf_z):
        if pz < 1e-16:
            continue
        # vogliamo: S + (W + 2Z) <= linea_ou  => W <= linea_ou - S - 2Z
        maxK_under = int(math.floor(float(linea_ou) - S - 2 * z))
        if maxK_under < 0:
            continue
        if maxK_under >= len(pmf_w) - 1:
            p_under += pz  # probabilità ~1
        else:
            p_under += pz * prefix[maxK_under + 1]

    p_under = min(max(p_under, 0.0), 1.0)
    p_over = min(max(1.0 - p_under, 0.0), 1.0)

    return p1, px, p2, p_under, p_over


def calcola_quota_reale(prob):
    return 1 / prob if prob > 0.001 else 999.0


# ==========================================
# 🎨 STREAMLIT DASHBOARD (UI OTTIMIZZATA MOBILE)
# ==========================================
st.set_page_config(page_title="Radar Pro Live", page_icon="⚡", layout="centered")

st.title("⚡ Radar Pro Live")
st.markdown("*Motore: Score Effects & Dinamica Cartellini*")

# --- 1. STATO PARTITA (LIVE) ---
st.header("⏱️ 1. Eventi Live")
minuto_gioco = st.slider("Minuto Attuale", 0, 90, 0, 1)

col_g1, col_g2 = st.columns(2)
with col_g1:
    gol_casa = st.number_input("⚽ Gol CASA", value=0, min_value=0)
with col_g2:
    gol_trasf = st.number_input("⚽ Gol TRASF.", value=0, min_value=0)

col_r1, col_r2 = st.columns(2)
with col_r1:
    rossi_casa = st.number_input("🟥 Rossi CASA", value=0, min_value=0, max_value=4)
with col_r2:
    rossi_trasf = st.number_input("🟥 Rossi TRASF.", value=0, min_value=0, max_value=4)

st.divider()

# --- 2. MERCATO ASIATICO E TARGET ---
st.header("📊 2. Spread Asiatici & Obiettivo")
col_a1, col_a2 = st.columns(2)
with col_a1:
    ah_op = st.number_input("AH Apertura", value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50, step=0.25)
with col_a2:
    ah_cur = st.number_input("AH Corrente", value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75, step=0.25)

st.markdown("**Mercato Under/Over da analizzare:**")
linea_target_ou = st.selectbox("Seleziona Linea U/O:", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], index=2)
cassa = st.number_input("💰 Tua Cassa (€)", value=1000.0, step=100.0)

st.divider()

if st.button("🚀 GENERA TARGET QUOTE", use_container_width=True, type="primary"):
    with st.spinner("Calcolo Matrice Stocastica Dinamica..."):
        # (1) mapping AH/Tot -> lambda rimanenti (gol attesi)
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur)

        # (2) time-decay + score effects + red cards
        xg1_live, xg2_live = time_decay_dinamico(
            xg1_base, xg2_base, minuto_gioco, gol_casa, gol_trasf, rossi_casa, rossi_trasf
        )

        # (3) probabilita esatte Poisson (Win/Draw/Win + Under/Over)
        mc_1, mc_x, mc_2, mc_u, mc_o = probabilita_poisson_esatta(
            xg1_live, xg2_live, gol_casa, gol_trasf, linea_target_ou
        )

        delta_ah = ah_cur - ah_op
        delta_tot = tot_cur - tot_op

    st.header(f"⚖️ Quote Reali (Min: {minuto_gioco}' | Ris: {gol_casa}-{gol_trasf})")

    c_1, c_x, c_2 = st.columns(3)
    c_1.metric("1 (Casa)", f"@{calcola_quota_reale(mc_1):.2f}")
    c_x.metric("X (Pareggio)", f"@{calcola_quota_reale(mc_x):.2f}")
    c_2.metric("2 (Trasf.)", f"@{calcola_quota_reale(mc_2):.2f}")

    cu, co = st.columns(2)
    cu.metric(f"Under {linea_target_ou}", f"@{calcola_quota_reale(mc_u):.2f}")
    co.metric(f"Over {linea_target_ou}", f"@{calcola_quota_reale(mc_o):.2f}")

    st.divider()

    st.header("🎯 AZIONI EXCHANGE RACCOMANDATE")
    segnali = False
    stake = cassa * 0.025

    if minuto_gioco >= 85:
        st.error("🛑 **FINE GIOCO:** Variabilità estrema, spread enormi sull'Exchange. Chiudere le posizioni o non entrare.")
        st.stop()

    if delta_ah <= -0.25:
        st.success(
            f"🟢 **PUNTA 1:** I soldi sono su CASA. Punta se trovi la quota **> @{calcola_quota_reale(mc_1) * 1.05:.2f}**"
        )
        st.write(f"Stake: € {stake:.2f}")
        segnali = True
    elif delta_ah >= 0.25:
        st.success(
            f"🟢 **PUNTA 2:** I soldi sono su TRASF. Punta se trovi la quota **> @{calcola_quota_reale(mc_2) * 1.05:.2f}**"
        )
        st.write(f"Stake: € {stake:.2f}")
        segnali = True

    # Logica Totale (mantengo la tua struttura decisionale)
    if delta_tot >= 0.25 and (gol_casa + gol_trasf) < linea_target_ou:
        st.warning(
            f"🔥 **PUNTA OVER {linea_target_ou}:** Punta se trovi la quota **> @{calcola_quota_reale(mc_o) * 1.05:.2f}**"
        )
        segnali = True
    elif delta_tot <= -0.25 and (gol_casa + gol_trasf) < linea_target_ou:
        st.info(
            f"🧊 **PUNTA UNDER {linea_target_ou}:** Punta se trovi la quota **> @{calcola_quota_reale(mc_u) * 1.05:.2f}**"
        )
        segnali = True

    if not segnali:
        st.error("⚖️ **NO BET:** Volumi stabili dall'apertura. Non c'è un vantaggio algoritmico sufficiente.")
