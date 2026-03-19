import streamlit as st
import math

# ==========================================
# ⚙️ MOTORE MATEMATICO: SCORE EFFECTS & RED CARDS
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, SCALE=0.25):
    """
    Mapping "più matematico" (punto 2):
    - fusione pesata di AH e Tot (apertura vs corrente)
    - AH negativo => casa favorita => Δ > 0
    - Δ scalato con sqrt(Tot) per avere un comportamento più stabile
    - conversione finale in lambda_home/lambda_away (medie Poisson dei gol rimanenti)
    """
    peso_prior = 0.35
    peso_evidenza = 0.65

    ah_bayes = (ah_op * peso_prior) + (ah_cur * peso_evidenza)
    tot_bayes = (tot_op * peso_prior) + (tot_cur * peso_evidenza)

    tot_bayes = max(0.2, float(tot_bayes))  # stabilità numerica

    # Convenzione: AH negativo => casa favorita => lambda_home > lambda_away
    delta = (-ah_bayes) * SCALE * math.sqrt(tot_bayes)

    # Cap per evitare lambda negative/assurde
    max_delta = 0.95 * tot_bayes
    delta = max(-max_delta, min(max_delta, delta))

    lambda_casa = 0.5 * (tot_bayes + delta)
    lambda_trasf = 0.5 * (tot_bayes - delta)

    return max(1e-6, lambda_casa), max(1e-6, lambda_trasf)


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

    # 2. Score Effects con cap (stabilità)
    differenza_reti = gol_casa - gol_trasf
    abs_diff = abs(differenza_reti)

    alpha = 0.15   # aggressività attacco per gol di scarto
    beta = 0.10    # riduzione avversario per gol di scarto

    max_attack_mult = 1.8
    min_def_mult = 0.6

    if differenza_reti < 0:  # Casa perde => casa attacca di più
        att = min(max_attack_mult, 1.0 + alpha * abs_diff)
        de = max(min_def_mult, 1.0 - beta * abs_diff)
        xg_c_live *= att
        xg_t_live *= de
    elif differenza_reti > 0:  # Trasferta perde => trasferta attacca di più
        att = min(max_attack_mult, 1.0 + alpha * abs_diff)
        de = max(min_def_mult, 1.0 - beta * abs_diff)
        xg_t_live *= att
        xg_c_live *= de

    # 3. Red Card Multipliers (Impatto devastante)
    if rossi_casa > 0:
        xg_c_live *= math.pow(0.65, rossi_casa)  # casa perde potenziale offensivo
        xg_t_live *= math.pow(1.35, rossi_casa)  # avversario guadagna potenziale
    if rossi_trasf > 0:
        xg_t_live *= math.pow(0.65, rossi_trasf)
        xg_c_live *= math.pow(1.35, rossi_trasf)

    return max(0.001, xg_c_live), max(0.001, xg_t_live)


# ==========================================
# 🎲 PROBABILITA' ESATTE: Poisson (niente Monte Carlo)
# ==========================================

def _poisson_pmf_normalized(mu, tail_mass=1e-12, max_k=160):
    """
    Calcola P(K=k) per k=0..kmax con taglio coda controllato e normalizzazione.
    Serve per evitare instabilità numeriche e garantire massa ~1.
    """
    if mu <= 0:
        return [1.0]

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
    mu_*_rem: gol rimanenti attesi (medie Poisson).
    gol_*    : gol già fatti.
    linea_ou : es. 2.5, 3.5, ...
    Definizione coerente col tuo originale:
    - Over: (S + K) > linea_ou
    - Under: (S + K) <= linea_ou
    """
    mu_casa_rem = max(1e-9, float(mu_casa_rem))
    mu_trasf_rem = max(1e-9, float(mu_trasf_rem))

    pmf_c = _poisson_pmf_normalized(mu_casa_rem)
    pmf_t = _poisson_pmf_normalized(mu_trasf_rem)

    p1 = px = p2 = 0.0

    # Convoluzione sui gol aggiuntivi
    for i, pi in enumerate(pmf_c):  # aggiuntivi casa
        if pi < 1e-16:
            continue
        for j, pj in enumerate(pmf_t):  # aggiuntivi trasferta
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

    # Under/Over sul totale finale
    S = gol_casa + gol_trasf
    mu_tot_rem = mu_casa_rem + mu_trasf_rem
    pmf_tot = _poisson_pmf_normalized(mu_tot_rem)

    # Under: S + K <= linea_ou  =>  K <= floor(linea_ou - S)
    maxK_under = int(math.floor(float(linea_ou) - S))
    if maxK_under < 0:
        p_under = 0.0
    else:
        maxK_under = min(maxK_under, len(pmf_tot) - 1)
        p_under = sum(pmf_tot[:maxK_under + 1])

    p_over = max(0.0, 1.0 - p_under)
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
