import streamlit as st
import math
import random

# ==========================================
# ⚙️ MOTORE MATEMATICO: SCORE EFFECTS & RED CARDS
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur):
    peso_prior = 0.35
    peso_evidenza = 0.65
    ah_bayes = (ah_op * peso_prior) + (ah_cur * peso_evidenza)
    tot_bayes = (tot_op * peso_prior) + (tot_cur * peso_evidenza)
    
    supremazia = -ah_bayes
    gamma = 1.0 + (abs(supremazia) * 0.05) 
    
    if supremazia > 0:
        xg_casa = (tot_bayes / 2.0) + (supremazia / 2.0) * gamma
        xg_trasf = tot_bayes - xg_casa
    else:
        xg_trasf = (tot_bayes / 2.0) + (abs(supremazia) / 2.0) * gamma
        xg_casa = tot_bayes - xg_trasf

    return max(0.01, xg_casa), max(0.01, xg_trasf)

def time_decay_dinamico(xg_casa, xg_trasf, minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    if minuto >= 90: return 0.001, 0.001
    
    # 1. Decadimento Weibull Base
    tempo_rimanente = 90 - minuto
    fattore_tempo = math.pow((tempo_rimanente / 90.0), 0.85)
    xg_c_live = xg_casa * fattore_tempo
    xg_t_live = xg_trasf * fattore_tempo

    # 2. Score Effects (Chi perde attacca di più, chi vince difende)
    differenza_reti = gol_casa - gol_trasf
    if differenza_reti < 0: # Casa perde
        xg_c_live *= (1.0 + 0.15 * abs(differenza_reti)) # +15% xG per ogni gol di scarto
        xg_t_live *= max(0.5, 1.0 - 0.10 * abs(differenza_reti)) # Trasferta si difende
    elif differenza_reti > 0: # Trasferta perde
        xg_t_live *= (1.0 + 0.15 * abs(differenza_reti))
        xg_c_live *= max(0.5, 1.0 - 0.10 * abs(differenza_reti))

    # 3. Red Card Multipliers (Impatto devastante)
    if rossi_casa > 0:
        xg_c_live *= math.pow(0.65, rossi_casa) # Perde 35% potenziale offensivo
        xg_t_live *= math.pow(1.35, rossi_casa) # Avversario guadagna 35%
    if rossi_trasf > 0:
        xg_t_live *= math.pow(0.65, rossi_trasf)
        xg_c_live *= math.pow(1.35, rossi_trasf)

    return max(0.01, xg_c_live), max(0.01, xg_t_live)

def monte_carlo_live_dinamico(mu1, mu2, g_casa, g_trasf, linea_ou, iterazioni=10000):
    vit_1 = pareggi = vit_2 = over_target = under_target = 0
    for _ in range(iterazioni):
        L1, L2 = math.exp(-mu1), math.exp(-mu2)
        k1, p1 = 0, 1.0
        while p1 > L1: k1 += 1; p1 *= random.random()
        
        k2, p2 = 0, 1.0
        while p2 > L2: k2 += 1; p2 *= random.random()
        
        tot_casa = g_casa + (k1 - 1)
        tot_trasf = g_trasf + (k2 - 1)
        
        if tot_casa > tot_trasf: vit_1 += 1
        elif tot_casa == tot_trasf: pareggi += 1
        else: vit_2 += 1
            
        if (tot_casa + tot_trasf) > linea_ou: over_target += 1
        else: under_target += 1
        
    return vit_1/iterazioni, pareggi/iterazioni, vit_2/iterazioni, under_target/iterazioni, over_target/iterazioni

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
with col_g1: gol_casa = st.number_input("⚽ Gol CASA", value=0, min_value=0)
with col_g2: gol_trasf = st.number_input("⚽ Gol TRASF.", value=0, min_value=0)

col_r1, col_r2 = st.columns(2)
with col_r1: rossi_casa = st.number_input("🟥 Rossi CASA", value=0, min_value=0, max_value=4)
with col_r2: rossi_trasf = st.number_input("🟥 Rossi TRASF.", value=0, min_value=0, max_value=4)

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
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur)
        xg1_live, xg2_live = time_decay_dinamico(xg1_base, xg2_base, minuto_gioco, gol_casa, gol_trasf, rossi_casa, rossi_trasf)
        
        mc_1, mc_x, mc_2, mc_u, mc_o = monte_carlo_live_dinamico(xg1_live, xg2_live, gol_casa, gol_trasf, linea_target_ou, 10000)
        
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
        st.success(f"🟢 **PUNTA 1:** I soldi sono su CASA. Punta se trovi la quota **> @{calcola_quota_reale(mc_1) * 1.05:.2f}**")
        st.write(f"Stake: € {stake:.2f}")
        segnali = True
    elif delta_ah >= 0.25:
        st.success(f"🟢 **PUNTA 2:** I soldi sono su TRASFERTA. Punta se trovi la quota **> @{calcola_quota_reale(mc_2) * 1.05:.2f}**")
        st.write(f"Stake: € {stake:.2f}")
        segnali = True

    if delta_tot >= 0.25 and (gol_casa + gol_trasf) < linea_target_ou:
        st.warning(f"🔥 **PUNTA OVER {linea_target_ou}:** Punta se trovi la quota **> @{calcola_quota_reale(mc_o) * 1.05:.2f}**")
        segnali = True
    elif delta_tot <= -0.25 and (gol_casa + gol_trasf) < linea_target_ou:
        st.info(f"🧊 **PUNTA UNDER {linea_target_ou}:** Punta se trovi la quota **> @{calcola_quota_reale(mc_u) * 1.05:.2f}**")
        segnali = True

    if not segnali:
        st.error("⚖️ **NO BET:** Volumi stabili dall'apertura. Non c'è un vantaggio algoritmico sufficiente.")
