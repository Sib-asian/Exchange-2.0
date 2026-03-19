import streamlit as st
import math
import random

# ==========================================
# ⚙️ MOTORE MATEMATICO ISTITUZIONALE NASCOSTO
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

def time_decay_weibull(xg_casa, xg_trasf, minuto):
    if minuto == 0: return xg_casa, xg_trasf
    if minuto >= 90: return 0.001, 0.001
    tempo_rimanente = 90 - minuto
    fattore_decadimento = math.pow((tempo_rimanente / 90.0), 0.85)
    return xg_casa * fattore_decadimento, xg_trasf * fattore_decadimento

def monte_carlo_simulator_full(mu1, mu2, iterazioni=10000):
    vit_1 = pareggi = vit_2 = over25 = under25 = 0
    for _ in range(iterazioni):
        L1, L2 = math.exp(-mu1), math.exp(-mu2)
        k1, p1_sim = 0, 1.0
        while p1_sim > L1:
            k1 += 1; p1_sim *= random.random()
        gol1 = k1 - 1
        
        k2, p2_sim = 0, 1.0
        while p2_sim > L2:
            k2 += 1; p2_sim *= random.random()
        gol2 = k2 - 1
        
        if gol1 > gol2: vit_1 += 1
        elif gol1 == gol2: pareggi += 1
        else: vit_2 += 1
            
        if (gol1 + gol2) > 2.5: over25 += 1
        else: under25 += 1
        
    return vit_1/iterazioni, pareggi/iterazioni, vit_2/iterazioni, under25/iterazioni, over25/iterazioni

def calcola_quota_reale(probabilita):
    return 1 / probabilita if probabilita > 0.001 else 999.0

# ==========================================
# 🎨 STREAMLIT DASHBOARD (SEMPLIFICATA E OPERATIVA)
# ==========================================
st.set_page_config(page_title="Radar Exchange Pro", page_icon="🎯", layout="centered")

st.title("🎯 Radar Exchange Pro")
st.markdown("*Modalità Operativa Live - Segnali Diretti*")

# --- 1. DATI INGRESSO ---
col_min, col_cassa = st.columns(2)
with col_min:
    minuto_gioco = st.slider("⏱️ Minuto Attuale (0 = Pre-match)", 0, 90, 0, 1)
with col_cassa:
    cassa = st.number_input("💰 Cassa Totale (€)", value=1000.0, step=100.0)

st.header("📊 Dati Mercato Asiatico")
col1, col2 = st.columns(2)
with col1:
    ah_op = st.number_input("AH Apertura", value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50, step=0.25)
with col2:
    ah_cur = st.number_input("AH Corrente", value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75, step=0.25)

st.divider()

if st.button("🚀 GENERA SEGNALI OPERATIVI", use_container_width=True, type="primary"):
    
    with st.spinner("Elaborazione dati Istituzionali in corso..."):
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur)
        xg1_live, xg2_live = time_decay_weibull(xg1_base, xg2_base, minuto_gioco)
        mc_1, mc_x, mc_2, mc_u25, mc_o25 = monte_carlo_simulator_full(xg1_live, xg2_live, 10000)
        
        delta_ah = ah_cur - ah_op
        delta_tot = tot_cur - tot_op

    # --- QUOTE REALI (IL TUO FARO) ---
    st.header(f"⚖️ Quote Reali (Minuto {minuto_gioco}')")
    st.markdown("Queste sono le quote matematiche pure. Il tuo obiettivo è **Puntare a una quota più ALTA** o **Bancare a una quota più BASSA**.")
    
    c_1, c_x, c_2 = st.columns(3)
    c_1.metric("1 (Casa)", f"@{calcola_quota_reale(mc_1):.2f}")
    c_x.metric("X (Pareggio)", f"@{calcola_quota_reale(mc_x):.2f}")
    c_2.metric("2 (Trasf.)", f"@{calcola_quota_reale(mc_2):.2f}")

    cu, co = st.columns(2)
    cu.metric("Under 2.5", f"@{calcola_quota_reale(mc_u25):.2f}")
    co.metric("Over 2.5", f"@{calcola_quota_reale(mc_o25):.2f}")

    st.divider()

    # --- SEGNALI OPERATIVI DIRETTI ---
    st.header("🎯 ISTRUZIONI EXCHANGE")
    segnali = False
    stake_sicuro = cassa * 0.025 # 2.5% della cassa come stake base raccomandato

    # Analisi Handicap (1X2)
    if delta_ah <= -0.25:
        q_target = calcola_quota_reale(mc_1) * 1.05 # Chiede almeno il 5% di vantaggio
        st.success("🟢 **SEGNALE: PUNTA 1 (Squadra di Casa)**")
        st.write(f"I volumi asiatici stanno schiacciando la casa. Apri l'Exchange:")
        st.write(f"👉 **Azione:** Se trovi una quota per l'1 **MAGGIORE di @{q_target:.2f}**, PUNTALA.")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True
    elif delta_ah >= 0.25:
        q_target = calcola_quota_reale(mc_2) * 1.05
        st.success("🟢 **SEGNALE: PUNTA 2 (Squadra in Trasferta)**")
        st.write(f"I flussi stanno crollando sulla squadra di casa. Apri l'Exchange:")
        st.write(f"👉 **Azione:** Se trovi una quota per il 2 **MAGGIORE di @{q_target:.2f}**, PUNTALA.")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True

    # Analisi Totale Gol
    if delta_tot >= 0.25:
        q_target_o = calcola_quota_reale(mc_o25) * 1.05
        q_target_x = calcola_quota_reale(mc_x) * 0.90 # Per bancare cerchiamo quota più bassa
        
        st.warning("🔥 **SEGNALE: PUNTA OVER 2.5** (o BANCA X)")
        st.write("I soldi pesanti si aspettano una pioggia di gol rispetto all'apertura.")
        st.write(f"👉 **Azione 1:** PUNTA l'Over 2.5 se trovi quota **MAGGIORE di @{q_target_o:.2f}**")
        if mc_x < 0.25:
            st.write(f"👉 **Azione 2 (Lay The Draw):** BANCA la X se trovi quota **MINORE di @{q_target_x:.2f}**")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True
    elif delta_tot <= -0.25:
        q_target_u = calcola_quota_reale(mc_u25) * 1.05
        st.info("🧊 **SEGNALE: PUNTA UNDER 2.5**")
        st.write("I professionisti stanno comprando l'Under. Partita tattica e chiusa.")
        st.write(f"👉 **Azione:** PUNTA l'Under 2.5 se trovi quota **MAGGIORE di @{q_target_u:.2f}**")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True

    if not segnali:
        st.error("⚖️ **NESSUN SEGNALE. MERCATO FERMO.**")
        st.write("I bookmaker e gli scommettitori sono d'accordo sulle quote (nessun Delta). Stai fermo, scommettere qui significa affidarsi al caso. Passa a un'altra partita.")
