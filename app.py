import streamlit as st
import math
import random

# ==========================================
# ⚙️ MOTORE MATEMATICO ISTITUZIONALE LIVE
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

def monte_carlo_simulator_live(mu1, mu2, gol_casa_attuali, gol_trasf_attuali, iterazioni=10000):
    vit_1 = pareggi = vit_2 = over25 = under25 = 0
    for _ in range(iterazioni):
        # 1. Simula i gol nel TEMPO RIMANENTE
        L1, L2 = math.exp(-mu1), math.exp(-mu2)
        k1, p1_sim = 0, 1.0
        while p1_sim > L1:
            k1 += 1; p1_sim *= random.random()
        gol_sim_1 = k1 - 1
        
        k2, p2_sim = 0, 1.0
        while p2_sim > L2:
            k2 += 1; p2_sim *= random.random()
        gol_sim_2 = k2 - 1
        
        # 2. Somma i gol simulati al RISULTATO ATTUALE
        totale_casa = gol_casa_attuali + gol_sim_1
        totale_trasf = gol_trasf_attuali + gol_sim_2
        
        if totale_casa > totale_trasf: vit_1 += 1
        elif totale_casa == totale_trasf: pareggi += 1
        else: vit_2 += 1
            
        if (totale_casa + totale_trasf) > 2.5: over25 += 1
        else: under25 += 1
        
    return vit_1/iterazioni, pareggi/iterazioni, vit_2/iterazioni, under25/iterazioni, over25/iterazioni

def calcola_quota_reale(probabilita):
    return 1 / probabilita if probabilita > 0.001 else 999.0

# ==========================================
# 🎨 STREAMLIT DASHBOARD (LIVE TRADING)
# ==========================================
st.set_page_config(page_title="Radar Exchange Pro", page_icon="🎯", layout="centered")

st.title("🎯 Radar Exchange Pro")
st.markdown("*Modalità Operativa Live - Segnali Diretti State-Space*")

# --- 1. DATI LIVE (RISULTATO E MINUTO) ---
st.header("⏱️ Stato della Partita (Live)")
col_min, col_cassa = st.columns(2)
with col_min:
    minuto_gioco = st.slider("Minuto Attuale (0 = Pre-match)", 0, 90, 0, 1)
with col_cassa:
    cassa = st.number_input("💰 Cassa Totale (€)", value=1000.0, step=100.0)

col_g1, col_g2 = st.columns(2)
with col_g1:
    gol_casa = st.number_input("⚽ Gol segnati CASA", value=0, min_value=0, step=1)
with col_g2:
    gol_trasf = st.number_input("⚽ Gol segnati TRASFERTA", value=0, min_value=0, step=1)

st.divider()

# --- 2. DATI MERCATO ---
st.header("📊 Dati Mercato Asiatico")
col1, col2 = st.columns(2)
with col1:
    ah_op = st.number_input("AH Apertura", value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50, step=0.25)
with col2:
    ah_cur = st.number_input("AH Corrente", value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75, step=0.25)

st.divider()

if st.button("🚀 GENERA SEGNALI OPERATIVI LIVE", use_container_width=True, type="primary"):
    
    with st.spinner("Elaborazione dati Istituzionali in corso..."):
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur)
        xg1_live, xg2_live = time_decay_weibull(xg1_base, xg2_base, minuto_gioco)
        
        # Simulazione tenendo conto del risultato attuale!
        mc_1, mc_x, mc_2, mc_u25, mc_o25 = monte_carlo_simulator_live(xg1_live, xg2_live, gol_casa, gol_trasf, 10000)
        
        delta_ah = ah_cur - ah_op
        delta_tot = tot_cur - tot_op

    # --- QUOTE REALI ---
    st.header(f"⚖️ Quote Reali (Minuto {minuto_gioco}' | Risultato {gol_casa}-{gol_trasf})")
    
    c_1, c_x, c_2 = st.columns(3)
    c_1.metric("1 (Casa)", f"@{calcola_quota_reale(mc_1):.2f}")
    c_x.metric("X (Pareggio)", f"@{calcola_quota_reale(mc_x):.2f}")
    c_2.metric("2 (Trasf.)", f"@{calcola_quota_reale(mc_2):.2f}")

    cu, co = st.columns(2)
    cu.metric("Under 2.5", f"@{calcola_quota_reale(mc_u25):.2f}")
    co.metric("Over 2.5", f"@{calcola_quota_reale(mc_o25):.2f}")

    st.divider()

    # --- SEGNALI OPERATIVI ---
    st.header("🎯 ISTRUZIONI EXCHANGE")
    segnali = False
    stake_sicuro = cassa * 0.025 

    # Se la partita è finita o quasi
    if minuto_gioco >= 85:
        st.error("🛑 **NO BET:** Mancano meno di 5 minuti. La liquidità è troppo bassa e la varianza troppo alta. Non operare.")
        st.stop()

    # Analisi Handicap
    if delta_ah <= -0.25:
        q_target = calcola_quota_reale(mc_1) * 1.05 
        st.success(f"🟢 **SEGNALE: PUNTA 1 (Squadra di Casa) a quota > @{q_target:.2f}**")
        st.write(f"I flussi asiatici sono sulla squadra di casa. Aggiungi il punteggio live ({gol_casa}-{gol_trasf}) a tuo vantaggio.")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True
    elif delta_ah >= 0.25:
        q_target = calcola_quota_reale(mc_2) * 1.05
        st.success(f"🟢 **SEGNALE: PUNTA 2 (Squadra Trasferta) a quota > @{q_target:.2f}**")
        st.write(f"Flussi crollati sulla casa, fiducia in trasferta. Controlla il mercato live.")
        st.write(f"💰 **Stake consigliato:** € {stake_sicuro:.2f}")
        segnali = True

    # Analisi Totale Gol
    if delta_tot >= 0.25 and (gol_casa + gol_trasf) < 3: # Ha senso puntare over 2.5 solo se non ci sono già 3 gol
        q_target_o = calcola_quota_reale(mc_o25) * 1.05
        q_target_x = calcola_quota_reale(mc_x) * 0.90 
        
        st.warning(f"🔥 **SEGNALE: PUNTA OVER 2.5 a quota > @{q_target_o:.2f}**")
        if mc_x < 0.25 and gol_casa == gol_trasf: # Suggerisce LTD solo se in pareggio
            st.write(f"👉 **Azione 2 (Lay The Draw):** Poiché il risultato è {gol_casa}-{gol_trasf}, puoi BANCARE la X se trovi quota < @{q_target_x:.2f}")
        segnali = True
        
    elif delta_tot <= -0.25 and (gol_casa + gol_trasf) < 3:
        q_target_u = calcola_quota_reale(mc_u25) * 1.05
        st.info(f"🧊 **SEGNALE: PUNTA UNDER 2.5 a quota > @{q_target_u:.2f}**")
        segnali = True

    if not segnali:
        st.error("⚖️ **NESSUN SEGNALE NETTO. MERCATO FERMO.**")
