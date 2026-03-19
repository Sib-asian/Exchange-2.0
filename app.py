import streamlit as st
import math
import random

# ==========================================
# ⚙️ MOTORE MATEMATICO ISTITUZIONALE (BAYES + WEIBULL + SHIN)
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur):
    """Calcola gli xG fondendo Apertura e Chiusura tramite logica pseudo-Bayesiana."""
    # Il mercato corrente ha un peso maggiore (Evidenza), l'apertura è il Prior.
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
    """
    Applica il decadimento temporale non-lineare (Weibull shape approx).
    I gol sono più frequenti nel secondo tempo.
    """
    if minuto == 0: return xg_casa, xg_trasf
    if minuto >= 90: return 0.001, 0.001
    
    # Il tempo rimanente non scala linearmente. 
    # Al 45' non resta il 50% dei gol, ma circa il 55% (curvatura 0.85).
    tempo_rimanente = 90 - minuto
    fattore_decadimento = math.pow((tempo_rimanente / 90.0), 0.85)
    
    return xg_casa * fattore_decadimento, xg_trasf * fattore_decadimento

def bessel_i_approx(k, z):
    sum_val = 0.0
    k = abs(int(k))
    for m in range(20):
        numeratore = math.pow(z / 2.0, 2 * m + k)
        denominatore = math.factorial(m) * math.factorial(m + k)
        sum_val += numeratore / denominatore
    return sum_val

def skellam_prob(k, mu1, mu2):
    term1 = math.exp(-(mu1 + mu2))
    term2 = math.pow(mu1 / mu2, k / 2.0)
    term3 = bessel_i_approx(k, 2 * math.sqrt(mu1 * mu2))
    return term1 * term2 * term3

def monte_carlo_simulator(mu1, mu2, iterazioni=10000):
    vittorie_1 = pareggi = vittorie_2 = 0
    for _ in range(iterazioni):
        L1, L2 = math.exp(-mu1), math.exp(-mu2)
        k1, p1_sim = 0, 1.0
        while p1_sim > L1:
            k1 += 1
            p1_sim *= random.random()
        gol1 = k1 - 1
        
        k2, p2_sim = 0, 1.0
        while p2_sim > L2:
            k2 += 1
            p2_sim *= random.random()
        gol2 = k2 - 1
        
        if gol1 > gol2: vittorie_1 += 1
        elif gol1 == gol2: pareggi += 1
        else: vittorie_2 += 1
        
    return vittorie_1/iterazioni, pareggi/iterazioni, vittorie_2/iterazioni

def calcola_quota_reale(probabilita):
    return 1 / probabilita if probabilita > 0.001 else 999.0

# ==========================================
# 🎨 STREAMLIT DASHBOARD (LIVE TRADING)
# ==========================================
st.set_page_config(page_title="Institutional Quant", page_icon="🏦", layout="centered")

st.title("🏦 Institutional Quant Terminal")
st.markdown("*Motore: Inferenza Bayesiana & Weibull Time-Decay (In-Play)*")

# --- 1. MODULO IN-PLAY (NUOVO) ---
st.header("⏱️ Controllo Temporale (Live Exchange)")
minuto_gioco = st.slider("Minuto di Gioco Attuale (0 = Pre-match)", min_value=0, max_value=90, value=0, step=1)
if minuto_gioco > 0:
    st.warning(f"📡 **MODALITÀ LIVE ATTIVA:** Il motore sta calcolando le probabilità per i restanti {90 - minuto_gioco} minuti usando la curvatura di Weibull.")

st.divider()

# --- 2. INPUT MERCATO ---
st.header("📊 Dati Mercato Asiatico")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prior (Apertura)")
    ah_op = st.number_input("AH Apertura", value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50, step=0.25)
with col2:
    st.subheader("Evidence (Corrente)")
    ah_cur = st.number_input("AH Corrente", value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75, step=0.25)

st.divider()

# --- 3. GESTIONE RISCHIO ---
st.header("💰 Risk Management")
col3, col4, col5 = st.columns(3)
with col3: cassa = st.number_input("Cassa (€)", value=1000.0, step=100.0)
with col4: segno_scelto = st.selectbox("Mercato", ["Vittoria Casa (1)", "Pareggio (X)", "Vittoria Trasf. (2)"])
with col5: quota_mercato = st.number_input("Quota Live", value=1.90, step=0.05)

if st.button("🧬 ESEGUI INFERENZA BAYESIANA", use_container_width=True, type="primary"):
    
    with st.spinner("Allineamento dei pesi Bayesiani e decadimento temporale..."):
        # 1. Calcolo xG puri (Bayesiani)
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur)
        
        # 2. Decadimento temporale (Live In-Play)
        xg1_live, xg2_live = time_decay_weibull(xg1_base, xg2_base, minuto_gioco)
        
        # 3. Simulazione Monte Carlo sul tempo rimanente
        mc_1, mc_x, mc_2 = monte_carlo_simulator(xg1_live, xg2_live, 10000)
        
        # 4. Generazione Curva Skellam
        margini = range(-3, 4) 
        prob_margini = [skellam_prob(k, xg1_live, xg2_live) for k in margini]

    st.header(f"⚖️ Fair Odds Istituzionali (Al minuto {minuto_gioco}')")
    c_1, c_x, c_2 = st.columns(3)
    c_1.metric("Segno 1", f"{mc_1*100:.1f}%", f"Fair: @{calcola_quota_reale(mc_1):.2f}", delta_color="off")
    c_x.metric("Segno X", f"{mc_x*100:.1f}%", f"Fair: @{calcola_quota_reale(mc_x):.2f}", delta_color="off")
    c_2.metric("Segno 2", f"{mc_2*100:.1f}%", f"Fair: @{calcola_quota_reale(mc_2):.2f}", delta_color="off")

    st.subheader("📊 Distribuzione Probabilità Margine Vittoria")
    st.bar_chart(dict(zip([f"{k} Gol" for k in margini], prob_margini)))

    st.divider()

    st.header("📈 Analisi Valore & Segnale Operativo")
    if segno_scelto == "Vittoria Casa (1)": prob_target = mc_1
    elif segno_scelto == "Pareggio (X)": prob_target = mc_x
    else: prob_target = mc_2

    ev = (prob_target * quota_mercato) - 1
    
    if ev > 0.02: # Edge minimo del 2% richiesto dai pro
        q = 1 - prob_target
        kelly_puro = ((prob_target * (quota_mercato - 1)) - q) / (quota_mercato - 1)
        stake_fraz = max(0, kelly_puro * 0.25)
        importo = stake_fraz * cassa
        eg = prob_target * math.log(1 + stake_fraz * (quota_mercato - 1)) + q * math.log(1 - stake_fraz)

        st.success("🎯 **SEGNALE FORTE: VALORE BAYESIANO CONFERMATO.**")
        col_ev, col_eg, col_stk, col_eur = st.columns(4)
        col_ev.metric("Expected Value", f"+{ev*100:.1f}%")
        col_eg.metric("Exp. Growth", f"+{eg*100:.2f}%")
        col_stk.metric("Kelly Fraz.", f"{stake_fraz*100:.1f}%")
        col_eur.metric("Stake Consigliato", f"€ {importo:.2f}")
        
    elif ev > -0.05 and ev <= 0.02:
        st.warning("⚠️ **VALORE MARGINALE / NEUTRO.** Il mercato è prezzato in modo quasi perfetto (Zero-Sum Game). Le commissioni dell'Exchange si mangeranno il profitto. Stare fermi.")
    else:
        st.error(f"🚫 **TRAPPOLA DEL MERCATO (EV NEGATIVO: {ev*100:.1f}%).**")
        st.write("La quota offerta include un aggio troppo alto o va contro l'evidenza Bayesiana dei flussi di denaro. Assolutamente NO BET.")
