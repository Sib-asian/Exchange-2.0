import streamlit as st
import math

# ==========================================
# ⚙️ MOTORE MATEMATICO STRATOSFERICO (DIXON-COLES)
# ==========================================
def calcola_xg(ah, totale):
    """Calcola gli xG tenendo conto delle dinamiche di spread asiatico"""
    xg_casa = (totale - ah) / 2.0
    xg_trasferta = (totale + ah) / 2.0
    return max(0.05, xg_casa), max(0.05, xg_trasferta) 

def probabilita_poisson(lam, k):
    return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)

def calcola_matrice_dixon_coles(xg_casa, xg_trasferta, rho=-0.15, max_gol=8):
    """Genera la matrice con correzione Dixon-Coles per la dipendenza dei gol"""
    prob_1 = prob_x = prob_2 = 0.0
    
    for gol_casa in range(max_gol):
        for gol_trasferta in range(max_gol):
            # Poisson Base
            prob = probabilita_poisson(xg_casa, gol_casa) * probabilita_poisson(xg_trasferta, gol_trasferta)
            
            # Correzione Dixon-Coles per i risultati a basso punteggio (aumenta i pareggi)
            if gol_casa == 0 and gol_trasferta == 0:
                prob *= (1 - (xg_casa * xg_trasferta * rho))
            elif gol_casa == 1 and gol_trasferta == 0:
                prob *= (1 + (xg_casa * rho))
            elif gol_casa == 0 and gol_trasferta == 1:
                prob *= (1 + (xg_trasferta * rho))
            elif gol_casa == 1 and gol_trasferta == 1:
                prob *= (1 - rho)
                
            # Assegnazione al segno 1X2
            if gol_casa > gol_trasferta: prob_1 += prob
            elif gol_casa == gol_trasferta: prob_x += prob
            else: prob_2 += prob
            
    # Normalizzazione per garantire somma 100% perfetta (Vig removal implicito)
    totale = prob_1 + prob_x + prob_2
    return prob_1/totale, prob_x/totale, prob_2/totale

def calcola_under_over_dc(xg_casa, xg_trasferta, linea, rho=-0.15, max_gol=8):
    """Calcola U/O applicando la matrice Dixon-Coles"""
    prob_under = prob_over = 0.0
    for gol_casa in range(max_gol):
        for gol_trasferta in range(max_gol):
            prob = probabilita_poisson(xg_casa, gol_casa) * probabilita_poisson(xg_trasferta, gol_trasferta)
            
            if gol_casa == 0 and gol_trasferta == 0: prob *= (1 - (xg_casa * xg_trasferta * rho))
            elif gol_casa == 1 and gol_trasferta == 0: prob *= (1 + (xg_casa * rho))
            elif gol_casa == 0 and gol_trasferta == 1: prob *= (1 + (xg_trasferta * rho))
            elif gol_casa == 1 and gol_trasferta == 1: prob *= (1 - rho)
                
            if (gol_casa + gol_trasferta) > linea: prob_over += prob
            else: prob_under += prob
            
    totale = prob_under + prob_over
    return prob_under/totale, prob_over/totale

def calcola_quota_reale(probabilita):
    if probabilita <= 0.001: return 999.0
    return 1 / probabilita

# ==========================================
# 🎨 INTERFACCIA GRAFICA STREAMLIT
# ==========================================
st.set_page_config(page_title="Radar Exchange Pro", page_icon="🧿", layout="centered")

st.title("🧿 Radar Exchange Pro")
st.markdown("*Motore Matematico: Poisson Multivariato + Dixon-Coles Adjustment*")

st.header("📝 Dati Mercato Asiatico")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Apertura")
    ah_op = st.number_input("Asian Handicap (Es: -0.25)", value=-0.25, step=0.25, format="%.2f", key="aho")
    tot_op = st.number_input("Linea Gol Totale", value=2.50, step=0.25, format="%.2f", key="to")

with col2:
    st.subheader("Corrente / Chiusura")
    ah_cur = st.number_input("Asian Handicap", value=-0.75, step=0.25, format="%.2f", key="ahc")
    tot_cur = st.number_input("Linea Gol Totale", value=2.75, step=0.25, format="%.2f", key="tc")

st.divider()

if st.button("🚀 GENERA ANALISI PROFONDA", use_container_width=True, type="primary"):
    
    # Calcolo xG puri
    xg_casa_op, xg_trasf_op = calcola_xg(ah_op, tot_op)
    xg_casa_cur, xg_trasf_cur = calcola_xg(ah_cur, tot_cur)

    # Calcolo Matrici Avanzate
    p1_cur, px_cur, p2_cur = calcola_matrice_dixon_coles(xg_casa_cur, xg_trasf_cur)
    u25_cur, o25_cur = calcola_under_over_dc(xg_casa_cur, xg_trasf_cur, 2.5)
    u15_cur, o15_cur = calcola_under_over_dc(xg_casa_cur, xg_trasf_cur, 1.5)

    delta_ah = ah_cur - ah_op
    delta_tot = tot_cur - tot_op

    st.header("⚖️ Quote Reali (Fair Odds Modello DC)")
    
    col_1, col_x, col_2 = st.columns(3)
    col_1.metric("Vittoria CASA (1)", f"{p1_cur*100:.1f}%", f"@ {calcola_quota_reale(p1_cur):.2f}", delta_color="off")
    col_x.metric("PAREGGIO (X)", f"{px_cur*100:.1f}%", f"@ {calcola_quota_reale(px_cur):.2f}", delta_color="off")
    col_2.metric("Vittoria TRASF. (2)", f"{p2_cur*100:.1f}%", f"@ {calcola_quota_reale(p2_cur):.2f}", delta_color="off")
    
    col_u, col_o = st.columns(2)
    col_u.metric("UNDER 2.5", f"{u25_cur*100:.1f}%", f"@ {calcola_quota_reale(u25_cur):.2f}", delta_color="off")
    col_o.metric("OVER 2.5", f"{o25_cur*100:.1f}%", f"@ {calcola_quota_reale(o25_cur):.2f}", delta_color="off")

    st.divider()

    st.header("🔮 Analisi Algoritmica Direzionale")
    
    segnali_trovati = False

    # Spread Analysis
    if delta_ah <= -0.25:
        st.success(f"💸 **Smart Money su CASA:** L'handicap è passato da {ah_op} a {ah_cur}. I professionisti stanno scommettendo forte sulla squadra di casa.")
        st.info("👉 **AZIONE EXCHANGE:** Punta 1 (se quota > Fair) oppure Banca 2.")
        segnali_trovati = True
    elif delta_ah >= 0.25:
        st.success(f"💸 **Smart Money su TRASFERTA:** L'handicap è passato da {ah_op} a {ah_cur}. Crollo di fiducia sulla squadra di casa.")
        st.info("👉 **AZIONE EXCHANGE:** Punta 2 (se quota > Fair) oppure Banca 1.")
        segnali_trovati = True

    # Total Lines Analysis
    if delta_tot >= 0.25:
        st.warning(f"🔥 **Trend OVER:** La linea gol è salita da {tot_op} a {tot_cur}. Attesi più gol del previsto.")
        if px_cur < 0.25:
            st.error("📉 **LAY THE DRAW (Banca X):** Trend da Over + Bassa probabilità di X pura. Ottimo spot per bancare il pareggio.")
        segnali_trovati = True
    elif delta_tot <= -0.25:
        st.info(f"🧊 **Trend UNDER:** La linea gol è scesa. Match che si preannuncia bloccato e tattico.")
        segnali_trovati = True

    # Valore Intrinseco Puro
    if p1_cur > 0.65 and delta_ah <= 0:
        st.success("🏰 **Roccaforte:** Probabilità di vittoria interna schiacciante (>65%) supportata dal mercato.")
        segnali_trovati = True

    if not segnali_trovati:
        st.warning("⚖️ **MERCATO EFFICIENTE / STABILE.** I volumi non hanno spostato le linee di apertura. Giocare qui significa sfidare la varianza senza un Edge direzionale. Skip.")

    st.caption("⚙️ Il modello utilizza una variabile di dipendenza ρ (rho) settata a -0.15 per correggere la distribuzione dei pareggi a basso punteggio, eliminando il bias del modello di Poisson tradizionale.")
