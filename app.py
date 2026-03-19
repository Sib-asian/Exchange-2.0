import streamlit as st
import math

# ==========================================
# ⚙️ MOTORE MATEMATICO
# ==========================================
def calcola_xg(ah, totale):
    xg_casa = (totale - ah) / 2
    xg_trasferta = (totale + ah) / 2
    return max(0.01, xg_casa), max(0.01, xg_trasferta) 

def probabilita_poisson(lam, k):
    return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)

def calcola_matrice_1x2(xg_casa, xg_trasferta, max_gol=8):
    prob_1 = prob_x = prob_2 = 0.0
    for gol_casa in range(max_gol):
        for gol_trasferta in range(max_gol):
            prob = probabilita_poisson(xg_casa, gol_casa) * probabilita_poisson(xg_trasferta, gol_trasferta)
            if gol_casa > gol_trasferta: prob_1 += prob
            elif gol_casa == gol_trasferta: prob_x += prob
            else: prob_2 += prob
    totale = prob_1 + prob_x + prob_2
    return prob_1/totale, prob_x/totale, prob_2/totale

def calcola_under_over(xg_casa, xg_trasferta, linea, max_gol=8):
    prob_under = prob_over = 0.0
    for gol_casa in range(max_gol):
        for gol_trasferta in range(max_gol):
            prob = probabilita_poisson(xg_casa, gol_casa) * probabilita_poisson(xg_trasferta, gol_trasferta)
            if (gol_casa + gol_trasferta) > linea: prob_over += prob
            else: prob_under += prob
    totale = prob_under + prob_over
    return prob_under/totale, prob_over/totale

def calcola_quota_reale(probabilita):
    if probabilita <= 0: return 999.0
    return 1 / probabilita

# ==========================================
# 🎨 INTERFACCIA GRAFICA STREAMLIT
# ==========================================
st.set_page_config(page_title="Radar Exchange", page_icon="🎯", layout="centered")

st.title("🏆 Pronosticatore Maniacale")
st.markdown("Inserisci i movimenti del mercato asiatico per ottenere le Fair Odds e il pronostico.")

# --- PANNELLO DI INPUT ---
st.header("📝 Dati Mercato Asiatico")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Apertura")
    ah_op = st.number_input("Asian Handicap (Es: -0.25)", value=-0.25, step=0.25, format="%.2f")
    tot_op = st.number_input("Linea Gol Totale (Es: 2.50)", value=2.50, step=0.25, format="%.2f")

with col2:
    st.subheader("Corrente")
    ah_cur = st.number_input("Asian Handicap Attuale", value=-0.75, step=0.25, format="%.2f")
    tot_cur = st.number_input("Linea Gol Totale Attuale", value=2.75, step=0.25, format="%.2f")

st.divider()

# --- PULSANTE DI CALCOLO ---
if st.button("🚀 Calcola Pronostico", use_container_width=True, type="primary"):
    
    # Elaborazione Dati
    xg_casa_op, xg_trasf_op = calcola_xg(ah_op, tot_op)
    xg_casa_cur, xg_trasf_cur = calcola_xg(ah_cur, tot_cur)

    p1_cur, px_cur, p2_cur = calcola_matrice_1x2(xg_casa_cur, xg_trasf_cur)
    u25_cur, o25_cur = calcola_under_over(xg_casa_cur, xg_trasf_cur, 2.5)

    delta_ah = ah_cur - ah_op
    delta_tot = tot_cur - tot_op

    # --- SEZIONE 1: QUOTE REALI ---
    st.header("⚖️ Quote Reali (Fair Odds)")
    
    col_1, col_x, col_2 = st.columns(3)
    col_1.metric(label="Vittoria CASA (1)", value=f"{p1_cur*100:.1f}%", delta=f"@ {calcola_quota_reale(p1_cur):.2f}", delta_color="off")
    col_x.metric(label="PAREGGIO (X)", value=f"{px_cur*100:.1f}%", delta=f"@ {calcola_quota_reale(px_cur):.2f}", delta_color="off")
    col_2.metric(label="Vittoria TRASF. (2)", value=f"{p2_cur*100:.1f}%", delta=f"@ {calcola_quota_reale(p2_cur):.2f}", delta_color="off")
    
    st.divider()

    # --- SEZIONE 2: PRONOSTICO AUTOMATICO ---
    st.header("🔮 Il Verdetto dell'Algoritmo")
    
    segnali_trovati = False

    if delta_ah <= -0.25:
        st.success("👉 **PUNTA 1 (o AH Casa):** I volumi asiatici stanno crollando pesantemente sulla squadra di casa.")
        st.error("👉 **EXCHANGE - BANCA 2:** La trasferta è totalmente sfavorita dal mercato.")
        segnali_trovati = True
    elif delta_ah >= 0.25:
        st.success("👉 **PUNTA 2 (o AH Trasferta):** I volumi asiatici spingono forte sulla squadra in trasferta.")
        st.error("👉 **EXCHANGE - BANCA 1:** La squadra di casa sta perdendo la fiducia degli investitori.")
        segnali_trovati = True
    elif p1_cur > 0.60:
        st.info("👉 **PUNTA 1:** Movimenti stabili, ma la probabilità matematica di base è altissima (>60%).")
        segnali_trovati = True
    elif p2_cur > 0.60:
        st.info("👉 **PUNTA 2:** Movimenti stabili, ma la probabilità matematica di base è altissima (>60%).")
        segnali_trovati = True

    if delta_tot >= 0.25:
        st.success("👉 **PUNTA OVER 2.5:** Il mercato ha alzato la linea, si aspetta sicuramente più reti dell'apertura.")
        if px_cur < 0.25:
            st.error("👉 **EXCHANGE - BANCA LA X:** Nelle partite da Over, il pareggio perde molto valore (Lay The Draw).")
        segnali_trovati = True
    elif delta_tot <= -0.25:
        st.warning("👉 **PUNTA UNDER 2.5:** Il mercato ha abbassato la linea, prevista partita bloccata e chiusa.")
        segnali_trovati = True

    if not segnali_trovati:
        st.warning("⚖️ **NO BET.** I movimenti di mercato e le probabilità sono troppo deboli per dare un segnale netto. Meglio passare oltre.")
        
    st.caption(f"Movimenti rilevati: Spread {delta_ah:+.2f} | Totale Gol {delta_tot:+.2f}")
