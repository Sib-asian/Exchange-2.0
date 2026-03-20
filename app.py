import streamlit as st
import math

# ==========================================
# ⚙️ MOTORE MATEMATICO
# ==========================================

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto):
    """
    FIX BUG 1 — normalizzazione temporale prima del blend bayesiano.

    Il problema originale: tot_op (full 90') e tot_cur (gol rimanenti) vivono
    su scale diverse. Blendarli direttamente produce un λ senza senso fisico.

    Soluzione: scaliamo la linea di apertura al tempo rimanente prima del blend,
    in modo che entrambi gli input rappresentino la stessa cosa: gol attesi
    dal minuto attuale al 90'.

    tot_op_rem = tot_op × (90 − minuto) / 90
    ah_op_rem  = ah_op  × (90 − minuto) / 90

    Il blend 35/65 opera poi su grandezze omogenee.
    """
    peso_prior     = 0.35
    peso_evidenza  = 0.65

    frac_rimasta   = max(0.05, (90.0 - minuto) / 90.0)
    tot_op_rem     = tot_op * frac_rimasta
    ah_op_rem      = ah_op  * frac_rimasta

    ah_bayes  = ah_op_rem * peso_prior + ah_cur * peso_evidenza
    tot_bayes = tot_op_rem * peso_prior + tot_cur * peso_evidenza
    tot_bayes = max(0.2, float(tot_bayes))

    eps = 1e-6

    def _poisson_pmf_for_ev(mu):
        tail_mass = 1e-12
        if mu <= 0:
            return [1.0]
        max_k = int(max(20.0, mu * 8.0 + 40.0))
        max_k = min(max_k, 300)
        p = math.exp(-mu)
        pmfs   = [p]
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
        pmf_h = _poisson_pmf_for_ev(mu_home)
        pmf_a = _poisson_pmf_for_ev(mu_away)
        ev = 0.0
        for i, pi in enumerate(pmf_h):
            if pi < 1e-18:
                continue
            for j, pj in enumerate(pmf_a):
                if pj < 1e-18:
                    continue
                s = (i - j) + h_half
                prob = pi * pj
                if s > 0:
                    ev += prob
                elif s < 0:
                    ev -= prob
        return ev

    def _asian_ev(mu_home, mu_away, ah):
        ah2 = float(ah) * 2.0
        if abs(ah2 - round(ah2)) < 1e-9:
            return _asian_ev_half(mu_home, mu_away, float(ah))
        h1 = math.floor(ah2) / 2.0
        h2 = h1 + 0.5
        return 0.5 * _asian_ev_half(mu_home, mu_away, h1) + \
               0.5 * _asian_ev_half(mu_home, mu_away, h2)

    def _ev_for_delta(delta):
        lam_h = max(eps, 0.5 * (tot_bayes + delta))
        lam_a = max(eps, 0.5 * (tot_bayes - delta))
        return _asian_ev(lam_h, lam_a, ah_bayes)

    lo    = -tot_bayes + eps
    hi    =  tot_bayes - eps
    ev_lo = _ev_for_delta(lo)
    ev_hi = _ev_for_delta(hi)

    if ev_lo == 0.0:
        delta_star = lo
    elif ev_hi == 0.0:
        delta_star = hi
    elif ev_lo * ev_hi > 0:
        delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
    else:
        increasing = ev_hi > ev_lo
        delta_l, delta_r = lo, hi
        for _ in range(18):
            mid    = 0.5 * (delta_l + delta_r)
            ev_mid = _ev_for_delta(mid)
            if increasing:
                if ev_mid > 0:
                    delta_r = mid
                else:
                    delta_l = mid
            else:
                if ev_mid > 0:
                    delta_l = mid
                else:
                    delta_r = mid
        delta_star = 0.5 * (delta_l + delta_r)

    lambda_casa  = max(eps, 0.5 * (tot_bayes + delta_star))
    lambda_trasf = max(eps, 0.5 * (tot_bayes - delta_star))
    return lambda_casa, lambda_trasf


def time_decay_dinamico(xg_casa, xg_trasf, minuto,
                        gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    """
    FIX BUG 2 — score effect rimosso (già prezzato nell'AH live).
    FIX BUG 3 — red card senza moltiplicatore frac.

    Principio: la linea AH corrente che l'utente inserisce riflette
    già il punteggio attuale. Applicare di nuovo uno score effect
    significa conteggiarlo due volte.

    Manteniamo solo un residuale comportamentale piccolo (max ±12%)
    per catturare l'urgency emotiva non ancora completamente prezzata.

    Red cards: l'effetto sul rate di gol rimanenti è costante dal
    momento dell'espulsione in poi, non deve scemare col tempo.
    math.pow(0.68, n_rossi) senza frac.
    """
    if minuto >= 90:
        return 0.001, 0.001

    # 1. Decadimento Weibull base
    fattore_tempo = math.pow((90.0 - minuto) / 90.0, 0.85)
    xg_c_live = xg_casa * fattore_tempo
    xg_t_live = xg_trasf * fattore_tempo

    # 2. Score effect RESIDUALE (solo componente comportamentale ≤ 12%)
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat      = abs(diff) / (2.0 + abs(diff))   # satura più lentamente
        residual = 0.12 * sat                        # max +12% / -12%
        if diff < 0:   # casa perde → spinge
            xg_c_live *= (1.0 + residual)
            xg_t_live *= (1.0 - residual)
        else:          # trasf. perde → spinge
            xg_t_live *= (1.0 + residual)
            xg_c_live *= (1.0 - residual)

    # 3. Red cards — effetto costante sul rate rimanente
    if rossi_casa > 0:
        xg_c_live *= math.pow(0.68, rossi_casa)   # casa perde potere offensivo
        xg_t_live *= math.pow(1.28, rossi_casa)   # avversario ne approfitta
    if rossi_trasf > 0:
        xg_t_live *= math.pow(0.68, rossi_trasf)
        xg_c_live *= math.pow(1.28, rossi_trasf)

    return max(0.001, xg_c_live), max(0.001, xg_t_live)


# ==========================================
# 🎲 PROBABILITÀ ESATTE — Poisson bivariata
# ==========================================

def _poisson_pmf_normalized(mu, tail_mass=1e-12, max_k=None):
    if mu <= 0:
        return [1.0]
    if max_k is None:
        max_k = min(int(max(20, mu * 12.0 + 60)), 500)
    p0     = math.exp(-mu)
    pmfs   = [p0]
    cumsum = p0
    k, p   = 0, p0
    while cumsum < (1.0 - tail_mass) and k < max_k:
        k += 1
        p = p * mu / k
        pmfs.append(p)
        cumsum += p
    return [x / cumsum for x in pmfs]


def probabilita_poisson_esatta(mu_casa_rem, mu_trasf_rem,
                                gol_casa, gol_trasf, linea_ou):
    mu_casa_rem  = max(1e-9, float(mu_casa_rem))
    mu_trasf_rem = max(1e-9, float(mu_trasf_rem))

    rho_corr = 0.12
    min_mu   = min(mu_casa_rem, mu_trasf_rem)
    lambda0  = max(0.0, min(rho_corr * min_mu, 0.95 * min_mu))

    mu_c_ind = max(1e-9, mu_casa_rem  - lambda0)
    mu_t_ind = max(1e-9, mu_trasf_rem - lambda0)

    pmf_c = _poisson_pmf_normalized(mu_c_ind)
    pmf_t = _poisson_pmf_normalized(mu_t_ind)

    p1 = px = p2 = 0.0
    for i, pi in enumerate(pmf_c):
        if pi < 1e-16:
            continue
        for j, pj in enumerate(pmf_t):
            if pj < 1e-16:
                continue
            tc = gol_casa  + i
            tt = gol_trasf + j
            if tc > tt:
                p1 += pi * pj
            elif tc == tt:
                px += pi * pj
            else:
                p2 += pi * pj

    psum = p1 + px + p2
    if psum > 0:
        p1 /= psum; px /= psum; p2 /= psum

    S      = gol_casa + gol_trasf
    pmf_z  = _poisson_pmf_normalized(lambda0)
    pmf_w  = _poisson_pmf_normalized(mu_c_ind + mu_t_ind)

    prefix = [0.0]
    c = 0.0
    for p in pmf_w:
        c += p
        prefix.append(c)

    p_under = 0.0
    for z, pz in enumerate(pmf_z):
        if pz < 1e-16:
            continue
        mk = int(math.floor(float(linea_ou) - S - 2 * z))
        if mk < 0:
            continue
        p_under += pz * (1.0 if mk >= len(pmf_w) - 1 else prefix[mk + 1])

    p_under = min(max(p_under, 0.0), 1.0)
    p_over  = 1.0 - p_under

    return p1, px, p2, p_under, p_over


def calcola_quota_reale(prob):
    return 1.0 / prob if prob > 0.001 else 999.0


def calcola_stake_kelly(prob_modello, quota_target, bankroll, frazione=0.5):
    """
    FIX BUG 5 — Kelly frazionato al posto dello stake piatto.

    edge = prob_modello − 1/quota_target   (vantaggio atteso)
    kelly% = edge / (quota_target − 1)     (formula Kelly)
    stake  = bankroll × kelly% × frazione  (½-Kelly per prudenza)

    Se edge ≤ 0 non c'è valore: stake = 0, non scommettere.
    Cap assoluto al 5% del bankroll per gestione del rischio.
    """
    prob_be = 1.0 / quota_target
    edge    = prob_modello - prob_be
    if edge <= 0:
        return 0.0
    kelly_pct = edge / (quota_target - 1.0)
    kelly_pct = min(kelly_pct, 0.05)
    return bankroll * kelly_pct * frazione


# ==========================================
# 🎨 UI STREAMLIT
# ==========================================
st.set_page_config(page_title="Radar Pro Live", page_icon="⚡", layout="centered")

st.title("⚡ Radar Pro Live")
st.caption("v41 · Fix: normalizzazione temporale · score effect unico · segnali da probabilità · Kelly")

# ── INPUT ────────────────────────────────────────────────────────────────────
st.header("⏱️ 1. Stato Partita")
minuto_gioco = st.slider("Minuto Attuale", 0, 90, 0, 1)

col_g1, col_g2 = st.columns(2)
with col_g1:
    gol_casa  = st.number_input("⚽ Gol CASA",   value=0, min_value=0)
with col_g2:
    gol_trasf = st.number_input("⚽ Gol TRASF.", value=0, min_value=0)

col_r1, col_r2 = st.columns(2)
with col_r1:
    rossi_casa  = st.number_input("🟥 Rossi CASA",   value=0, min_value=0, max_value=4)
with col_r2:
    rossi_trasf = st.number_input("🟥 Rossi TRASF.", value=0, min_value=0, max_value=4)

st.divider()

st.header("📊 2. Linee Asiatiche")
col_a1, col_a2 = st.columns(2)
with col_a1:
    st.markdown("**Apertura — full 90'**")
    ah_op  = st.number_input("AH Apertura",     value=-0.25, step=0.25)
    tot_op = st.number_input("Totale Apertura", value=2.50,  step=0.25)
with col_a2:
    st.markdown("**Corrente — gol rimanenti**")
    ah_cur  = st.number_input("AH Corrente",     value=-0.75, step=0.25)
    tot_cur = st.number_input("Totale Corrente", value=2.75,  step=0.25)

linea_target_ou = st.selectbox("Linea U/O da analizzare:", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], index=2)
cassa = st.number_input("💰 Bankroll (€)", value=1000.0, step=100.0)

st.divider()

# ── CALCOLO ──────────────────────────────────────────────────────────────────
if st.button("🚀 ANALIZZA", use_container_width=True, type="primary"):
    with st.spinner("Calcolo..."):
        xg1_base, xg2_base = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, minuto_gioco
        )
        xg1_live, xg2_live = time_decay_dinamico(
            xg1_base, xg2_base, minuto_gioco,
            gol_casa, gol_trasf, rossi_casa, rossi_trasf
        )
        mc_1, mc_x, mc_2, mc_u, mc_o = probabilita_poisson_esatta(
            xg1_live, xg2_live, gol_casa, gol_trasf, linea_target_ou
        )
        delta_ah  = ah_cur  - ah_op
        delta_tot = tot_cur - tot_op

    # ── QUOTE FAIR ───────────────────────────────────────────────────────────
    st.header(f"⚖️ Quote Fair  ·  {minuto_gioco}' | {gol_casa}–{gol_trasf}")

    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa",      f"@{calcola_quota_reale(mc_1):.2f}", f"{mc_1:.1%}")
    cx.metric("X — Pareggio",  f"@{calcola_quota_reale(mc_x):.2f}", f"{mc_x:.1%}")
    c2.metric("2 — Trasf.",    f"@{calcola_quota_reale(mc_2):.2f}", f"{mc_2:.1%}")

    cu, co = st.columns(2)
    cu.metric(f"Under {linea_target_ou}", f"@{calcola_quota_reale(mc_u):.2f}", f"{mc_u:.1%}")
    co.metric(f"Over  {linea_target_ou}", f"@{calcola_quota_reale(mc_o):.2f}", f"{mc_o:.1%}")

    st.divider()

    # ── SEGNALI (FIX BUG 4) ──────────────────────────────────────────────────
    st.header("🎯 Segnali Exchange")

    if minuto_gioco >= 85:
        st.error("🛑 Fine partita — spread enormi, non entrare.")
        st.stop()

    # Soglie dinamiche: crescono con il tempo rimasto
    frac_giocata = minuto_gioco / 90.0
    soglia_1x2   = 0.45 + 0.15 * frac_giocata   # 0.45 (kick-off) → 0.60 (85')
    soglia_ou    = 0.52 + 0.13 * frac_giocata   # 0.52 → 0.65

    gol_attuali  = gol_casa + gol_trasf
    gol_mancanti = linea_target_ou - gol_attuali
    segnali      = False

    # ── 1X2 ──────────────────────────────────────────────────────────────────
    if delta_ah <= -0.25:
        if mc_1 > soglia_1x2 and (mc_1 - soglia_1x2 > 0.05 or minuto_gioco < 60):
            q_fair   = calcola_quota_reale(mc_1)
            q_target = q_fair * 1.05
            stake    = calcola_stake_kelly(mc_1, q_target, cassa)
            if stake > 0:
                st.success(
                    f"🟢 **PUNTA 1 — CASA**\n\n"
                    f"Prob. modello: **{mc_1:.1%}** · Quota fair: **@{q_fair:.2f}** · "
                    f"Entra solo se trovi **> @{q_target:.2f}**"
                )
                st.write(f"Stake ½-Kelly: **€ {stake:.2f}**")
                segnali = True
            else:
                st.warning(f"⚠️ CASA segnalata dal mercato ma edge insufficiente (@{q_fair:.2f}). No bet.")
        else:
            st.info(
                f"Mercato → CASA, ma modello: **{mc_1:.1%}** "
                f"(soglia: {soglia_1x2:.0%}). Valore non confermato."
            )

    elif delta_ah >= 0.25:
        if mc_2 > soglia_1x2 and (mc_2 - soglia_1x2 > 0.05 or minuto_gioco < 60):
            q_fair   = calcola_quota_reale(mc_2)
            q_target = q_fair * 1.05
            stake    = calcola_stake_kelly(mc_2, q_target, cassa)
            if stake > 0:
                st.success(
                    f"🟢 **PUNTA 2 — TRASFERTA**\n\n"
                    f"Prob. modello: **{mc_2:.1%}** · Quota fair: **@{q_fair:.2f}** · "
                    f"Entra solo se trovi **> @{q_target:.2f}**"
                )
                st.write(f"Stake ½-Kelly: **€ {stake:.2f}**")
                segnali = True
            else:
                st.warning(f"⚠️ TRASF. segnalata dal mercato ma edge insufficiente (@{q_fair:.2f}). No bet.")
        else:
            st.info(
                f"Mercato → TRASF., ma modello: **{mc_2:.1%}** "
                f"(soglia: {soglia_1x2:.0%}). Valore non confermato."
            )

    # ── OVER / UNDER ─────────────────────────────────────────────────────────
    if delta_tot >= 0.25 and gol_attuali < linea_target_ou:
        if minuto_gioco >= 75 and gol_mancanti >= 2:
            st.info(
                f"⏱️ Linea salita ma al {minuto_gioco}' mancano ancora "
                f"{gol_mancanti:.0f} gol ({mc_o:.1%}). Troppo tardi."
            )
        elif mc_o > soglia_ou:
            q_fair   = calcola_quota_reale(mc_o)
            q_target = q_fair * 1.05
            stake    = calcola_stake_kelly(mc_o, q_target, cassa)
            if stake > 0:
                st.warning(
                    f"🔥 **OVER {linea_target_ou}**\n\n"
                    f"Prob. modello: **{mc_o:.1%}** · Quota fair: **@{q_fair:.2f}** · "
                    f"Entra solo se trovi **> @{q_target:.2f}**"
                )
                st.write(f"Stake ½-Kelly: **€ {stake:.2f}**")
                segnali = True
            else:
                st.info(f"OVER segnalato ma edge insufficiente. No bet.")
        else:
            st.info(
                f"Mercato → OVER ma modello: **{mc_o:.1%}** "
                f"(soglia: {soglia_ou:.0%}). No bet."
            )

    elif delta_tot <= -0.25 and gol_attuali < linea_target_ou:
        if mc_u > soglia_ou:
            q_fair   = calcola_quota_reale(mc_u)
            q_target = q_fair * 1.05
            stake    = calcola_stake_kelly(mc_u, q_target, cassa)
            if stake > 0:
                st.info(
                    f"🧊 **UNDER {linea_target_ou}**\n\n"
                    f"Prob. modello: **{mc_u:.1%}** · Quota fair: **@{q_fair:.2f}** · "
                    f"Entra solo se trovi **> @{q_target:.2f}**"
                )
                st.write(f"Stake ½-Kelly: **€ {stake:.2f}**")
                segnali = True
            else:
                st.info(f"UNDER segnalato ma edge insufficiente. No bet.")
        else:
            st.info(
                f"Mercato → UNDER ma modello: **{mc_u:.1%}** "
                f"(soglia: {soglia_ou:.0%}). No bet."
            )

    if not segnali:
        msg = (
            "⚖️ **NO BET** — Linee stabili, nessun movimento da sfruttare."
            if delta_ah == 0 and delta_tot == 0 else
            "⚖️ **NO BET** — Movimento di linea non confermato dal modello."
        )
        st.error(msg)

    # ── DEBUG ─────────────────────────────────────────────────────────────────
    with st.expander("🔍 Dettaglio calcoli interni"):
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Probabilità modello**")
            st.write(f"P(1) = {mc_1:.4f}")
            st.write(f"P(X) = {mc_x:.4f}")
            st.write(f"P(2) = {mc_2:.4f}")
            st.write(f"P(U{linea_target_ou}) = {mc_u:.4f}")
            st.write(f"P(O{linea_target_ou}) = {mc_o:.4f}")
        with c_right:
            st.markdown("**Soglie e parametri**")
            st.write(f"Soglia 1X2 = {soglia_1x2:.3f}")
            st.write(f"Soglia U/O = {soglia_ou:.3f}")
            st.write(f"Δ AH  = {delta_ah:+.2f}")
            st.write(f"Δ Tot = {delta_tot:+.2f}")
            st.write(f"λ casa  (rim.) = {xg1_live:.4f}")
            st.write(f"λ trasf (rim.) = {xg2_live:.4f}")
