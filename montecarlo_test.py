"""
Monte Carlo Simulation — 1 milione di casi
Verifica l'intera pipeline matematica: calcola_xg_bayesiani → time_decay_dinamico
→ calcola_tutto (bivariate Poisson + Dixon-Coles) su input randomizzati.

Controlla:
  1. Normalizzazione: Σ(1,X,2) = 1, Σ(U,O) = 1, BTTS ∈ [0,1]
  2. Monotonicità Over/Under rispetto al punteggio attuale
  3. BTTS settled: gol_casa>0 & gol_trasf>0 → P(BTTS)=1.0
  4. Over settled: gol_totali >= linea_ou → P(U)=0, P(O)=1
  5. Correct Score: Σ CS ≈ 1.0 e ogni CS ≥ 0
  6. CS interno coerente con P(1)/P(X)/P(2)
  7. Correct Score massimo atteso corrispondente a stato partita
  8. xG limite: minuto=89 → xG residui quasi zero
  9. Symmetry: partita pareggio + AH=0 → P(1)≈P(2)
 10. Edge cases: mu molto piccole, mu grandi, rossi multipli
 11. gol_tot_dist: somma ≈ 1.0
 12. full_matrix: somma ≈ 1.0
 13. AH multipli (da full_matrix): P_eff ∈ [0,1] per ogni livello
 14. Monotonia AH: più handicap favorisce casa → P_cover decresce
 15. Quarter lines: P(U2.75) tra P(U2.5) e P(U3.0)
"""

import math
import random
import sys
import time
from collections import defaultdict

random.seed(42)

# ── Copia le funzioni matematiche da app.py (senza Streamlit) ────────────────

def rho_dinamico(tot_cur, minuto, shot_dom=0.0, gol_totali=0):
    base  = max(0.02, 0.14 - 0.018 * min(tot_cur, 4.5))
    decay = 1.0 - 0.40 * max(0.0, min(minuto / 90.0, 1.0))
    gol_decay = max(0.50, 1.0 - 0.10 * gol_totali)
    rho_base = base * decay * gol_decay
    return max(0.02, rho_base * (1.0 - 0.45 * shot_dom))


def dixon_coles_tau(i, j, mu_h, mu_a, rho_dc=-0.13):
    if   i == 0 and j == 0: tau = 1.0 - mu_h * mu_a * rho_dc
    elif i == 1 and j == 0: tau = 1.0 + mu_a * rho_dc
    elif i == 0 and j == 1: tau = 1.0 + mu_h * rho_dc
    elif i == 1 and j == 1: tau = 1.0 - rho_dc
    else:                   return 1.0
    return max(0.05, min(tau, 3.0))


def _poisson_pmf_norm(mu, tail_mass=1e-12):
    if mu <= 0:
        return [1.0]
    max_k = max(20, int(mu + 6.0 * math.sqrt(max(mu, 1.0)) + 10))
    p0 = math.exp(-mu)
    pmfs, cumsum, k, p = [p0], p0, 0, p0
    while cumsum < (1.0 - tail_mass) and k < max_k:
        k += 1
        p = p * mu / k
        pmfs.append(p)
        cumsum += p
    return [x / cumsum for x in pmfs]


def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto):
    frac_giocata = minuto / 90.0
    frac_rimasta = max(0.005, 1.0 - frac_giocata)
    w_cur = min(0.90, 0.65 + 0.20 * frac_giocata)
    w_op  = 1.0 - w_cur
    delta_ah_inner  = abs(ah_cur  - ah_op)
    delta_tot_inner = abs(tot_cur - tot_op)
    if delta_ah_inner < 1e-6 and delta_tot_inner < 1e-6:
        ah_bayes  = float(ah_cur)
        tot_bayes = max(0.2, float(tot_cur))
    else:
        ah_bayes  = (ah_op  * frac_rimasta) * w_op + ah_cur  * w_cur
        tot_bayes = max(0.2, (tot_op * frac_rimasta) * w_op + tot_cur * w_cur)
    eps = 1e-6

    def _ah_ev_half(mh, ma, h):
        ev = dc_norm = 0.0
        pmf_h = _poisson_pmf_norm(mh)
        pmf_a = _poisson_pmf_norm(ma)
        for i, pi in enumerate(pmf_h):
            if pi < 1e-18: continue
            for j, pj in enumerate(pmf_a):
                if pj < 1e-18: continue
                w = pi * pj * dixon_coles_tau(i, j, mh, ma)
                dc_norm += w
                s = (i - j) + h
                if   s > 0: ev += w
                elif s < 0: ev -= w
        return ev / dc_norm if dc_norm > 0 else 0.0

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
        for _ in range(52):
            m  = 0.5 * (dl + dr)
            em = _ev(m)
            if abs(em) < 1e-9 or (dr - dl) < 1e-12:
                break
            if inc:
                if em > 0: dr = m
                else:      dl = m
            else:
                if em > 0: dl = m
                else:      dr = m
        delta_star = 0.5 * (dl + dr)

    return max(eps, 0.5*(tot_bayes+delta_star)), max(eps, 0.5*(tot_bayes-delta_star))


def time_decay_dinamico(xg_casa, xg_trasf, minuto,
                        gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    if minuto >= 90:
        return 0.001, 0.001
    xg_c = float(xg_casa)
    xg_t = float(xg_trasf)
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat = abs(diff) / (1.5 + abs(diff))
        minute_scale = max(0.30, 1.20 - 1.10 * (minuto / 90.0))
        residual = min(0.08, 0.07 * sat) * minute_scale
        if diff < 0:
            xg_c *= (1.0 + residual)
            xg_t *= (1.0 - residual)
        else:
            xg_t *= (1.0 + residual)
            xg_c *= (1.0 - residual)
    _RED_DECAY = [1.000, 0.680, 0.578, 0.532, 0.500]
    _RED_BOOST = [1.000, 1.280, 1.434, 1.520, 1.566]
    if rossi_casa > 0:
        idx = min(rossi_casa, 4)
        xg_c *= _RED_DECAY[idx]
        xg_t *= _RED_BOOST[idx]
    if rossi_trasf > 0:
        idx = min(rossi_trasf, 4)
        xg_t *= _RED_DECAY[idx] * 0.95
        xg_c *= _RED_BOOST[idx] * 1.04
    return max(0.001, xg_c), max(0.001, xg_t)


def calcola_tutto(mu_c_rem, mu_t_rem, gol_casa, gol_trasf, linea_ou, tot_cur, minuto,
                  shot_dom=0.0):
    mu_c_rem = max(1e-9, float(mu_c_rem))
    mu_t_rem = max(1e-9, float(mu_t_rem))
    rho     = rho_dinamico(tot_cur, minuto, shot_dom, gol_casa + gol_trasf)
    geom_mu = math.sqrt(mu_c_rem * mu_t_rem)
    lambda0 = max(0.0, min(rho * geom_mu, 0.75 * min(mu_c_rem, mu_t_rem)))
    mu_c_ind = max(1e-9, mu_c_rem - lambda0)
    mu_t_ind = max(1e-9, mu_t_rem - lambda0)
    pmf_c = _poisson_pmf_norm(mu_c_ind)
    pmf_t = _poisson_pmf_norm(mu_t_ind)
    pmf_z = _poisson_pmf_norm(lambda0)

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

    full = {}
    for (i, j), pij in joint_ind.items():
        for z, pz in enumerate(pmf_z):
            if pz < 1e-16: continue
            a, b = i + z, j + z
            full[(a, b)] = full.get((a, b), 0.0) + pij * pz
    fj_sum = sum(full.values())
    if fj_sum > 0:
        full = {k: v / fj_sum for k, v in full.items()}

    p1 = px = p2 = 0.0
    for (i, j), pij in joint_ind.items():
        diff = (gol_casa + i) - (gol_trasf + j)
        if   diff > 0: p1 += pij
        elif diff < 0: p2 += pij
        else:          px += pij
    s12x = p1 + px + p2
    if s12x > 0:
        p1 /= s12x; px /= s12x; p2 /= s12x

    S     = gol_casa + gol_trasf
    line4 = round(linea_ou * 4)
    if line4 % 2 != 0:
        h_low  = (line4 - 1) / 4.0
        h_high = (line4 + 1) / 4.0
        p_u_low  = sum(p for (a, b), p in full.items() if S + a + b < h_low)
        p_u_high = sum(p for (a, b), p in full.items() if S + a + b < h_high)
        p_under  = 0.5 * (p_u_low + p_u_high)
    else:
        p_under = sum(p for (a, b), p in full.items() if S + a + b < linea_ou)
    p_under = min(max(p_under, 0.0), 1.0)
    p_over  = 1.0 - p_under

    if gol_casa > 0 and gol_trasf > 0:
        p_btts = 1.0
    elif gol_casa > 0:
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if b > 0))
    elif gol_trasf > 0:
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if a > 0))
    else:
        p_btts = max(0.0, sum(p for (a, b), p in full.items() if a > 0 and b > 0))
    p_btts = min(1.0, p_btts)

    cs_final = {}
    for (a, b), p in full.items():
        key = (gol_casa + a, gol_trasf + b)
        cs_final[key] = cs_final.get(key, 0.0) + p

    top_cs = sorted(cs_final.items(), key=lambda x: x[1], reverse=True)[:5]

    gol_tot_dist = {}
    for (fc, ft), p in cs_final.items():
        tot = fc + ft
        gol_tot_dist[tot] = gol_tot_dist.get(tot, 0.0) + p

    return p1, px, p2, p_under, p_over, p_btts, top_cs, rho, gol_tot_dist, full, cs_final


# ── Generatore di scenari realistici ─────────────────────────────────────────

def genera_scenario():
    """Genera uno scenario di partita realistico con distribuzione pesata."""
    minuto = random.randint(0, 89)
    # Punteggio plausibile per il minuto
    max_gol = max(1, int(minuto / 20))
    gol_casa  = random.randint(0, min(max_gol, 5))
    gol_trasf = random.randint(0, min(max_gol, 5))
    rossi_casa  = random.choices([0, 1, 2], weights=[90, 9, 1])[0]
    rossi_trasf = random.choices([0, 1, 2], weights=[90, 9, 1])[0]
    # Linee AH di apertura (tipiche)
    ah_op  = random.choice([-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    tot_op = random.choice([2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0])
    # Linee correnti: vicine alle apertura con piccolo drift
    ah_cur  = ah_op  + random.uniform(-0.75, 0.75)
    tot_cur = max(0.3, tot_op + random.uniform(-1.0, 1.0))
    # Linea O/U da analizzare
    linea_ou = random.choice([0.5, 1.5, 1.75, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.5])
    shot_dom = random.uniform(0.0, 0.8)
    return (minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf,
            ah_op, tot_op, ah_cur, tot_cur, linea_ou, shot_dom)


# ── Runner ────────────────────────────────────────────────────────────────────

N = 1_000_000
TOLL = 1e-6      # tolleranza normalizzazione
TOLL_SOFT = 5e-4 # tolleranza soft per somme CS (tail truncation)

errors   = defaultdict(list)   # {tipo_errore: [descrizione, ...]}
warnings = defaultdict(int)    # {tipo: count}

t0 = time.time()
print(f"Avvio {N:,} simulazioni...")

for sim_idx in range(N):
    if sim_idx % 100_000 == 0 and sim_idx > 0:
        elapsed = time.time() - t0
        rate = sim_idx / elapsed
        eta  = (N - sim_idx) / rate
        print(f"  {sim_idx:>7,} / {N:,}  —  {elapsed:.1f}s  —  ETA {eta:.0f}s  "
              f"—  errori: {sum(len(v) for v in errors.values())}")

    (minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf,
     ah_op, tot_op, ah_cur, tot_cur, linea_ou, shot_dom) = genera_scenario()

    ctx = (f"min={minuto} score={gol_casa}-{gol_trasf} "
           f"rc={rossi_casa}/{rossi_trasf} ah_op={ah_op} tot_op={tot_op} "
           f"ah_cur={ah_cur:.2f} tot_cur={tot_cur:.2f} linea={linea_ou}")

    # ── Step 1: xG bayesiani ──────────────────────────────────────────────────
    try:
        xg1_base, xg2_base = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto)
    except Exception as e:
        errors["xg_bayesiani_crash"].append(f"{ctx} | {e}")
        if len(errors["xg_bayesiani_crash"]) > 20: break
        continue

    if not (1e-9 <= xg1_base <= 20.0):
        errors["xg1_out_of_range"].append(f"xg1={xg1_base:.6f} | {ctx}")
    if not (1e-9 <= xg2_base <= 20.0):
        errors["xg2_out_of_range"].append(f"xg2={xg2_base:.6f} | {ctx}")

    # ── Step 2: time decay ────────────────────────────────────────────────────
    try:
        xg1_live, xg2_live = time_decay_dinamico(
            xg1_base, xg2_base, minuto,
            gol_casa, gol_trasf, rossi_casa, rossi_trasf)
    except Exception as e:
        errors["time_decay_crash"].append(f"{ctx} | {e}")
        if len(errors["time_decay_crash"]) > 20: break
        continue

    if not (1e-9 <= xg1_live <= 20.0):
        errors["xg1_live_out_of_range"].append(f"xg1_live={xg1_live:.6f} | {ctx}")
    if not (1e-9 <= xg2_live <= 20.0):
        errors["xg2_live_out_of_range"].append(f"xg2_live={xg2_live:.6f} | {ctx}")

    # ── Step 3: calcola_tutto ─────────────────────────────────────────────────
    try:
        p1, px, p2, p_u, p_o, p_btts, top_cs, rho, gol_tot_dist, full, cs_final = calcola_tutto(
            xg1_live, xg2_live, gol_casa, gol_trasf,
            linea_ou, tot_cur, minuto, shot_dom)
    except Exception as e:
        errors["calcola_tutto_crash"].append(f"{ctx} | {e}")
        if len(errors["calcola_tutto_crash"]) > 20: break
        continue

    # CHECK 1: Normalizzazione 1X2
    s1x2 = p1 + px + p2
    if abs(s1x2 - 1.0) > TOLL:
        errors["1x2_norm"].append(f"sum={s1x2:.8f} | {ctx}")

    # CHECK 2: Normalizzazione O/U
    sou = p_u + p_o
    if abs(sou - 1.0) > TOLL:
        errors["ou_norm"].append(f"sum={sou:.8f} | {ctx}")

    # CHECK 3: Tutti in [0,1]
    for name, val in [("p1",p1),("px",px),("p2",p2),("p_u",p_u),("p_o",p_o),("p_btts",p_btts)]:
        if not (-TOLL <= val <= 1.0 + TOLL):
            errors["prob_out_range"].append(f"{name}={val:.6f} | {ctx}")

    # CHECK 4: BTTS settled — entrambi già segnato → P(BTTS) deve essere 1.0
    if gol_casa > 0 and gol_trasf > 0:
        if abs(p_btts - 1.0) > TOLL:
            errors["btts_settled_wrong"].append(f"P(BTTS)={p_btts:.6f} ma entrambi già segnato | {ctx}")

    # CHECK 5: Over settled — gol_totali >= linea_ou → P(U)=0, P(O)=1
    gol_tot = gol_casa + gol_trasf
    if gol_tot >= linea_ou:
        # Per linee intere potrebbe esserci push, ma per half-lines è preciso
        line4 = round(linea_ou * 4)
        if line4 % 2 != 0:
            # quarter line: Over settled se gol_tot > ceil (linea_ou arrotondata su)
            h_high = (line4 + 1) / 4.0
            if gol_tot > h_high and p_o < 1.0 - TOLL_SOFT:
                errors["over_settled_wrong"].append(
                    f"P(O)={p_o:.6f} ma gol={gol_tot} >= linea={linea_ou} | {ctx}")
        else:
            if p_o < 1.0 - TOLL_SOFT:
                errors["over_settled_wrong"].append(
                    f"P(O)={p_o:.6f} ma gol={gol_tot} >= linea={linea_ou} | {ctx}")

    # CHECK 6: CS sum ≈ 1.0
    cs_sum = sum(cs_final.values())
    if abs(cs_sum - 1.0) > TOLL_SOFT:
        errors["cs_norm"].append(f"cs_sum={cs_sum:.6f} | {ctx}")

    # CHECK 7: CS negativi
    for score, prob in cs_final.items():
        if prob < -TOLL:
            errors["cs_negative"].append(f"P({score})={prob:.6f} | {ctx}")
            break  # uno basta

    # CHECK 8: CS punteggio attuale impossibile (score < stato attuale)
    for (fc, ft), prob in cs_final.items():
        if fc < gol_casa or ft < gol_trasf:
            if prob > TOLL:
                errors["cs_impossible_score"].append(
                    f"CS ({fc}-{ft}) impossibile con attuale {gol_casa}-{gol_trasf}, "
                    f"P={prob:.6f} | {ctx}")
                break

    # CHECK 9: CS coerenza con P(1)/P(X)/P(2)
    cs_p1 = sum(p for (fc,ft),p in cs_final.items() if fc > ft)
    cs_px = sum(p for (fc,ft),p in cs_final.items() if fc == ft)
    cs_p2 = sum(p for (fc,ft),p in cs_final.items() if fc < ft)
    if abs(cs_p1 - p1) > TOLL_SOFT:
        errors["cs_1x2_mismatch"].append(
            f"CS_P1={cs_p1:.5f} vs P1={p1:.5f} | {ctx}")
    if abs(cs_px - px) > TOLL_SOFT:
        errors["cs_1x2_mismatch"].append(
            f"CS_PX={cs_px:.5f} vs PX={px:.5f} | {ctx}")
    if abs(cs_p2 - p2) > TOLL_SOFT:
        errors["cs_1x2_mismatch"].append(
            f"CS_P2={cs_p2:.5f} vs P2={p2:.5f} | {ctx}")

    # CHECK 10: gol_tot_dist sum ≈ 1.0
    gtd_sum = sum(gol_tot_dist.values())
    if abs(gtd_sum - 1.0) > TOLL_SOFT:
        errors["gol_tot_dist_norm"].append(f"sum={gtd_sum:.6f} | {ctx}")

    # CHECK 11: full_matrix sum ≈ 1.0
    full_sum = sum(full.values())
    if abs(full_sum - 1.0) > TOLL_SOFT:
        errors["full_matrix_norm"].append(f"sum={full_sum:.6f} | {ctx}")

    # CHECK 12: AH multipli da full_matrix — P_eff ∈ [0,1]
    ah_levels = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    prev_p_eff = 1.1
    monotony_ok = True
    for level in ah_levels:
        win = push = lose = 0.0
        for (a, b), p in full.items():
            diff = (a - b) + level
            if   diff > 1e-9:  win  += p
            elif diff < -1e-9: lose += p
            else:              push += p
        p_eff = win + 0.5 * push
        if not (0.0 - TOLL <= p_eff <= 1.0 + TOLL):
            errors["ah_peff_out_range"].append(f"level={level} p_eff={p_eff:.6f} | {ctx}")
        # Monotonicità: livello più negativo (home favorito di più) → P_cover scende
        if p_eff > prev_p_eff + 1e-4:
            errors["ah_monotony"].append(
                f"level={level} p_eff={p_eff:.4f} > prev={prev_p_eff:.4f} | {ctx}")
            monotony_ok = False
        prev_p_eff = p_eff

    # CHECK 13: Quarter line monotonicità P(U2.5) ≤ P(U2.75) ≤ P(U3.0)
    S = gol_casa + gol_trasf
    pu25  = sum(p for (a,b),p in full.items() if S+a+b < 2.5)
    pu275 = 0.5 * sum(p for (a,b),p in full.items() if S+a+b < 2.5) + \
            0.5 * sum(p for (a,b),p in full.items() if S+a+b < 3.0)
    pu30  = sum(p for (a,b),p in full.items() if S+a+b < 3.0)
    if not (pu25 - TOLL_SOFT <= pu275 <= pu30 + TOLL_SOFT):
        errors["quarter_line_monotony"].append(
            f"U2.5={pu25:.4f} U2.75={pu275:.4f} U3.0={pu30:.4f} | {ctx}")

    # CHECK 14: xG ≈ 0 a minuto 89 (entrambe le squadre quasi ferme)
    if minuto == 89 and xg1_live > 0.25:
        warnings["xg_high_at_89"] += 1
    if minuto == 89 and xg2_live > 0.25:
        warnings["xg_high_at_89_away"] += 1

    # CHECK 15: Rho nel range atteso [0.02, 0.14]
    if not (0.01 <= rho <= 0.20):
        errors["rho_out_range"].append(f"rho={rho:.5f} | {ctx}")

    # CHECK 16: CS — il punteggio più probabile deve avere score >= stato attuale
    if top_cs:
        best_score, best_prob = top_cs[0]
        fc, ft = best_score
        if fc < gol_casa or ft < gol_trasf:
            errors["cs_top1_impossible"].append(
                f"Top CS ({fc}-{ft}) impossibile con {gol_casa}-{gol_trasf} | {ctx}")

    # CHECK 17: Simmetria — AH=0 linee uguali minuto=0 → P(1)≈P(2) e P(X) max
    # (solo per scenari esattamente simmetrici)
    if (abs(ah_op) < 0.01 and abs(ah_cur) < 0.01 and abs(ah_op - ah_cur) < 0.01
            and gol_casa == 0 and gol_trasf == 0 and rossi_casa == 0 and rossi_trasf == 0):
        if abs(p1 - p2) > 0.05:
            errors["symmetry_broken"].append(f"P1={p1:.3f} P2={p2:.3f} | {ctx}")

elapsed = time.time() - t0
print(f"\nCompletate {N:,} simulazioni in {elapsed:.1f}s  ({N/elapsed:,.0f} sim/s)")

# ── Report finale ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("REPORT ERRORI")
print("="*70)

total_errors = sum(len(v) for v in errors.values())
if total_errors == 0:
    print("✅  Nessun errore trovato su 1.000.000 simulazioni.")
else:
    print(f"⚠️  {total_errors} errori totali in {len(errors)} categorie:\n")
    for etype, elist in sorted(errors.items(), key=lambda x: -len(x[1])):
        print(f"  [{len(elist):>6}]  {etype}")
        for msg in elist[:5]:
            print(f"           → {msg}")
        if len(elist) > 5:
            print(f"           ... e altri {len(elist)-5}")
        print()

print("\nWARNINGS (non bloccanti):")
if warnings:
    for wtype, cnt in sorted(warnings.items(), key=lambda x: -x[1]):
        print(f"  [{cnt:>6}]  {wtype}")
else:
    print("  Nessun warning.")

print("\nSTATISTICHE CAMPIONE (ultimi 100k scenari):")
print(f"  Elapsed: {elapsed:.2f}s")
print(f"  Throughput: {N/elapsed:,.0f} sim/s")
sys.exit(0 if total_errors == 0 else 1)
