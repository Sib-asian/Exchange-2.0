"""Debug rapido: 5000 sim per identificare gli errori prima del run completo."""
import math, random, sys, time
from collections import defaultdict

random.seed(42)

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
    if mu <= 0: return [1.0]
    max_k = max(20, int(mu + 6.0 * math.sqrt(max(mu, 1.0)) + 10))
    p0 = math.exp(-mu)
    pmfs, cumsum, k, p = [p0], p0, 0, p0
    while cumsum < (1.0 - tail_mass) and k < max_k:
        k += 1; p = p * mu / k; pmfs.append(p); cumsum += p
    return [x / cumsum for x in pmfs]

def calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto):
    frac_giocata = minuto / 90.0
    frac_rimasta = max(0.005, 1.0 - frac_giocata)
    w_cur = min(0.90, 0.65 + 0.20 * frac_giocata)
    w_op  = 1.0 - w_cur
    if abs(ah_cur - ah_op) < 1e-6 and abs(tot_cur - tot_op) < 1e-6:
        ah_bayes  = float(ah_cur)
        tot_bayes = max(0.2, float(tot_cur))
    else:
        ah_bayes  = (ah_op * frac_rimasta) * w_op + ah_cur * w_cur
        tot_bayes = max(0.2, (tot_op * frac_rimasta) * w_op + tot_cur * w_cur)
    eps = 1e-6

    def _ah_ev_half(mh, ma, h):
        ev = dc_norm = 0.0
        for i, pi in enumerate(_poisson_pmf_norm(mh)):
            if pi < 1e-18: continue
            for j, pj in enumerate(_poisson_pmf_norm(ma)):
                if pj < 1e-18: continue
                w = pi * pj * dixon_coles_tau(i, j, mh, ma)
                dc_norm += w
                s = (i - j) + h
                if s > 0: ev += w
                elif s < 0: ev -= w
        return ev / dc_norm if dc_norm > 0 else 0.0

    def _ah_ev(mh, ma, ah):
        ah2 = float(ah) * 2.0
        if abs(ah2 - round(ah2)) < 1e-9: return _ah_ev_half(mh, ma, float(ah))
        h1 = math.floor(ah2) / 2.0
        return 0.5 * _ah_ev_half(mh, ma, h1) + 0.5 * _ah_ev_half(mh, ma, h1 + 0.5)

    def _ev(delta):
        lh = max(eps, 0.5 * (tot_bayes + delta))
        la = max(eps, 0.5 * (tot_bayes - delta))
        return _ah_ev(lh, la, ah_bayes)

    lo, hi = -tot_bayes + eps, tot_bayes - eps
    ev_lo, ev_hi = _ev(lo), _ev(hi)
    if ev_lo == 0.0: delta_star = lo
    elif ev_hi == 0.0: delta_star = hi
    elif ev_lo * ev_hi > 0:
        delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
    else:
        inc = ev_hi > ev_lo
        dl, dr = lo, hi
        for _ in range(52):
            m = 0.5 * (dl + dr); em = _ev(m)
            if abs(em) < 1e-9 or (dr - dl) < 1e-12: break
            if inc:
                if em > 0: dr = m
                else: dl = m
            else:
                if em > 0: dl = m
                else: dr = m
        delta_star = 0.5 * (dl + dr)
    return max(eps, 0.5*(tot_bayes+delta_star)), max(eps, 0.5*(tot_bayes-delta_star))

def time_decay_dinamico(xg_casa, xg_trasf, minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf):
    if minuto >= 90: return 0.001, 0.001
    xg_c, xg_t = float(xg_casa), float(xg_trasf)
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat = abs(diff) / (1.5 + abs(diff))
        minute_scale = max(0.30, 1.20 - 1.10 * (minuto / 90.0))
        residual = min(0.08, 0.07 * sat) * minute_scale
        if diff < 0: xg_c *= (1.0 + residual); xg_t *= (1.0 - residual)
        else: xg_t *= (1.0 + residual); xg_c *= (1.0 - residual)
    _RED_DECAY = [1.000, 0.680, 0.578, 0.532, 0.500]
    _RED_BOOST = [1.000, 1.280, 1.434, 1.520, 1.566]
    if rossi_casa > 0:
        idx = min(rossi_casa, 4); xg_c *= _RED_DECAY[idx]; xg_t *= _RED_BOOST[idx]
    if rossi_trasf > 0:
        idx = min(rossi_trasf, 4); xg_t *= _RED_DECAY[idx] * 0.95; xg_c *= _RED_BOOST[idx] * 1.04
    return max(0.001, xg_c), max(0.001, xg_t)

def calcola_tutto(mu_c_rem, mu_t_rem, gol_casa, gol_trasf, linea_ou, tot_cur, minuto, shot_dom=0.0):
    mu_c_rem = max(1e-9, float(mu_c_rem)); mu_t_rem = max(1e-9, float(mu_t_rem))
    rho = rho_dinamico(tot_cur, minuto, shot_dom, gol_casa + gol_trasf)
    geom_mu = math.sqrt(mu_c_rem * mu_t_rem)
    lambda0 = max(0.0, min(rho * geom_mu, 0.75 * min(mu_c_rem, mu_t_rem)))
    mu_c_ind = max(1e-9, mu_c_rem - lambda0); mu_t_ind = max(1e-9, mu_t_rem - lambda0)
    pmf_c = _poisson_pmf_norm(mu_c_ind); pmf_t = _poisson_pmf_norm(mu_t_ind)
    pmf_z = _poisson_pmf_norm(lambda0)
    joint_ind = {}; dc_sum = 0.0
    for i, pi in enumerate(pmf_c):
        if pi < 1e-16: continue
        for j, pj in enumerate(pmf_t):
            if pj < 1e-16: continue
            tau = dixon_coles_tau(i, j, mu_c_ind, mu_t_ind)
            val = pi * pj * tau; joint_ind[(i, j)] = val; dc_sum += val
    if dc_sum > 0: joint_ind = {k: v/dc_sum for k,v in joint_ind.items()}
    full = {}
    for (i,j), pij in joint_ind.items():
        for z, pz in enumerate(pmf_z):
            if pz < 1e-16: continue
            a, b = i+z, j+z; full[(a,b)] = full.get((a,b), 0.0) + pij*pz
    fj_sum = sum(full.values())
    if fj_sum > 0: full = {k: v/fj_sum for k,v in full.items()}
    p1 = px = p2 = 0.0
    for (i,j), pij in joint_ind.items():
        diff = (gol_casa+i) - (gol_trasf+j)
        if diff > 0: p1 += pij
        elif diff < 0: p2 += pij
        else: px += pij
    s12x = p1+px+p2
    if s12x > 0: p1/=s12x; px/=s12x; p2/=s12x
    S = gol_casa + gol_trasf; line4 = round(linea_ou * 4)
    if line4 % 2 != 0:
        h_low = (line4-1)/4.0; h_high = (line4+1)/4.0
        p_u_low  = sum(p for (a,b),p in full.items() if S+a+b < h_low)
        p_u_high = sum(p for (a,b),p in full.items() if S+a+b < h_high)
        p_under  = 0.5*(p_u_low + p_u_high)
    else:
        p_under = sum(p for (a,b),p in full.items() if S+a+b < linea_ou)
    p_under = min(max(p_under, 0.0), 1.0); p_over = 1.0 - p_under
    if gol_casa > 0 and gol_trasf > 0: p_btts = 1.0
    elif gol_casa > 0: p_btts = max(0.0, sum(p for (a,b),p in full.items() if b > 0))
    elif gol_trasf > 0: p_btts = max(0.0, sum(p for (a,b),p in full.items() if a > 0))
    else: p_btts = max(0.0, sum(p for (a,b),p in full.items() if a > 0 and b > 0))
    p_btts = min(1.0, p_btts)
    cs_final = {}
    for (a,b), p in full.items():
        key = (gol_casa+a, gol_trasf+b); cs_final[key] = cs_final.get(key, 0.0) + p
    return p1, px, p2, p_under, p_over, p_btts, cs_final, rho, full

# ── 5000 sim di debug con output verboso ─────────────────────────────────────
errors = defaultdict(list)
N = 5000
t0 = time.time()

for sim_idx in range(N):
    minuto      = random.randint(0, 89)
    max_gol     = max(1, int(minuto / 20))
    gol_casa    = random.randint(0, min(max_gol, 5))
    gol_trasf   = random.randint(0, min(max_gol, 5))
    rossi_casa  = random.choices([0,1,2], weights=[90,9,1])[0]
    rossi_trasf = random.choices([0,1,2], weights=[90,9,1])[0]
    ah_op    = random.choice([-2.0,-1.5,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])
    tot_op   = random.choice([2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0])
    ah_cur   = ah_op  + random.uniform(-0.75, 0.75)
    tot_cur  = max(0.3, tot_op + random.uniform(-1.0, 1.0))
    linea_ou = random.choice([0.5,1.5,1.75,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.5])
    shot_dom = random.uniform(0.0, 0.8)

    ctx = (f"sim#{sim_idx} min={minuto} {gol_casa}-{gol_trasf} r={rossi_casa}/{rossi_trasf} "
           f"ah_op={ah_op} tot_op={tot_op} ah_cur={ah_cur:.2f} tot_cur={tot_cur:.2f} L={linea_ou}")

    try:
        xg1, xg2 = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto)
        xg1, xg2 = time_decay_dinamico(xg1, xg2, minuto, gol_casa, gol_trasf, rossi_casa, rossi_trasf)
        p1, px, p2, p_u, p_o, p_btts, cs_final, rho, full = calcola_tutto(
            xg1, xg2, gol_casa, gol_trasf, linea_ou, tot_cur, minuto, shot_dom)
    except Exception as e:
        errors["CRASH"].append(f"{ctx} | {e}")
        continue

    TOLL = 1e-4

    if abs(p1+px+p2 - 1.0) > TOLL:
        errors["1X2_norm"].append(f"sum={p1+px+p2:.7f} | {ctx}")
    if abs(p_u+p_o - 1.0) > TOLL:
        errors["OU_norm"].append(f"sum={p_u+p_o:.7f} | {ctx}")

    cs_sum = sum(cs_final.values())
    if abs(cs_sum - 1.0) > 1e-3:
        errors["CS_norm"].append(f"cs_sum={cs_sum:.6f} | {ctx}")

    # CS impossibili
    for (fc,ft), prob in cs_final.items():
        if (fc < gol_casa or ft < gol_trasf) and prob > TOLL:
            errors["CS_impossible"].append(f"CS({fc}-{ft}) impossibile con {gol_casa}-{gol_trasf}, P={prob:.5f} | {ctx}")
            break

    # CS coerenza con 1X2
    cs_p1 = sum(p for (fc,ft),p in cs_final.items() if fc > ft)
    cs_px = sum(p for (fc,ft),p in cs_final.items() if fc == ft)
    cs_p2 = sum(p for (fc,ft),p in cs_final.items() if fc < ft)
    if abs(cs_p1-p1) > 5e-4:
        errors["CS_1X2_mismatch"].append(f"CS_P1={cs_p1:.5f} P1={p1:.5f} | {ctx}")
    if abs(cs_px-px) > 5e-4:
        errors["CS_1X2_mismatch"].append(f"CS_PX={cs_px:.5f} PX={px:.5f} | {ctx}")

    # BTTS settled
    if gol_casa > 0 and gol_trasf > 0 and abs(p_btts - 1.0) > TOLL:
        errors["BTTS_settled"].append(f"BTTS={p_btts:.5f} | {ctx}")

    # Over settled
    gol_tot = gol_casa + gol_trasf
    if gol_tot >= linea_ou and p_o < 1.0 - 1e-3:
        errors["Over_settled"].append(f"P(O)={p_o:.5f} gol={gol_tot} >= L={linea_ou} | {ctx}")

    # AH monotony
    prev = 1.1
    for lv in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        win = push = 0.0
        for (a,b),p in full.items():
            d = (a-b)+lv
            if d > 1e-9: win += p
            elif abs(d) <= 1e-9: push += p
        p_eff = win + 0.5*push
        if p_eff > prev + 1e-3:
            errors["AH_monotony"].append(f"lv={lv} p_eff={p_eff:.4f} > prev={prev:.4f} | {ctx}")
        prev = p_eff

    # Quarter line monotonicità
    S = gol_casa + gol_trasf
    pu25  = sum(p for (a,b),p in full.items() if S+a+b < 2.5)
    pu275 = 0.5*sum(p for (a,b),p in full.items() if S+a+b < 2.5) + \
            0.5*sum(p for (a,b),p in full.items() if S+a+b < 3.0)
    pu30  = sum(p for (a,b),p in full.items() if S+a+b < 3.0)
    if not (pu25 - 1e-4 <= pu275 <= pu30 + 1e-4):
        errors["QL_monotony"].append(f"U2.5={pu25:.4f} U2.75={pu275:.4f} U3.0={pu30:.4f} | {ctx}")

print(f"\n{N} sim in {time.time()-t0:.1f}s")
print(f"Errori totali: {sum(len(v) for v in errors.values())}")
for etype, elist in sorted(errors.items(), key=lambda x: -len(x[1])):
    print(f"\n[{len(elist):>5}] {etype}")
    for msg in elist[:8]:
        print(f"  → {msg}")
    if len(elist) > 8:
        print(f"  ... e altri {len(elist)-8}")
