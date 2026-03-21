"""
Monte Carlo 1.000.000 simulazioni — versione ottimizzata con multiprocessing.
Tutti i check matematici con direzione corretta.
"""
import math, random, time, sys, os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ── Funzioni matematiche (copia da app.py) ────────────────────────────────────

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
    w_cur = min(0.90, 0.65 + 0.20 * frac_giocata); w_op = 1.0 - w_cur
    if abs(ah_cur - ah_op) < 1e-6 and abs(tot_cur - tot_op) < 1e-6:
        ah_bayes = float(ah_cur); tot_bayes = max(0.2, float(tot_cur))
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
                w = pi * pj * dixon_coles_tau(i, j, mh, ma); dc_norm += w
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
        lh = max(eps, 0.5*(tot_bayes+delta)); la = max(eps, 0.5*(tot_bayes-delta))
        return _ah_ev(lh, la, ah_bayes)

    lo, hi = -tot_bayes + eps, tot_bayes - eps
    ev_lo, ev_hi = _ev(lo), _ev(hi)
    if ev_lo == 0.0: delta_star = lo
    elif ev_hi == 0.0: delta_star = hi
    elif ev_lo * ev_hi > 0: delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
    else:
        inc = ev_hi > ev_lo; dl, dr = lo, hi
        for _ in range(52):
            m = 0.5*(dl+dr); em = _ev(m)
            if abs(em) < 1e-9 or (dr-dl) < 1e-12: break
            if inc:
                if em > 0: dr = m
                else: dl = m
            else:
                if em > 0: dl = m
                else: dr = m
        delta_star = 0.5*(dl+dr)
    return max(eps, 0.5*(tot_bayes+delta_star)), max(eps, 0.5*(tot_bayes-delta_star))

def time_decay_dinamico(xg_casa, xg_trasf, minuto, gol_casa, gol_trasf, rc, rt):
    if minuto >= 90: return 0.001, 0.001
    xg_c, xg_t = float(xg_casa), float(xg_trasf)
    diff = gol_casa - gol_trasf
    if diff != 0:
        sat = abs(diff)/(1.5+abs(diff))
        ms  = max(0.30, 1.20 - 1.10*(minuto/90.0))
        r   = min(0.08, 0.07*sat)*ms
        if diff < 0: xg_c *= (1+r); xg_t *= (1-r)
        else:        xg_t *= (1+r); xg_c *= (1-r)
    D = [1.000, 0.680, 0.578, 0.532, 0.500]
    B = [1.000, 1.280, 1.434, 1.520, 1.566]
    if rc > 0: i=min(rc,4); xg_c *= D[i]; xg_t *= B[i]
    if rt > 0: i=min(rt,4); xg_t *= D[i]*0.95; xg_c *= B[i]*1.04
    return max(0.001, xg_c), max(0.001, xg_t)

def calcola_tutto(mu_c, mu_t, gol_casa, gol_trasf, linea_ou, tot_cur, minuto, shot_dom=0.0):
    mu_c = max(1e-9, float(mu_c)); mu_t = max(1e-9, float(mu_t))
    rho = rho_dinamico(tot_cur, minuto, shot_dom, gol_casa+gol_trasf)
    geom_mu = math.sqrt(mu_c * mu_t)
    lam0 = max(0.0, min(rho*geom_mu, 0.75*min(mu_c, mu_t)))
    mc = max(1e-9, mu_c-lam0); mt = max(1e-9, mu_t-lam0)
    pmf_c = _poisson_pmf_norm(mc); pmf_t = _poisson_pmf_norm(mt); pmf_z = _poisson_pmf_norm(lam0)
    ji = {}; dcs = 0.0
    for i,pi in enumerate(pmf_c):
        if pi < 1e-16: continue
        for j,pj in enumerate(pmf_t):
            if pj < 1e-16: continue
            v = pi*pj*dixon_coles_tau(i,j,mc,mt); ji[(i,j)] = v; dcs += v
    if dcs > 0: ji = {k: v/dcs for k,v in ji.items()}
    full = {}
    for (i,j),pij in ji.items():
        for z,pz in enumerate(pmf_z):
            if pz < 1e-16: continue
            a,b = i+z,j+z; full[(a,b)] = full.get((a,b),0.0)+pij*pz
    fs = sum(full.values())
    if fs > 0: full = {k: v/fs for k,v in full.items()}
    p1=px=p2=0.0
    for (i,j),pij in ji.items():
        d=(gol_casa+i)-(gol_trasf+j)
        if d>0: p1+=pij
        elif d<0: p2+=pij
        else: px+=pij
    s=p1+px+p2
    if s>0: p1/=s; px/=s; p2/=s
    S=gol_casa+gol_trasf; l4=round(linea_ou*4)
    if l4%2!=0:
        hl=(l4-1)/4.0; hh=(l4+1)/4.0
        pu=0.5*sum(p for (a,b),p in full.items() if S+a+b<hl)+\
           0.5*sum(p for (a,b),p in full.items() if S+a+b<hh)
    else:
        pu=sum(p for (a,b),p in full.items() if S+a+b<linea_ou)
    pu=min(max(pu,0.0),1.0); po=1.0-pu
    if gol_casa>0 and gol_trasf>0: pb=1.0
    elif gol_casa>0: pb=max(0.0,sum(p for (a,b),p in full.items() if b>0))
    elif gol_trasf>0: pb=max(0.0,sum(p for (a,b),p in full.items() if a>0))
    else: pb=max(0.0,sum(p for (a,b),p in full.items() if a>0 and b>0))
    pb=min(1.0,pb)
    cs={}
    for (a,b),p in full.items():
        k=(gol_casa+a,gol_trasf+b); cs[k]=cs.get(k,0.0)+p
    return p1,px,p2,pu,po,pb,cs,rho,full

# ── Worker per multiprocessing ────────────────────────────────────────────────

def run_batch(args):
    """Esegui un batch di simulazioni e restituisci dizionario errori."""
    batch_id, n_batch, seed_offset = args
    rng = random.Random(seed_offset)
    errs = defaultdict(list)
    TOLL = 1e-4; TOLL_S = 5e-4

    for _ in range(n_batch):
        minuto   = rng.randint(0, 89)
        mg       = max(1, int(minuto/20))
        gc       = rng.randint(0, min(mg, 5))
        gt       = rng.randint(0, min(mg, 5))
        rc       = rng.choices([0,1,2], weights=[90,9,1])[0]
        rt       = rng.choices([0,1,2], weights=[90,9,1])[0]
        ah_op    = rng.choice([-2.0,-1.5,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])
        tot_op   = rng.choice([2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0])
        ah_cur   = ah_op  + rng.uniform(-0.75, 0.75)
        tot_cur  = max(0.3, tot_op + rng.uniform(-1.0, 1.0))
        linea_ou = rng.choice([0.5,1.5,1.75,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.5])
        sd       = rng.uniform(0.0, 0.8)
        ctx = f"min={minuto} {gc}-{gt} r={rc}/{rt} ah_op={ah_op} tot_op={tot_op} tot_cur={tot_cur:.2f} L={linea_ou}"

        try:
            xg1, xg2 = calcola_xg_bayesiani(ah_op, tot_op, ah_cur, tot_cur, minuto)
            xg1, xg2 = time_decay_dinamico(xg1, xg2, minuto, gc, gt, rc, rt)
            p1,px,p2,pu,po,pb,cs,rho,full = calcola_tutto(xg1,xg2,gc,gt,linea_ou,tot_cur,minuto,sd)
        except Exception as e:
            errs["CRASH"].append(f"{ctx} | {e}"); continue

        # 1. Normalizzazione 1X2
        if abs(p1+px+p2-1.0) > TOLL:
            errs["1X2_norm"].append(f"sum={p1+px+p2:.7f} | {ctx}")

        # 2. Normalizzazione O/U
        if abs(pu+po-1.0) > TOLL:
            errs["OU_norm"].append(f"sum={pu+po:.7f} | {ctx}")

        # 3. Valori in [0,1]
        for nm,v in [("p1",p1),("px",px),("p2",p2),("pu",pu),("po",po),("pb",pb)]:
            if not (-TOLL <= v <= 1+TOLL):
                errs["PROB_range"].append(f"{nm}={v:.6f} | {ctx}")

        # 4. BTTS settled
        if gc>0 and gt>0 and abs(pb-1.0)>TOLL:
            errs["BTTS_settled"].append(f"pb={pb:.5f} | {ctx}")

        # 5. Over settled
        gtot = gc+gt
        if gtot >= linea_ou:
            l4 = round(linea_ou*4)
            hh = (l4+1)/4.0 if l4%2!=0 else linea_ou
            if gtot > hh and po < 1.0-TOLL_S:
                errs["Over_settled"].append(f"P(O)={po:.5f} gol={gtot}>=L={linea_ou} | {ctx}")

        # 6. CS norm
        cs_sum = sum(cs.values())
        if abs(cs_sum-1.0) > TOLL_S:
            errs["CS_norm"].append(f"cs_sum={cs_sum:.6f} | {ctx}")

        # 7. CS punteggi impossibili (al di sotto dello stato attuale)
        for (fc,ft),p in cs.items():
            if (fc < gc or ft < gt) and p > TOLL:
                errs["CS_impossible"].append(f"CS({fc}-{ft}) < {gc}-{gt}, P={p:.5f} | {ctx}")
                break

        # 8. CS coerente con 1X2
        cp1 = sum(p for (f,t),p in cs.items() if f>t)
        cpx = sum(p for (f,t),p in cs.items() if f==t)
        cp2 = sum(p for (f,t),p in cs.items() if f<t)
        if abs(cp1-p1)>TOLL_S: errs["CS_1X2"].append(f"cp1={cp1:.5f} p1={p1:.5f} | {ctx}")
        if abs(cpx-px)>TOLL_S: errs["CS_1X2"].append(f"cpx={cpx:.5f} px={px:.5f} | {ctx}")
        if abs(cp2-p2)>TOLL_S: errs["CS_1X2"].append(f"cp2={cp2:.5f} p2={p2:.5f} | {ctx}")

        # 9. AH monotonicità CORRETTA: da -2.5 a +2.5 P(cover) deve AUMENTARE
        prev = -0.01
        for lv in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            win=push=0.0
            for (a,b),p in full.items():
                d=(a-b)+lv
                if d>1e-9: win+=p
                elif abs(d)<=1e-9: push+=p
            p_eff = win+0.5*push
            if p_eff < prev - 1e-3:  # deve AUMENTARE, non diminuire
                errs["AH_monotony"].append(f"lv={lv} p_eff={p_eff:.4f} < prev={prev:.4f} | {ctx}")
            prev = p_eff

        # 10. Quarter line: U2.5 <= U2.75 <= U3.0
        S = gc+gt
        pu25  = sum(p for (a,b),p in full.items() if S+a+b<2.5)
        pu275 = 0.5*sum(p for (a,b),p in full.items() if S+a+b<2.5)+\
                0.5*sum(p for (a,b),p in full.items() if S+a+b<3.0)
        pu30  = sum(p for (a,b),p in full.items() if S+a+b<3.0)
        if not (pu25-TOLL_S <= pu275 <= pu30+TOLL_S):
            errs["QL_monotony"].append(f"U25={pu25:.4f} U275={pu275:.4f} U30={pu30:.4f} | {ctx}")

        # 11. rho in range atteso
        if not (0.01 <= rho <= 0.20):
            errs["rho_range"].append(f"rho={rho:.5f} | {ctx}")

        # 12. full_matrix sum
        fs = sum(full.values())
        if abs(fs-1.0) > TOLL_S:
            errs["full_norm"].append(f"full_sum={fs:.6f} | {ctx}")

        # 13. CS negativi
        for sc,p in cs.items():
            if p < -TOLL:
                errs["CS_negative"].append(f"CS{sc}={p:.6f} | {ctx}"); break

        # 14. xG range ragionevole
        if not (1e-9 <= xg1 <= 15.0): errs["xG_range"].append(f"xg1={xg1:.4f} | {ctx}")
        if not (1e-9 <= xg2 <= 15.0): errs["xG_range"].append(f"xg2={xg2:.4f} | {ctx}")

    return dict(errs)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N        = 1_000_000
    N_CPU    = max(1, cpu_count() - 1)
    BATCH    = N // N_CPU
    batches  = [(i, BATCH + (N % N_CPU if i == N_CPU-1 else 0), i*100003)
                for i in range(N_CPU)]

    print(f"Monte Carlo — {N:,} simulazioni su {N_CPU} CPU")
    t0 = time.time()

    with Pool(N_CPU) as pool:
        results = pool.map(run_batch, batches)

    elapsed = time.time() - t0

    # Merge risultati
    merged = defaultdict(list)
    for r in results:
        for k, v in r.items():
            merged[k].extend(v)

    total_errors = sum(len(v) for v in merged.values())
    print(f"\nCompletate {N:,} sim in {elapsed:.1f}s  ({N/elapsed:,.0f} sim/s)")
    print("="*70)
    print("REPORT FINALE")
    print("="*70)

    if total_errors == 0:
        print(f"\n✅  ZERO errori su {N:,} simulazioni — il motore è matematicamente corretto.")
    else:
        print(f"\n⚠️  {total_errors} errori in {len(merged)} categorie:\n")
        for etype, elist in sorted(merged.items(), key=lambda x: -len(x[1])):
            print(f"  [{len(elist):>7,}]  {etype}")
            for msg in elist[:4]:
                print(f"             → {msg}")
            if len(elist) > 4:
                print(f"             ... e altri {len(elist)-4:,}")
            print()

    sys.exit(0 if total_errors == 0 else 1)
