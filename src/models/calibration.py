"""
calibration.py — Calibrazione Bayesiana degli xG dalle linee di mercato e dai tiri.

Implementa:
  - Blend Bayesiano a pesi dinamici (apertura + corrente)
  - Bisection per estrarre gli xG impliciti dall'Asian Handicap
  - Blend T+D degli xG da linee di mercato con evidenze empiriche dei tiri

Riferimenti:
  Karlis & Ntzoufras (2003)
  Dixon & Coles (1997)
  Brechot & Flepp (2020)
"""

from __future__ import annotations

import math

from src.config import BAYES, POISSON, SHOTS
from src.models.poisson import dixon_coles_tau, poisson_pmf

# ---------------------------------------------------------------------------
# AH EV helper
# ---------------------------------------------------------------------------

def _ah_ev_half(
    mu_h: float,
    mu_a: float,
    handicap: float,
) -> float:
    """
    Calcola l'EV dell'Asian Handicap con correzione Dixon-Coles inclusa.

    La coerenza con il modello finale (bivariate Poisson + DC) è essenziale:
    estrarre xG con un modello diverso (Poisson indipendente) introduce bias
    sistematico del 2-5% sulle probabilità dei punteggi bassi.

    Il termine Z si cancella nella differenza (i-j) → usiamo solo la
    componente indipendente + DC (Z non influenza l'EV dell'AH).

    Args:
        mu_h: Lambda casa (componente indipendente).
        mu_a: Lambda trasferta (componente indipendente).
        handicap: Handicap applicato alla casa (es. -0.75, +0.5).

    Returns:
        EV in [-1, 1]: positivo = valore per la casa.
    """
    pmf_h = poisson_pmf(mu_h)
    pmf_a = poisson_pmf(mu_a)
    ev = 0.0
    dc_norm = 0.0

    for i, pi in enumerate(pmf_h):
        if pi < 1e-18:
            continue
        for j, pj in enumerate(pmf_a):
            if pj < 1e-18:
                continue
            w = pi * pj * dixon_coles_tau(i, j, mu_h, mu_a)
            dc_norm += w
            s = (i - j) + handicap
            if s > 0:
                ev += w
            elif s < 0:
                ev -= w

    return ev / dc_norm if dc_norm > 0 else 0.0


def _ah_ev(mu_h: float, mu_a: float, ah: float) -> float:
    """
    EV dell'AH con interpolazione continua tra le half-lines adiacenti.

    Per la calibrazione degli xG, l'AH dal blend bayesiano può essere qualsiasi float
    (es. -0.212, -0.406), non solo valori standard di mercato (±0.25, ±0.50, ...).

    L'EV esatto della distribuzione di Poisson è una funzione a gradini:
    costante tra due interi consecutivi, con discontinuità agli interi (dove il push cambia).
    Usare il gradino direttamente per valori non-standard causa errori sistematici
    del 15-35% nella calibrazione degli xG.

    Soluzione: interpolazione lineare tra le due half-lines adiacenti.
    Questo produce una funzione EV continua che:
    - Ai valori half-line (±0.5, ±1.0): coincide con l'EV esatto (con push)
    - Alle quarter lines (±0.25, ±0.75): dà il corretto split 50/50
    - Per valori non-standard (dal blend): interpola smoothly

    Esempio: ah = -0.212
      h_low = -0.5, h_high = 0.0
      t = (-0.212 - (-0.5)) / 0.5 = 0.576
      EV = 0.424 × EV(-0.5) + 0.576 × EV(0.0)

    Args:
        mu_h, mu_a: Lambda casa e trasferta.
        ah: Linea AH (standard di mercato o valore dal blend).

    Returns:
        EV in [-1, 1].
    """
    ah_f = float(ah)
    ah2 = ah_f * 2.0

    # Controlla se è esattamente una half-line (es. -0.5, 0.0, +0.5, -1.0)
    if abs(ah2 - round(ah2)) < POISSON.EPS:
        return _ah_ev_half(mu_h, mu_a, ah_f)

    # Per tutti gli altri valori (quarter lines standard e valori non-standard dal blend):
    # interpolazione con correzione di curvatura (Hermite-like).
    # L'interpolazione lineare pura ha errore O(t²) ≈ ±2.5% sui quarter-line.
    # La correzione quadratica riduce l'errore a O(t⁴) ≈ ±0.2%.
    h_low = math.floor(ah2) / 2.0
    h_high = h_low + 0.5
    t = (ah_f - h_low) / 0.5  # posizione in [0, 1): 0.5 per quarter lines standard

    ev_low = _ah_ev_half(mu_h, mu_a, h_low)
    ev_high = _ah_ev_half(mu_h, mu_a, h_high)
    ev_linear = (1.0 - t) * ev_low + t * ev_high

    # Correzione curvatura: derivata seconda approssimata con 4-point stencil
    h_ll = h_low - 0.5
    h_hh = h_high + 0.5
    ev_ll = _ah_ev_half(mu_h, mu_a, h_ll)
    ev_hh = _ah_ev_half(mu_h, mu_a, h_hh)
    # f'' ≈ (ev_hh - ev_high) - (ev_low - ev_ll) = curvatura discretizzata
    f_pp = (ev_hh - ev_high) - (ev_low - ev_ll)
    # Correzione Hermite: t*(1-t) pesa 0 ai bordi, max 0.25 al centro
    return ev_linear + 0.05 * t * (1.0 - t) * f_pp


# ---------------------------------------------------------------------------
# Blend Bayesiano + bisection
# ---------------------------------------------------------------------------

def calcola_xg_bayesiani(
    ah_op: float,
    tot_op: float,
    ah_cur: float,
    tot_cur: float,
    minuto: int,
    gol_diff: int = 0,
    gol_tot: int = 0,
) -> tuple[float, float]:
    """
    Estrae gli xG impliciti dal blend Bayesiano delle linee di mercato.

    Pipeline:
    1. Calcola i pesi time-varying: w_cur cresce da 65% (t=0) a 90% (t=90').
    2. Blend linea apertura (scalata al tempo rimanente) + linea corrente.
    3. Bisection su delta = mu_h - mu_a per trovare il punto dove AH EV = 0.

    Pesi time-varying (Brechot & Flepp 2020):
        w_cur = min(0.90, 0.65 + 0.20 * frac_giocata)
        w_op  = 1 - w_cur

    Motivazione: la linea live incorpora progressivamente più informazione
    (eventi accaduti, condizioni di gioco, liquidità del mercato).
    Il prior di apertura viene scalato al tempo rimanente prima del blend:
    entrambi gli input diventano omogenei (gol rimanenti).

    Quando le linee sono "flat" (nessun movimento): usa direttamente ah_cur/tot_cur
    senza blend per evitare deriva artificiale col passare del tempo.

    Args:
        ah_op: AH apertura (handicap full 90').
        tot_op: Total apertura (gol attesi full 90').
        ah_cur: AH corrente (gol rimanenti).
        tot_cur: Total corrente (gol rimanenti).
        minuto: Minuto attuale [0, 90].
        gol_diff: gol_casa - gol_trasf (per eliminare l'offset del punteggio dal delta AH).
        gol_tot: gol_casa + gol_trasf (per eliminare l'offset del punteggio dal delta Total).

    Returns:
        (xg_h, xg_a): xG rimanenti attesi per casa e trasferta.
    """
    eps = POISSON.EPS
    frac_giocata = minuto / 90.0
    frac_rimasta = max(BAYES.FRAC_RIMASTA_FLOOR, 1.0 - frac_giocata)

    # Pesi time-varying (esponenziali):
    # w_op decade più velocemente early game (eventi compound) ma mantiene un residuo.
    # t=0: w_op=0.35, t=30': ~0.22, t=60': ~0.14, t=90': ~0.10
    w_op_raw = (1.0 - BAYES.W_CUR_MIN) * math.exp(-BAYES.W_OP_EXP_RATE * frac_giocata ** 2) \
        + (1.0 - BAYES.W_CUR_MAX) * (1.0 - frac_giocata)
    w_op = max(1.0 - BAYES.W_CUR_MAX, min(1.0 - BAYES.W_CUR_MIN, w_op_raw))
    w_cur = 1.0 - w_op

    # Rilevamento linee flat: confronto in full-game space per non confondere il
    # movimento del mercato con l'effetto meccanico del punteggio.
    # Se il mercato non si è mosso e il punteggio è gol_diff/gol_tot:
    #   ah_cur  = ah_op  + gol_diff   (puro effetto punteggio)
    #   tot_cur = tot_op - gol_tot    (puro effetto punteggio)
    expected_ah_cur = ah_op + gol_diff
    expected_tot_cur = max(0.0, tot_op - gol_tot)  # i gol rimanenti non possono essere negativi
    delta_ah_inner = abs(ah_cur - expected_ah_cur)
    delta_tot_inner = abs(tot_cur - expected_tot_cur)
    flat = delta_ah_inner < BAYES.FLAT_LINE_THRESHOLD and delta_tot_inner < BAYES.FLAT_LINE_THRESHOLD

    if flat:
        ah_bayes = float(ah_cur)
        tot_bayes = max(BAYES.TOT_BAYES_MIN, float(tot_cur))
    else:
        ah_bayes = (ah_op * frac_rimasta) * w_op + ah_cur * w_cur
        tot_bayes = max(BAYES.TOT_BAYES_MIN, (tot_op * frac_rimasta) * w_op + tot_cur * w_cur)

    # Bisection: trova delta* tale che AH EV = 0
    # mu_h = (tot_bayes + delta) / 2
    # mu_a = (tot_bayes - delta) / 2

    def _ev(delta: float) -> float:
        lh = max(eps, 0.5 * (tot_bayes + delta))
        la = max(eps, 0.5 * (tot_bayes - delta))
        return _ah_ev(lh, la, ah_bayes)

    lo = -tot_bayes + eps
    hi = tot_bayes - eps
    ev_lo, ev_hi = _ev(lo), _ev(hi)

    if ev_lo == 0.0:
        delta_star = lo
    elif ev_hi == 0.0:
        delta_star = hi
    elif ev_lo * ev_hi > 0:
        # Stessa direzione: la bisection non può trovare una radice.
        # Questo accade tipicamente a fine partita con tot_bayes basso e |ah_bayes| significativo:
        # P(0,0) domina e l'AH EV è negativo per qualsiasi split.
        #
        # Fallback: approssimazione lineare delta ≈ -ah_bayes.
        # Per Poisson con mu piccolo, E[X-Y] ≈ mu_h - mu_a = delta,
        # e l'AH EV → 0 quando E[D] → -ah (la differenza attesa bilancia l'handicap).
        # Questo produce uno split (mu_h, mu_a) molto più equilibrato
        # rispetto al bound estremo, evitando xG_a ≈ 0 irrealistici.
        delta_linear = -ah_bayes
        delta_star = max(lo, min(hi, delta_linear))
    else:
        # Bisection standard
        increasing = ev_hi > ev_lo
        dl, dr = lo, hi
        for _ in range(POISSON.BISECTION_ITERS):
            m = 0.5 * (dl + dr)
            em = _ev(m)
            if abs(em) < POISSON.BISECTION_TOL or (dr - dl) < 1e-12:
                break
            if increasing:
                if em > 0:
                    dr = m
                else:
                    dl = m
            else:
                if em > 0:
                    dl = m
                else:
                    dr = m
        delta_star = 0.5 * (dl + dr)

    # Cap su |delta_star| per garantire xg_h/xg_a ≤ XG_RATIO_CAP.
    # Da (tot+d)/(tot-d) = R → d_max = tot·(R-1)/(R+1).
    # A fine partita con tot_bayes basso, la bisection/fallback può produrre
    # rapporti estremi (>10:1) perché P(0,0) domina e l'EV è negativo
    # per qualsiasi split, o perché |ah_bayes| > tot_bayes.
    cap = BAYES.XG_RATIO_CAP
    delta_max = tot_bayes * (cap - 1.0) / (cap + 1.0)
    delta_star = max(-delta_max, min(delta_max, delta_star))

    xg_h = max(eps, 0.5 * (tot_bayes + delta_star))
    xg_a = max(eps, 0.5 * (tot_bayes - delta_star))

    return xg_h, xg_a


# ---------------------------------------------------------------------------
# Blend xG da tiri + linee
# ---------------------------------------------------------------------------

def blend_xg_shots(
    mu_h_line: float,
    mu_a_line: float,
    sot_h: int,
    soff_h: int,
    sot_a: int,
    soff_a: int,
    gol_h: int,
    gol_a: int,
    minuto: int,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Integra le stime xG da linee di mercato con le evidenze empiriche dei tiri.

    ── xG per tiro (valori calibrati su top-5 leagues) ──────────────────────
      XG_SOT  = 0.30  (tiro in porta medio, senza dati posizionali)
      XG_SOFF = 0.05  (tiro fuori)

      Correzione game-state ±7% per gol di vantaggio (cap 15%):
        - Squadra in vantaggio: tiri in contropiede su difese aperte → qualità ↑
        - Squadra in svantaggio: pressing disperato → tiri da lontano → qualità ↓

    ── Proiezione al tempo rimanente ─────────────────────────────────────────
      rate_h = xg_h_accum / frac_giocata   (xG per 90' implicato dai tiri)
      mu_h_shots = rate_h × frac_rimasta

    ── Blend in spazio T+D ───────────────────────────────────────────────────
      Totale e Differenziale separati → pesi diversi:
        α_T (Total):         max 0.25 — il Total-line è molto efficiente
        α_D (Differenziale): max 0.70 — i tiri rivelano il dominio meglio del mercato

      Entrambi i pesi crescono con:
        • shot_info = min(1, n_shots/15)  — campione sufficiente
        • frac_giocata (o radice) — dati live più affidabili col tempo

      T_blend = (1−α_T)·T_line + α_T·T_shots
      D_blend = (1−α_D)·D_line + α_D·D_shots
      mu_h = (T_blend + D_blend) / 2
      mu_a = (T_blend − D_blend) / 2

    Args:
        mu_h_line, mu_a_line: xG stimati dalle linee (baseline).
        sot_h, soff_h: Tiri in porta e fuori della casa.
        sot_a, soff_a: Tiri in porta e fuori della trasferta.
        gol_h, gol_a: Gol attuali.
        minuto: Minuto attuale.

    Returns:
        (mu_h_final, mu_a_final, xg_h_accum, xg_a_accum, alpha_T, alpha_D, shot_dom)
    """
    gol_totali_now = float(gol_h + gol_a)
    # Cap dinamico: in gare ad alto punteggio il cap fisso 3.5 sottostima il ritmo
    mu_max = max(3.5, gol_totali_now * 1.2)

    frac_giocata = max(minuto, 1) / 90.0

    # Correzione game-state DIFFERENZIATA per tipo di tiro, modulata dal minuto.
    # SOT (in porta): qualità alta, sensibile al game-state (contropiede vs pressing)
    # SOFF (fuori porta): qualità bassa, meno sensibile (tiri disperati = rumore)
    #
    # Early game: la squadra in vantaggio preme → SOT qualità ↑ (contropiede)
    # Late game: la squadra in vantaggio difende → SOT qualità ↓ (si abbassa)
    # La squadra in svantaggio tardi forza pressing → SOFF ↑ (volume), SOT ↓ (qualità)
    diff_score = gol_h - gol_a
    gs_minute_scale = max(0.6, 1.2 - 0.8 * frac_giocata)

    gs_sot = min(SHOTS.GAME_STATE_CAP, abs(diff_score) * SHOTS.GAME_STATE_RATE_SOT) * gs_minute_scale
    gs_soff = min(SHOTS.GAME_STATE_CAP, abs(diff_score) * SHOTS.GAME_STATE_RATE_SOFF) * gs_minute_scale

    if diff_score > 0:
        # Casa in vantaggio: SOT casa ↑ (contropiede), SOT trasf ↓ (pressing disperato)
        # SOFF trasf ↑ (volume disperazione), SOFF casa invariato
        k_sot_h, k_sot_a = 1.0 + gs_sot, 1.0 - gs_sot
        k_soff_h, k_soff_a = 1.0, 1.0 + gs_soff
    elif diff_score < 0:
        k_sot_h, k_sot_a = 1.0 - gs_sot, 1.0 + gs_sot
        k_soff_h, k_soff_a = 1.0 + gs_soff, 1.0
    else:
        k_sot_h, k_sot_a = 1.0, 1.0
        k_soff_h, k_soff_a = 1.0, 1.0

    xg_h_accum = sot_h * SHOTS.XG_SOT * k_sot_h + soff_h * SHOTS.XG_SOFF * k_soff_h
    xg_a_accum = sot_a * SHOTS.XG_SOT * k_sot_a + soff_a * SHOTS.XG_SOFF * k_soff_a

    frac_rimasta = max(BAYES.FRAC_RIMASTA_FLOOR, (90.0 - minuto) / 90.0)

    # Proiezione rate → mu rimanenti (shot-based)
    # Smorzamento esponenziale: il tasso di tiri osservato su un campione breve
    # sovrastima il ritmo effettivo (regressione alla media). Curva esponenziale:
    # converge rapidamente dopo il 30' (campione diventa affidabile) ma resta
    # conservativa nei primi 15' (alta varianza del campione).
    # 0.75 a inizio → ~0.92 a 30' → ~0.98 a 60' → 1.0 a fine
    dampening = 1.0 - (1.0 - SHOTS.RATE_DAMP_FLOOR) * math.exp(-SHOTS.RATE_DAMP_DECAY * frac_giocata)
    rate_h = xg_h_accum / frac_giocata * dampening
    rate_a = xg_a_accum / frac_giocata * dampening
    mu_h_shots = rate_h * frac_rimasta
    mu_a_shots = rate_a * frac_rimasta

    # Indice di dominio tiri (solo in porta, più informativi)
    sot_tot = sot_h + sot_a
    shot_dom = abs(sot_h - sot_a) / sot_tot if sot_tot > 0 else 0.0

    # Pesi blend dinamici
    n_shots = sot_h + soff_h + sot_a + soff_a
    shot_info = min(1.0, n_shots / SHOTS.SHOT_INFO_THRESHOLD)
    alpha_t = min(SHOTS.ALPHA_T_MAX, shot_info * frac_giocata * SHOTS.ALPHA_T_RATE)
    alpha_d = min(SHOTS.ALPHA_D_MAX, shot_info * math.sqrt(frac_giocata))

    # Blend in spazio T+D
    t_line = mu_h_line + mu_a_line
    d_line = mu_h_line - mu_a_line
    t_shots = mu_h_shots + mu_a_shots
    d_shots = mu_h_shots - mu_a_shots

    t_blend = (1.0 - alpha_t) * t_line + alpha_t * t_shots
    d_blend = (1.0 - alpha_d) * d_line + alpha_d * d_shots

    eps = POISSON.EPS
    mu_h_final = max(eps, min((t_blend + d_blend) / 2.0, mu_max))
    mu_a_final = max(eps, min((t_blend - d_blend) / 2.0, mu_max))

    return mu_h_final, mu_a_final, xg_h_accum, xg_a_accum, alpha_t, alpha_d, shot_dom
