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
    EV dell'AH con supporto per quarter lines (2.25, 2.75, 3.25...).

    Quarter line: split 50/50 tra la half-line inferiore e superiore.

    Args:
        mu_h, mu_a: Lambda casa e trasferta (componenti indipendenti).
        ah: Linea AH (es. -0.75, +0.25, -0.50).

    Returns:
        EV in [-1, 1].
    """
    ah2 = float(ah) * 2.0
    if abs(ah2 - round(ah2)) < POISSON.EPS:
        # Linea intera o half-line: es. -0.5, -1.0, +0.75
        return _ah_ev_half(mu_h, mu_a, float(ah))
    # Quarter line: es. -0.75 = 50% di -0.5 + 50% di -1.0
    h_low = math.floor(ah2) / 2.0
    return 0.5 * _ah_ev_half(mu_h, mu_a, h_low) + 0.5 * _ah_ev_half(mu_h, mu_a, h_low + 0.5)


# ---------------------------------------------------------------------------
# Blend Bayesiano + bisection
# ---------------------------------------------------------------------------

def calcola_xg_bayesiani(
    ah_op: float,
    tot_op: float,
    ah_cur: float,
    tot_cur: float,
    minuto: int,
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

    Returns:
        (xg_h, xg_a): xG rimanenti attesi per casa e trasferta.
    """
    eps = POISSON.EPS
    frac_giocata = minuto / 90.0
    frac_rimasta = max(BAYES.FRAC_RIMASTA_FLOOR, 1.0 - frac_giocata)

    # Pesi time-varying
    w_cur = min(BAYES.W_CUR_MAX, BAYES.W_CUR_MIN + BAYES.W_CUR_SLOPE * frac_giocata)
    w_op = 1.0 - w_cur

    # Rilevamento linee flat (nessun movimento intrapartita)
    delta_ah_inner = abs(ah_cur - ah_op)
    delta_tot_inner = abs(tot_cur - tot_op)
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
        # Stessa direzione: usa il bound con EV più vicino a zero
        delta_star = lo if abs(ev_lo) < abs(ev_hi) else hi
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

    # Correzione game-state sulla qualità dei tiri
    diff_score = gol_h - gol_a
    gs_adj = min(SHOTS.GAME_STATE_CAP, abs(diff_score) * SHOTS.GAME_STATE_RATE)
    if diff_score > 0:
        k_h, k_a = 1.0 + gs_adj, 1.0 - gs_adj  # casa in vantaggio → contropiede
    elif diff_score < 0:
        k_h, k_a = 1.0 - gs_adj, 1.0 + gs_adj  # trasferta in vantaggio
    else:
        k_h, k_a = 1.0, 1.0

    xg_h_accum = sot_h * SHOTS.XG_SOT * k_h + soff_h * SHOTS.XG_SOFF
    xg_a_accum = sot_a * SHOTS.XG_SOT * k_a + soff_a * SHOTS.XG_SOFF

    frac_giocata = max(minuto, 1) / 90.0
    frac_rimasta = max(0.0, (90.0 - minuto) / 90.0)

    # Proiezione rate → mu rimanenti (shot-based)
    rate_h = xg_h_accum / frac_giocata
    rate_a = xg_a_accum / frac_giocata
    mu_h_shots = rate_h * frac_rimasta
    mu_a_shots = rate_a * frac_rimasta

    # Indice di dominio tiri (solo in porta, più informativi)
    sot_tot = sot_h + sot_a
    shot_dom = abs(sot_h - sot_a) / sot_tot if sot_tot > 0 else 0.0

    # Pesi blend dinamici
    n_shots = sot_h + soff_h + sot_a + soff_a
    shot_info = min(1.0, n_shots / SHOTS.SHOT_INFO_THRESHOLD)
    alpha_t = min(SHOTS.ALPHA_T_MAX, shot_info * frac_giocata * 0.30)
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
