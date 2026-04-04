"""
outputs.py — Rendering dei risultati dell'analisi su Streamlit.

Centralizza tutta la logica di output dell'interfaccia utente:
  - Quote fair (sempre visibili)
  - Correct Score e distribuzione gol
  - Tabella Asian Handicap
  - Momentum di mercato
  - Segnali rapidi e avanzati
  - Pannello debug parametri interni
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.config import DECAY, SIGNALS, UI
from src.config import MOMENTUM as MOMENTUM_CFG
from src.engine import ExchangeQuotes, ProbabilitaModello
from src.markets.asian_handicap import calcola_asian_handicap
from src.signals import Signal

if TYPE_CHECKING:
    from src.engine import MatchState


def _prob_ah_cover(p1: float, px: float, p2: float, linea_ah: float) -> tuple[float, float]:
    """
    Restituisce (p_home_eq, p_away_eq) per l'AH dato.

    Usa la formula EV per le linee quarter (±0.25, ±0.75):
      Q_fair_home = 1 + (P(X)/2 + P(2)) / P(1)   per linea -0.25
    Per le linee intere (0, ±0.5) usa le probabilità dirette.
    P_eq = 1 / Q_fair, e p_home + p_away = 1 (no push nelle quarter).
    """
    eps = 1e-6
    ah = round(linea_ah * 4) / 4   # normalizza a multipli di 0.25

    # AH -0.5 (home dà 0.5 gol): home vince solo se vince la partita
    if ah <= -0.5:
        p_h = p1
        p_a = px + p2
    # AH 0 (PK): pareggio = push
    elif ah == 0.0:
        denom = p1 + p2
        if denom < eps:
            p_h = p_a = 0.5
        else:
            p_h = p1 / denom
            p_a = p2 / denom
    # AH -0.25 (quarter verso -0.5): pareggio = metà perdita per home
    elif ah < 0.0:  # -0.25
        denom = p1 + px / 2.0 + p2
        p_h = p1 / denom if denom > eps else 0.5
        p_a = 1.0 - p_h
    # AH +0.25 (quarter verso 0): pareggio = metà vincita per home
    elif 0.0 < ah < 0.5:
        denom = p1 + px / 2.0 + p2
        p_h = (p1 + px / 2.0) / denom if denom > eps else 0.5
        p_a = 1.0 - p_h
    else:  # AH >= +0.5: home copre anche con pari o meglio
        p_h = p1 + px
        p_a = p2

    return (round(p_h, 4), round(p_a, 4))


def _q_fair(prob: float) -> float:
    return 1.0 / prob if prob > 0.001 else 999.0


# ---------------------------------------------------------------------------
# Pronostici Rapidi — solo percentuali, niente quote
# ---------------------------------------------------------------------------

def _poisson_ht_1x2(lam_h: float, lam_a: float) -> tuple[float, float, float, float]:
    """1X2 al HT + P(over 0.5) da due Poisson indipendenti."""

    def _pp(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    p_ht1 = p_htx = p_ht2 = 0.0
    for h in range(6):
        for a in range(6):
            p = _pp(h, lam_h) * _pp(a, lam_a)
            if h > a:
                p_ht1 += p
            elif h == a:
                p_htx += p
            else:
                p_ht2 += p
    p_ht_over05 = 1.0 - _pp(0, lam_h) * _pp(0, lam_a)
    return p_ht1, p_htx, p_ht2, p_ht_over05


def _ht_goal_share_1h(g1h: float, g2h: float, *, default: float = 0.46) -> float:
    """Quota dei gol di squadra segnati nel 1T rispetto al totale 1T+2T (stagione)."""
    a = max(0.0, float(g1h or 0.0))
    b = max(0.0, float(g2h or 0.0))
    if a + b < 1e-6:
        return default
    return max(0.34, min(0.56, a / (a + b)))


def _calcola_ht_probs(
    prematch: Any,
    xg_h_fallback: float = 0.0,
    xg_a_fallback: float = 0.0,
    *,
    p1_ft: float | None = None,
    px_ft: float | None = None,
    p2_ft: float | None = None,
    risultati: ProbabilitaModello | None = None,
    late_goals_pct_h: float = 0.0,
    late_goals_pct_a: float = 0.0,
    early_conceded_pct_h: float = 0.0,
    early_conceded_pct_a: float = 0.0,
) -> tuple[float, float, float, float, bool] | None:
    """
    Stima le probabilità del risultato al 1° tempo usando tutti i segnali disponibili:

    - **λ 1T**: media gol 1T a partita (stagione) combinata con **xG FT × quota 1T/90′**
      per squadra (`goals_1h / (goals_1h+goals_2h)`), non un fisso 46%.
    - **Forma** (`forma_mult_*`) e **strength** Nowgoal: leggero tilt su λ.
    - **Goal timing** (MatchState): chi segna tardi → λ 1T leggermente più basso; chi subisce
      presto → λ avversario 1T leggermente più alto.
    - **Standings HT** + **H2H HT** (peso ridotto con pochi match).
    - **Ancoraggio 1X2 FT** del modello + **p_over_1.5** per calibrare Over 0.5 a HT.

    Restituisce (p_ht1, p_htx, p_ht2, p_ht_over05, is_estimate) o None se nessun dato.
    is_estimate=True quando λ 1T deriva solo da xG FT (nessun dato HT diretto in tabella).
    """
    xg_h = max(0.0, float(xg_h_fallback or 0.0))
    xg_a = max(0.0, float(xg_a_fallback or 0.0))

    # ── 1. Dati gol per tempo (stagione) e partite ─────────────────────────────
    g1h_h = getattr(prematch, "home_goals_1h", 0.0) or 0.0
    g2h_h = getattr(prematch, "home_goals_2h", 0.0) or 0.0
    g1h_a = getattr(prematch, "away_goals_1h", 0.0) or 0.0
    g2h_a = getattr(prematch, "away_goals_2h", 0.0) or 0.0
    hm = max(1, getattr(prematch, "home_matches", 1) or 1)
    am = max(1, getattr(prematch, "away_matches", 1) or 1)

    phi_h = _ht_goal_share_1h(g1h_h, g2h_h)
    phi_a = _ht_goal_share_1h(g1h_a, g2h_a)

    lam_s_h = float(g1h_h) / hm  # media gol segnati 1T / partita (casa)
    lam_s_a = float(g1h_a) / am
    lam_x_h = xg_h * phi_h
    lam_x_a = xg_a * phi_a

    def _blend_side(lam_s: float, lam_x: float, n_matches: int, xg_ok: bool) -> float:
        if lam_s > 0.015 and xg_ok:
            w = 0.48 * min(1.0, n_matches / 10.0)
            return (1.0 - w) * lam_x + w * lam_s
        if lam_s > 0.015:
            return lam_s
        if xg_ok:
            return lam_x
        return 0.0

    xg_ok_h = xg_h > 0.008
    xg_ok_a = xg_a > 0.008
    lam_h = _blend_side(lam_s_h, lam_x_h, hm, xg_ok_h)
    lam_a = _blend_side(lam_s_a, lam_x_a, am, xg_ok_a)

    # Forma / strength estratti (Nowgoal)
    fm_h = float(getattr(prematch, "forma_mult_h", 1.0) or 1.0)
    fm_a = float(getattr(prematch, "forma_mult_a", 1.0) or 1.0)
    lam_h *= max(0.82, min(1.18, fm_h**0.28))
    lam_a *= max(0.82, min(1.18, fm_a**0.28))

    sh = int(getattr(prematch, "strength_home", 0) or 0)
    sa = int(getattr(prematch, "strength_away", 0) or 0)
    if sh > 0 or sa > 0:
        delta = max(-0.4, min(0.4, (sh - sa) / 160.0))
        lam_h *= 1.0 + 0.11 * delta
        lam_a *= 1.0 - 0.11 * delta

    # Goal timing (effetto sul 1T)
    if late_goals_pct_h > 18.0:
        lam_h *= 1.0 - 0.065 * min(1.0, (late_goals_pct_h - 18.0) / 32.0)
    if late_goals_pct_a > 18.0:
        lam_a *= 1.0 - 0.065 * min(1.0, (late_goals_pct_a - 18.0) / 32.0)
    if early_conceded_pct_h > 14.0:
        lam_a *= 1.0 + 0.048 * min(1.0, (early_conceded_pct_h - 14.0) / 38.0)
    if early_conceded_pct_a > 14.0:
        lam_h *= 1.0 + 0.048 * min(1.0, (early_conceded_pct_a - 14.0) / 38.0)

    # H2H basso scoring → intensità 1T leggermente più bassa
    h2h_n = int(getattr(prematch, "h2h_matches_count", 0) or 0)
    h2h_av = float(getattr(prematch, "h2h_avg_goals_home", 0.0) or 0.0)
    h2h_av += float(getattr(prematch, "h2h_avg_goals_away", 0.0) or 0.0)
    if h2h_n >= 2 and h2h_av > 0.1 and h2h_av < 2.15:
        lam_h *= 0.94
        lam_a *= 0.94

    has_xg = lam_h > 0.01 or lam_a > 0.01

    # ── 2. Standings HT (dati estratti da Nowgoal) ─────────────────────────────
    # home_ht_win/draw/lose = risultati al HT della squadra di casa
    # away_ht_win/draw/lose = risultati al HT della squadra di trasferta
    home_ht_w = getattr(prematch, "home_ht_win", 0) or 0
    home_ht_d = getattr(prematch, "home_ht_draw", 0) or 0
    home_ht_l = getattr(prematch, "home_ht_lose", 0) or 0
    away_ht_w = getattr(prematch, "away_ht_win", 0) or 0
    away_ht_d = getattr(prematch, "away_ht_draw", 0) or 0
    away_ht_l = getattr(prematch, "away_ht_lose", 0) or 0

    # Calcola percentuali dai risultati HT
    home_ht_total = home_ht_w + home_ht_d + home_ht_l
    away_ht_total = away_ht_w + away_ht_d + away_ht_l

    has_standings_ht = home_ht_total > 0 and away_ht_total > 0

    if has_standings_ht:
        # % risultati al HT per ogni squadra
        home_w_pct = home_ht_w / home_ht_total  # Casa vince al HT
        home_d_pct = home_ht_d / home_ht_total  # Casa pareggia al HT
        home_l_pct = home_ht_l / home_ht_total  # Casa perde al HT

        away_w_pct = away_ht_w / away_ht_total  # Trasferta vince al HT (nel suo stadio)
        away_d_pct = away_ht_d / away_ht_total  # Trasferta pareggia al HT
        away_l_pct = away_ht_l / away_ht_total  # Trasferta perde al HT

        # Deriva probabilità HT per QUESTA partita:
        # - p_ht1 = prob che casa vince al HT = casa forte in casa + trasferta debole fuori
        # - p_ht2 = prob che trasferta vince al HT = trasferta forte fuori + casa debole in casa
        # Formula: media tra % vittorie casa in casa e % sconfitte trasferta fuori
        standings_p1 = (home_w_pct + away_l_pct) / 2
        standings_px = (home_d_pct + away_d_pct) / 2
        standings_p2 = (home_l_pct + away_w_pct) / 2

        # Normalizza
        _tot = standings_p1 + standings_px + standings_p2
        if _tot > 0:
            standings_p1 /= _tot
            standings_px /= _tot
            standings_p2 /= _tot

    # ── 3. H2H HT (se disponibili) ─────────────────────────────────────────────
    h2h_h_pct = getattr(prematch, "h2h_ht_home_win_pct", 0.0) or 0.0
    h2h_d_pct = getattr(prematch, "h2h_ht_draw_pct", 0.0) or 0.0
    h2h_a_pct = getattr(prematch, "h2h_ht_away_win_pct", 0.0) or 0.0
    has_h2h = h2h_h_pct + h2h_d_pct + h2h_a_pct > 0.5  # in scala 0-100

    # ── 4. Determina se abbiamo dati HT diretti ─────────────────────────────────
    has_ht_direct = has_xg or has_standings_ht or has_h2h

    if not has_ht_direct:
        # Solo xG FT: λ 1T = xG × quota gol in 1T per squadra (o default 46%)
        if xg_h > 0.01 or xg_a > 0.01:
            lam_h = xg_h * phi_h
            lam_a = xg_a * phi_a
            has_xg = True
            is_estimate = True
        else:
            return None
    else:
        is_estimate = False

    # ── 5. Poisson indipendente (se abbiamo xG HT) ──────────────────────────────
    if has_xg:
        p_ht1, p_htx, p_ht2, p_ht_over05 = _poisson_ht_1x2(lam_h, lam_a)
    else:
        p_ht1 = p_htx = p_ht2 = p_ht_over05 = 0.0

    # ── 6. Blend con Standings HT ────────────────────────────────────────────────
    if has_standings_ht and has_xg:
        _conf = float(risultati.model_confidence) if risultati is not None else 0.5
        alpha = 0.40 - (0.07 if _conf >= 0.63 else 0.0)
        if lam_s_h < 0.04 and lam_s_a < 0.04:
            alpha *= 0.88
        alpha = max(0.28, min(0.44, alpha))
        p_ht1 = (1 - alpha) * p_ht1 + alpha * standings_p1
        p_htx = (1 - alpha) * p_htx + alpha * standings_px
        p_ht2 = (1 - alpha) * p_ht2 + alpha * standings_p2
    elif has_standings_ht and not has_xg:
        # Usa solo standings HT
        p_ht1, p_htx, p_ht2 = standings_p1, standings_px, standings_p2
        # Stima over 0.5 basata su probabilità di 0-0
        p_ht_over05 = 1.0 - (p_htx * 0.7)  # Approssimazione: 0-0 è circa 70% dei pareggi

    # ── 7. Blend con H2H HT (peso ↓ se pochi match HT) ──────────────────────────
    if has_h2h:
        _tot_h2h = h2h_h_pct + h2h_d_pct + h2h_a_pct
        h2h_p1 = h2h_h_pct / _tot_h2h
        h2h_px = h2h_d_pct / _tot_h2h
        h2h_p2 = h2h_a_pct / _tot_h2h
        _n_ht_m = int(getattr(prematch, "h2h_ht_matches_count", 0) or 0)
        if _n_ht_m > 0:
            h2h_scale = min(1.0, _n_ht_m / 5.0)
        else:
            # Conteggio sconosciuto: non dare pieno peso a percentuali H2H HT spesso rumorose
            h2h_scale = 0.55
        alpha_h2h = 0.20 * h2h_scale
        if has_xg or has_standings_ht:
            p_ht1 = (1 - alpha_h2h) * p_ht1 + alpha_h2h * h2h_p1
            p_htx = (1 - alpha_h2h) * p_htx + alpha_h2h * h2h_px
            p_ht2 = (1 - alpha_h2h) * p_ht2 + alpha_h2h * h2h_p2
        else:
            p_ht1, p_htx, p_ht2 = h2h_p1, h2h_px, h2h_p2

    # ── 8. Coerenza con 1X2 FT del modello (evita HT opposto al favorito FT) ───
    if (
        p1_ft is not None
        and px_ft is not None
        and p2_ft is not None
        and (p1_ft + px_ft + p2_ft) > 0.01
    ):
        sft = p1_ft + px_ft + p2_ft
        f1, fx, f2 = p1_ft / sft, px_ft / sft, p2_ft / sft
        mix_ft = 0.38 if not is_estimate else 0.28
        if risultati is not None:
            mix_ft += 0.06 * max(0.0, min(1.0, (risultati.model_confidence - 0.52) / 0.38))
        mix_ft = max(0.22, min(0.48, mix_ft))
        p_ht1 = (1.0 - mix_ft) * p_ht1 + mix_ft * f1
        p_htx = (1.0 - mix_ft) * p_htx + mix_ft * fx
        p_ht2 = (1.0 - mix_ft) * p_ht2 + mix_ft * f2

    # ── 9. Over 0.5 a HT: ancoraggio leggero a P(Over 1.5) FT (stesso motore) ───
    if risultati is not None and risultati.p_over_15 > 0.02 and p_ht_over05 > 0.02:
        o15 = float(risultati.p_over_15)
        anchor_o = 0.26 + 0.58 * (o15**0.82)
        anchor_o = max(0.30, min(0.92, anchor_o))
        p_ht_over05 = 0.62 * p_ht_over05 + 0.38 * anchor_o

    _ht_sum = p_ht1 + p_htx + p_ht2
    if _ht_sum < 0.01:
        return None

    p_ht1 /= _ht_sum
    p_htx /= _ht_sum
    p_ht2 /= _ht_sum

    return p_ht1, p_htx, p_ht2, p_ht_over05, is_estimate


def render_pronostici_rapidi(
    risultati: ProbabilitaModello,
    linea_ou: float,
    minuto: int = 0,
    gol_casa: int = 0,
    gol_trasf: int = 0,
    linea_ah: float = -0.25,
    prematch: Any = None,
    match_state: Any = None,
) -> None:
    """
    Blocco pronostici completo: 1X2, AH, O/U, xG, Top CS, Primo Tempo.

    match_state: opzionale (MatchState) per goal timing early/late nel calcolo 1T.
    """
    if minuto > 0:
        st.subheader(f"Pronostici — {minuto}' | {gol_casa}–{gol_trasf}")
    else:
        st.subheader("Pronostici Prematch")

    # ── 1X2 ──────────────────────────────────────────────────────────────────
    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa",     f"{risultati.p1:.0%}")
    cx.metric("X — Pareggio", f"{risultati.px:.0%}")
    c2.metric("2 — Trasf.",   f"{risultati.p2:.0%}")

    # ── Contesto squadre (rank, strength, forma) ──────────────────────────────
    if prematch is not None:
        ctx_parts: list[str] = []
        _hr = getattr(prematch, "home_rank", None)
        _ar = getattr(prematch, "away_rank", None)
        _sh = getattr(prematch, "strength_home", None) or 0
        _sa = getattr(prematch, "strength_away", None) or 0
        _hl6w = getattr(prematch, "home_last6_win", None)
        _al6w = getattr(prematch, "away_last6_win", None)
        if _hr or _ar:
            ctx_parts.append(f"Rank: #{_hr or '—'} vs #{_ar or '—'}")
        if _sh > 0 or _sa > 0:
            ctx_parts.append(f"Strength: {_sh or '—'}/100 vs {_sa or '—'}/100")
        if _hl6w is not None and _al6w is not None:
            _hl6d = getattr(prematch, "home_last6_draw", 0) or 0
            _hl6l = getattr(prematch, "home_last6_lose", 0) or 0
            _al6d = getattr(prematch, "away_last6_draw", 0) or 0
            _al6l = getattr(prematch, "away_last6_lose", 0) or 0
            ctx_parts.append(
                f"Forma (ult.6): {_hl6w}V-{_hl6d}P-{_hl6l}S  vs  {_al6w}V-{_al6d}P-{_al6l}S"
            )
        if ctx_parts:
            st.caption("  ·  ".join(ctx_parts))

    st.divider()

    # ── AH cover ─────────────────────────────────────────────────────────────
    p_ah_h, p_ah_a = _prob_ah_cover(risultati.p1, risultati.px, risultati.p2, linea_ah)
    if linea_ah == 0.0:
        _lbl_casa  = "AH Casa (PK)"
        _lbl_trasf = "AH Trasf. (PK)"
    else:
        _sign_c = "+" if linea_ah > 0 else ""
        _lbl_casa = f"AH Casa ({_sign_c}{linea_ah:g})"
        _sign_t = "+" if linea_ah < 0 else "-"
        _lbl_trasf = f"AH Trasf. ({_sign_t}{abs(linea_ah):g})"
    cah1, cah2 = st.columns(2)
    delta_h = p_ah_h - 0.5
    delta_a = p_ah_a - 0.5
    cah1.metric(
        _lbl_casa,
        f"{p_ah_h:.0%}",
        delta=f"{delta_h:+.0%} vs 50%",
        delta_color="normal",
    )
    cah2.metric(
        _lbl_trasf,
        f"{p_ah_a:.0%}",
        delta=f"{delta_a:+.0%} vs 50%",
        delta_color="normal",
    )

    st.divider()

    # ── Over/Under (1.5 + linea mercato) + BTTS + xG ──────────────────────────
    c15o, c15u, co, cu, cgg, cng = st.columns(6)
    c15o.metric("Over 1.5", f"{risultati.p_over_15:.0%}")
    c15u.metric("Under 1.5", f"{risultati.p_under_15:.0%}")
    co.metric(f"Over {linea_ou}", f"{risultati.p_over:.0%}")
    cu.metric(f"Under {linea_ou}", f"{risultati.p_under:.0%}")
    cgg.metric("GG (sì)", f"{risultati.p_btts:.0%}")
    cng.metric("NG (no)", f"{1 - risultati.p_btts:.0%}")
    st.caption(
        "Over 1.5 = almeno **2 gol** totali; Under 1.5 = **0 o 1 gol**. "
        f"Over/Under **{linea_ou}** = linea mercato nel modulo."
    )
    if minuto == 0 and prematch is not None:
        _qgg = float(getattr(prematch, "mkt_init_gg", 0.0) or 0.0)
        _qng = float(getattr(prematch, "mkt_init_ng", 0.0) or 0.0)
        if _qgg > 1.01 and _qng > 1.01:
            st.caption(
                "GG/NG = P(BTTS) del modello, con leggero ancoraggio alle quote consensus Nowgoal (se valide)."
            )
        else:
            st.caption(
                "GG/NG = P(BTTS) del modello (entrambe segnano almeno una volta); "
                "non è automaticamente la quota del bookmaker."
            )

    xh = risultati.xg_h_final
    xa = risultati.xg_a_final
    if xh + xa > 0.01:
        cxh, cxa = st.columns(2)
        cxh.metric("xG Casa attesi",  f"{xh:.2f}")
        cxa.metric("xG Trasf. attesi", f"{xa:.2f}")

    # ── Top 3 Correct Score ──────────────────────────────────────────────────
    if risultati.top_cs:
        st.divider()
        st.caption("**Punteggi più probabili**")
        cs_cols = st.columns(3)
        for idx, ((fc, ft), prob) in enumerate(risultati.top_cs[:3]):
            cs_cols[idx].metric(f"{fc}–{ft}", f"{prob:.1%}", f"fair @{_q_fair(prob):.2f}")

    # ── Primo Tempo ──────────────────────────────────────────────────────────
    if prematch is not None and minuto == 0:
        _ms = match_state
        ht = _calcola_ht_probs(
            prematch,
            xg_h_fallback=risultati.xg_h_final,
            xg_a_fallback=risultati.xg_a_final,
            p1_ft=risultati.p1,
            px_ft=risultati.px,
            p2_ft=risultati.p2,
            risultati=risultati,
            late_goals_pct_h=float(getattr(_ms, "late_goals_pct_h", 0.0) or 0.0) if _ms else 0.0,
            late_goals_pct_a=float(getattr(_ms, "late_goals_pct_a", 0.0) or 0.0) if _ms else 0.0,
            early_conceded_pct_h=float(getattr(_ms, "early_conceded_pct_h", 0.0) or 0.0) if _ms else 0.0,
            early_conceded_pct_a=float(getattr(_ms, "early_conceded_pct_a", 0.0) or 0.0) if _ms else 0.0,
        )
        if ht is not None:
            p_ht1, p_htx, p_ht2, p_ht_o05, ht_is_est = ht
            st.divider()
            _ht_label = "**Primo Tempo (stima da xG)**" if ht_is_est else "**Primo Tempo**"
            st.caption(_ht_label)
            ch1, chx, ch2, cho = st.columns(4)
            ch1.metric("1T Casa",     f"{p_ht1:.0%}")
            chx.metric("1T Pareggio", f"{p_htx:.0%}")
            ch2.metric("1T Trasf.",   f"{p_ht2:.0%}")
            if p_ht_o05 > 0.01:
                cho.metric("1T Over 0.5", f"{p_ht_o05:.0%}")
            st.caption(
                "1T: λ da **gol 1T stagionali** + **xG FT × quota gol 1T/(1T+2T)** per squadra, "
                "timing (se disponibile), forma/strength, standings/H2H a HT, "
                "ancoraggio 1X2 FT e **Over 1.5** per Over 0.5 a HT."
            )
            if ht_is_est:
                st.caption("_Nessuna tabella HT in estrazione — λ 1T solo da xG FT × quota 1T_")

    # ── Confidenza ───────────────────────────────────────────────────────────
    conf = risultati.model_confidence
    icon = "🟢" if conf >= 0.70 else "🟡" if conf >= 0.40 else "🔴"
    parts = [f"{icon} Confidenza: **{conf:.0%}**"]
    if risultati.stale_line:
        parts.append(" · linea invariata")
    if risultati.model_agreement < 0.70:
        parts.append(f" · accordo modelli: {risultati.model_agreement:.0%}")
    st.caption("  ".join(parts))


# ---------------------------------------------------------------------------
# Analisi Dinamica Live
# ---------------------------------------------------------------------------

def render_analisi_dinamica(
    risultati: ProbabilitaModello,
    state: MatchState,
    gol_tot: int,
    scenario_h: ProbabilitaModello | None = None,
    scenario_a: ProbabilitaModello | None = None,
) -> None:
    """
    Sezione sempre visibile con insight dinamici:
      1. Prossimo gol P(home/away) + P(gol nei 15')
      2. Chi sta dominando (live) + momentum linee
      3. xG breakdown sorgenti (solo se tiri disponibili)
      4. AH linee vicine
      5. Mercati O/U disponibili da gol_tot_dist
      6. Scenario prossimo gol (expander, solo live)
    """
    minuto = state.minuto

    # ── 1. Prossimo gol ───────────────────────────────────────────────────────
    xg_h = risultati.xg_h_final
    xg_a = risultati.xg_a_final
    if xg_h + xg_a > 0.01:
        p_ng_h = xg_h / (xg_h + xg_a)
        if risultati.xg_h_accum + risultati.xg_a_accum > 0.01:
            acc_h = risultati.xg_h_accum
            acc_a = risultati.xg_a_accum
            p_ng_h = 0.5 * p_ng_h + 0.5 * acc_h / (acc_h + acc_a)
        p_ng_a = 1.0 - p_ng_h

        st.markdown("**Prossimo gol**")
        cng1, cng2 = st.columns(2)
        cng1.metric("P(gol Casa)",  f"{p_ng_h:.0%}", delta=f"{p_ng_h-0.5:+.0%} vs 50%", delta_color="normal")
        cng2.metric("P(gol Trasf.)", f"{p_ng_a:.0%}", delta=f"{p_ng_a-0.5:+.0%} vs 50%", delta_color="normal")

        # P(gol nei prossimi 15' e 30')
        if minuto > 0:
            min_rimasti = max(1, 90 - minuto)
            lam_tot = xg_h + xg_a
            p15 = 1.0 - math.exp(-lam_tot * min(15, min_rimasti) / min_rimasti)
            p30 = 1.0 - math.exp(-lam_tot * min(30, min_rimasti) / min_rimasti)
            st.caption(
                f"Prob. gol nei prossimi **15'**: {p15:.0%}  ·  "
                f"nei prossimi **30'**: {p30:.0%}"
            )

    # ── 2. Chi sta dominando + momentum linee ─────────────────────────────────
    if minuto > 0:
        indicatori: list[float] = []

        n_shots = state.sot_h + state.soff_h + state.sot_a + state.soff_a
        if n_shots > 0:
            shots_h = state.sot_h + state.soff_h
            shots_a = state.sot_a + state.soff_a
            indicatori.append((shots_h - shots_a) / (shots_h + shots_a))
        if state.possesso_h > 0 and state.possesso_a > 0:
            indicatori.append((state.possesso_h - state.possesso_a) / 100.0)
        att_tot = state.att_pericolosi_h + state.att_pericolosi_a
        if att_tot > 0:
            indicatori.append((state.att_pericolosi_h - state.att_pericolosi_a) / att_tot)

        caption_parts = []
        if indicatori:
            score = sum(indicatori) / len(indicatori)
            if abs(score) < 0.08:
                dom_label = "Partita equilibrata"
            elif score > 0:
                dom_label = f"Casa domina {'nettamente' if score > 0.25 else 'leggermente'}"
            else:
                dom_label = f"Trasf. domina {'nettamente' if score < -0.25 else 'leggermente'}"
            caption_parts.append(f"Pressione: **{dom_label}**{_dom_detail(state, score)}")

        # Momentum linee di mercato
        if abs(risultati.delta_ah) >= 0.20:
            if risultati.delta_ah > 0:
                caption_parts.append(f"Mercato riduce favore casa (AH {risultati.delta_ah:+.2f})")
            else:
                caption_parts.append(f"Mercato aumenta favore casa (AH {risultati.delta_ah:+.2f})")

        if caption_parts:
            st.caption("  ·  ".join(caption_parts))

    # ── 3. xG sorgenti (solo live con tiri) ───────────────────────────────────
    if minuto > 0 and risultati.xg_h_accum + risultati.xg_a_accum > 0.01:
        st.caption(
            f"xG — linee: {risultati.xg_h_base:.2f}/{risultati.xg_a_base:.2f}  ·  "
            f"tiri accum.: {risultati.xg_h_accum:.2f}/{risultati.xg_a_accum:.2f}  ·  "
            f"blend finale: **{risultati.xg_h_final:.2f}/{risultati.xg_a_final:.2f}** "
            f"(α_D={risultati.alpha_d:.2f})"
        )

    # ── 4. AH linee vicine (solo live) ───────────────────────────────────────
    if minuto > 0:
        ah_cur = state.ah_cur
        ah_rows = []
        for delta in (-0.25, 0.0, +0.25):
            alt = ah_cur + delta
            ph, pa = _prob_ah_cover(risultati.p1, risultati.px, risultati.p2, alt)
            tag = " ◄" if delta == 0.0 else ""
            ah_rows.append((f"AH {alt:+g}{tag}", ph, pa))

        if ah_rows:
            st.markdown("**AH linee vicine**")
            cols = st.columns(len(ah_rows))
            for i, (lbl, ph, pa) in enumerate(ah_rows):
                cols[i].metric(lbl, f"Casa {ph:.0%}", delta=f"Trasf. {pa:.0%}", delta_color="off")

    # ── 5. Mercati O/U disponibili (solo live) ───────────────────────────────
    if minuto > 0:
        dist = risultati.gol_tot_dist
        if dist:
            total_p = sum(dist.values())
            cum: dict[int, float] = {
                k: sum(v for key, v in dist.items() if key >= k) / (total_p or 1.0)
                for k in range(1, 6)
            }
            mercati_ou = []
            for extra in (1, 2, 3):
                linea_f = gol_tot + extra - 0.5
                p_o = cum.get(extra, 0.0)
                mercati_ou += [(f"Over {linea_f:.1f}", p_o), (f"Under {linea_f:.1f}", 1.0 - p_o)]

            interessanti = [(lbl, p) for lbl, p in mercati_ou if 0.15 <= p <= 0.85]
            if interessanti:
                st.markdown("**Mercati O/U disponibili**")
                cols = st.columns(min(len(interessanti), 4))
                for i, (lbl, p) in enumerate(interessanti[:4]):
                    cols[i].metric(lbl, f"{p:.0%}")

    # ── 6. Scenario prossimo gol ──────────────────────────────────────────────
    if minuto > 0 and scenario_h and scenario_a:
        with st.expander("🎯 Scenario: se segna subito..."):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Casa segna** → {state.gol_casa+1}–{state.gol_trasf}")
                sc1.metric("1 Casa",  f"{scenario_h.p1:.0%}", f"{scenario_h.p1-risultati.p1:+.0%}")
                sc1.metric("X",       f"{scenario_h.px:.0%}", f"{scenario_h.px-risultati.px:+.0%}")
                sc1.metric("2 Trasf.",f"{scenario_h.p2:.0%}", f"{scenario_h.p2-risultati.p2:+.0%}")
                ah_sh, ah_sa = _prob_ah_cover(scenario_h.p1, scenario_h.px, scenario_h.p2, ah_cur)
                sc1.caption(f"AH Casa {ah_sh:.0%} · AH Trasf. {ah_sa:.0%}")
            with sc2:
                st.markdown(f"**Trasf. segna** → {state.gol_casa}–{state.gol_trasf+1}")
                sc2.metric("1 Casa",  f"{scenario_a.p1:.0%}", f"{scenario_a.p1-risultati.p1:+.0%}")
                sc2.metric("X",       f"{scenario_a.px:.0%}", f"{scenario_a.px-risultati.px:+.0%}")
                sc2.metric("2 Trasf.",f"{scenario_a.p2:.0%}", f"{scenario_a.p2-risultati.p2:+.0%}")
                ah_ah, ah_aa = _prob_ah_cover(scenario_a.p1, scenario_a.px, scenario_a.p2, ah_cur)
                sc2.caption(f"AH Casa {ah_ah:.0%} · AH Trasf. {ah_aa:.0%}")


def _dom_detail(state: MatchState, score: float) -> str:
    """Costruisce una breve stringa con i dati che guidano il giudizio di dominio."""
    parts = []
    if state.sot_h + state.soff_h + state.sot_a + state.soff_a > 0:
        parts.append(f"tiri {state.sot_h + state.soff_h}–{state.sot_a + state.soff_a}")
    if state.possesso_h > 0:
        parts.append(f"poss. {state.possesso_h:.0f}%–{state.possesso_a:.0f}%")
    return f" ({', '.join(parts)})" if parts else ""


# ---------------------------------------------------------------------------
# Quote Fair
# ---------------------------------------------------------------------------

def _ci_label(ci: dict, key: str, prob: float) -> str:
    """Restituisce la stringa CI inline 'prob · CI @lo–@hi' se disponibile."""
    if key not in ci:
        return f"{prob:.1%}"
    lo, hi = ci[key]
    q_lo = _q_fair(hi) if hi > 0.001 else 999.0
    q_hi = _q_fair(lo) if lo > 0.001 else 999.0
    return f"{prob:.1%} · CI @{q_lo:.2f}–@{q_hi:.2f}"


def render_quote_fair(
    risultati: ProbabilitaModello,
    minuto: int,
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
) -> None:
    """Render delle quote fair con intervalli di credibilità inline (Fix #12)."""
    st.header(f"Quote Fair  —  {minuto}' | {gol_casa}–{gol_trasf}")
    st.caption("Confronta queste quote con quelle sull'exchange: se vedi di meglio, c'è valore. "
               "CI = intervallo di credibilità multi-modello.")

    ci = risultati.credible_intervals

    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa", f"@{_q_fair(risultati.p1):.2f}", _ci_label(ci, "p1", risultati.p1))
    cx.metric("X — Pareggio", f"@{_q_fair(risultati.px):.2f}", _ci_label(ci, "px", risultati.px))
    c2.metric("2 — Trasf.", f"@{_q_fair(risultati.p2):.2f}", _ci_label(ci, "p2", risultati.p2))

    c15u, c15o, cu, co, cb = st.columns(5)
    c15u.metric("Under 1.5", f"@{_q_fair(risultati.p_under_15):.2f}")
    c15o.metric("Over 1.5", f"@{_q_fair(risultati.p_over_15):.2f}")
    cu.metric(f"Under {linea_ou}", f"@{_q_fair(risultati.p_under):.2f}", _ci_label(ci, "p_under", risultati.p_under))
    co.metric(f"Over  {linea_ou}", f"@{_q_fair(risultati.p_over):.2f}", _ci_label(ci, "p_over", risultati.p_over))
    cb.metric("BTTS — Sì", f"@{_q_fair(risultati.p_btts):.2f}", _ci_label(ci, "p_btts", risultati.p_btts))


# ---------------------------------------------------------------------------
# Confidence Band + Model Confidence
# ---------------------------------------------------------------------------


def render_model_confidence(risultati: ProbabilitaModello) -> None:
    """Render del model confidence score con indicazione visuale."""
    conf = risultati.model_confidence
    if conf >= 0.70:
        label = f"Alta ({conf:.0%})"
        icon = "🟢"
    elif conf >= 0.40:
        label = f"Media ({conf:.0%})"
        icon = "🟡"
    else:
        label = f"Bassa ({conf:.0%})"
        icon = "🔴"
    parts = [f"{icon} Confidenza modello: **{label}**"]
    if risultati.stale_line:
        parts.append(" · Linea stantia (nessun movimento)")
    agreement_pct = risultati.model_agreement
    if agreement_pct < 0.70:
        parts.append(f" · Accordo modelli: {agreement_pct:.0%}")
    st.caption("".join(parts))


def render_confidence_bands(
    risultati: ProbabilitaModello,
    linea_ou: float,
) -> None:
    """Render degli intervalli di credibilità basati sullo spread tra modelli."""
    ci = risultati.credible_intervals
    if not ci:
        return

    label_map = {
        "p1": "1 Casa", "px": "X Pareggio", "p2": "2 Trasf.",
        "p_over": f"Over {linea_ou}", "p_under": f"Under {linea_ou}",
        "p_btts": "BTTS Sì",
    }
    prob_map = {
        "p1": risultati.p1, "px": risultati.px, "p2": risultati.p2,
        "p_over": risultati.p_over, "p_under": risultati.p_under,
        "p_btts": risultati.p_btts,
    }

    with st.expander("Intervalli di credibilità (multi-modello)"):
        st.caption("Basati sullo spread tra Bivariate Poisson, CMP+Copula e Markov Chain")
        rows = []
        for key in ["p1", "px", "p2", "p_over", "p_under", "p_btts"]:
            if key not in ci:
                continue
            prob = prob_map[key]
            lo, hi = ci[key]
            q_fair = _q_fair(prob)
            q_lo = _q_fair(hi) if hi > 0.001 else 999.0  # inverso: alta prob → bassa quota
            q_hi = _q_fair(lo) if lo > 0.001 else 999.0
            spread = hi - lo
            icon = "🟢" if spread < 0.02 else ("🟡" if spread < 0.05 else "🔴")
            rows.append(
                f"{icon} **{label_map[key]}**: @{q_fair:.2f} "
                f"(CI: @{q_lo:.2f} – @{q_hi:.2f} · spread {spread:.1%})"
            )
        st.markdown("  \n".join(rows))


# ---------------------------------------------------------------------------
# Coerenza tra mercati
# ---------------------------------------------------------------------------

def render_coerenza_mercati(risultati: ProbabilitaModello, linea_ou: float) -> None:
    """
    Cross-validazione delle probabilità tra mercati diversi.

    Controlla che le probabilità 1X2, O/U e BTTS siano coerenti tra loro.
    Incoerenze possono segnalare instabilità numerica o input errati.
    """
    issues: list[str] = []

    # 1. P(decisione) vs P(Over): se la partita finisce con decisione (non draw),
    #    è probabile che ci siano gol → Over dovrebbe essere moderato/alto
    p_decision = risultati.p1 + risultati.p2
    if p_decision > 0.80 and risultati.p_over < 0.35 and linea_ou <= 2.5:
        issues.append(
            f"Alta probabilità di risultato ({p_decision:.0%}) ma Over {linea_ou} solo {risultati.p_over:.0%} "
            f"— possibile partita decisa da 1-0"
        )

    # 2. BTTS alto + Under bassa linea
    if risultati.p_btts > 0.55 and risultati.p_under > 0.55 and linea_ou <= 2.0:
        issues.append(
            f"BTTS Sì alto ({risultati.p_btts:.0%}) con Under {linea_ou} ({risultati.p_under:.0%}) "
            f"— quasi impossibile: BTTS richiede ≥2 gol"
        )

    # 3. Over alto + BTTS basso → gol concentrati su una squadra
    if risultati.p_over > 0.65 and risultati.p_btts < 0.30:
        issues.append(
            f"Over {linea_ou} probabile ({risultati.p_over:.0%}) ma BTTS solo {risultati.p_btts:.0%} "
            f"— gol previsti da una sola squadra"
        )

    # 4. xG ratio vs 1X2 consistency
    ratio = risultati.xg_h_final / max(risultati.xg_a_final, 0.001)
    if ratio > 2.0 and risultati.p2 > risultati.p1:
        issues.append(
            f"xG ratio {ratio:.1f}:1 a favore casa ma P(2) > P(1) — possibile disallineamento"
        )
    elif ratio < 0.5 and risultati.p1 > risultati.p2:
        issues.append(
            f"xG ratio 1:{1/ratio:.1f} a favore trasf ma P(1) > P(2) — possibile disallineamento"
        )

    if issues:
        with st.expander(f"Coerenza mercati — {len(issues)} avviso/i"):
            for issue in issues:
                st.warning(f"⚠️ {issue}")


# ---------------------------------------------------------------------------
# Allineamento modello vs mercato
# ---------------------------------------------------------------------------

def render_allineamento_mercato(
    risultati: ProbabilitaModello,
    quotes: ExchangeQuotes,
) -> None:
    """Mostra il gap tra probabilità del modello e probabilità implicite del mercato."""
    if not quotes.any_active:
        return

    markets = [
        ("1 Casa", risultati.p1, quotes.q_1),
        ("X Pareggio", risultati.px, quotes.q_x),
        ("2 Trasf.", risultati.p2, quotes.q_2),
        ("BTTS Sì", risultati.p_btts, quotes.q_btts_si),
    ]

    rows = []
    for label, p_mod, q_exc in markets:
        if q_exc <= 1.0:
            continue
        p_imp = 1.0 / q_exc
        gap = p_mod - p_imp
        if abs(gap) < 0.02:
            icon = "🟢"
        elif abs(gap) < 0.05:
            icon = "🟡"
        else:
            icon = "🔴"
        direction = "modello sopra" if gap > 0 else "mercato sopra"
        rows.append(f"{icon} **{label}**: Modello {p_mod:.1%} · Mercato {p_imp:.1%} · Gap **{abs(gap):.1%}** ({direction})")

    if rows:
        with st.expander("Modello vs Mercato"):
            st.markdown("  \n".join(rows))


# ---------------------------------------------------------------------------
# Riepilogo Modello (xG + top score + bias mercato)
# ---------------------------------------------------------------------------

def render_riepilogo_modello(
    risultati: ProbabilitaModello,
    linea_ou: float,
    minuto: int,
) -> None:
    """
    Blocco sempre visibile con i dati chiave del modello:
    - xG attesi per entrambe le squadre
    - Squadra favorita con probabilità
    - Orientamento Over/Under
    - Top 3 correct score attesi
    """
    xh = risultati.xg_h_final
    xa = risultati.xg_a_final

    st.subheader("Riepilogo Modello")

    # ── Riga 1: xG attesi + favorita ────────────────────────────────────────
    col_xg1, col_xg2, col_xg3 = st.columns(3)
    col_xg1.metric("xG Casa", f"{xh:.2f}", help="Goal attesi rimanenti — casa")
    col_xg2.metric("xG Trasf.", f"{xa:.2f}", help="Goal attesi rimanenti — trasferta")

    # Favorita + orientamento mercato gol
    if risultati.p1 > risultati.p2 + 0.08:
        _fav = f"Casa · {risultati.p1:.0%}"
        _fav_delta = f"vs Trasf. {risultati.p2:.0%}"
    elif risultati.p2 > risultati.p1 + 0.08:
        _fav = f"Trasf. · {risultati.p2:.0%}"
        _fav_delta = f"vs Casa {risultati.p1:.0%}"
    else:
        _fav = "Equilibrata"
        _fav_delta = f"1: {risultati.p1:.0%} · X: {risultati.px:.0%} · 2: {risultati.p2:.0%}"
    col_xg3.metric("Favorita", _fav, _fav_delta)

    # ── Riga 2: O/U 1.5 + O/U linea + BTTS + Draw ───────────────────────────
    col_15, col_ou, col_btts, col_x = st.columns(4)
    col_15.metric(
        "Over 1.5",
        f"{risultati.p_over_15:.0%}",
        f"Under {risultati.p_under_15:.0%}",
    )
    _ou_lbl = f"Over {linea_ou:.1f}"
    _ou_val = f"{risultati.p_over:.0%}"
    _ou_dir = f"Under {risultati.p_under:.0%}"
    col_ou.metric(_ou_lbl, _ou_val, _ou_dir)
    col_btts.metric("BTTS Sì", f"{risultati.p_btts:.0%}", f"No {1 - risultati.p_btts:.0%}")
    col_x.metric("Pareggio", f"{risultati.px:.0%}", f"Fair @{_q_fair(risultati.px):.2f}")

    # ── Top 3 Correct Score ─────────────────────────────────────────────────
    if risultati.top_cs:
        st.caption("**Punteggi più probabili**")
        cs_cols = st.columns(min(3, len(risultati.top_cs)))
        for idx, ((fc, ft), prob) in enumerate(risultati.top_cs[:3]):
            cs_cols[idx].metric(
                label=f"{fc}–{ft}",
                value=f"{prob:.1%}",
                delta=f"fair @{_q_fair(prob):.2f}",
            )


# ---------------------------------------------------------------------------
# Correct Score + Distribuzione Gol
# ---------------------------------------------------------------------------

def render_correct_score(
    risultati: ProbabilitaModello,
) -> None:
    """Render del Correct Score top-N e della distribuzione dei gol totali."""
    with st.expander("Correct Score — Top 5 + Totali"):
        cs_cols = st.columns(UI.TOP_CS_COUNT)
        for idx, ((fc, ft), prob) in enumerate(risultati.top_cs):
            cs_cols[idx].metric(
                label=f"{fc}–{ft}",
                value=f"@{_q_fair(prob):.2f}",
                delta=f"{prob:.2%}",
            )

        st.caption("**Distribuzione gol totali partita**")
        max_tot = max(risultati.gol_tot_dist.keys()) if risultati.gol_tot_dist else 6
        tot_rows = []
        cum = 0.0
        for t in range(min(max_tot + 1, UI.MAX_GOL_DIST)):
            p = risultati.gol_tot_dist.get(t, 0.0)
            cum += p
            tot_rows.append(f"**{t} gol**: {p:.1%} · cumul. ≤{t}: {cum:.1%}")
        st.markdown("  \n".join(tot_rows))


# ---------------------------------------------------------------------------
# Asian Handicap
# ---------------------------------------------------------------------------

def render_asian_handicap(full_matrix: dict) -> None:
    """Render della tabella AH a diversi livelli."""
    with st.expander("Asian Handicap — Probabilità a diversi livelli"):
        st.caption(
            "Handicap sui **gol rimanenti** — coerente con la linea AH Corrente inserita sopra. "
            "Usa la matrice bivariate completa (Dixon-Coles + correlazione)."
        )
        risultati_ah = calcola_asian_handicap(full_matrix, UI.AH_LEVELS)
        ah_rows = []
        for r in risultati_ah:
            sign = "+" if r["level"] >= 0 else ""
            push_txt = f" · push {r['p_push']:.1%}" if r["p_push"] > 0.005 else ""
            ah_rows.append(
                f"AH {sign}{r['level']:.1f} → Casa copre: **{r['p_eff']:.1%}** "
                f"· fair @{r['quota_fair']:.2f}{push_txt}"
            )
        st.markdown("  \n".join(ah_rows))


# ---------------------------------------------------------------------------
# Clean Sheet & Win to Nil
# ---------------------------------------------------------------------------

def render_clean_sheet(
    full_matrix: dict,
    gol_casa: int,
    gol_trasf: int,
) -> tuple[float, float]:
    """
    Render delle probabilità Clean Sheet e Win to Nil.

    Returns:
        (p_clean_casa, p_clean_trasf) per uso in altri componenti.
    """
    from src.markets.clean_sheet import calcola_clean_sheet

    p_clean_casa, p_clean_trasf = calcola_clean_sheet(full_matrix, gol_casa, gol_trasf)

    with st.expander("🧤 Clean Sheet & Win to Nil"):
        # Determina lo stato dei clean sheet
        if gol_trasf > 0 and gol_casa > 0:
            st.info("🚫 Clean Sheet impossibile per entrambe le squadre (hanno già segnato)")
            return p_clean_casa, p_clean_trasf

        cs_cols = st.columns(2)

        # Clean Sheet Casa
        if gol_trasf > 0:
            cs_cols[0].metric(
                "🧤 Clean Sheet Casa",
                value="IMPOSSIBILE",
                delta="Trasferta ha segnato",
                delta_color="off"
            )
        else:
            cs_cols[0].metric(
                "🧤 Clean Sheet Casa",
                value=f"{p_clean_casa:.1%}",
                delta=f"@{_q_fair(p_clean_casa):.2f}"
            )

        # Clean Sheet Trasferta
        if gol_casa > 0:
            cs_cols[1].metric(
                "🧤 Clean Sheet Trasferta",
                value="IMPOSSIBILE",
                delta="Casa ha segnato",
                delta_color="off"
            )
        else:
            cs_cols[1].metric(
                "🧤 Clean Sheet Trasferta",
                value=f"{p_clean_trasf:.1%}",
                delta=f"@{_q_fair(p_clean_trasf):.2f}"
            )

        st.caption(
            "Clean Sheet = la squadra non subisce gol per tutta la partita. "
            "Se l'avversario ha già segnato, il mercato è chiuso."
        )

        # Win to Nil (solo se clean sheet è possibile)
        if (gol_trasf == 0 or gol_casa == 0):
            st.divider()
            st.markdown("**🏆 Win to Nil** (vince senza subire gol)")

            # Calcola Win to Nil usando le probabilità 1X2 se disponibili
            # Per ora usiamo una stima basata sulla matrice
            p_wtn_casa = sum(p for (a, b), p in full_matrix.items()
                             if gol_casa + a > gol_trasf + b and b == 0)
            p_wtn_trasf = sum(p for (a, b), p in full_matrix.items()
                              if gol_trasf + b > gol_casa + a and a == 0)

            wtn_cols = st.columns(2)

            if gol_trasf > 0:
                wtn_cols[0].metric("Casa WtN", value="—", delta_color="off")
            else:
                wtn_cols[0].metric(
                    "Casa WtN",
                    value=f"{p_wtn_casa:.1%}",
                    delta=f"@{_q_fair(p_wtn_casa):.2f}"
                )

            if gol_casa > 0:
                wtn_cols[1].metric("Trasferta WtN", value="—", delta_color="off")
            else:
                wtn_cols[1].metric(
                    "Trasferta WtN",
                    value=f"{p_wtn_trasf:.1%}",
                    delta=f"@{_q_fair(p_wtn_trasf):.2f}"
                )

            st.caption(
                "Win to Nil = la squadra vince senza subire gol. "
                "Richiede che la squadra vinca E mantenga il clean sheet."
            )

    return p_clean_casa, p_clean_trasf


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def render_momentum(
    momentum: float,
    delta_ah: float = 0.0,
    delta_tot: float = 0.0,
) -> None:
    """Render della barra del momentum con decomposizione AH/Total (Fix #14)."""
    if momentum < MOMENTUM_CFG.STABLE_THRESHOLD:
        label = f"Mercato stabile [{momentum:.2f}/6.0]"
    elif momentum < MOMENTUM_CFG.MODERATE_THRESHOLD:
        label = f"Movimento moderato [{momentum:.2f}/6.0]"
    elif momentum < MOMENTUM_CFG.SIGNIFICANT_THRESHOLD:
        label = f"Movimento significativo [{momentum:.2f}/6.0]"
    else:
        label = f"Movimento estremo — verifica eventi non registrati [{momentum:.2f}/6.0]"

    st.progress(min(momentum / MOMENTUM_CFG.MOMENTUM_CAP, 1.0), text=label)

    # Decomposizione AH vs Total (Fix #14):
    # Delta AH → suggerisce revisione forza relativa → impatto su 1X2 / AH
    # Delta Total → suggerisce revisione gol attesi → impatto su Over/Under
    # Market shock warning
    from src.config import MOMENTUM as _MOM_CFG
    if momentum >= _MOM_CFG.MOMENTUM_SHOCK_THRESHOLD:
        st.error(
            "⚡ **MARKET SHOCK** — Le linee si sono mosse in modo anomalo rispetto al tempo giocato. "
            "Possibile informazione asimmetrica (infortuni, formazioni, notizie non registrate). "
            "I segnali del modello hanno affidabilità ridotta fino all'aggiornamento delle linee."
        )

    if abs(delta_ah) > 0.05 or abs(delta_tot) > 0.05:
        ah_dir = ("+" if delta_ah > 0 else "") + f"{delta_ah:+.2f}"
        tot_dir = ("+" if delta_tot > 0 else "") + f"{delta_tot:+.2f}"
        ah_impact = "→ AH mosso: revisiona 1X2 e Asian Handicap" if abs(delta_ah) > abs(delta_tot) else ""
        tot_impact = "→ Total mosso: revisiona Over/Under" if abs(delta_tot) >= abs(delta_ah) else ""
        st.caption(
            f"Δ AH: **{ah_dir}** {ah_impact}  ·  Δ Total: **{tot_dir}** {tot_impact}"
        )


# ---------------------------------------------------------------------------
# Segnali
# ---------------------------------------------------------------------------

def _bet_difficulty(quota_fair: float) -> tuple[str, str]:
    """
    Calcola un indicatore di difficoltà nel trovare il segnale a valore.

    La difficoltà dipende dalla liquidità del mercato:
    - Quote basse (< 1.5): mercato molto liquido, hard to beat (spread stretto)
    - Quote medie (1.5-4.0): bilanciato
    - Quote alte (> 4.0): mercato meno efficiente, più facile trovare value

    Returns:
        (emoji, label) descrittivi della difficoltà.
    """
    if quota_fair < 1.50:
        return "🔴", "Alta liquidità — spread stretto"
    elif quota_fair < 2.50:
        return "🟡", "Liquidità media"
    elif quota_fair < 4.00:
        return "🟢", "Buona liquidità"
    else:
        return "🔵", "Mercato di nicchia — spread più ampio"


def render_segnali_rapidi(segnali: list[Signal]) -> bool:
    """
    Render dei segnali rapidi (senza quote exchange).

    Returns:
        True se almeno un segnale è stato mostrato.
    """
    if not segnali:
        st.info("Nessun candidato forte al momento — il modello non vede probabilità dominanti.")
        return False

    for s in segnali:
        # Fix #13: mostra probabilità implicita della quota target (1/quota)
        prob_impl_back = 1.0 / s.quota_exc if s.quota_exc > 1.0 else 0.0
        diff_emoji, diff_label = _bet_difficulty(s.quota_fair)
        cluster_note = ""
        cluster_items = [r for r in s.riduzioni if r.startswith("cluster:")]
        if cluster_items:
            cluster_note = f"\n\n⚠️ _Correlato con: {cluster_items[0].replace('cluster: ', '')}_"
        if s.tipo == "INFO_BACK":
            impl_txt = f" · Prob. implicita @{s.quota_exc:.2f} = {prob_impl_back:.1%}" if prob_impl_back > 0 else ""
            st.success(
                f"**BACK candidato — {s.mercato}** · Modello {s.prob_mod:.1%} · Fair @{s.quota_fair:.2f}\n\n"
                f"✅ Cerca sull'exchange **almeno @{s.quota_exc:.2f}**{impl_txt}\n\n"
                f"{diff_emoji} {diff_label}{cluster_note}"
            )
        elif s.tipo == "INFO_LAY":
            prob_impl_lay = 1.0 / s.quota_exc if s.quota_exc > 1.0 else 0.0
            impl_txt = f" · Prob. implicita = {prob_impl_lay:.1%}" if prob_impl_lay > 0 else ""
            st.warning(
                f"**LAY candidato — {s.mercato}** · Modello {s.prob_mod:.1%} · Fair @{s.quota_fair:.2f}\n\n"
                f"✅ Banca se la quota sull'exchange è **al massimo @{s.quota_exc:.2f}**{impl_txt}\n\n"
                f"{diff_emoji} {diff_label}{cluster_note}"
            )

    return True


def render_segnali_avanzati(segnali: list[Signal], any_exc_quote: bool) -> None:
    """Render dei segnali avanzati con Kelly/EV."""
    for s in segnali:
        rid_txt = f" _({', '.join(s.riduzioni)})_" if s.riduzioni else ""
        prob_imp_txt = f"({s.prob_implicita:.1%})" if s.quota_exc > 0 else ""

        if s.tipo == "BACK":
            st.success(
                f"**PUNTA {s.mercato}** — "
                f"Modello {s.prob_mod:.1%} · Mercato @{s.quota_exc:.2f} {prob_imp_txt} · "
                f"Edge netto **+{s.edge*100:.1f}%** · EV **€{s.ev_euro:+.2f}**"
            )
            # Fix #15: breakdown della stake Kelly
            if s.kelly_raw > 0 and s.kelly_raw != s.stake:
                fraction_applied = s.stake / s.kelly_raw if s.kelly_raw > 0 else 1.0
                st.write(
                    f"Stake Kelly: **€{s.stake:.2f}**{rid_txt}  "
                    f"_(Raw 100% Kelly: €{s.kelly_raw:.2f} → ×{fraction_applied:.2f} fattori applicati)_"
                )
            else:
                st.write(f"Stake Kelly: **€{s.stake:.2f}**{rid_txt}")

        elif s.tipo == "LAY":
            st.warning(
                f"**BANCA {s.mercato}** — "
                f"Modello {s.prob_mod:.1%} · Mercato @{s.quota_exc:.2f} {prob_imp_txt} · "
                f"Edge lay netto **+{s.edge*100:.1f}%** · EV **€{s.ev_euro:+.2f}**"
            )
            st.write(f"Stake lay: **€{s.stake:.2f}** · Liability: **€{s.liability:.2f}**{rid_txt}")

        elif s.tipo == "INFO_BACK":
            st.info(
                f"📈 **{s.mercato}**: modello {s.prob_mod:.1%} (fair @{s.quota_fair:.2f}) "
                f"— potenziale back · inserisci quota per edge preciso"
            )

        elif s.tipo == "INFO_LAY":
            st.info(
                f"📉 **{s.mercato}**: modello {s.prob_mod:.1%} (fair @{s.quota_fair:.2f}) "
                f"— mercato probabilmente sopravvaluta · considera lay · inserisci quota per conferma"
            )

    if not segnali:
        if any_exc_quote:
            st.error("NO BET — Quote exchange allineate al modello, nessun edge sufficiente.")
        else:
            st.info("Nessuna indicazione forte. Il modello non vede probabilità dominanti.")


# ---------------------------------------------------------------------------
# Mercati chiusi
# ---------------------------------------------------------------------------

def render_mercati_chiusi(
    gol_attuali: int,
    linea_ou: float,
    gol_casa: int,
    gol_trasf: int,
    minuto: int,
    p_btts: float,
) -> dict[str, bool]:
    """
    Renderizza i messaggi per i mercati già chiusi o quasi.

    Returns:
        Dict {mercato: settled} per escludere i mercati chiusi dalla valutazione.
    """
    settled = {
        "ou_vinto": False,
        "btts_si_settled": False,
        "btts_no_settled": False,
    }

    if gol_attuali >= linea_ou:
        settled["ou_vinto"] = True

    btts_si_settled = gol_casa > 0 and gol_trasf > 0
    btts_no_settled = (
        minuto >= SIGNALS.BTTS_NO_SETTLED_MINUTE
        and (gol_casa == 0 or gol_trasf == 0)
        and p_btts < SIGNALS.BTTS_NO_SETTLED_PROB_THRESHOLD
    )

    if btts_si_settled:
        settled["btts_si_settled"] = True
    elif btts_no_settled:
        settled["btts_no_settled"] = True

    # Mostra i mercati già chiusi in un'unica riga compatta
    chiusi = []
    if settled["ou_vinto"]:
        chiusi.append(f"Over {linea_ou} ✅ vinto · Under {linea_ou} ❌ perso")
    if settled["btts_si_settled"]:
        chiusi.append("BTTS Sì ✅ vinto · BTTS No ❌ perso")
    elif settled["btts_no_settled"]:
        chiusi.append("BTTS Sì ❌ quasi impossibile · BTTS No ✅ quasi vinto")
    if chiusi:
        st.info("Mercati chiusi: " + " · ".join(chiusi))

    return settled


# ---------------------------------------------------------------------------
# Impatto cartellini rossi (Fix #16)
# ---------------------------------------------------------------------------

def render_red_card_impact(
    rossi_casa: int,
    rossi_trasf: int,
    minuto: int,
) -> None:
    """Mostra il delta xG quantificato per i cartellini rossi inseriti (Fix #16)."""
    if rossi_casa == 0 and rossi_trasf == 0:
        return

    time_remaining_pct = max(0.0, (90.0 - minuto) / 90.0)
    red_time_mult = DECAY.RED_TIME_FLOOR + (1.0 - DECAY.RED_TIME_FLOOR) * time_remaining_pct

    rows = []
    if rossi_casa > 0:
        idx = min(rossi_casa, len(DECAY.RED_DECAY) - 1)
        decay_eff = 1.0 + (DECAY.RED_DECAY[idx] - 1.0) * red_time_mult
        boost_eff = 1.0 + (DECAY.RED_BOOST[idx] - 1.0) * red_time_mult
        pct_decay = (decay_eff - 1.0) * 100
        pct_boost = (boost_eff - 1.0) * 100
        rows.append(
            f"🟥 **{rossi_casa} rosso/i CASA**: attacco casa **{pct_decay:+.0f}%** xG · "
            f"attacco trasf **{pct_boost:+.0f}%** xG · "
            f"(effetto temporale {red_time_mult:.0%})"
        )
    if rossi_trasf > 0:
        idx = min(rossi_trasf, len(DECAY.RED_DECAY) - 1)
        decay_eff = 1.0 + (DECAY.RED_DECAY[idx] - 1.0) * red_time_mult * DECAY.RED_AWAY_PENALTY
        boost_eff = 1.0 + (DECAY.RED_BOOST[idx] - 1.0) * red_time_mult * DECAY.RED_HOME_BOOST
        pct_decay = (decay_eff - 1.0) * 100
        pct_boost = (boost_eff - 1.0) * 100
        rows.append(
            f"🟥 **{rossi_trasf} rosso/i TRASF**: attacco trasf **{pct_decay:+.0f}%** xG · "
            f"attacco casa **{pct_boost:+.0f}%** xG · "
            f"(penalità trasferta +5% inclusa, effetto temporale {red_time_mult:.0%})"
        )

    if rows:
        with st.expander("Impatto cartellini rossi"):
            for r in rows:
                st.markdown(r)


# ---------------------------------------------------------------------------
# Avvisi di affidabilità
# ---------------------------------------------------------------------------

def render_avvisi_affidabilita(
    flat_lines: bool,
    n_shots_tot: int,
    minuto: int,
) -> None:
    """Render degli avvisi quando l'affidabilità del modello è ridotta."""
    if flat_lines and n_shots_tot == 0 and minuto >= 30:
        st.warning(
            "⚠️ **Affidabilità ridotta**: linee di mercato invariate e nessun tiro live. "
            "Il modello opera su prior di prematch — inserisci i tiri per aumentare la precisione."
        )
    elif n_shots_tot == 0 and minuto >= 45:
        st.warning(
            "⚠️ **Nessun dato tiri live**: oltre il 45' senza tiri, "
            "il modello ha affidabilità ridotta. Inserisci i tiri per sbloccare l'alpha."
        )


def render_lines_need_update(risultati: ProbabilitaModello) -> None:
    """Render avviso se le linee sembrano non aggiornate dopo i gol."""
    if risultati.lines_need_update:
        st.error(
            "🚨 **ATTENZIONE**: Le linee di mercato sembrano NON AGGIORNATE! "
            "Sono stati segnati gol ma AH e Total sono ancora ai valori d'apertura. "
            "**Aggiorna le linee live** per ottenere previsioni accurate."
        )


# ---------------------------------------------------------------------------
# Avvisi incoerenza
# ---------------------------------------------------------------------------

def render_avvisi_incoerenza(
    p_btts: float,
    p_under: float,
    p_over: float,
    linea_ou: float,
    gol_attuali: int,
    btts_settled: bool,
) -> None:
    """Render degli avvisi di incoerenza logica tra mercati."""
    # BTTS Sì + Under bassa: impossibile (BTTS richiede ≥2 gol)
    if (
        p_btts > SIGNALS.BTTS_UNDER_INCOHERENCE_BTTS
        and p_under > SIGNALS.BTTS_UNDER_INCOHERENCE_UNDER
        and linea_ou <= SIGNALS.BTTS_UNDER_MAX_LINE
        and not btts_settled
    ):
        st.warning(
            f"⚠️ Incoerenza interna: BTTS Sì ({p_btts:.0%}) e Under {linea_ou} "
            f"({p_under:.0%}) sono incompatibili — BTTS Sì richiede ≥2 gol totali. "
            "Verifica le linee inserite."
        )

    # Over + BTTS No improbabile
    if (
        p_over > SIGNALS.OVER_BTTS_INCOHERENCE_OVER
        and p_btts < SIGNALS.OVER_BTTS_INCOHERENCE_BTTS
        and gol_attuali < linea_ou
    ):
        st.warning(
            f"⚠️ Incoerenza: Over {linea_ou} probabile ({p_over:.0%}) "
            f"ma BTTS solo {p_btts:.0%}. "
            "Over senza BTTS richiede gol multipli dalla stessa squadra."
        )


# ---------------------------------------------------------------------------
# Debug panel
# ---------------------------------------------------------------------------

def render_debug(
    risultati: ProbabilitaModello,
    linea_ou: float,
    minuto: int,
    soglie: dict,
    comm_pct: float,
    momentum_factor: float,
    n_shots_tot: int,
) -> None:
    """Render del pannello di debug con tutti i parametri interni del motore."""
    with st.expander("Parametri interni del motore"):
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Probabilità modello**")
            st.write(f"P(1)       = {risultati.p1:.4f}")
            st.write(f"P(X)       = {risultati.px:.4f}")
            st.write(f"P(2)       = {risultati.p2:.4f}")
            sum_1x2 = risultati.p1 + risultati.px + risultati.p2
            ok = "✅" if abs(sum_1x2 - 1.0) < 0.001 else "⚠️"
            st.write(f"Σ(1+X+2)   = {sum_1x2:.4f} {ok}")
            st.divider()
            st.write(f"P(U{linea_ou}) = {risultati.p_under:.4f}")
            st.write(f"P(O{linea_ou}) = {risultati.p_over:.4f}")
            sum_ou = risultati.p_under + risultati.p_over
            ok = "✅" if abs(sum_ou - 1.0) < 0.001 else "⚠️"
            st.write(f"Σ(U+O)     = {sum_ou:.4f} {ok}")
            st.divider()
            st.write(f"P(BTTS)    = {risultati.p_btts:.4f}")
            st.write(f"P(BTTSno)  = {1.0 - risultati.p_btts:.4f}")

        with col_r:
            st.markdown("**Parametri motore**")
            st.write(f"rho dinamico  = {risultati.rho:.4f}")
            st.write(f"lambda casa   = {risultati.xg_h_final:.4f}")
            st.write(f"lambda trasf  = {risultati.xg_a_final:.4f}")
            st.write(f"xG base casa  = {risultati.xg_h_base:.4f}")
            st.write(f"xG base trasf = {risultati.xg_a_base:.4f}")
            st.divider()

            if n_shots_tot > 0:
                st.markdown("**Tiri & blend**")
                st.write(f"xG accum casa  = {risultati.xg_h_accum:.3f}")
                st.write(f"xG accum trasf = {risultati.xg_a_accum:.3f}")
                st.write(f"xG blend casa  = {risultati.xg_h_blend:.4f}")
                st.write(f"xG blend trasf = {risultati.xg_a_blend:.4f}")
                st.write(f"α_T (Total)    = {risultati.alpha_t:.3f}")
                st.write(f"α_D (Diff)     = {risultati.alpha_d:.3f}")
                st.write(f"Shot dominance = {risultati.shot_dom:.3f}")
                st.divider()

            gol_bonus = soglie.get("gol_mancanti", 0.0)
            st.write(f"Soglia 1X2     = {soglie.get('1x2', 0):.3f}")
            st.write(f"Soglia Over    = {soglie.get('ou_over', 0):.3f}  (+{min(SIGNALS.OVER_GOL_BONUS_CAP, max(0.0, (gol_bonus - 1.0) * SIGNALS.OVER_GOL_BONUS_RATE)):.2f} gol-bonus)")
            st.write(f"Soglia Under   = {soglie.get('ou_under', 0):.3f}")
            st.write(f"Soglia BTTS    = {soglie.get('btts_si', 0):.3f}")
            st.divider()
            st.write(f"Momentum       = {risultati.momentum:.2f}/6.0")
            st.write(f"Stake factor   = ×{momentum_factor:.2f}")
            st.write(f"Comm           = {comm_pct:.1f}%")
            st.write(f"Confidence     = {risultati.model_confidence:.2f}")


# ---------------------------------------------------------------------------
# Value Bet Detection
# ---------------------------------------------------------------------------

def render_value_bets(
    risultati: ProbabilitaModello,
    quotes: ExchangeQuotes,
    min_edge: float = 0.03,
) -> list:
    """
    Render automatic value bet detection.

    Args:
        risultati: Model output.
        quotes: Exchange quotes.
        min_edge: Minimum edge threshold.

    Returns:
        List of detected value bets.
    """
    from src.utils.analytics import detect_value_bets

    value_bets = detect_value_bets(risultati, quotes, min_edge=min_edge)

    if not value_bets:
        if quotes.any_active:
            st.info("📊 Nessun value bet rilevato — mercato allineato al modello.")
        return []

    st.subheader("🎯 Value Bet Rilevati")

    for vb in value_bets[:5]:  # Show top 5
        # Determine color based on quality
        if vb.quality_score >= 0.7:
            color = "🟢"
        elif vb.quality_score >= 0.4:
            color = "🟡"
        else:
            color = "🟠"

        with st.container():
            st.markdown(f"{color} **{vb.bet_type} {vb.market}** @{vb.odds:.2f}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Edge", f"+{vb.edge_percentage:.1f}%")
            col2.metric("EV", f"+{vb.ev_percentage:.1f}%")
            col3.metric("Kelly", f"{vb.kelly_fraction:.1%}")
            st.caption(
                f"Modello: {vb.prob_model:.1%} · Mercato: {vb.prob_market:.1%} · "
                f"Confidenza: {vb.confidence:.0%} · Quality: {vb.quality_score:.0%}"
            )
            st.divider()

    return value_bets


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def render_anomalies(state: MatchState) -> list:
    """
    Render anomaly detection warnings.

    Args:
        state: Match state to analyze.

    Returns:
        List of detected anomalies.
    """
    from src.utils.anomaly import detect_input_anomalies

    anomalies = detect_input_anomalies(state)

    if not anomalies:
        return []

    # Count by severity
    errors = [a for a in anomalies if a.severity == "error"]
    warnings = [a for a in anomalies if a.severity == "warning"]
    infos = [a for a in anomalies if a.severity == "info"]

    if errors:
        st.error(f"🔴 **{len(errors)} errore/i critico/i** nei dati inseriti")
    if warnings:
        st.warning(f"🟡 **{len(warnings)} warning** — verificare i dati")
    if infos and (errors or warnings):
        st.info(f"🔵 {len(infos)} notifiche addizionali")

    # Show details in expander
    with st.expander("🔍 Dettaglio anomalie", expanded=bool(errors)):
        for a in anomalies:
            icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(a.severity, "⚪")
            st.markdown(f"{icon} **{a.message}**")
            if a.suggestion:
                st.caption(f"💡 _{a.suggestion}_")

    return anomalies


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

def render_risk_metrics(
    segnali: list,
    bankroll: float,
) -> dict:
    """
    Render risk metrics for potential bets.

    Args:
        segnali: List of signals with stake/ev info.
        bankroll: Current bankroll.

    Returns:
        Dict with risk metrics.
    """
    from src.utils.analytics import calculate_risk_metrics

    if not segnali:
        return {}

    # Extract expected returns and probabilities
    expected_returns = []
    probabilities = []

    for s in segnali:
        if s.stake > 0:
            ev_pct = s.ev_euro / s.stake if s.stake > 0 else 0
            expected_returns.append(ev_pct)
            probabilities.append(s.prob_mod)

    if not expected_returns:
        return {}

    metrics = calculate_risk_metrics(expected_returns, probabilities, bankroll)

    # Display
    with st.expander("📊 Metriche di Rischio"):
        col1, col2, col3 = st.columns(3)
        col1.metric("EV Totale", f"€{metrics['total_ev']:.2f}")
        col2.metric("Rischio (σ)", f"€{metrics['total_risk']:.2f}")
        col3.metric("Risk/Bankroll", f"{metrics['risk_per_bankroll']:.1%}")

        st.divider()

        col4, col5 = st.columns(2)
        col4.metric("Diversificazione", f"{metrics['diversification']:.0%}")
        col5.metric("Max Rischio Singolo", f"€{metrics['max_single_risk']:.2f}")

        # Interpretation
        if metrics['risk_per_bankroll'] > 0.10:
            st.warning("⚠️ Rischio totale >10% del bankroll — considera ridurre le stake")
        elif metrics['diversificazione'] < 0.3:
            st.info("ℹ️ Bassa diversificazione — gli esiti sono correlati")
        else:
            st.success("✅ Profil rischio bilanciato")

    return metrics


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OCR Market Divergence Warning (#12)
# ---------------------------------------------------------------------------

_OCR_DIVERGENCE_THRESHOLD = 0.15  # 15% di scarto per mostrare warning


def render_ocr_market_divergence(
    risultati: ProbabilitaModello,
    ocr_quota_1: float,
    ocr_quota_x: float,
    ocr_quota_2: float,
    ocr_quota_over: float,
    ocr_quota_under: float,
) -> None:
    """
    Mostra un warning se le quote OCR divergono >15% dalle probabilità del modello.

    Le quote OCR sono quelle estratte dallo screenshot del sito di scommesse.
    Il modello stima le probabilità dalle linee AH/Total.
    Una divergenza >15% indica:
      - Possibile errore nell'inserimento manuale AH/Total
      - Quote OCR da mercato diverso (es. European 1X2 vs Asian)
      - Vera inefficienza di mercato da investigare

    Args:
        risultati: Probabilità calcolate dal modello.
        ocr_quota_*: Quote estratte dall'OCR (0 = non disponibile).
    """
    divergences: list[str] = []

    # Controlla mercato 1X2
    if ocr_quota_1 > 1.0 and ocr_quota_x > 1.0 and ocr_quota_2 > 1.0:
        inv1 = 1.0 / ocr_quota_1
        invx = 1.0 / ocr_quota_x
        inv2 = 1.0 / ocr_quota_2
        tot = inv1 + invx + inv2
        if tot > 0:
            ocr_p1 = inv1 / tot
            ocr_px = invx / tot
            ocr_p2 = inv2 / tot
            if abs(risultati.p1 - ocr_p1) > _OCR_DIVERGENCE_THRESHOLD:
                divergences.append(
                    f"**1 (Casa)**: modello {risultati.p1:.1%} · mercato OCR {ocr_p1:.1%} "
                    f"(Δ {abs(risultati.p1 - ocr_p1):.1%})"
                )
            if abs(risultati.px - ocr_px) > _OCR_DIVERGENCE_THRESHOLD:
                divergences.append(
                    f"**X (Pareggio)**: modello {risultati.px:.1%} · mercato OCR {ocr_px:.1%} "
                    f"(Δ {abs(risultati.px - ocr_px):.1%})"
                )
            if abs(risultati.p2 - ocr_p2) > _OCR_DIVERGENCE_THRESHOLD:
                divergences.append(
                    f"**2 (Trasf.)**: modello {risultati.p2:.1%} · mercato OCR {ocr_p2:.1%} "
                    f"(Δ {abs(risultati.p2 - ocr_p2):.1%})"
                )

    # Controlla mercato O/U
    if ocr_quota_over > 1.0 and ocr_quota_under > 1.0:
        inv_o = 1.0 / ocr_quota_over
        inv_u = 1.0 / ocr_quota_under
        tot_ou = inv_o + inv_u
        if tot_ou > 0:
            ocr_p_over = inv_o / tot_ou
            if abs(risultati.p_over - ocr_p_over) > _OCR_DIVERGENCE_THRESHOLD:
                divergences.append(
                    f"**Over**: modello {risultati.p_over:.1%} · mercato OCR {ocr_p_over:.1%} "
                    f"(Δ {abs(risultati.p_over - ocr_p_over):.1%})"
                )

    if not divergences:
        return

    with st.expander("⚠️ Divergenza modello–mercato OCR", expanded=True):
        st.warning(
            "Le quote OCR estratte dallo screenshot divergono dal modello. "
            "Possibili cause: AH/Total inseriti non coerenti con le quote, "
            "mercato diverso (European vs Asian), o vera opportunità di value."
        )
        for d in divergences:
            st.markdown(f"• {d}")


# ---------------------------------------------------------------------------
# Session Statistics
# ---------------------------------------------------------------------------

def render_session_stats() -> None:
    """Render session statistics including cache performance."""
    from src.utils.memo import get_cache_stats

    stats = get_cache_stats()

    if stats["size"] > 0:
        with st.expander("📈 Statistiche Sessione"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Cache Hit Rate", f"{stats['hit_rate']:.0%}")
            col2.metric("Cache Size", f"{stats['size']}/{stats['max_size']}")
            col3.metric("TTL", f"{stats['ttl_seconds']:.0f}s")
