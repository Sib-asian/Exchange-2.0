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
from typing import TYPE_CHECKING

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

def render_pronostici_rapidi(
    risultati: ProbabilitaModello,
    linea_ou: float,
    minuto: int = 0,
    gol_casa: int = 0,
    gol_trasf: int = 0,
    linea_ah: float = -0.25,
) -> None:
    """
    Blocco percentuali pulito per tutti i mercati principali.
    Niente quote, niente CI, niente robe strane — solo le probabilità.
    """
    if minuto > 0:
        st.subheader(f"Pronostici — {minuto}' | {gol_casa}–{gol_trasf}")
    else:
        st.subheader("Pronostici Prematch")

    # 1X2
    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa",     f"{risultati.p1:.0%}")
    cx.metric("X — Pareggio", f"{risultati.px:.0%}")
    c2.metric("2 — Trasf.",   f"{risultati.p2:.0%}")

    st.divider()

    # AH cover probability — sempre visibile, direttamente sull'Asian Handicap
    p_ah_h, p_ah_a = _prob_ah_cover(risultati.p1, risultati.px, risultati.p2, linea_ah)
    ah_sign = "+" if linea_ah >= 0 else ""
    cah1, cah2 = st.columns(2)
    delta_h = p_ah_h - 0.5
    delta_a = p_ah_a - 0.5
    cah1.metric(
        f"AH Casa ({ah_sign}{linea_ah:g})",
        f"{p_ah_h:.0%}",
        delta=f"{delta_h:+.0%} vs 50%",
        delta_color="normal",
    )
    _ah_away_sign = "+" if linea_ah <= 0 else "-"
    cah2.metric(
        f"AH Trasf. ({_ah_away_sign}{abs(linea_ah):g})",
        f"{p_ah_a:.0%}",
        delta=f"{delta_a:+.0%} vs 50%",
        delta_color="normal",
    )

    st.divider()

    # Over/Under + BTTS
    co, cu, cgg, cng = st.columns(4)
    co.metric(f"Over {linea_ou}",  f"{risultati.p_over:.0%}")
    cu.metric(f"Under {linea_ou}", f"{risultati.p_under:.0%}")
    cgg.metric("GG (sì)",          f"{risultati.p_btts:.0%}")
    cng.metric("NG (no)",          f"{1 - risultati.p_btts:.0%}")

    # Confidenza modello
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

    # ── 4. AH linee vicine ────────────────────────────────────────────────────
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

    # ── 5. Mercati O/U disponibili ────────────────────────────────────────────
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

    cu, co, cb = st.columns(3)
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

    # ── Riga 2: O/U + BTTS + Draw ───────────────────────────────────────────
    col_ou, col_btts, col_x = st.columns(3)
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
