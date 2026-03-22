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

import streamlit as st

from src.config import MOMENTUM as MOMENTUM_CFG
from src.config import SIGNALS, UI
from src.engine import ExchangeQuotes, ProbabilitaModello
from src.markets.asian_handicap import calcola_asian_handicap
from src.signals import Signal


def _q_fair(prob: float) -> float:
    return 1.0 / prob if prob > 0.001 else 999.0


# ---------------------------------------------------------------------------
# Quote Fair
# ---------------------------------------------------------------------------

def render_quote_fair(
    risultati: ProbabilitaModello,
    minuto: int,
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
) -> None:
    """Render delle quote fair per tutti i mercati principali."""
    st.header(f"Quote Fair  —  {minuto}' | {gol_casa}–{gol_trasf}")
    st.caption("Confronta queste quote con quelle sull'exchange: se vedi di meglio, c'è valore.")

    c1, cx, c2 = st.columns(3)
    c1.metric("1 — Casa", f"@{_q_fair(risultati.p1):.2f}", f"{risultati.p1:.1%}")
    cx.metric("X — Pareggio", f"@{_q_fair(risultati.px):.2f}", f"{risultati.px:.1%}")
    c2.metric("2 — Trasf.", f"@{_q_fair(risultati.p2):.2f}", f"{risultati.p2:.1%}")

    cu, co, cb = st.columns(3)
    cu.metric(f"Under {linea_ou}", f"@{_q_fair(risultati.p_under):.2f}", f"{risultati.p_under:.1%}")
    co.metric(f"Over  {linea_ou}", f"@{_q_fair(risultati.p_over):.2f}", f"{risultati.p_over:.1%}")
    cb.metric("BTTS — Sì", f"@{_q_fair(risultati.p_btts):.2f}", f"{risultati.p_btts:.1%}")


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
# Momentum
# ---------------------------------------------------------------------------

def render_momentum(momentum: float) -> None:
    """Render della barra del momentum con etichetta testuale."""
    if momentum < MOMENTUM_CFG.STABLE_THRESHOLD:
        label = f"Mercato stabile [{momentum:.2f}/6.0]"
    elif momentum < MOMENTUM_CFG.MODERATE_THRESHOLD:
        label = f"Movimento moderato [{momentum:.2f}/6.0]"
    elif momentum < MOMENTUM_CFG.SIGNIFICANT_THRESHOLD:
        label = f"Movimento significativo [{momentum:.2f}/6.0]"
    else:
        label = f"Movimento estremo — verifica eventi non registrati [{momentum:.2f}/6.0]"

    st.progress(min(momentum / MOMENTUM_CFG.MOMENTUM_CAP, 1.0), text=label)


# ---------------------------------------------------------------------------
# Segnali
# ---------------------------------------------------------------------------

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
        if s.tipo == "INFO_BACK":
            st.success(
                f"**BACK candidato — {s.mercato}** · Modello {s.prob_mod:.1%} · Fair @{s.quota_fair:.2f}\n\n"
                f"✅ Cerca sull'exchange **almeno @{s.quota_exc:.2f}** per avere edge"
            )
        elif s.tipo == "INFO_LAY":
            st.warning(
                f"**LAY candidato — {s.mercato}** · Modello {s.prob_mod:.1%} · Fair @{s.quota_fair:.2f}\n\n"
                f"✅ Banca se la quota sull'exchange è **al massimo @{s.quota_exc:.2f}**"
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
        st.success(f"✅ Over {linea_ou} già **VINTO** — {gol_attuali:.0f} gol totali. Mercato chiuso.")
        st.error(f"❌ Under {linea_ou} già **PERSO**. Mercato chiuso.")
        settled["ou_vinto"] = True

    btts_si_settled = gol_casa > 0 and gol_trasf > 0
    btts_no_settled = (
        minuto >= SIGNALS.BTTS_NO_SETTLED_MINUTE
        and (gol_casa == 0 or gol_trasf == 0)
        and p_btts < SIGNALS.BTTS_NO_SETTLED_PROB_THRESHOLD
    )

    if btts_si_settled:
        st.success("✅ BTTS SÌ già VINTO — entrambe le squadre hanno segnato. Mercato chiuso.")
        st.error("❌ BTTS NO già PERSO. Mercato chiuso.")
        settled["btts_si_settled"] = True
    elif btts_no_settled:
        st.error("❌ BTTS SÌ quasi impossibile. Mercato chiuso praticamente.")
        st.success("✅ BTTS NO quasi VINTO — non entrare ora.")
        settled["btts_no_settled"] = True

    return settled


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
