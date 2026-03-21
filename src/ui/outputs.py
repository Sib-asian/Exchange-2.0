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
from src.engine import ProbabilitaModello
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
            st.write(f"Momentum     = {risultati.momentum:.2f}/6.0")
            st.write(f"Stake factor = ×{momentum_factor:.2f}")
            st.write(f"Comm         = {comm_pct:.1f}%")
