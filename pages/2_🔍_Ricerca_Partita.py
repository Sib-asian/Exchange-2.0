"""
pages/2_🔍_Ricerca_Partita.py — Ricerca autonoma contesto partita.

Pagina isolata: usa Gemini + Google Search per trovare autonomamente
formazioni, infortuni, forma recente e head-to-head prima di una partita.
Non dipende dall'analisi principale (app.py).
"""

import streamlit as st

st.set_page_config(
    page_title="Ricerca Partita — Radar Pro Live",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 Ricerca Autonoma Pre-Partita")
st.caption("Gemini cerca su Google: infortuni, forma, H2H e contesto — in automatico.")

st.info(
    "Inserisci i nomi delle due squadre. Gemini cercherà autonomamente su Google "
    "le informazioni più recenti e suggerirà aggiustamenti ai parametri del modello.",
    icon="ℹ️",
)

st.divider()

# ---------------------------------------------------------------------------
# Input squadre
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    squadra_casa = st.text_input(
        "🏠 Squadra Casa",
        placeholder="es. Arsenal",
        help="Nome della squadra che gioca in casa",
    )
with col2:
    squadra_trasf = st.text_input(
        "✈️ Squadra Trasferta",
        placeholder="es. Chelsea",
        help="Nome della squadra ospite",
    )

competizione = st.text_input(
    "🏆 Competizione (opzionale)",
    placeholder="es. Premier League, Serie A, Champions League...",
    help="Aiuta Gemini a trovare le informazioni giuste per la competizione corretta",
)

st.divider()

# ---------------------------------------------------------------------------
# Bottone ricerca
# ---------------------------------------------------------------------------

cerca_btn = st.button(
    "🔎 Avvia Ricerca Autonoma",
    use_container_width=True,
    type="primary",
    disabled=not (squadra_casa.strip() and squadra_trasf.strip()),
)

if not squadra_casa.strip() or not squadra_trasf.strip():
    st.caption("Inserisci entrambi i nomi delle squadre per avviare la ricerca.")

# ---------------------------------------------------------------------------
# Esecuzione ricerca
# ---------------------------------------------------------------------------

if cerca_btn and squadra_casa.strip() and squadra_trasf.strip():

    from src.research import ricerca_contesto_partita

    with st.spinner(f"Gemini sta cercando informazioni su {squadra_casa} vs {squadra_trasf}..."):
        risultato = ricerca_contesto_partita(squadra_casa, squadra_trasf, competizione)

    # Salva in session state per uso futuro
    st.session_state["ricerca_risultato"] = risultato

# ---------------------------------------------------------------------------
# Mostra risultati (da session state o appena calcolati)
# ---------------------------------------------------------------------------

if "ricerca_risultato" in st.session_state:
    r = st.session_state["ricerca_risultato"]

    if not r.success:
        st.error(f"❌ Ricerca fallita: {r.error}")
        if "API key" in r.error:
            st.warning(
                "Aggiungi **GEMINI_API_KEY** nei secrets di Streamlit: "
                "`Settings → Secrets → GEMINI_API_KEY = \"la-tua-chiave\"`"
            )
    else:
        # ── Titolo partita ────────────────────────────────────────────────
        comp_str = f" · {r.competizione}" if r.competizione else ""
        st.success(f"✅ Ricerca completata{comp_str}")
        st.subheader(f"{r.squadra_casa}  vs  {r.squadra_trasf}")

        # ── Affidabilità ──────────────────────────────────────────────────
        aff_color = {"alta": "🟢", "media": "🟡", "bassa": "🔴"}.get(r.affidabilita, "⚪")
        st.caption(f"Affidabilità dati trovati: {aff_color} **{r.affidabilita.upper()}**")

        st.divider()

        # ── Assenze / Infortuni ───────────────────────────────────────────
        col_ass1, col_ass2 = st.columns(2)

        with col_ass1:
            st.markdown(f"**🏠 Assenze {r.squadra_casa}**")
            if r.assenze_casa:
                for a in r.assenze_casa:
                    st.markdown(f"- ❌ {a}")
            else:
                st.markdown("- ✅ Nessuna assenza rilevata")

        with col_ass2:
            st.markdown(f"**✈️ Assenze {r.squadra_trasf}**")
            if r.assenze_trasf:
                for a in r.assenze_trasf:
                    st.markdown(f"- ❌ {a}")
            else:
                st.markdown("- ✅ Nessuna assenza rilevata")

        st.divider()

        # ── Forma recente ─────────────────────────────────────────────────
        col_forma1, col_forma2 = st.columns(2)

        def _forma_colored(forma: str) -> str:
            """Trasforma stringa forma in emoji colorate."""
            if not forma:
                return "N/D"
            out = []
            for c in forma.upper():
                if c == "W":
                    out.append("🟢")
                elif c == "D":
                    out.append("🟡")
                elif c == "L":
                    out.append("🔴")
                else:
                    out.append(c)
            return " ".join(out)

        with col_forma1:
            st.markdown(f"**📊 Forma {r.squadra_casa}**")
            if r.forma_casa:
                st.markdown(_forma_colored(r.forma_casa))
                st.caption(r.forma_casa + " (ultimi 5 risultati)")
            else:
                st.markdown("N/D")

        with col_forma2:
            st.markdown(f"**📊 Forma {r.squadra_trasf}**")
            if r.forma_trasf:
                st.markdown(_forma_colored(r.forma_trasf))
                st.caption(r.forma_trasf + " (ultimi 5 risultati)")
            else:
                st.markdown("N/D")

        st.divider()

        # ── Head to Head ──────────────────────────────────────────────────
        st.markdown("**⚔️ Head-to-Head recente**")
        if r.h2h_sommario:
            st.markdown(r.h2h_sommario)
        if r.h2h_media_gol > 0:
            st.metric("Media gol negli H2H", f"{r.h2h_media_gol:.1f}")
        elif not r.h2h_sommario:
            st.markdown("N/D")

        st.divider()

        # ── Contesto ──────────────────────────────────────────────────────
        if r.contesto:
            st.markdown("**📝 Contesto partita**")
            st.info(r.contesto)
            st.divider()

        # ── Aggiustamenti suggeriti ───────────────────────────────────────
        st.markdown("**⚙️ Aggiustamenti suggeriti al modello**")

        col_adj1, col_adj2 = st.columns(2)
        with col_adj1:
            delta_tot_str = f"{r.adj_tot:+.2f}" if r.adj_tot != 0 else "0.00 (nessuno)"
            colore_tot = "normal" if r.adj_tot == 0 else ("inverse" if r.adj_tot < 0 else "off")
            st.metric(
                label="Δ Total atteso (tot_op)",
                value=delta_tot_str,
                delta="abbassa totale" if r.adj_tot < 0 else ("alza totale" if r.adj_tot > 0 else None),
                delta_color=colore_tot,
            )
        with col_adj2:
            delta_ah_str = f"{r.adj_ah:+.2f}" if r.adj_ah != 0 else "0.00 (nessuno)"
            st.metric(
                label="Δ Handicap (ah_op)",
                value=delta_ah_str,
                delta="favorisce trasferta" if r.adj_ah > 0 else ("favorisce casa" if r.adj_ah < 0 else None),
                delta_color="inverse" if r.adj_ah != 0 else "normal",
            )

        if r.note_aggiustamento:
            st.caption(f"💬 {r.note_aggiustamento}")

        # ── Come usare gli aggiustamenti ──────────────────────────────────
        if r.adj_tot != 0 or r.adj_ah != 0:
            with st.expander("📋 Come applicare questi aggiustamenti", expanded=False):
                tot_adj_example = f"tot_op attuale ± {r.adj_tot:+.2f}"
                ah_adj_example = f"ah_op attuale ± {r.adj_ah:+.2f}"
                st.markdown(f"""
Nella pagina principale **Radar Pro Live**:
- **Total apertura**: aggiungi `{r.adj_tot:+.2f}` al tuo valore → `{tot_adj_example}`
- **Spread apertura**: aggiungi `{r.adj_ah:+.2f}` al tuo valore → `{ah_adj_example}`

Questi aggiustamenti sono **suggerimenti conservativi** basati sulle informazioni trovate.
Valuta tu stesso se applicarli in base alla tua conoscenza della partita.
""")

        st.divider()

        # ── Fonti ─────────────────────────────────────────────────────────
        if r.fonti:
            with st.expander(f"🔗 Fonti ({len(r.fonti)})", expanded=False):
                for url in r.fonti:
                    # Mostra dominio invece dell'URL completo
                    try:
                        from urllib.parse import urlparse
                        dominio = urlparse(url).netloc.replace("www.", "")
                        st.markdown(f"- [{dominio}]({url})")
                    except Exception:
                        st.markdown(f"- {url}")

        # ── Pulsante reset ────────────────────────────────────────────────
        st.divider()
        if st.button("🔄 Nuova ricerca", use_container_width=False):
            del st.session_state["ricerca_risultato"]
            st.rerun()
