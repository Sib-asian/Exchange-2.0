"""
ui.py — Componente Streamlit per il Prediction Logging.

Integrazione:
1. Salvataggio automatico dopo ogni analisi
2. Dashboard per chiudere le partite
3. Statistiche di performance
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

from src.tracking.prediction_log import (
    PredictionLog,
    PredictionRecord,
    get_prediction_log,
    create_record_from_analysis,
)
from src.tracking.stats import PerformanceStats, MarketStats


# ---------------------------------------------------------------------------
# Salvataggio automatico
# ---------------------------------------------------------------------------

def auto_save_prediction(
    squadra_casa: str,
    squadra_trasf: str,
    lega: str,
    input_data: dict,
    predictions: dict,
    market_quotes: dict | None = None,
) -> PredictionRecord | None:
    """
    Salva automaticamente una previsione.

    Chiamare questa funzione DOPO aver calcolato le probabilità.

    Args:
        squadra_casa: Nome squadra casa
        squadra_trasf: Nome squadra trasferta
        lega: Nome lega/campionato
        input_data: Dict con ah_op, tot_op, xg_h, xg_a
        predictions: Dict con p1, px, p2, p_over, p_under, p_btts, model_confidence
        market_quotes: Dict con quote mercato (opzionale)

    Returns:
        PredictionRecord salvato, o None se dati insufficienti
    """
    # Non salvare se mancano dati essenziali
    if not squadra_casa or not squadra_trasf:
        return None

    log = get_prediction_log()
    record = create_record_from_analysis(
        squadra_casa=squadra_casa,
        squadra_trasf=squadra_trasf,
        lega=lega,
        input_data=input_data,
        predictions=predictions,
        market_quotes=market_quotes,
    )

    log.add(record)
    return record


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

def render_tracking_tab() -> None:
    """
    Renderizza la tab del tracking system.

    Da chiamare come tab aggiuntiva nell'app principale.
    """
    log = get_prediction_log()
    counts = log.count()

    # Metriche rapide
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Totale Previsioni", counts["total"])
    with col2:
        st.metric("In Attesa", counts["pending"], delta_color="off")
    with col3:
        st.metric("Completate", counts["completed"])

    st.divider()

    # Tab interne
    tab_pending, tab_completed, tab_stats = st.tabs([
        "🕐 Partite da Chiudere",
        "✅ Cronologia",
        "📊 Statistiche",
    ])

    with tab_pending:
        _render_pending_tab(log)

    with tab_completed:
        _render_completed_tab(log)

    with tab_stats:
        _render_stats_tab(log)


def _render_pending_tab(log: PredictionLog) -> None:
    """Renderizza la lista delle partite da chiudere."""
    pending = log.get_pending()

    if not pending:
        st.success("✅ Nessuna partita in attesa di risultato!")
        st.info("Le previsioni verranno salvate automaticamente quando analizzi una partita.")
        return

    st.write(f"**{len(pending)} partite in attesa di risultato:**")

    for record in pending:
        _render_pending_card(record, log)


def _render_pending_card(record: PredictionRecord, log: PredictionLog) -> None:
    """Renderizza una singola card per partita pending."""
    with st.container():
        col_info, col_result, col_action = st.columns([3, 2, 1])

        with col_info:
            # Data e squadre
            try:
                dt = datetime.fromisoformat(record.timestamp)
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except:
                date_str = record.timestamp[:16]

            st.markdown(f"**{record.squadra_casa}** vs **{record.squadra_trasf}**")
            st.caption(f"{record.lega} · {date_str}")

            # Previsioni
            prev_str = f"1={record.p1*100:.0f}% X={record.px*100:.0f}% 2={record.p2*100:.0f}%"
            if record.p_over_25 > 0:
                prev_str += f" | O2.5={record.p_over_25*100:.0f}%"
            if record.p_btts > 0:
                prev_str += f" | BTTS={record.p_btts*100:.0f}%"
            st.caption(prev_str)

        with col_result:
            # Input risultato
            c1, c2 = st.columns(2)
            with c1:
                gol_casa = st.number_input(
                    "Gol Casa",
                    min_value=0,
                    max_value=20,
                    key=f"gol_casa_{record.id}",
                    label_visibility="collapsed",
                    placeholder="Gol Casa",
                )
            with c2:
                gol_trasf = st.number_input(
                    "Gol Trasf",
                    min_value=0,
                    max_value=20,
                    key=f"gol_trasf_{record.id}",
                    label_visibility="collapsed",
                    placeholder="Gol Trasf",
                )

        with col_action:
            if st.button("✓ Conferma", key=f"confirm_{record.id}", type="primary"):
                if log.complete(record.id, gol_casa, gol_trasf):
                    st.success("Salvato!")
                    st.rerun()
                else:
                    st.error("Errore nel salvataggio")

        st.divider()


def _render_completed_tab(log: PredictionLog) -> None:
    """Renderizza la cronologia delle partite completate."""
    completed = log.get_completed()

    if not completed:
        st.info("Nessuna partita completata. Chiudi alcune previsioni per vedere la cronologia.")
        return

    # Filtri
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        show_count = st.selectbox("Mostra", [10, 25, 50, 100], index=1)
    with col_f2:
        filter_market = st.selectbox(
            "Filtra per esito",
            ["Tutti", "Vittorie Casa (1)", "Pareggi (X)", "Vittorie Trasf (2)"],
        )

    # Filtra
    filtered = completed
    if filter_market == "Vittorie Casa (1)":
        filtered = [r for r in completed if r.risultato_1x2 == "1"]
    elif filter_market == "Pareggi (X)":
        filtered = [r for r in completed if r.risultato_1x2 == "X"]
    elif filter_market == "Vittorie Trasf (2)":
        filtered = [r for r in completed if r.risultato_1x2 == "2"]

    # Mostra le più recenti
    filtered = sorted(filtered, key=lambda r: r.timestamp, reverse=True)[:show_count]

    # Tabella
    for record in filtered:
        _render_completed_row(record, log)


def _render_completed_row(record: PredictionRecord, log: PredictionLog) -> None:
    """Renderizza una riga di partita completata."""
    col_info, col_result, col_analysis = st.columns([3, 1, 2])

    with col_info:
        st.markdown(f"**{record.squadra_casa}** vs **{record.squadra_trasf}**")
        try:
            dt = datetime.fromisoformat(record.timestamp)
            st.caption(dt.strftime("%d/%m/%Y"))
        except:
            pass

    with col_result:
        result_color = {
            "1": "🟢",
            "X": "🟡",
            "2": "🔴",
        }.get(record.risultato_1x2, "⚪")

        st.markdown(f"### {result_color} {record.gol_casa}-{record.gol_trasf}")
        st.caption(f"Risultato: {record.risultato_1x2}")

    with col_analysis:
        # Analisi esiti
        hits = []
        if record.over_25_hit is not None:
            hits.append(f"O2.5: {'✓' if record.over_25_hit else '✗'}")
        if record.btts_hit is not None:
            hits.append(f"BTTS: {'✓' if record.btts_hit else '✗'}")
        st.caption(" | ".join(hits))

    st.divider()


def _render_stats_tab(log: PredictionLog) -> None:
    """Renderizza le statistiche di performance."""
    completed = log.get_completed()

    if len(completed) < 5:
        st.info("📊 Servono almeno 5 partite completate per le statistiche.")
        st.write(f"Completate: **{len(completed)}** / 5 minime")
        return

    # Calcola statistiche
    stats = PerformanceStats.compute_all_stats(completed)
    # Alert automatico quando la qualità recente peggiora.
    recent = sorted(completed, key=lambda r: r.completed_at or r.timestamp, reverse=True)[:50]
    if len(recent) >= 20:
        recent_stats = PerformanceStats.compute_all_stats(recent)
        valid_recent = [s for s in recent_stats.values() if s.total_predictions >= 8]
        if valid_recent:
            worst_recent = max(valid_recent, key=lambda s: s.brier_score)
            if worst_recent.brier_score > 0.28:
                st.warning(
                    f"⚠️ Alert qualità: Brier alto ({worst_recent.brier_score:.3f}) su {worst_recent.market_name} "
                    f"nelle ultime {len(recent)} partite."
                )

    # Tabella statistiche
    st.subheader("📊 Performance per Mercato")

    # Prepara dati per dataframe
    market_names = {
        "1X2_1": "1X2 Casa",
        "1X2_X": "1X2 Pareggio",
        "1X2_2": "1X2 Trasferta",
        "OVER_25": "Over 2.5",
        "UNDER_25": "Under 2.5",
        "BTTS_SI": "BTTS Sì",
        "BTTS_NO": "BTTS No",
    }

    data = []
    for key, s in stats.items():
        if s.total_predictions > 0:
            data.append({
                "Mercato": market_names.get(key, key),
                "Previsioni": s.total_predictions,
                "Win Rate": f"{s.win_rate*100:.1f}%",
                "Brier Score": f"{s.brier_score:.3f}",
                "Edge Medio": f"{s.avg_edge*100:+.1f}%",
                "ROI": f"{s.roi*100:+.1f}%",
            })

    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Calibrazione 1X2 (multiclasse)")
    _mb = PerformanceStats.compute_multiclass_brier_1x2(completed)
    _ll = PerformanceStats.compute_log_loss_1x2(completed)
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric(
            "Brier 1X2 (vector)",
            f"{_mb:.4f}" if _mb is not None else "—",
        )
    with c_m2:
        st.metric(
            "Log-loss 1X2",
            f"{_ll:.4f}" if _ll is not None else "—",
        )

    _by_l = PerformanceStats.segment_by_league(completed)
    _by_t = PerformanceStats.segment_by_tot_band(completed)
    with st.expander("Per lega (N≥3)", expanded=False):
        rows_l = []
        for lega, sub in sorted(_by_l.items(), key=lambda x: -len(x[1])):
            if len(sub) < 3:
                continue
            b = PerformanceStats.compute_multiclass_brier_1x2(sub)
            ll = PerformanceStats.compute_log_loss_1x2(sub)
            rows_l.append({
                "Lega": lega[:48],
                "N": len(sub),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
            })
        if rows_l:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows_l), use_container_width=True, hide_index=True)
        else:
            st.caption("Servono almeno 3 partite completate per lega.")

    with st.expander("Per fascia tot_op (N≥3)", expanded=False):
        rows_t = []
        for band, sub in sorted(_by_t.items(), key=lambda x: -len(x[1])):
            if len(sub) < 3:
                continue
            b = PerformanceStats.compute_multiclass_brier_1x2(sub)
            ll = PerformanceStats.compute_log_loss_1x2(sub)
            rows_t.append({
                "Fascia total": band,
                "N": len(sub),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
            })
        if rows_t:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)
        else:
            st.caption("Servono almeno 3 partite per fascia.")

    # Best/Worst
    st.divider()
    col_best, col_worst = st.columns(2)

    with col_best:
        best = PerformanceStats.get_best_market(stats)
        if best:
            best_name = market_names.get(best.market_name, best.market_name)
            st.metric(
                "🏆 Miglior Mercato",
                best_name,
                delta=f"{best.avg_edge*100:+.1f}% edge",
            )

    with col_worst:
        worst = PerformanceStats.get_worst_market(stats)
        if worst:
            worst_name = market_names.get(worst.market_name, worst.market_name)
            st.metric(
                "⚠️ Da Migliorare",
                worst_name,
                delta=f"{worst.avg_edge*100:+.1f}% edge",
                delta_color="inverse",
            )

    # Export
    st.divider()
    st.subheader("💾 Esporta Dati")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("📥 Esporta CSV"):
            csv_data = _export_to_csv(completed)
            st.download_button(
                "Scarica predictions.csv",
                csv_data,
                "predictions.csv",
                "text/csv",
            )

    with col_exp2:
        if st.button("🗑️ Cancella Tutto", type="secondary"):
            if st.checkbox("Confermo l'eliminazione di TUTTE le previsioni"):
                log.clear_all()
                st.success("Eliminato!")
                st.rerun()


def _export_to_csv(records: list[PredictionRecord]) -> str:
    """Esporta le previsioni in CSV."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "id", "timestamp", "squadra_casa", "squadra_trasf", "lega",
        "ah_op", "tot_op", "xg_h", "xg_a",
        "p1", "px", "p2", "p_over_25", "p_btts",
        "gol_casa", "gol_trasf", "risultato_1x2", "over_25_hit", "btts_hit",
        "status",
    ])

    # Dati
    for r in records:
        writer.writerow([
            r.id, r.timestamp, r.squadra_casa, r.squadra_trasf, r.lega,
            r.ah_op, r.tot_op, r.xg_h, r.xg_a,
            r.p1, r.px, r.p2, r.p_over_25, r.p_btts,
            r.gol_casa or "", r.gol_trasf or "", r.risultato_1x2,
            r.over_25_hit if r.over_25_hit is not None else "",
            r.btts_hit if r.btts_hit is not None else "",
            r.status,
        ])

    return output.getvalue()
