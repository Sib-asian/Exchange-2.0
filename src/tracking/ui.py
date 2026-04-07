"""
ui.py — Componente Streamlit per il Prediction Logging.

Integrazione:
1. Salvataggio automatico dopo ogni analisi
2. Dashboard per chiudere le partite
3. Statistiche di performance
"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime

import streamlit as st

from src.tracking.prediction_log import (
    PredictionLog,
    PredictionRecord,
    create_record_from_analysis,
    get_prediction_log,
)
from src.tracking.stats import PerformanceStats

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
            except Exception:
                date_str = record.timestamp[:16]

            st.markdown(f"**{record.squadra_casa}** vs **{record.squadra_trasf}**")
            st.caption(f"{record.lega} · {date_str} · min {record.minuto}'")

            # Previsioni
            prev_str = f"1={record.p1*100:.0f}% X={record.px*100:.0f}% 2={record.p2*100:.0f}%"
            if record.p_over_25 > 0:
                _ol = getattr(record, "ou_line", 2.5) or 2.5
                prev_str += f" | O{_ol:g}={record.p_over_25*100:.0f}%"
            if record.p_btts > 0:
                prev_str += f" | BTTS={record.p_btts*100:.0f}%"
            st.caption(prev_str)
            _q1 = (record.quota_1 or 0) > 1.0
            _qx = (record.quota_x or 0) > 1.0
            _q2 = (record.quota_2 or 0) > 1.0
            if _q1 and _qx and _q2:
                st.caption(
                    f"Quote 1X2 salvate: {record.quota_1:.2f} / {record.quota_x:.2f} / {record.quota_2:.2f}"
                )
            else:
                st.caption("Quote 1X2 non in log — edge/ROI 1X2 saranno vuoti; Brier sì.")

        with col_result:
            st.caption("Risultato finale")
            c1, c2 = st.columns(2)
            with c1:
                gol_casa = st.number_input(
                    "Casa",
                    min_value=0,
                    max_value=20,
                    value=0,
                    key=f"gol_casa_{record.id}",
                )
            with c2:
                gol_trasf = st.number_input(
                    "Trasferta",
                    min_value=0,
                    max_value=20,
                    value=0,
                    key=f"gol_trasf_{record.id}",
                )

        with col_action:
            st.write("")  # allinea il bottone
            if st.button("✓ Conferma", key=f"confirm_{record.id}", type="primary"):
                if log.complete(record.id, int(gol_casa), int(gol_trasf)):
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
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_count = st.selectbox("Mostra", [10, 25, 50, 100], index=1)
    with col_f2:
        filter_market = st.selectbox(
            "Filtra per esito",
            ["Tutti", "Vittorie Casa (1)", "Pareggi (X)", "Vittorie Trasf (2)"],
        )
    with col_f3:
        team_q = st.text_input("Cerca squadra", placeholder="nome parziale")

    # Filtra
    filtered = PerformanceStats.sort_completed_newest_first(completed)
    if filter_market == "Vittorie Casa (1)":
        filtered = [r for r in filtered if r.risultato_1x2 == "1"]
    elif filter_market == "Pareggi (X)":
        filtered = [r for r in filtered if r.risultato_1x2 == "X"]
    elif filter_market == "Vittorie Trasf (2)":
        filtered = [r for r in filtered if r.risultato_1x2 == "2"]

    if team_q.strip():
        q = team_q.strip().lower()
        filtered = [
            r
            for r in filtered
            if q in (r.squadra_casa or "").lower() or q in (r.squadra_trasf or "").lower()
        ]

    filtered = filtered[:show_count]

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
        except Exception:
            pass
        if record.completed_at:
            ca = record.completed_at.replace("T", " ")[:16]
            st.caption(f"Chiusa: {ca}")

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
            _ol = getattr(record, "ou_line", 2.5) or 2.5
            hits.append(f"O{_ol:g}: {'✓' if record.over_25_hit else '✗'}")
        if record.btts_hit is not None:
            hits.append(f"BTTS: {'✓' if record.btts_hit else '✗'}")
        st.caption(" | ".join(hits))
        _qq = str(getattr(record, "quote_quality", "") or "").strip().lower()
        if _qq == "trusted":
            st.caption("Quote quality: ✅ trusted")
        elif _qq == "untrusted":
            st.caption("Quote quality: ⚠️ untrusted")

    st.divider()


def _render_stats_tab(log: PredictionLog) -> None:
    """Renderizza le statistiche di performance."""
    completed = log.get_completed()

    if len(completed) < 5:
        st.info("📊 Servono almeno 5 partite completate per le statistiche.")
        st.write(f"Completate: **{len(completed)}** / 5 minime")
        return

    min_conf = st.slider(
        "Confidenza minima modello (filtro report)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Mostra statistiche solo su partite con model_confidence >= soglia.",
    )
    completed_view = [
        r for r in completed
        if float(getattr(r, "model_confidence", 0.0) or 0.0) >= float(min_conf)
    ]
    if len(completed_view) < 5:
        st.info("Con il filtro confidenza attuale servono almeno 5 partite.")
        st.write(f"Partite filtrate: **{len(completed_view)}** / {len(completed)}")
        return

    # Calcola statistiche
    stats = PerformanceStats.compute_all_stats(completed_view)
    stats_trusted = PerformanceStats.compute_all_stats(completed_view, trusted_only_quotes=True)
    trusted_n = len(PerformanceStats.filter_records_by_quote_quality(completed_view, trusted_only=True))
    st.caption(
        f"Partite nel report: **{len(completed_view)}** (filtro confidenza ≥ {min_conf:.2f}) · "
        f"Record con quote trusted: **{trusted_n}/{len(completed_view)}**"
    )
    # Alert: Brier peggiore sulle ultime partite (solo mercati con volume).
    recent = PerformanceStats.sort_completed_newest_first(completed_view)[:50]
    if len(recent) >= 20:
        recent_stats = PerformanceStats.compute_all_stats(recent)
        valid_recent = [s for s in recent_stats.values() if s.total_predictions >= 8]
        if valid_recent:
            worst_recent = max(valid_recent, key=lambda s: s.brier_score)
            if worst_recent.brier_score > 0.28:
                st.warning(
                    f"⚠️ Alert qualità: Brier alto ({worst_recent.brier_score:.3f}) su "
                    f"{worst_recent.market_name} nelle ultime {len(recent)} partite."
                )

    roll_10 = PerformanceStats.rolling_1x2_metrics(completed_view, last_n=10)
    roll_20 = PerformanceStats.rolling_1x2_metrics(completed_view, last_n=20)
    roll_30 = PerformanceStats.rolling_1x2_metrics(completed_view, last_n=30)
    if roll_10 or roll_20 or roll_30:
        _chunks: list[str] = []
        for lbl, roll in (("10", roll_10), ("20", roll_20), ("30", roll_30)):
            if not roll:
                continue
            _chunks.append(
                f"R{lbl}: Brier **{roll['brier_1x2']:.4f}**, "
                f"LL **{roll['log_loss_1x2']:.4f}**, ECE **{roll['ece_1x2']:.4f}**"
            )
        if _chunks:
            st.caption(" · ".join(_chunks))

    # Learning status (autonomous calibration)
    try:
        from src.models.prematch_history_calibration import estimate_calibration_signals

        sig = estimate_calibration_signals(league="")
        st.subheader("🧠 Learning Status")
        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        with c_l1:
            st.metric("Campioni usati", int(getattr(sig, "samples", 0) or 0))
        with c_l2:
            st.metric("Peso calibrazione", f"{float(getattr(sig, 'weight', 0.0) or 0.0):.3f}")
        with c_l3:
            st.metric("Scope", str(getattr(sig, "scope", "global")))
        with c_l4:
            _phase = "full" if float(getattr(sig, "weight", 0.0) or 0.0) >= 0.04 else "warmup/off"
            st.metric("Fase", _phase)
    except Exception:
        pass

    # Tabella statistiche
    st.subheader("📊 Performance per Mercato")

    # Prepara dati per dataframe
    market_names = {
        "1X2_1": "1X2 Casa",
        "1X2_X": "1X2 Pareggio",
        "1X2_2": "1X2 Trasferta",
        "OVER_25": "Over (linea salvata)",
        "UNDER_25": "Under (linea salvata)",
        "BTTS_SI": "BTTS Sì",
        "BTTS_NO": "BTTS No",
    }

    data = []
    for key, s in stats.items():
        if s.total_predictions > 0:
            qn = s.predictions_with_quote
            tot = s.total_predictions
            edge_s = f"{s.avg_edge*100:+.1f}%" if qn > 0 else "—"
            roi_s = f"{s.roi*100:+.1f}%" if qn > 0 else "—"
            data.append({
                "Mercato": market_names.get(key, key),
                "Previsioni": tot,
                "Con quota": f"{qn}/{tot}",
                "Win Rate": f"{s.win_rate*100:.1f}%",
                "Brier": f"{s.brier_score:.3f}",
                "ECE": f"{s.ece_score:.3f}",
                "CLV": f"{s.avg_clv*100:+.2f}%" if s._clv_n > 0 else "—",
                "Edge (su quota)": edge_s,
                "ROI (su quota)": roi_s,
            })

    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Edge e ROI sono calcolati solo sulle partite in cui è stata salvata la quota per quel mercato. "
            "CLV usa anche la quota closing se disponibile. "
            "Over/Under: ogni riga usa la **linea O/U scelta** al momento dell'analisi (es. 1.5 o 2.5)."
        )

    st.subheader("📊 Performance quote affidabili (trusted-only)")
    trusted_data = []
    for key, s in stats_trusted.items():
        if s.total_predictions > 0:
            qn = s.predictions_with_quote
            tot = s.total_predictions
            edge_s = f"{s.avg_edge*100:+.1f}%" if qn > 0 else "—"
            roi_s = f"{s.roi*100:+.1f}%" if qn > 0 else "—"
            trusted_data.append({
                "Mercato": market_names.get(key, key),
                "Previsioni": tot,
                "Con quota trusted": f"{qn}/{tot}",
                "Win Rate": f"{s.win_rate*100:.1f}%",
                "Brier": f"{s.brier_score:.3f}",
                "ECE": f"{s.ece_score:.3f}",
                "CLV": f"{s.avg_clv*100:+.2f}%" if s._clv_n > 0 else "—",
                "Edge (trusted)": edge_s,
                "ROI (trusted)": roi_s,
            })
    if trusted_data:
        import pandas as pd
        st.dataframe(pd.DataFrame(trusted_data), use_container_width=True, hide_index=True)
        st.caption("Questa vista usa solo record con `quote_quality=trusted` per Edge/ROI/CLV.")

    with st.expander("📉 Report per linea O/U e lega (Brier / log-loss)", expanded=False):
        from src.tracking.deep_report import render_deep_report_streamlit

        render_deep_report_streamlit(completed_view, min_n=3)

    st.divider()
    st.subheader("Calibrazione 1X2 (multiclasse)")
    _mb = PerformanceStats.compute_multiclass_brier_1x2(completed_view)
    _ll = PerformanceStats.compute_log_loss_1x2(completed_view)
    _ece = PerformanceStats.compute_multiclass_ece_1x2(completed_view)
    _clv = PerformanceStats.compute_clv_proxy_1x2(completed_view)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
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
    with c_m3:
        if roll_30:
            st.metric(
                "Rolling 30",
                f"{roll_30['brier_1x2']:.4f}",
                help="Brier 1X2 sulle ultime 30 partite chiuse",
            )
        else:
            st.metric("Rolling 30", "—")
    with c_m4:
        st.metric(
            "ECE 1X2",
            f"{_ece:.4f}" if _ece is not None else "—",
            help="Expected Calibration Error multiclasse (più basso è meglio).",
        )
    if _clv is not None:
        st.caption(f"CLV proxy 1X2 medio (open→close): **{_clv*100:+.2f}%**")

    _by_l = PerformanceStats.segment_by_league(completed_view)
    _by_t = PerformanceStats.segment_by_tot_band(completed_view)
    prem, live = PerformanceStats.segment_by_prematch(completed_view)
    with st.expander("Prematch vs live (1X2)", expanded=False):
        rows_pm = []
        if len(prem) >= 3:
            b = PerformanceStats.compute_multiclass_brier_1x2(prem)
            ll = PerformanceStats.compute_log_loss_1x2(prem)
            ece = PerformanceStats.compute_multiclass_ece_1x2(prem)
            rows_pm.append({
                "Contesto": "Prematch",
                "N": len(prem),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
                "ECE": f"{ece:.4f}" if ece is not None else "—",
            })
        if len(live) >= 3:
            b = PerformanceStats.compute_multiclass_brier_1x2(live)
            ll = PerformanceStats.compute_log_loss_1x2(live)
            ece = PerformanceStats.compute_multiclass_ece_1x2(live)
            rows_pm.append({
                "Contesto": "Live",
                "N": len(live),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
                "ECE": f"{ece:.4f}" if ece is not None else "—",
            })
        if rows_pm:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows_pm), use_container_width=True, hide_index=True)
        else:
            st.caption("Servono almeno 3 partite per almeno una delle due categorie.")

    with st.expander("Per lega (N≥3)", expanded=False):
        rows_l = []
        for lega, sub in sorted(_by_l.items(), key=lambda x: -len(x[1])):
            if len(sub) < 3:
                continue
            b = PerformanceStats.compute_multiclass_brier_1x2(sub)
            ll = PerformanceStats.compute_log_loss_1x2(sub)
            ece = PerformanceStats.compute_multiclass_ece_1x2(sub)
            rows_l.append({
                "Lega": lega[:48],
                "N": len(sub),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
                "ECE": f"{ece:.4f}" if ece is not None else "—",
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
            ece = PerformanceStats.compute_multiclass_ece_1x2(sub)
            rows_t.append({
                "Fascia total": band,
                "N": len(sub),
                "Brier 1X2": f"{b:.4f}" if b is not None else "—",
                "Log-loss": f"{ll:.4f}" if ll is not None else "—",
                "ECE": f"{ece:.4f}" if ece is not None else "—",
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
        best, how_b = PerformanceStats.pick_best_market(stats)
        if best:
            best_name = market_names.get(best.market_name, best.market_name)
            if how_b == "edge":
                st.metric(
                    "🏆 Miglior mercato (edge)",
                    best_name,
                    delta=f"{best.avg_edge*100:+.1f}% su {best.predictions_with_quote} con quota",
                )
            else:
                st.metric(
                    "🏆 Miglior calibrazione (Brier)",
                    best_name,
                    delta=f"Brier {best.brier_score:.3f}",
                    help="Poche quote salvate: classifica per Brier, non per edge.",
                )

    with col_worst:
        worst, how_w = PerformanceStats.pick_worst_market(stats)
        if worst:
            worst_name = market_names.get(worst.market_name, worst.market_name)
            if how_w == "edge":
                st.metric(
                    "⚠️ Edge più basso",
                    worst_name,
                    delta=f"{worst.avg_edge*100:+.1f}% su {worst.predictions_with_quote} con quota",
                    delta_color="inverse",
                )
            else:
                st.metric(
                    "⚠️ Brier più alto",
                    worst_name,
                    delta=f"Brier {worst.brier_score:.3f}",
                    delta_color="inverse",
                    help="Poche quote salvate: classifica per Brier.",
                )

    with st.expander("Riepilogo testuale (copia/incolla)", expanded=False):
        st.code(PerformanceStats.format_summary(stats), language="text")

    # Export
    st.divider()
    st.subheader("💾 Esporta Dati")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_data = _export_to_csv(log.get_all())
        st.download_button(
            "📥 Scarica predictions.csv (tutti i record)",
            csv_data,
            file_name="predictions.csv",
            mime="text/csv",
            key="dl_csv_predictions",
        )

    with col_exp2:
        if st.button("🔧 Retro-tag quote quality", key="btn_backfill_quote_quality"):
            out = log.backfill_quote_quality(overwrite=False)
            st.success(
                "Quote quality aggiornate: "
                f"updated={out['updated']} trusted={out['trusted']} untrusted={out['untrusted']}"
            )
            st.rerun()
        if st.button("🗑️ Cancella Tutto", type="secondary", key="btn_clear_all"):
            st.session_state["_tracking_confirm_clear"] = True
        if st.session_state.get("_tracking_confirm_clear", False):
            c_y, c_n = st.columns(2)
            with c_n:
                if st.button("Annulla", key="btn_clear_cancel"):
                    st.session_state["_tracking_confirm_clear"] = False
                    st.rerun()
            with c_y:
                if st.checkbox("Confermo l'eliminazione di TUTTE le previsioni", key="cb_clear_all"):
                    log.clear_all()
                    st.session_state["_tracking_confirm_clear"] = False
                    st.success("Eliminato!")
                    st.rerun()


def _csv_cell(v: object) -> str | float | int:
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    return v


def _export_to_csv(records: list[PredictionRecord]) -> str:
    """Esporta tutti i campi del record (include pending e completate)."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    col_names = [f.name for f in fields(PredictionRecord)]
    writer.writerow(col_names)
    for r in records:
        row = asdict(r)
        writer.writerow([_csv_cell(row.get(c)) for c in col_names])
    return output.getvalue()
