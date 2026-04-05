"""
Report per segmenti (linea O/U, lega) su previsioni chiuse.

Usabile da CLI (`scripts/prediction_report.py`) e dalla UI Streamlit.
"""

from __future__ import annotations

import math
from typing import Any

from src.tracking.prediction_log import PredictionRecord
from src.tracking.stats import PerformanceStats


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def build_segment_rows(
    completed: list[PredictionRecord],
    segments: dict[str, list[PredictionRecord]],
    *,
    min_n: int,
) -> list[dict[str, Any]]:
    """Una riga per segmento con N sufficiente: metriche 1X2 + Over."""
    rows: list[dict[str, Any]] = []
    for name in sorted(
        segments.keys(),
        key=lambda k: (1 if k == "(senza lega)" else 0, k.lower()),
    ):
        recs = segments[name]
        if len(recs) < min_n:
            continue
        b1 = PerformanceStats.compute_multiclass_brier_1x2(recs)
        ll = PerformanceStats.compute_log_loss_1x2(recs)
        over = PerformanceStats.compute_market_stats(recs, "OVER_25")
        ece = PerformanceStats.compute_multiclass_ece_1x2(recs)
        clv = PerformanceStats.compute_clv_proxy_1x2(recs)
        rows.append({
            "Segmento": name,
            "N": len(recs),
            "Brier 1X2": b1,
            "Log-loss 1X2": ll,
            "ECE 1X2": ece,
            "CLV 1X2": clv,
            "Brier Over": over.brier_score if over.total_predictions else None,
            "WR Over": over.win_rate if over.total_predictions else None,
        })
    return rows


def print_deep_report(completed: list[PredictionRecord], *, min_n: int = 3) -> None:
    """Scrive su stdout un riepilogo testuale (per terminale / CI)."""
    done = [r for r in completed if r.is_completed()]
    print(f"Partite completate: {len(done)}")
    if not done:
        print("(nessun record chiuso)")
        return

    print(f"\n--- Per linea Over/Under (analisi), N>={min_n} ---")
    ou = PerformanceStats.segment_by_ou_line(done)
    for row in build_segment_rows(done, ou, min_n=min_n):
        wr = row["WR Over"]
        wr_s = f"  WRover={wr * 100:.1f}%" if wr is not None else ""
        print(
            f"  O/U {row['Segmento']}: N={row['N']}  "
            f"Brier1X2={_fmt_float(row['Brier 1X2'])}  "
            f"log-loss={_fmt_float(row['Log-loss 1X2'])}  "
            f"ECE={_fmt_float(row['ECE 1X2'])}  "
            f"CLV={_fmt_float(row['CLV 1X2'])}  "
            f"BrierOver={_fmt_float(row['Brier Over'], 3)}{wr_s}"
        )

    print(f"\n--- Per lega, N>={min_n} ---")
    leg = PerformanceStats.segment_by_league(done)
    for row in build_segment_rows(done, leg, min_n=min_n):
        wr = row["WR Over"]
        wr_s = f"  WRover={wr * 100:.1f}%" if wr is not None else ""
        print(
            f"  {row['Segmento']}: N={row['N']}  "
            f"Brier1X2={_fmt_float(row['Brier 1X2'])}  "
            f"log-loss={_fmt_float(row['Log-loss 1X2'])}  "
            f"ECE={_fmt_float(row['ECE 1X2'])}  "
            f"CLV={_fmt_float(row['CLV 1X2'])}  "
            f"BrierOver={_fmt_float(row['Brier Over'], 3)}{wr_s}"
        )


def render_deep_report_streamlit(completed: list[PredictionRecord], *, min_n: int = 3) -> None:
    """Tabella in Streamlit (tab statistiche)."""
    import streamlit as st

    done = [r for r in completed if r.is_completed()]
    if len(done) < min_n:
        st.caption(f"Servono almeno **{min_n}** partite completate per il report a segmenti.")
        return

    ou = PerformanceStats.segment_by_ou_line(done)
    rows_ou = build_segment_rows(done, ou, min_n=min_n)
    st.markdown("**Per linea O/U** (quella scelta in analisi)")
    if not rows_ou:
        st.caption(f"Nessun gruppo con almeno {min_n} partite.")
    else:
        import pandas as pd

        disp = []
        for r in rows_ou:
            disp.append({
                "Linea O/U": r["Segmento"],
                "N": r["N"],
                "Brier 1X2": r["Brier 1X2"],
                "Log-loss 1X2": r["Log-loss 1X2"],
                "ECE 1X2": r["ECE 1X2"],
                "CLV 1X2": r["CLV 1X2"],
                "Brier Over": r["Brier Over"],
                "Win rate Over": r["WR Over"],
            })
        df = pd.DataFrame(disp)
        st.dataframe(df, use_container_width=True, hide_index=True)

    leg = PerformanceStats.segment_by_league(done)
    rows_l = build_segment_rows(done, leg, min_n=min_n)
    st.markdown("**Per lega**")
    if not rows_l:
        st.caption(f"Nessuna lega con almeno {min_n} partite.")
    else:
        import pandas as pd

        disp = []
        for r in rows_l:
            disp.append({
                "Lega": r["Segmento"],
                "N": r["N"],
                "Brier 1X2": r["Brier 1X2"],
                "Log-loss 1X2": r["Log-loss 1X2"],
                "ECE 1X2": r["ECE 1X2"],
                "CLV 1X2": r["CLV 1X2"],
                "Brier Over": r["Brier Over"],
                "Win rate Over": r["WR Over"],
            })
        st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)

    st.caption(
        "Brier/log-loss/ECE 1X2 = qualità probabilità multiclasse 1-X-2. "
        "CLV 1X2 = proxy open->close sulle quote salvate. "
        "Brier Over = calibrazione probabilità Over sulla linea salvata. "
        f"Solo segmenti con N≥{min_n}."
    )
