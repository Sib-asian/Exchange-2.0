#!/usr/bin/env python3
"""
Simulazione end-to-end: fetch Nowgoal via Jina, estrazione prematch, pipeline analisi.

Uso (dalla root del progetto):
  python scripts/run_nowgoal_e2e.py [URL]

Default: h2h Mallorca vs Real Madrid (match id usato in test interni).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import UI
from src.ocr import extract_prematch_analysis_from_url
from src.prematch_app_bridge import build_match_state_from_prematch_analysis
from src.pipeline import run_analysis_pipeline


def main() -> int:
    p = argparse.ArgumentParser(description="E2E Nowgoal prematch + motore")
    p.add_argument(
        "url",
        nargs="?",
        default="https://live5.nowgoal26.com/match/h2h-2804589",
        help="URL Nowgoal (h2h / live / detail)",
    )
    args = p.parse_args()

    print("URL:", args.url)
    print("--- Estrazione (Jina + regex/HTML) ---")
    pa = extract_prematch_analysis_from_url(args.url)
    if not pa.extraction_success:
        print("FAIL extraction:", pa.error_message)
        return 1

    print("  teams:", pa.home_team, "|", pa.away_team)
    print("  league:", pa.league_name)
    print("  coverage:", round(pa.extraction_coverage, 3))
    print("  fixture days (H/A):", pa.fixture_next_days_home, "/", pa.fixture_next_days_away)
    print("  team_stats goals R10 (H/A):", pa.team_stats_home_goals, "/", pa.team_stats_away_goals)
    print("  team_stats goals R3 (H/A):", pa.team_stats3_home_goals, "/", pa.team_stats3_away_goals)
    print("  forma_mult (H/A):", round(pa.forma_mult_h, 4), "/", round(pa.forma_mult_a, 4))
    notes = getattr(pa, "extraction_notes", None) or []
    if notes:
        print("  notes:", notes[:5])

    match = {
        "minuto": 0,
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
        "gialli_casa": 0,
        "gialli_trasf": 0,
        "sot_h": 0,
        "soff_h": 0,
        "sot_a": 0,
        "soff_a": 0,
        "blk_h": 0,
        "blk_a": 0,
        "corner_h": 0,
        "corner_a": 0,
        "possesso_h": 0.0,
        "possesso_a": 0.0,
        "att_pericolosi_h": 0,
        "att_pericolosi_a": 0,
        "att_h": 0,
        "att_a": 0,
        "falli_casa": 0,
        "falli_trasf": 0,
    }
    lines = {
        "ah_op": -0.5,
        "tot_op": 2.5,
        "ah_cur": -0.5,
        "tot_cur": 2.5,
        "tot_cur_raw": 2.5,
    }
    linea_ou = 2.5
    bankroll = float(UI.BANKROLL_DEFAULT)
    comm_rate = float(UI.COMM_DEFAULT) / 100.0

    print("--- MatchState + pipeline (prematch) ---")
    try:
        state, lega, cov = build_match_state_from_prematch_analysis(
            pa,
            match=match,
            lines=lines,
            linea_ou=linea_ou,
            bankroll=bankroll,
            comm_rate=comm_rate,
        )
    except (AssertionError, ValueError) as e:
        print("FAIL build_match_state:", e)
        return 2

    print("  forma_mult engine (H/A):", round(state.forma_mult_h, 4), "/", round(state.forma_mult_a, 4))
    print("  prev_avg scored (H/A):", round(state.prev_avg_scored_h, 3), "/", round(state.prev_avg_scored_a, 3))

    try:
        ris, sig, _trace = run_analysis_pipeline(
            state,
            league=lega,
            apply_prematch_calibration=True,
            extraction_coverage=float(cov) if cov else 1.0,
        )
    except Exception as e:
        print("FAIL pipeline:", e)
        import traceback
        traceback.print_exc()
        return 3

    print("  OK pipeline cal_sig:", sig)
    print("  P(1/X/2):", f"{ris.p1:.1%}", f"{ris.px:.1%}", f"{ris.p2:.1%}")
    print("  xG final:", round(ris.xg_h_final, 3), "/", round(ris.xg_a_final, 3))
    print("  P(Over2.5):", f"{ris.p_over:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
