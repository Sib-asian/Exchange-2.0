#!/usr/bin/env python3
"""
Report Brier / log-loss 1X2 e Over per linea O/U e per lega (solo previsioni chiuse).

Uso dalla root del repo:
  python scripts/prediction_report.py
  python scripts/prediction_report.py --min-n 5

Legge lo stesso storage dell'app (file locale ``data/predictions.json`` o Supabase se configurato).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Report metriche su previsioni chiuse")
    p.add_argument(
        "--min-n",
        type=int,
        default=3,
        metavar="N",
        help="Numero minimo di partite per segmento (default: 3)",
    )
    args = p.parse_args()

    from src.tracking.deep_report import print_deep_report
    from src.tracking.prediction_log import get_prediction_log

    log = get_prediction_log()
    print_deep_report(log.get_all(), min_n=max(1, args.min_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
