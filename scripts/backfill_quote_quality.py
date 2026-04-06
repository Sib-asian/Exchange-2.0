"""
Retro-tagging quote quality sui record tracking esistenti.

Uso:
  python scripts/backfill_quote_quality.py
  python scripts/backfill_quote_quality.py --overwrite
"""

from __future__ import annotations

import argparse

from src.tracking.prediction_log import get_prediction_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill quote_quality in prediction log")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ricalcola quote_quality anche su record già classificati.",
    )
    args = parser.parse_args()

    log = get_prediction_log()
    out = log.backfill_quote_quality(overwrite=bool(args.overwrite))
    print(
        "Backfill quote_quality completed:",
        f"updated={out['updated']}",
        f"trusted={out['trusted']}",
        f"untrusted={out['untrusted']}",
        f"total={out['total']}",
    )


if __name__ == "__main__":
    main()
