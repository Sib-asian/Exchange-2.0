"""
calibration_tracking.py — Infrastructure for tracking prediction calibration.

Provides CalibrationTracker class that records (predicted_prob, actual_outcome)
pairs for any market and computes diagnostic metrics:
  - Mean absolute calibration error
  - Brier score
  - Hit rate at various probability thresholds

This is the foundation for future Supabase-based validation and
model performance monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CalibrationRecord:
    """Single predicted/actual pair for one market."""
    predicted_prob: float
    actual_outcome: float  # 1.0 or 0.0
    market: str
    fixture_id: str = ""
    timestamp: float = 0.0


class CalibrationTracker:
    """Tracks prediction calibration across markets.

    Usage:
        tracker = CalibrationTracker()
        tracker.record("1x2_home", predicted=0.55, actual=1.0)
        tracker.record("1x2_home", predicted=0.30, actual=0.0)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._records: Dict[str, List[CalibrationRecord]] = {}

    def record(
        self,
        market: str,
        predicted_prob: float,
        actual_outcome: float,
        fixture_id: str = "",
        timestamp: float = 0.0,
    ) -> None:
        """Record a single prediction-outcome pair.

        Args:
            market: Market identifier (e.g. "1x2_home", "over_25", "btts_yes").
            predicted_prob: Model's predicted probability for the outcome.
            actual_outcome: 1.0 if the outcome occurred, 0.0 otherwise.
            fixture_id: Optional fixture identifier for deduplication.
            timestamp: Optional timestamp for temporal analysis.
        """
        predicted_prob = max(0.0, min(1.0, float(predicted_prob)))
        actual_outcome = float(actual_outcome)

        if market not in self._records:
            self._records[market] = []
        self._records[market].append(
            CalibrationRecord(
                predicted_prob=predicted_prob,
                actual_outcome=actual_outcome,
                market=market,
                fixture_id=fixture_id,
                timestamp=timestamp,
            )
        )

    def calibration_error(self, market: str) -> float:
        """Mean absolute calibration error |predicted - actual| for a market.

        Returns:
            MAE in [0, 1]. Returns 0.0 if no records.
        """
        records = self._records.get(market, [])
        if not records:
            return 0.0
        return sum(abs(r.predicted_prob - r.actual_outcome) for r in records) / len(records)

    def brier_score(self, market: str) -> float:
        """Brier score (mean squared error) for a market.

        Lower is better. Perfect calibration = 0.0.

        Returns:
            Brier score in [0, 1]. Returns 0.0 if no records.
        """
        records = self._records.get(market, [])
        if not records:
            return 0.0
        return sum((r.predicted_prob - r.actual_outcome) ** 2 for r in records) / len(records)

    def hit_rate(self, market: str, threshold: float = 0.50) -> Tuple[float, int, int]:
        """Hit rate when the model predicts probability above a threshold.

        Args:
            market: Market identifier.
            threshold: Minimum predicted probability to count as a "pick".

        Returns:
            (hit_rate, hits, total_picks) — hit_rate in [0, 1].
        """
        records = self._records.get(market, [])
        picks = [r for r in records if r.predicted_prob >= threshold]
        if not picks:
            return 0.0, 0, 0
        hits = sum(1 for r in picks if r.actual_outcome > 0.5)
        return hits / len(picks), hits, len(picks)

    def count(self, market: Optional[str] = None) -> int:
        """Number of records for a market (or all markets)."""
        if market is not None:
            return len(self._records.get(market, []))
        return sum(len(v) for v in self._records.values())

    def markets(self) -> List[str]:
        """List of all tracked markets."""
        return sorted(self._records.keys())

    def reliability_data(
        self, market: str, n_bins: int = 10,
    ) -> List[Tuple[float, float, int]]:
        """Reliability diagram data: (avg_predicted, avg_actual, count) per bin.

        Useful for plotting calibration curves.

        Args:
            market: Market identifier.
            n_bins: Number of probability bins.

        Returns:
            List of (bin_center_predicted, bin_actual, bin_count) tuples.
        """
        records = self._records.get(market, [])
        if not records:
            return []

        bins: List[List[CalibrationRecord]] = [[] for _ in range(n_bins)]
        for r in records:
            idx = min(int(r.predicted_prob * n_bins), n_bins - 1)
            idx = max(0, idx)
            bins[idx].append(r)

        data: List[Tuple[float, float, int]] = []
        for i, bin_records in enumerate(bins):
            if not bin_records:
                continue
            avg_pred = sum(r.predicted_prob for r in bin_records) / len(bin_records)
            avg_actual = sum(r.actual_outcome for r in bin_records) / len(bin_records)
            data.append((avg_pred, avg_actual, len(bin_records)))
        return data

    def summary(self, market: Optional[str] = None) -> Dict[str, dict]:
        """Summary diagnostics for one or all markets.

        Args:
            market: If specified, return summary for that market only.
                    Otherwise, return summary for all markets.

        Returns:
            Dict mapping market name to diagnostics dict.
        """
        markets = [market] if market is not None else self.markets()
        result: Dict[str, dict] = {}
        for m in markets:
            hr, hits, total = self.hit_rate(m, threshold=0.50)
            hr60, hits60, total60 = self.hit_rate(m, threshold=0.60)
            result[m] = {
                "n": self.count(m),
                "calibration_error": round(self.calibration_error(m), 4),
                "brier_score": round(self.brier_score(m), 4),
                "hit_rate_50": round(hr, 4),
                "hits_50": hits,
                "picks_50": total,
                "hit_rate_60": round(hr60, 4),
                "hits_60": hits60,
                "picks_60": total60,
            }
        return result

    def reset(self, market: Optional[str] = None) -> None:
        """Clear records for a market or all markets."""
        if market is not None:
            self._records.pop(market, None)
        else:
            self._records.clear()
