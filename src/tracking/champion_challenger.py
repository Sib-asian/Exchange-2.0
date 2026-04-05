"""
Utilities per confronto champion/challenger su storico completato.
"""

from __future__ import annotations

from src.tracking.prediction_log import PredictionRecord
from src.tracking.stats import ChampionChallengeEvaluation, PerformanceStats


def evaluate_champion_challenger_records(
    champion_records: list[PredictionRecord],
    challenger_records: list[PredictionRecord],
) -> ChampionChallengeEvaluation:
    """
    Wrapper esplicito per uso CLI/UI: delega al motore statistico.
    """
    return PerformanceStats.evaluate_champion_challenger(champion_records, challenger_records)


__all__ = ["evaluate_champion_challenger_records", "ChampionChallengeEvaluation"]
