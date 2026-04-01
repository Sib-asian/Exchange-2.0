"""
stats.py — Calcolo statistiche di performance dalle previsioni.

Metriche calcolate:
- Win rate per mercato
- Brier score (accuratezza probabilità)
- Edge medio realizzato
- ROI per mercato
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tracking.prediction_log import PredictionRecord


@dataclass
class MarketStats:
    """Statistiche per un singolo mercato."""

    market_name: str
    total_predictions: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    brier_score: float = 0.0
    avg_edge: float = 0.0
    roi: float = 0.0

    # Detti per calcolo
    _brier_sum: float = 0.0
    _edge_sum: float = 0.0
    _profit_sum: float = 0.0
    _stake_sum: float = 0.0


class PerformanceStats:
    """
    Calcolatore di statistiche dalle previsioni completate.
    """

    @staticmethod
    def compute_market_stats(
        records: list["PredictionRecord"],
        market: str,
    ) -> MarketStats:
        """
        Calcola statistiche per un mercato specifico.

        Mercati supportati:
        - "1X2_1": Vittoria casa
        - "1X2_X": Pareggio
        - "1X2_2": Vittoria trasferta
        - "OVER_25": Over 2.5 gol
        - "UNDER_25": Under 2.5 gol
        - "BTTS_SI": Entrambe segnano
        - "BTTS_NO": Almeno una non segna

        Args:
            records: Lista di record COMPLETATI
            market: Nome del mercato

        Returns:
            MarketStats con tutte le metriche
        """
        stats = MarketStats(market_name=market)

        for r in records:
            if not r.is_completed():
                continue

            # Estrai probabilità e esito in base al mercato
            prob_model, outcome, quote = PerformanceStats._get_market_data(r, market)

            if prob_model is None:
                continue

            stats.total_predictions += 1

            # Brier score: (p - outcome)^2
            brier = (prob_model - outcome) ** 2
            stats._brier_sum += brier

            # Win/Loss
            if outcome == 1:
                stats.wins += 1
            else:
                stats.losses += 1

            # Edge e ROI (solo se c'è quota mercato)
            if quote > 1.0:
                prob_market = 1.0 / quote
                edge = prob_model - prob_market
                stats._edge_sum += edge

                # Profitto ipotizzando stake unitario
                if outcome == 1:
                    stats._profit_sum += (quote - 1.0)
                else:
                    stats._profit_sum -= 1.0
                stats._stake_sum += 1.0

        # Calcola medie
        if stats.total_predictions > 0:
            stats.win_rate = stats.wins / stats.total_predictions
            stats.brier_score = stats._brier_sum / stats.total_predictions
            stats.avg_edge = stats._edge_sum / stats.total_predictions

        if stats._stake_sum > 0:
            stats.roi = stats._profit_sum / stats._stake_sum

        return stats

    @staticmethod
    def _get_market_data(
        record: "PredictionRecord",
        market: str,
    ) -> tuple[float | None, int, float]:
        """
        Estrae probabilità modello, esito e quota per un mercato.

        Returns:
            (prob_model, outcome, quote)
            - prob_model: Probabilità dal modello (None se non disponibile)
            - outcome: 1 se vinto, 0 se perso
            - quote: Quota mercato (0 se non disponibile)
        """
        if market == "1X2_1":
            prob = record.p1
            outcome = 1 if record.risultato_1x2 == "1" else 0
            return prob, outcome, record.quota_1

        elif market == "1X2_X":
            prob = record.px
            outcome = 1 if record.risultato_1x2 == "X" else 0
            return prob, outcome, record.quota_x

        elif market == "1X2_2":
            prob = record.p2
            outcome = 1 if record.risultato_1x2 == "2" else 0
            return prob, outcome, record.quota_2

        elif market == "OVER_25":
            prob = record.p_over_25
            outcome = 1 if record.over_25_hit else 0
            return prob, outcome, record.quota_over

        elif market == "UNDER_25":
            prob = record.p_under_25
            outcome = 0 if record.over_25_hit else 1  # Under = NOT Over
            return prob, outcome, record.quota_under

        elif market == "BTTS_SI":
            prob = record.p_btts
            outcome = 1 if record.btts_hit else 0
            return prob, outcome, record.quota_btts_si

        elif market == "BTTS_NO":
            prob = 1.0 - record.p_btts
            outcome = 0 if record.btts_hit else 1
            return prob, outcome, record.quota_btts_no

        return None, 0, 0.0

    @staticmethod
    def compute_all_stats(
        records: list["PredictionRecord"],
    ) -> dict[str, MarketStats]:
        """
        Calcola statistiche per tutti i mercati.

        Returns:
            Dict mercato -> MarketStats
        """
        markets = [
            "1X2_1", "1X2_X", "1X2_2",
            "OVER_25", "UNDER_25",
            "BTTS_SI", "BTTS_NO",
        ]

        return {
            m: PerformanceStats.compute_market_stats(records, m)
            for m in markets
        }

    @staticmethod
    def get_best_market(stats: dict[str, MarketStats]) -> MarketStats | None:
        """Trova il mercato con miglior edge medio."""
        valid = [s for s in stats.values() if s.total_predictions >= 5]
        if not valid:
            return None
        return max(valid, key=lambda s: s.avg_edge)

    @staticmethod
    def get_worst_market(stats: dict[str, MarketStats]) -> MarketStats | None:
        """Trova il mercato con peggiore edge medio."""
        valid = [s for s in stats.values() if s.total_predictions >= 5]
        if not valid:
            return None
        return min(valid, key=lambda s: s.avg_edge)

    @staticmethod
    def format_summary(stats: dict[str, MarketStats]) -> str:
        """Genera un riepilogo testuale delle statistiche."""
        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            "│  📊 PERFORMANCE TRACKER                                  │",
            "├─────────────────────────────────────────────────────────┤",
        ]

        # Header tabella
        lines.append(
            "│  MERCATO      │ PREV │ WIN%   │ BRIER │ EDGE    │ ROI    │"
        )
        lines.append(
            "│  ─────────────┼──────┼────────┼───────┼─────────┼────────│"
        )

        market_names = {
            "1X2_1": "1X2 Casa",
            "1X2_X": "1X2 X",
            "1X2_2": "1X2 Trasf",
            "OVER_25": "Over 2.5",
            "UNDER_25": "Under 2.5",
            "BTTS_SI": "BTTS Sì",
            "BTTS_NO": "BTTS No",
        }

        for market_key, market_stats in stats.items():
            name = market_names.get(market_key, market_key)
            win_pct = market_stats.win_rate * 100
            edge_pct = market_stats.avg_edge * 100
            roi_pct = market_stats.roi * 100

            # Indicatori visivi
            edge_icon = "✓" if edge_pct > 2 else ("⚠" if edge_pct < 0 else "")

            lines.append(
                f"│  {name:<12} │ {market_stats.total_predictions:>4} │ "
                f"{win_pct:>5.1f}% │ {market_stats.brier_score:.3f} │ "
                f"{edge_pct:>+6.1f}% {edge_icon}│ {roi_pct:>+6.1f}% │"
            )

        lines.append("└─────────────────────────────────────────────────────────┘")

        # Best/worst
        best = PerformanceStats.get_best_market(stats)
        worst = PerformanceStats.get_worst_market(stats)

        if best:
            best_name = market_names.get(best.market_name, best.market_name)
            lines.append(f"  🏆 MIGLIOR: {best_name} ({best.avg_edge*100:+.1f}% edge)")
        if worst:
            worst_name = market_names.get(worst.market_name, worst.market_name)
            lines.append(f"  ⚠️ DA RIVEDERE: {worst_name} ({worst.avg_edge*100:+.1f}% edge)")

        return "\n".join(lines)
