"""
stats.py — Calcolo statistiche di performance dalle previsioni.

Metriche calcolate:
- Win rate per mercato
- Brier score (accuratezza probabilità)
- Edge medio realizzato
- ROI per mercato
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tracking.prediction_log import PredictionRecord


@dataclass
class MarketStats:
    """
    Statistiche per un singolo mercato.

    total_predictions: partite con esito definito per quel mercato.
    predictions_with_quote: sottoinsieme con quota > 1 (edge/ROI sensati solo lì).
    """

    market_name: str
    total_predictions: int = 0
    predictions_with_quote: int = 0
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
                stats.predictions_with_quote += 1
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
            if stats.predictions_with_quote > 0:
                stats.avg_edge = stats._edge_sum / stats.predictions_with_quote

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
    def pick_best_market(
        stats: dict[str, MarketStats],
        min_n: int = 5,
        min_with_quote: int = 3,
    ) -> tuple[MarketStats | None, str]:
        """
        Sceglie il mercato "migliore": prima per edge (se abbastanza quote),
        altrimenti per Brier più basso (calibrazione).
        """
        edge_ok = [
            s
            for s in stats.values()
            if s.total_predictions >= min_n and s.predictions_with_quote >= min_with_quote
        ]
        if edge_ok:
            return max(edge_ok, key=lambda s: s.avg_edge), "edge"
        brier_ok = [s for s in stats.values() if s.total_predictions >= min_n]
        if not brier_ok:
            return None, ""
        return min(brier_ok, key=lambda s: s.brier_score), "brier"

    @staticmethod
    def pick_worst_market(
        stats: dict[str, MarketStats],
        min_n: int = 5,
        min_with_quote: int = 3,
    ) -> tuple[MarketStats | None, str]:
        """Peggiore mercato: edge più basso se ci sono quote, altrimenti Brier più alto."""
        edge_ok = [
            s
            for s in stats.values()
            if s.total_predictions >= min_n and s.predictions_with_quote >= min_with_quote
        ]
        if edge_ok:
            return min(edge_ok, key=lambda s: s.avg_edge), "edge"
        brier_ok = [s for s in stats.values() if s.total_predictions >= min_n]
        if not brier_ok:
            return None, ""
        return max(brier_ok, key=lambda s: s.brier_score), "brier"

    @staticmethod
    def get_best_market(stats: dict[str, MarketStats]) -> MarketStats | None:
        """Retrocompatibile: stessa logica di pick_best_market (solo l'oggetto)."""
        s, _ = PerformanceStats.pick_best_market(stats)
        return s

    @staticmethod
    def get_worst_market(stats: dict[str, MarketStats]) -> MarketStats | None:
        s, _ = PerformanceStats.pick_worst_market(stats)
        return s

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
            qn = market_stats.predictions_with_quote
            tot = market_stats.total_predictions
            if qn > 0:
                edge_pct = market_stats.avg_edge * 100
                roi_pct = market_stats.roi * 100
                edge_cell = f"{edge_pct:>+6.1f}%"
                roi_cell = f"{roi_pct:>+6.1f}%"
            else:
                edge_cell = "   —  "
                roi_cell = "   —  "
            edge_icon = ""
            if qn > 0:
                edge_icon = "✓" if market_stats.avg_edge * 100 > 2 else (
                    "⚠" if market_stats.avg_edge < 0 else ""
                )

            lines.append(
                f"│  {name:<12} │ {tot:>4} │ "
                f"{win_pct:>5.1f}% │ {market_stats.brier_score:.3f} │ "
                f"{edge_cell} {edge_icon}│ {roi_cell} │"
            )

        lines.append("└─────────────────────────────────────────────────────────┘")
        lines.append("  Edge/ROI: solo partite con quota salvata per quel mercato.")

        # Best/worst
        best, best_how = PerformanceStats.pick_best_market(stats)
        worst, worst_how = PerformanceStats.pick_worst_market(stats)

        if best:
            best_name = market_names.get(best.market_name, best.market_name)
            if best_how == "edge":
                lines.append(f"  🏆 MIGLIOR: {best_name} ({best.avg_edge*100:+.1f}% edge su quota)")
            else:
                lines.append(
                    f"  🏆 MIGLIOR CALIBRAZIONE: {best_name} (Brier {best.brier_score:.3f})"
                )
        if worst:
            worst_name = market_names.get(worst.market_name, worst.market_name)
            if worst_how == "edge":
                lines.append(f"  ⚠️ DA RIVEDERE: {worst_name} ({worst.avg_edge*100:+.1f}% edge su quota)")
            else:
                lines.append(
                    f"  ⚠️ CALIBRAZIONE: {worst_name} (Brier {worst.brier_score:.3f})"
                )

        return "\n".join(lines)

    @staticmethod
    def compute_multiclass_brier_1x2(records: list["PredictionRecord"]) -> float | None:
        """Brier medio sul vettore 1X2 (somma (p-o)^2 sui tre esiti, diviso N)."""
        acc = 0.0
        n = 0
        for r in records:
            if not r.is_completed() or r.risultato_1x2 not in ("1", "X", "2"):
                continue
            o1, ox, o2 = (1.0, 0.0, 0.0) if r.risultato_1x2 == "1" else (
                (0.0, 1.0, 0.0) if r.risultato_1x2 == "X" else (0.0, 0.0, 1.0)
            )
            acc += (r.p1 - o1) ** 2 + (r.px - ox) ** 2 + (r.p2 - o2) ** 2
            n += 1
        if n == 0:
            return None
        return acc / n

    @staticmethod
    def compute_log_loss_1x2(records: list["PredictionRecord"]) -> float | None:
        """Log-loss (naturale) medio per classe vincitrice 1X2."""
        eps = 1e-6
        acc = 0.0
        n = 0
        for r in records:
            if not r.is_completed() or r.risultato_1x2 not in ("1", "X", "2"):
                continue
            if r.risultato_1x2 == "1":
                p = max(eps, min(1.0 - eps, r.p1))
            elif r.risultato_1x2 == "X":
                p = max(eps, min(1.0 - eps, r.px))
            else:
                p = max(eps, min(1.0 - eps, r.p2))
            acc += -math.log(p)
            n += 1
        if n == 0:
            return None
        return acc / n

    @staticmethod
    def segment_by_league(records: list["PredictionRecord"]) -> dict[str, list["PredictionRecord"]]:
        out: dict[str, list] = {}
        for r in records:
            if not r.is_completed():
                continue
            key = (r.lega or "").strip() or "(senza lega)"
            out.setdefault(key, []).append(r)
        return out

    @staticmethod
    def segment_by_tot_band(records: list["PredictionRecord"]) -> dict[str, list["PredictionRecord"]]:
        from src.tracking.prediction_log import tot_op_band

        out: dict[str, list] = {}
        for r in records:
            if not r.is_completed():
                continue
            key = (r.tot_band or "").strip() or tot_op_band(r.tot_op)
            out.setdefault(key, []).append(r)
        return out

    @staticmethod
    def sort_completed_newest_first(
        records: list["PredictionRecord"],
    ) -> list["PredictionRecord"]:
        """Ordina le completate per data chiusura (o timestamp analisi) decrescente."""

        def _key(r: "PredictionRecord") -> str:
            return (r.completed_at or r.timestamp or "")

        done = [r for r in records if r.is_completed()]
        return sorted(done, key=_key, reverse=True)

    @staticmethod
    def segment_by_prematch(
        records: list["PredictionRecord"],
    ) -> tuple[list["PredictionRecord"], list["PredictionRecord"]]:
        """(prematch, live) tra i record completati."""
        done = [r for r in records if r.is_completed()]
        prem = [r for r in done if r.is_prematch]
        live = [r for r in done if not r.is_prematch]
        return prem, live

    @staticmethod
    def rolling_1x2_metrics(
        records: list["PredictionRecord"],
        last_n: int = 30,
    ) -> dict[str, Any] | None:
        """Brier e log-loss 1X2 sulle ultime N partite chiuse (ordine cronologico inverso)."""
        sub = PerformanceStats.sort_completed_newest_first(records)[: max(0, last_n)]
        if len(sub) < 3:
            return None
        b = PerformanceStats.compute_multiclass_brier_1x2(sub)
        ll = PerformanceStats.compute_log_loss_1x2(sub)
        return {"n": len(sub), "brier_1x2": b, "log_loss_1x2": ll}
