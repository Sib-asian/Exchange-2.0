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
    ece_score: float = 0.0
    avg_edge: float = 0.0
    avg_clv: float = 0.0
    roi: float = 0.0

    # Detti per calcolo
    _brier_sum: float = 0.0
    _edge_sum: float = 0.0
    _clv_sum: float = 0.0
    _clv_n: int = 0
    _profit_sum: float = 0.0
    _stake_sum: float = 0.0


@dataclass
class ChampionChallengeEvaluation:
    """Esito del confronto champion/challenger su metriche di calibrazione."""

    promote: bool
    samples: int
    delta_brier_1x2: float | None = None
    delta_logloss_1x2: float | None = None
    delta_ece_1x2: float | None = None
    delta_clv_1x2: float | None = None
    reasons: list[str] | None = None


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
        from src.config import PRECISION

        stats = MarketStats(market_name=market)
        _probs: list[float] = []
        _outs: list[int] = []

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
            _probs.append(float(prob_model))
            _outs.append(int(outcome))

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

            close_quote = PerformanceStats._get_market_close_quote(r, market)
            if quote > 1.0 and close_quote > 1.0:
                clv = (1.0 / quote) - (1.0 / close_quote)
                stats._clv_sum += clv
                stats._clv_n += 1

        # Calcola medie
        if stats.total_predictions > 0:
            stats.win_rate = stats.wins / stats.total_predictions
            stats.brier_score = stats._brier_sum / stats.total_predictions
            ece = PerformanceStats.compute_ece_binary(_probs, _outs, bins=PRECISION.ECE_BINS)
            stats.ece_score = ece if ece is not None else 0.0
            if stats.predictions_with_quote > 0:
                stats.avg_edge = stats._edge_sum / stats.predictions_with_quote
            if stats._clv_n > 0:
                stats.avg_clv = stats._clv_sum / stats._clv_n

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
    def _get_market_close_quote(record: "PredictionRecord", market: str) -> float:
        """Quote closing associate al mercato (0.0 se non disponibili)."""
        if market == "1X2_1":
            return float(getattr(record, "quota_1_close", 0.0) or 0.0)
        if market == "1X2_X":
            return float(getattr(record, "quota_x_close", 0.0) or 0.0)
        if market == "1X2_2":
            return float(getattr(record, "quota_2_close", 0.0) or 0.0)
        if market == "OVER_25":
            return float(getattr(record, "quota_over_close", 0.0) or 0.0)
        if market == "UNDER_25":
            return float(getattr(record, "quota_under_close", 0.0) or 0.0)
        if market == "BTTS_SI":
            return float(getattr(record, "quota_btts_si_close", 0.0) or 0.0)
        if market == "BTTS_NO":
            return float(getattr(record, "quota_btts_no_close", 0.0) or 0.0)
        return 0.0

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
            "┌────────────────────────────────────────────────────────────────────────────┐",
            "│  📊 PERFORMANCE TRACKER                                                     │",
            "├────────────────────────────────────────────────────────────────────────────┤",
        ]

        # Header tabella
        lines.append(
            "│  MERCATO      │ PREV │ WIN%   │ BRIER │  ECE  │  CLV   │ EDGE    │ ROI    │"
        )
        lines.append(
            "│  ─────────────┼──────┼────────┼───────┼───────┼────────┼─────────┼────────│"
        )

        market_names = {
            "1X2_1": "1X2 Casa",
            "1X2_X": "1X2 X",
            "1X2_2": "1X2 Trasf",
            "OVER_25": "Over (linea salvata)",
            "UNDER_25": "Under (linea salvata)",
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
            if market_stats._clv_n > 0:
                clv_cell = f"{market_stats.avg_clv*100:+6.2f}%"
            else:
                clv_cell = "   —   "
            edge_icon = ""
            if qn > 0:
                edge_icon = "✓" if market_stats.avg_edge * 100 > 2 else (
                    "⚠" if market_stats.avg_edge < 0 else ""
                )

            lines.append(
                f"│  {name:<12} │ {tot:>4} │ "
                f"{win_pct:>5.1f}% │ {market_stats.brier_score:.3f} │ "
                f"{market_stats.ece_score:.3f} │ {clv_cell} │ "
                f"{edge_cell} {edge_icon}│ {roi_cell} │"
            )

        lines.append("└────────────────────────────────────────────────────────────────────────────┘")
        lines.append("  Edge/ROI: solo partite con quota salvata per quel mercato.")
        lines.append("  CLV: proxy (open->close) su quote closing disponibili.")

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
    def segment_by_ou_line(records: list["PredictionRecord"]) -> dict[str, list["PredictionRecord"]]:
        """Raggruppa le completate per linea Over/Under usata in analisi (``ou_line``)."""
        out: dict[str, list] = {}
        for r in records:
            if not r.is_completed():
                continue
            ol = float(getattr(r, "ou_line", 2.5) or 2.5)
            key = f"{ol:g}"
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
        ece = PerformanceStats.compute_multiclass_ece_1x2(sub)
        clv = PerformanceStats.compute_clv_proxy_1x2(sub)
        return {"n": len(sub), "brier_1x2": b, "log_loss_1x2": ll, "ece_1x2": ece, "clv_1x2": clv}

    @staticmethod
    def compute_ece_binary(
        probs: list[float],
        outcomes: list[int],
        *,
        bins: int = 10,
    ) -> float | None:
        """Expected Calibration Error binario."""
        if not probs or len(probs) != len(outcomes) or bins <= 0:
            return None
        n = len(probs)
        bucket_counts = [0 for _ in range(bins)]
        bucket_prob_sum = [0.0 for _ in range(bins)]
        bucket_out_sum = [0.0 for _ in range(bins)]
        for p_raw, o_raw in zip(probs, outcomes):
            p = max(0.0, min(1.0, float(p_raw)))
            o = 1.0 if int(o_raw) == 1 else 0.0
            idx = min(bins - 1, int(p * bins))
            bucket_counts[idx] += 1
            bucket_prob_sum[idx] += p
            bucket_out_sum[idx] += o
        ece = 0.0
        for i in range(bins):
            c = bucket_counts[i]
            if c == 0:
                continue
            conf = bucket_prob_sum[i] / c
            acc = bucket_out_sum[i] / c
            ece += (c / n) * abs(acc - conf)
        return ece

    @staticmethod
    def compute_multiclass_ece_1x2(
        records: list["PredictionRecord"],
        *,
        bins: int = 10,
    ) -> float | None:
        """
        ECE multiclasse per 1X2 usando confidence della classe predetta.
        """
        probs: list[float] = []
        outcomes: list[int] = []
        for r in records:
            if not r.is_completed() or r.risultato_1x2 not in ("1", "X", "2"):
                continue
            arr = [float(r.p1), float(r.px), float(r.p2)]
            pred_idx = max(range(3), key=lambda i: arr[i])
            conf = arr[pred_idx]
            out_idx = 0 if r.risultato_1x2 == "1" else (1 if r.risultato_1x2 == "X" else 2)
            probs.append(conf)
            outcomes.append(1 if pred_idx == out_idx else 0)
        return PerformanceStats.compute_ece_binary(probs, outcomes, bins=bins)

    @staticmethod
    def compute_clv_proxy_1x2(records: list["PredictionRecord"]) -> float | None:
        """
        CLV proxy atteso 1X2: sum(p_i * (imp_open_i - imp_close_i)).
        Positivo = closing migliore del prezzo preso (buon segnale di qualità).
        """
        acc = 0.0
        n = 0
        for r in records:
            if not r.is_completed():
                continue
            q_open = [float(r.quota_1), float(r.quota_x), float(r.quota_2)]
            q_close = [
                float(getattr(r, "quota_1_close", 0.0) or 0.0),
                float(getattr(r, "quota_x_close", 0.0) or 0.0),
                float(getattr(r, "quota_2_close", 0.0) or 0.0),
            ]
            if not all(q > 1.0 for q in q_open) or not all(q > 1.0 for q in q_close):
                continue
            p = [float(r.p1), float(r.px), float(r.p2)]
            acc += sum(p_i * ((1.0 / qo) - (1.0 / qc)) for p_i, qo, qc in zip(p, q_open, q_close))
            n += 1
        if n == 0:
            return None
        return acc / n

    @staticmethod
    def evaluate_champion_challenger(
        champion_records: list["PredictionRecord"],
        challenger_records: list["PredictionRecord"],
    ) -> ChampionChallengeEvaluation:
        """
        Gate multi-metrica per promuovere un challenger.

        Usa l'intersezione per `id` delle partite completate disponibili in entrambi
        i set per un confronto fair.
        """
        from src.config import PRECISION
        ch_map = {
            r.id: r for r in champion_records
            if r.is_completed() and r.risultato_1x2 in ("1", "X", "2")
        }
        cg_map = {
            r.id: r for r in challenger_records
            if r.is_completed() and r.risultato_1x2 in ("1", "X", "2")
        }
        common_ids = [rid for rid in ch_map.keys() if rid in cg_map]
        if len(common_ids) < PRECISION.CHAMPION_MIN_SAMPLES:
            return ChampionChallengeEvaluation(
                promote=False,
                samples=len(common_ids),
                reasons=[f"campione insufficiente ({len(common_ids)}/{PRECISION.CHAMPION_MIN_SAMPLES})"],
            )

        ch = [ch_map[rid] for rid in common_ids]
        cg = [cg_map[rid] for rid in common_ids]

        ch_b = PerformanceStats.compute_multiclass_brier_1x2(ch)
        cg_b = PerformanceStats.compute_multiclass_brier_1x2(cg)
        ch_ll = PerformanceStats.compute_log_loss_1x2(ch)
        cg_ll = PerformanceStats.compute_log_loss_1x2(cg)
        ch_ece = PerformanceStats.compute_multiclass_ece_1x2(ch, bins=PRECISION.ECE_BINS)
        cg_ece = PerformanceStats.compute_multiclass_ece_1x2(cg, bins=PRECISION.ECE_BINS)
        ch_clv = PerformanceStats.compute_clv_proxy_1x2(ch)
        cg_clv = PerformanceStats.compute_clv_proxy_1x2(cg)

        delta_b = (cg_b - ch_b) if (ch_b is not None and cg_b is not None) else None
        delta_ll = (cg_ll - ch_ll) if (ch_ll is not None and cg_ll is not None) else None
        delta_ece = (cg_ece - ch_ece) if (ch_ece is not None and cg_ece is not None) else None
        delta_clv = (cg_clv - ch_clv) if (ch_clv is not None and cg_clv is not None) else None

        reasons: list[str] = []
        promote = True
        if delta_b is None or delta_b > PRECISION.CHAMPION_MAX_DELTA_BRIER:
            promote = False
            reasons.append("Brier 1X2 non migliora abbastanza")
        if delta_ll is None or delta_ll > PRECISION.CHAMPION_MAX_DELTA_LOGLOSS:
            promote = False
            reasons.append("Log-loss 1X2 non migliora abbastanza")
        if delta_ece is None or delta_ece > PRECISION.CHAMPION_MAX_DELTA_ECE:
            promote = False
            reasons.append("ECE 1X2 non migliora abbastanza")
        if delta_clv is not None and delta_clv < PRECISION.CHAMPION_MIN_DELTA_CLV:
            promote = False
            reasons.append("CLV peggiora oltre soglia")

        return ChampionChallengeEvaluation(
            promote=promote,
            samples=len(common_ids),
            delta_brier_1x2=delta_b,
            delta_logloss_1x2=delta_ll,
            delta_ece_1x2=delta_ece,
            delta_clv_1x2=delta_clv,
            reasons=reasons or ["ok"],
        )
