"""
analytics.py — Value bet tracking and risk metrics.

Provides:
  - Value bet detection and tracking
  - Sharpe Ratio calculation
  - Expected Value analysis
  - Kelly-optimal fraction computation
  - Risk-adjusted return metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ValueBet:
    """Represents a detected value bet opportunity."""

    market: str
    bet_type: str
    prob_model: float
    prob_market: float
    edge: float
    odds: float
    ev: float
    kelly_fraction: float
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def edge_percentage(self) -> float:
        """Edge as percentage."""
        return self.edge * 100

    @property
    def ev_percentage(self) -> float:
        """EV as percentage."""
        return self.ev * 100

    @property
    def quality_score(self) -> float:
        """Combined quality score [0, 1]."""
        edge_score = min(1.0, abs(self.edge) / 0.10)
        ev_score = min(1.0, abs(self.ev) / 0.15)
        conf_score = self.confidence
        return 0.4 * edge_score + 0.3 * ev_score + 0.3 * conf_score


@dataclass
class ValueBetTracker:
    """Tracks value bets over time for performance analysis."""

    bets: list[ValueBet] = field(default_factory=list)
    max_history: int = 100

    def add_bet(self, bet: ValueBet) -> None:
        """Add a value bet to the tracker."""
        self.bets.append(bet)
        if len(self.bets) > self.max_history:
            self.bets = self.bets[-self.max_history:]

    def get_stats(self) -> dict[str, Any]:
        """Calculate aggregate statistics."""
        if not self.bets:
            return {
                "total_bets": 0,
                "avg_edge": 0.0,
                "avg_ev": 0.0,
                "avg_confidence": 0.0,
                "back_count": 0,
                "lay_count": 0,
                "markets": {},
            }

        back_bets = [b for b in self.bets if b.bet_type == "BACK"]
        lay_bets = [b for b in self.bets if b.bet_type == "LAY"]

        market_counts: dict[str, int] = {}
        for bet in self.bets:
            market_counts[bet.market] = market_counts.get(bet.market, 0) + 1

        return {
            "total_bets": len(self.bets),
            "avg_edge": sum(b.edge for b in self.bets) / len(self.bets),
            "avg_ev": sum(b.ev for b in self.bets) / len(self.bets),
            "avg_confidence": sum(b.confidence for b in self.bets) / len(self.bets),
            "avg_quality": sum(b.quality_score for b in self.bets) / len(self.bets),
            "back_count": len(back_bets),
            "lay_count": len(lay_bets),
            "markets": market_counts,
        }

    def get_recent(self, n: int = 10) -> list[ValueBet]:
        """Get n most recent value bets."""
        return self.bets[-n:]

    def clear(self) -> None:
        """Clear bet history."""
        self.bets.clear()


def calculate_expected_value(
    prob_model: float,
    odds: float,
    commission_rate: float = 0.0,
    bet_type: str = "BACK",
) -> float:
    """Calculate expected value for a bet."""
    if bet_type == "BACK":
        odds_net = 1 + (odds - 1) * (1 - commission_rate)
        return prob_model * odds_net - 1.0
    else:  # LAY
        return (1 - prob_model) * (1 - commission_rate) - prob_model * (odds - 1)


def calculate_kelly_optimal_fraction(
    prob_model: float,
    odds: float,
    commission_rate: float = 0.0,
    bet_type: str = "BACK",
) -> float:
    """Calculate optimal Kelly fraction."""
    if bet_type == "BACK":
        odds_net = 1 + (odds - 1) * (1 - commission_rate)
        if odds_net <= 1:
            return 0.0
        b = odds_net - 1
        kelly = (prob_model * odds_net - 1) / b
        return max(0.0, min(kelly, 0.25))
    else:  # LAY
        if commission_rate >= 1.0:
            return 0.0
        denom = odds - commission_rate
        if denom <= 0:
            return 0.0
        break_even = (1 - commission_rate) / denom
        kelly = break_even - prob_model
        return max(0.0, min(kelly, 0.25))


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate Sharpe Ratio from a series of returns."""
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_dev


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """Calculate Sortino Ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    downside_returns = [r for r in returns if r < target_return]

    if not downside_returns:
        return float('inf') if mean_return > risk_free_rate else 0.0

    downside_variance = sum((r - target_return) ** 2 for r in downside_returns) / len(returns)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

    if downside_std == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0

    return (mean_return - risk_free_rate) / downside_std


def detect_value_bets(
    risultati: Any,
    quotes: Any,
    min_edge: float = 0.03,
    min_confidence: float = 0.50,
) -> list[ValueBet]:
    """Detect all value bets from model vs market comparison."""
    if not quotes.any_active:
        return []

    if risultati.model_confidence < min_confidence:
        return []

    value_bets: list[ValueBet] = []

    markets = [
        ("1 Casa", risultati.p1, quotes.q_1),
        ("X Pareggio", risultati.px, quotes.q_x),
        ("2 Trasf.", risultati.p2, quotes.q_2),
        ("Over", risultati.p_over, quotes.q_over),
        ("Under", risultati.p_under, quotes.q_under),
        ("BTTS Sì", risultati.p_btts, quotes.q_btts_si),
        ("BTTS No", 1 - risultati.p_btts, quotes.q_btts_no),
    ]

    for market, prob_model, odds in markets:
        if odds <= 1.0:
            continue

        prob_market = 1.0 / odds
        edge = prob_model - prob_market

        if abs(edge) >= min_edge:
            ev = calculate_expected_value(prob_model, odds, bet_type="BACK")
            kelly = calculate_kelly_optimal_fraction(prob_model, odds)

            if ev > 0:
                value_bets.append(ValueBet(
                    market=market,
                    bet_type="BACK",
                    prob_model=prob_model,
                    prob_market=prob_market,
                    edge=edge,
                    odds=odds,
                    ev=ev,
                    kelly_fraction=kelly,
                    confidence=risultati.model_confidence,
                ))

    value_bets.sort(key=lambda b: b.quality_score, reverse=True)
    return value_bets


def calculate_risk_metrics(
    expected_returns: list[float],
    probabilities: list[float],
    bankroll: float,
) -> dict[str, float]:
    """Calculate comprehensive risk metrics for a set of potential bets."""
    if not expected_returns:
        return {
            "total_ev": 0.0,
            "total_risk": 0.0,
            "risk_per_bankroll": 0.0,
            "diversification": 0.0,
            "max_single_risk": 0.0,
        }

    total_ev = sum(er * p for er, p in zip(expected_returns, probabilities, strict=False))

    variance = 0.0
    for er, p in zip(expected_returns, probabilities, strict=False):
        variance += p * (1 - p) * (er ** 2)

    total_risk = math.sqrt(variance)
    risk_per_bankroll = total_risk / bankroll if bankroll > 0 else 0.0

    n = len(expected_returns)
    if n > 1:
        weights = [p / sum(probabilities) for p in probabilities]
        herfindahl = sum(w ** 2 for w in weights)
        diversification = 1 - herfindahl
    else:
        diversification = 0.0

    max_single_risk = max(
        abs(er) * p for er, p in zip(expected_returns, probabilities, strict=False)
    )

    return {
        "total_ev": total_ev,
        "total_risk": total_risk,
        "risk_per_bankroll": risk_per_bankroll,
        "diversification": diversification,
        "max_single_risk": max_single_risk,
    }
