"""
anomaly.py — Automatic anomaly detection in match data.

Provides:
  - Detection of anomalous input combinations
  - Market line consistency checks
  - Statistical outlier detection
  - Warning generation for suspicious data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Anomaly thresholds
_SHOTS_PER_MINUTE_HIGH = 1.5
_SHOTS_PER_MINUTE_LOW = 0.1
_GOAL_RATE_HIGH = 5.0
_AH_TOT_MISMATCH = 2.0


@dataclass
class Anomaly:
    """Represents a detected anomaly in input data."""

    severity: str
    category: str
    message: str
    field: str
    value: float
    expected_range: tuple[float, float]
    suggestion: str = ""


class AnomalyDetector:
    """Detects anomalies in match input data."""

    def __init__(
        self,
        shots_per_minute_high: float = _SHOTS_PER_MINUTE_HIGH,
        shots_per_minute_low: float = _SHOTS_PER_MINUTE_LOW,
        goal_rate_high: float = _GOAL_RATE_HIGH,
        ah_tot_mismatch: float = _AH_TOT_MISMATCH,
    ):
        self.shots_per_minute_high = shots_per_minute_high
        self.shots_per_minute_low = shots_per_minute_low
        self.goal_rate_high = goal_rate_high
        self.ah_tot_mismatch = ah_tot_mismatch

    def detect(self, state: Any) -> list[Anomaly]:
        """Run all anomaly checks on match state."""
        anomalies: list[Anomaly] = []

        anomalies.extend(self._check_shots_anomalies(state))
        anomalies.extend(self._check_goal_anomalies(state))
        anomalies.extend(self._check_market_anomalies(state))
        anomalies.extend(self._check_time_anomalies(state))
        anomalies.extend(self._check_red_card_anomalies(state))
        anomalies.extend(self._check_line_consistency(state))

        severity_order = {"error": 0, "warning": 1, "info": 2}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 3))

        return anomalies

    def _check_shots_anomalies(self, state: Any) -> list[Anomaly]:
        """Check for anomalous shot counts."""
        anomalies: list[Anomaly] = []

        if state.minuto == 0:
            return anomalies

        total_shots = state.sot_h + state.soff_h + state.sot_a + state.soff_a
        shots_per_minute = total_shots / state.minuto

        if shots_per_minute > self.shots_per_minute_high:
            anomalies.append(Anomaly(
                severity="warning",
                category="input",
                message=f"Tiri molto elevati: {total_shots} in {state.minuto}' ({shots_per_minute:.2f}/min)",
                field="shots",
                value=total_shots,
                expected_range=(0, self.shots_per_minute_high * state.minuto),
                suggestion="Verifica che i tiri siano totali da inizio gara",
            ))

        if state.minuto >= 30 and total_shots < state.minuto * self.shots_per_minute_low:
            anomalies.append(Anomaly(
                severity="info",
                category="statistical",
                message=f"Pochi tiri per partita live: {total_shots} al {state.minuto}'",
                field="shots",
                value=total_shots,
                expected_range=(state.minuto * self.shots_per_minute_low, state.minuto * 0.8),
                suggestion="Partita difensiva o dati mancanti",
            ))

        return anomalies

    def _check_goal_anomalies(self, state: Any) -> list[Anomaly]:
        """Check for anomalous goal patterns."""
        anomalies: list[Anomaly] = []

        total_goals = state.gol_casa + state.gol_trasf

        if state.minuto > 0:
            goal_rate = total_goals / state.minuto * 90

            if goal_rate > self.goal_rate_high:
                anomalies.append(Anomaly(
                    severity="warning",
                    category="statistical",
                    message=f"Rate gol molto alto: {goal_rate:.1f} gol/90min",
                    field="goals",
                    value=goal_rate,
                    expected_range=(0.0, self.goal_rate_high),
                    suggestion="Verifica che i gol siano corretti",
                ))

        return anomalies

    def _check_market_anomalies(self, state: Any) -> list[Anomaly]:
        """Check for market line anomalies."""
        anomalies: list[Anomaly] = []

        ah_abs = abs(state.ah_cur)
        if ah_abs > state.tot_cur + self.ah_tot_mismatch:
            anomalies.append(Anomaly(
                severity="warning",
                category="market",
                message=f"AH ({state.ah_cur:+.2f}) incoerente con Total ({state.tot_cur:.2f})",
                field="ah_cur",
                value=state.ah_cur,
                expected_range=(-state.tot_cur - self.ah_tot_mismatch, state.tot_cur + self.ah_tot_mismatch),
                suggestion="Verifica le linee inserite",
            ))

        ah_movement = abs(state.ah_cur - state.ah_op)
        if ah_movement > 1.0 and state.minuto < 30:
            anomalies.append(Anomaly(
                severity="info",
                category="market",
                message=f"Grande movimento AH: {state.ah_op:+.2f} → {state.ah_cur:+.2f}",
                field="ah_movement",
                value=ah_movement,
                expected_range=(0.0, 1.0),
                suggestion="Verifica eventi non registrati",
            ))

        return anomalies

    def _check_time_anomalies(self, state: Any) -> list[Anomaly]:
        """Check for time-related anomalies."""
        anomalies: list[Anomaly] = []

        if state.minuto >= 75 and state.tot_cur > 0.8:
            anomalies.append(Anomaly(
                severity="info",
                category="statistical",
                message=f"Al {state.minuto}' con ancora {state.tot_cur:.2f} gol attesi",
                field="late_game_total",
                value=state.tot_cur,
                expected_range=(0.0, 0.8),
                suggestion="Late game volatile — usare cautela",
            ))

        total_shots = state.sot_h + state.soff_h + state.sot_a + state.soff_a
        if state.minuto == 0 and total_shots > 0:
            anomalies.append(Anomaly(
                severity="warning",
                category="input",
                message=f"Tiri inseriti ({total_shots}) ma minuto = 0",
                field="shots_at_minute_0",
                value=total_shots,
                expected_range=(0, 0),
                suggestion="Se è prematch, lascia i tiri a 0",
            ))

        return anomalies

    def _check_red_card_anomalies(self, state: Any) -> list[Anomaly]:
        """Check for red card anomalies."""
        anomalies: list[Anomaly] = []

        total_reds = state.rossi_casa + state.rossi_trasf

        if total_reds >= 3:
            anomalies.append(Anomaly(
                severity="info",
                category="input",
                message=f"Partita con {total_reds} espulsioni",
                field="red_cards",
                value=total_reds,
                expected_range=(0, 2),
                suggestion="Previsioni meno affidabili",
            ))

        return anomalies

    def _check_line_consistency(self, state: Any) -> list[Anomaly]:
        """Check for internal consistency of line data."""
        anomalies: list[Anomaly] = []

        gol_diff = state.gol_casa - state.gol_trasf
        expected_ah_live = state.ah_op + gol_diff
        ah_deviation = abs(state.ah_cur - expected_ah_live)

        if ah_deviation > 0.75 and state.minuto > 0:
            anomalies.append(Anomaly(
                severity="info",
                category="market",
                message=f"AH devia di {ah_deviation:.2f} dal valore atteso",
                field="ah_consistency",
                value=ah_deviation,
                expected_range=(0.0, 0.75),
                suggestion="Possibile edge",
            ))

        return anomalies


def detect_input_anomalies(state: Any) -> list[Anomaly]:
    """Detect anomalies in match input data."""
    detector = AnomalyDetector()
    return detector.detect(state)


def format_anomalies(anomalies: list[Anomaly]) -> str:
    """Format anomalies for display."""
    if not anomalies:
        return "✅ Nessuna anomalia rilevata"

    lines = []
    for a in anomalies:
        icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(a.severity, "⚪")
        lines.append(f"{icon} **{a.message}**")
        if a.suggestion:
            lines.append(f"   _{a.suggestion}_")

    return "\n".join(lines)
