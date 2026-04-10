"""
Diagnostica prematch: coerenza linee, tracciamento pipeline, tightness CI per Kelly/firewall.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrematchPipelineTrace:
    """Snapshot probabilità dopo ogni stadio della pipeline prematch (solo minuto 0)."""

    p1_engine: float = 0.0
    px_engine: float = 0.0
    p2_engine: float = 0.0
    p_over_engine: float = 0.0
    p_after_league_p1: float | None = None
    p_after_platt_p1: float | None = None
    p_after_drawlearn_p1: float | None = None
    final_p1: float = 0.0
    final_px: float = 0.0
    final_p2: float = 0.0
    final_p_over: float = 0.0
    league_cal_weight: float = 0.0
    platt_strength: float = 1.0
    platt_applied: bool = False
    draw_learning_applied: bool = False
    line_coherence_warnings: tuple[str, ...] = ()
    pipeline_log: tuple[str, ...] = ()

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(
            f"Motore → 1X2 ({self.p1_engine:.1%}/{self.px_engine:.1%}/{self.p2_engine:.1%}), "
            f"Over linea {self.p_over_engine:.1%}"
        )
        if self.league_cal_weight > 0:
            lines.append(f"Calibrazione storica lega (peso {self.league_cal_weight:.1%})")
        else:
            lines.append("Calibrazione storica lega: non attiva o assente")
        if self.platt_applied:
            lines.append(f"Platt da log (forza relativa {self.platt_strength:.0%})")
        else:
            lines.append("Platt da log: non applicato o dati insufficienti")
        if self.draw_learning_applied:
            lines.append("Micro-aggiustamento draw da parameter_learning")
        if self.p_after_league_p1 is not None and abs(self.p_after_league_p1 - self.p1_engine) > 1e-5:
            lines.append(
                f"Δ P(1) dopo lega: {self.p_after_league_p1 - self.p1_engine:+.2%}"
            )
        if self.p_after_platt_p1 is not None and self.p_after_league_p1 is not None:
            if abs(self.p_after_platt_p1 - self.p_after_league_p1) > 1e-5:
                lines.append(
                    f"Δ P(1) dopo Platt: {self.p_after_platt_p1 - self.p_after_league_p1:+.2%}"
                )
        for w in self.line_coherence_warnings:
            lines.append(f"⚠ {w}")
        for msg in self.pipeline_log:
            lines.append(msg)
        return lines


def line_coherence_warnings(
    *,
    ah_op: float,
    tot_op: float,
    linea_ou: float,
    p1: float,
    p2: float,
    p_over: float,
) -> tuple[str, ...]:
    """Euristiche leggere AH / total / 1X2 / Over (best-effort, non bloccanti)."""
    out: list[str] = []
    if ah_op < -0.15 and p2 > p1 + 0.08:
        out.append("Segno AH (casa avanti) vs 1X2: verifica coerenza quote e handicap.")
    if ah_op > 0.15 and p1 > p2 + 0.08:
        out.append("Segno AH (ospite avanti) vs 1X2: verifica coerenza quote e handicap.")
    if tot_op >= 2.75 and linea_ou >= 2.25 and p_over < 0.36:
        out.append("Total apertura alto ma P(Over) bassa: controllare linee O/U e total.")
    if 0 < tot_op <= 2.15 and linea_ou <= 2.75 and p_over > 0.64:
        out.append("Total apertura basso ma P(Over) alta: controllare linee O/U e total.")
    return tuple(out)


def ci_tightness_score(credible_intervals: dict[str, tuple[float, float]]) -> float:
    """
    Score [0,1]: 1 = intervalli stretti (modelli concordi), 0 = molto incerti.
    Usato per Kelly e firewall prematch.
    """
    keys = ("p1", "p_over", "p_btts")
    widths: list[float] = []
    for k in keys:
        t = credible_intervals.get(k)
        if not t:
            continue
        lo, hi = t
        if hi > lo:
            widths.append(hi - lo)
    if not widths:
        return 0.55
    m = sum(widths) / len(widths)
    return max(0.0, min(1.0, 1.0 - m * 2.45))


__all__ = [
    "PrematchPipelineTrace",
    "ci_tightness_score",
    "line_coherence_warnings",
]
