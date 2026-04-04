"""Coerenza probabilità 1T vs FT in _calcola_ht_probs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ui.outputs import _calcola_ht_probs, _ht_goal_share_1h


def test_ht_probs_pull_toward_ft_favorite():
    """Con standings HT estremamente pareggio-heavy, l'ancoraggio FT alza P(1 HT casa)."""
    pm = SimpleNamespace(
        home_goals_1h=0.0,
        away_goals_1h=0.0,
        home_matches=5,
        away_matches=5,
        home_ht_win=0,
        home_ht_draw=5,
        home_ht_lose=0,
        away_ht_win=0,
        away_ht_draw=5,
        away_ht_lose=0,
        h2h_ht_home_win_pct=8.0,
        h2h_ht_draw_pct=53.0,
        h2h_ht_away_win_pct=39.0,
        h2h_ht_matches_count=3,
    )
    out = _calcola_ht_probs(
        pm,
        xg_h_fallback=2.2,
        xg_a_fallback=0.75,
        p1_ft=0.67,
        px_ft=0.21,
        p2_ft=0.12,
    )
    assert out is not None
    p1, px, p2, _, _ = out
    assert p1 > 0.22
    assert p1 > p2


def test_ht_goal_share_clamped():
    assert _ht_goal_share_1h(0, 0) == 0.46
    assert _ht_goal_share_1h(22, 30) == pytest.approx(22 / 52, rel=1e-5)


def test_ht_probs_without_ft_args_unchanged_direction():
    """Senza p*_ft resta solo euristica HT (retrocompatibilità chiamate)."""
    pm = SimpleNamespace(
        home_goals_1h=1.0,
        away_goals_1h=0.5,
        home_matches=5,
        away_matches=5,
        home_ht_win=1,
        home_ht_draw=2,
        home_ht_lose=2,
        away_ht_win=0,
        away_ht_draw=3,
        away_ht_lose=2,
        h2h_ht_home_win_pct=0.0,
        h2h_ht_draw_pct=0.0,
        h2h_ht_away_win_pct=0.0,
        h2h_ht_matches_count=0,
    )
    out = _calcola_ht_probs(pm, xg_h_fallback=0.0, xg_a_fallback=0.0)
    assert out is not None
    p1, px, p2, _, _ = out
    assert abs(p1 + px + p2 - 1.0) < 1e-5
