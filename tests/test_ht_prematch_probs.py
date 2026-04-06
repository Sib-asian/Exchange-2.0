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


def test_ht_probs_htft_matrix_nudges_toward_stronger_away_ht():
    """Matrice HT/FT (senza standings HT) sposta P(2 HT) rispetto al solo Poisson."""
    base = {
        "home_goals_1h": 1.0,
        "away_goals_1h": 1.0,
        "home_matches": 10,
        "away_matches": 10,
        "home_ht_win": 0,
        "home_ht_draw": 0,
        "home_ht_lose": 0,
        "away_ht_win": 0,
        "away_ht_draw": 0,
        "away_ht_lose": 0,
        "h2h_ht_home_win_pct": 0.0,
        "h2h_ht_draw_pct": 0.0,
        "h2h_ht_away_win_pct": 0.0,
        "h2h_ht_matches_count": 0,
    }
    flat = SimpleNamespace(**base)
    out_flat = _calcola_ht_probs(flat, xg_h_fallback=1.35, xg_a_fallback=1.35, p1_ft=0.33, px_ft=0.34, p2_ft=0.33)
    assert out_flat is not None
    p2_flat = out_flat[2]
    skew = SimpleNamespace(
        **base,
        htft_home_htw_ftw=0,
        htft_home_htw_ftd=0,
        htft_home_htw_ftl=0,
        htft_home_htd_ftw=0,
        htft_home_htd_ftd=2,
        htft_home_htd_ftl=0,
        htft_home_htl_ftw=0,
        htft_home_htl_ftd=0,
        htft_home_htl_ftl=8,
        htft_away_htw_ftw=5,
        htft_away_htw_ftd=2,
        htft_away_htw_ftl=0,
        htft_away_htd_ftw=0,
        htft_away_htd_ftd=2,
        htft_away_htd_ftl=1,
        htft_away_htl_ftw=0,
        htft_away_htl_ftd=0,
        htft_away_htl_ftl=0,
    )
    out_skew = _calcola_ht_probs(skew, xg_h_fallback=1.35, xg_a_fallback=1.35, p1_ft=0.33, px_ft=0.34, p2_ft=0.33)
    assert out_skew is not None
    assert out_skew[2] > p2_flat


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
