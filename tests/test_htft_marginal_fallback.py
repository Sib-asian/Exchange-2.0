"""Fallback 1X2 da % HT H2H quando manca la matrice HT→FT."""

from src.models.htft_model import compute_htft_adjustment


def test_htft_uses_h2h_ht_marginals_when_matrix_sparse() -> None:
    p1 = px = p2 = 1.0 / 3.0
    out = compute_htft_adjustment(
        p1,
        px,
        p2,
        htft_blend_scale=1.0,
        htft_home_htw_ftw=0,
        htft_home_htw_ftd=0,
        htft_home_htw_ftl=0,
        htft_home_htd_ftw=0,
        htft_home_htd_ftd=0,
        htft_home_htd_ftl=0,
        htft_home_htl_ftw=0,
        htft_home_htl_ftd=0,
        htft_home_htl_ftl=0,
        htft_away_htw_ftw=0,
        htft_away_htw_ftd=0,
        htft_away_htw_ftl=0,
        htft_away_htd_ftw=0,
        htft_away_htd_ftd=0,
        htft_away_htd_ftl=0,
        htft_away_htl_ftw=0,
        htft_away_htl_ftd=0,
        htft_away_htl_ftl=0,
        h2h_ht_home_win_pct=55.0,
        h2h_ht_draw_pct=30.0,
        h2h_ht_away_win_pct=15.0,
        h2h_ht_matches_count=8,
        extraction_coverage=1.0,
        minuto=0,
        ht_result="",
    )
    r1, rx, r2 = out
    assert abs(r1 + rx + r2 - 1.0) < 1e-6
    assert r1 > p1
    assert r2 < p2


def test_htft_live_skips_marginal_without_transitions() -> None:
    p1, px, p2 = 0.5, 0.25, 0.25
    out = compute_htft_adjustment(
        p1,
        px,
        p2,
        htft_blend_scale=1.0,
        htft_home_htw_ftw=0,
        htft_home_htw_ftd=0,
        htft_home_htw_ftl=0,
        htft_home_htd_ftw=0,
        htft_home_htd_ftd=0,
        htft_home_htd_ftl=0,
        htft_home_htl_ftw=0,
        htft_home_htl_ftd=0,
        htft_home_htl_ftl=0,
        htft_away_htw_ftw=0,
        htft_away_htw_ftd=0,
        htft_away_htw_ftl=0,
        htft_away_htd_ftw=0,
        htft_away_htd_ftd=0,
        htft_away_htd_ftl=0,
        htft_away_htl_ftw=0,
        htft_away_htl_ftd=0,
        htft_away_htl_ftl=0,
        h2h_ht_home_win_pct=90.0,
        h2h_ht_draw_pct=5.0,
        h2h_ht_away_win_pct=5.0,
        h2h_ht_matches_count=10,
        extraction_coverage=1.0,
        minuto=50,
        ht_result="W",
    )
    assert out == (p1, px, p2)
