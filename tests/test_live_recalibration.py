"""Live recalibration: ancoraggio al mercato (tot_cur rimanente)."""

from src.models.live_recalibration import compute_live_recalibration_factor


def test_live_recal_returns_one_premature():
    assert compute_live_recalibration_factor(2.5, 0, 10) == 1.0


def test_market_anchor_bounded_factor():
    f = compute_live_recalibration_factor(2.8, 1, 55, tot_cur_remaining=1.1)
    assert 0.88 <= f <= 1.12
