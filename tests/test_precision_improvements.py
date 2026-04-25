"""
test_precision_improvements.py — Comprehensive tests for precision improvements.

Tests:
  1. Probability coherence (p1+px+p2=1, p_over+p_under=1)
  2. Correct score probabilities sum to <= 1.0
  3. BTTS consistency with score matrix
  4. No probability > 1.0 or < 0.0 anywhere
  5. Edge cases: extreme lines (ah=-2.5, tot=5.5), default state, live at minute 85
  6. Ensemble weight stability (weights don't change > 20% between similar inputs)
  7. Over/Under consistency at different lines (p_over_25 > p_over_15)
"""

import math
import pytest

from src.engine import MatchState, analizza


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_prematch():
    """Standard prematch state."""
    return MatchState(
        minuto=0,
        gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.25, tot_op=2.5,
        ah_cur=-0.25, tot_cur=2.5,
        linea_ou=2.5,
    )


@pytest.fixture
def extreme_lines():
    """Extreme AH and total lines."""
    return MatchState(
        minuto=0,
        gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-2.5, tot_op=5.5,
        ah_cur=-2.5, tot_cur=5.5,
        linea_ou=5.5,
    )


@pytest.fixture
def live_minute_85():
    """Late-game live state at minute 85."""
    return MatchState(
        minuto=85,
        gol_casa=2, gol_trasf=1,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.5, tot_op=2.75,
        ah_cur=-1.5, tot_cur=0.75,
        linea_ou=3.5,
        sot_h=8, soff_h=3,
        sot_a=4, soff_a=2,
    )


@pytest.fixture
def live_minute_60():
    """Live state at minute 60 with shots."""
    return MatchState(
        minuto=60,
        gol_casa=1, gol_trasf=1,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.5, tot_op=2.5,
        ah_cur=0.0, tot_cur=1.5,
        linea_ou=2.5,
        sot_h=7, soff_h=4,
        sot_a=3, soff_a=2,
    )


@pytest.fixture
def away_favorite():
    """Away team favored (positive AH)."""
    return MatchState(
        minuto=0,
        gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=1.0, tot_op=2.25,
        ah_cur=1.0, tot_cur=2.25,
        linea_ou=2.25,
    )


# ---------------------------------------------------------------------------
# 1. Probability coherence
# ---------------------------------------------------------------------------

class TestProbabilityCoherence:
    """Ensure p1+px+p2=1 and p_over+p_under=1 for all states."""

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85",
        "live_minute_60", "away_favorite",
    ])
    def test_1x2_sum_to_one(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        total = r.p1 + r.px + r.p2
        assert abs(total - 1.0) < 1e-9, f"1X2 sum={total:.12f} for {fixture_name}"

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85",
        "live_minute_60", "away_favorite",
    ])
    def test_ou_sum_to_one(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        total = r.p_over + r.p_under
        assert abs(total - 1.0) < 1e-9, f"OU sum={total:.12f} for {fixture_name}"


# ---------------------------------------------------------------------------
# 2. Correct score probabilities sum <= 1.0
# ---------------------------------------------------------------------------

class TestCorrectScoreCoherence:
    """Top correct scores should sum to <= 1.0."""

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85", "away_favorite",
    ])
    def test_top_cs_sum_lte_one(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        cs_sum = sum(p for _, p in r.top_cs)
        assert cs_sum <= 1.0 + 1e-9, f"CS sum={cs_sum:.6f} > 1.0 for {fixture_name}"

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines",
    ])
    def test_gol_tot_dist_sum_to_one(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        total = sum(r.gol_tot_dist.values())
        assert abs(total - 1.0) < 1e-9, f"gol_tot_dist sum={total:.12f}"


# ---------------------------------------------------------------------------
# 3. BTTS consistency with score matrix
# ---------------------------------------------------------------------------

class TestBTTSConsistency:
    """BTTS should be consistent with the full_matrix score probabilities."""

    def test_btts_matches_matrix_derived(self, default_prematch):
        r = analizza(default_prematch)
        # Derive BTTS from matrix
        p_home_0 = sum(p for (a, _), p in r.full_matrix.items() if a == 0)
        p_away_0 = sum(p for ( _, b), p in r.full_matrix.items() if b == 0)
        p_0_0 = sum(p for (a, b), p in r.full_matrix.items() if a == 0 and b == 0)
        p_btts_matrix = 1.0 - p_home_0 - p_away_0 + p_0_0

        # After coherence enforcement, BTTS should be close to matrix-derived
        gap = abs(r.p_btts - p_btts_matrix)
        # The coherence enforcement applies 10% correction when gap > 0.5%,
        # so after one correction, gap should be < 0.5% + 90% of original
        assert gap < 0.10, (
            f"BTTS gap from matrix: {gap:.4f} "
            f"(p_btts={r.p_btts:.4f}, matrix={p_btts_matrix:.4f})"
        )

    def test_btts_range(self, default_prematch):
        r = analizza(default_prematch)
        assert 0.0 <= r.p_btts <= 1.0


# ---------------------------------------------------------------------------
# 4. No probability out of [0, 1] range
# ---------------------------------------------------------------------------

class TestProbabilityBounds:
    """All output probabilities must be in [0, 1]."""

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85",
        "live_minute_60", "away_favorite",
    ])
    def test_main_probabilities_in_range(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        for name, val in [
            ("p1", r.p1), ("px", r.px), ("p2", r.p2),
            ("p_over", r.p_over), ("p_under", r.p_under),
            ("p_btts", r.p_btts),
            ("p_over_15", r.p_over_15), ("p_under_15", r.p_under_15),
            ("p_over_25_ref", r.p_over_25_ref), ("p_under_25_ref", r.p_under_25_ref),
        ]:
            assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9, (
                f"{name}={val:.10f} out of [0,1] for {fixture_name}"
            )

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85",
    ])
    def test_correct_score_probs_in_range(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        for (score, prob) in r.top_cs:
            assert 0.0 <= prob <= 1.0, f"CS {score} prob={prob:.6f} out of [0,1]"

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines",
    ])
    def test_gol_tot_dist_in_range(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        for goals, prob in r.gol_tot_dist.items():
            assert 0.0 <= prob <= 1.0, f"gol_tot_dist[{goals}]={prob:.6f} out of [0,1]"

    @pytest.mark.parametrize("fixture_name", [
        "default_prematch", "extreme_lines", "live_minute_85",
    ])
    def test_per_model_probs_in_range(self, request, fixture_name):
        state = request.getfixturevalue(fixture_name)
        r = analizza(state)
        for name, val in [
            ("p1_bp", r.p1_bp), ("px_bp", r.px_bp), ("p2_bp", r.p2_bp),
            ("p1_cop", r.p1_cop), ("px_cop", r.px_cop), ("p2_cop", r.p2_cop),
            ("p1_mk", r.p1_mk), ("px_mk", r.px_mk), ("p2_mk", r.p2_mk),
        ]:
            assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9, (
                f"{name}={val:.10f} out of [0,1]"
            )


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Extreme inputs should not produce NaN, inf, or invalid probabilities."""

    def test_extreme_ah_tot(self, extreme_lines):
        r = analizza(extreme_lines)
        assert not math.isnan(r.p1)
        assert not math.isnan(r.p2)
        assert not math.isnan(r.p_over)
        assert not math.isinf(r.p_btts)
        # With ah=-2.5, home should be strongly favored
        assert r.p1 > 0.3

    def test_live_minute_85_no_nan(self, live_minute_85):
        r = analizza(live_minute_85)
        assert not math.isnan(r.p1)
        assert not math.isnan(r.p_over)
        assert not math.isnan(r.p_btts)

    def test_live_btts_both_scored(self, live_minute_60):
        """When both scored, BTTS should be 1.0 (settled market)."""
        r = analizza(live_minute_60)
        assert r.p_btts == pytest.approx(1.0, abs=1e-6)

    def test_high_total_line(self):
        """Very high total line (5.5) should produce sensible probabilities."""
        state = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=0.0, tot_op=5.5,
            ah_cur=0.0, tot_cur=5.5,
            linea_ou=5.5,
        )
        r = analizza(state)
        assert r.p_over < 0.5  # Over 5.5 should be less than 50%
        assert abs(r.p1 + r.px + r.p2 - 1.0) < 1e-9

    def test_zero_minute_no_red_cards(self, default_prematch):
        """Simplest possible input."""
        r = analizza(default_prematch)
        assert 0.0 < r.p1 < 1.0
        assert 0.0 < r.p2 < 1.0

    def test_home_strong_favorite(self):
        """Home strongly favored (AH = -2.0)."""
        state = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-2.0, tot_op=3.0,
            ah_cur=-2.0, tot_cur=3.0,
            linea_ou=3.0,
        )
        r = analizza(state)
        assert r.p1 > r.p2 * 1.5

    def test_draw_heavy(self):
        """Lines suggesting a draw (AH ~ 0, low total)."""
        state = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=0.0, tot_op=1.75,
            ah_cur=0.0, tot_cur=1.75,
            linea_ou=1.75,
        )
        r = analizza(state)
        assert r.px > 0.25  # Low total → more draws expected


# ---------------------------------------------------------------------------
# 6. Ensemble weight stability
# ---------------------------------------------------------------------------

class TestEnsembleWeightStability:
    """Weights should not oscillate wildly between similar inputs."""

    def test_weights_similar_inputs(self):
        """Small input changes should produce similar weights."""
        base = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-0.25, tot_op=2.5,
            ah_cur=-0.25, tot_cur=2.5,
            linea_ou=2.5,
        )
        perturbed = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-0.25, tot_op=2.5,
            ah_cur=-0.30, tot_cur=2.55,
            linea_ou=2.5,
        )
        r_base = analizza(base)
        r_pert = analizza(perturbed)

        # Check that no single weight changed by more than 20%
        for name, w_base, w_pert in [
            ("bp", r_base.consensus_w_bp, r_pert.consensus_w_bp),
            ("cop", r_base.consensus_w_cop, r_pert.consensus_w_cop),
            ("mk", r_base.consensus_w_mk, r_pert.consensus_w_mk),
        ]:
            if w_base > 1e-6:
                change = abs(w_pert - w_base) / w_base
                assert change < 0.20, (
                    f"Weight {name} changed by {change:.1%} "
                    f"({w_base:.4f} → {w_pert:.4f})"
                )

    def test_weights_sum_to_one(self, default_prematch):
        r = analizza(default_prematch)
        w_sum = r.consensus_w_bp + r.consensus_w_cop + r.consensus_w_mk
        assert abs(w_sum - 1.0) < 1e-6, f"Consensus weights sum={w_sum:.6f}"

    def test_weights_positive(self, default_prematch):
        r = analizza(default_prematch)
        assert r.consensus_w_bp > 0
        assert r.consensus_w_cop > 0
        assert r.consensus_w_mk > 0


# ---------------------------------------------------------------------------
# 7. Over/Under consistency across lines
# ---------------------------------------------------------------------------

class TestOverUnderConsistency:
    """p_over should be monotonically related to the line."""

    def test_over_25_greater_than_over_15(self, default_prematch):
        """Over 2.5 should have lower probability than Over 1.5."""
        r = analizza(default_prematch)
        if r.p_over_15 > 0 and r.p_over_25_ref > 0:
            assert r.p_over_25_ref < r.p_over_15, (
                f"p_over_2.5={r.p_over_25_ref:.4f} should be < "
                f"p_over_1.5={r.p_over_15:.4f}"
            )

    def test_over_under_15_sum_to_one(self, default_prematch):
        r = analizza(default_prematch)
        assert abs(r.p_over_15 + r.p_under_15 - 1.0) < 1e-9

    def test_over_under_25_ref_sum_to_one(self, default_prematch):
        r = analizza(default_prematch)
        assert abs(r.p_over_25_ref + r.p_under_25_ref - 1.0) < 1e-9

    def test_over_15_less_than_1(self, default_prematch):
        r = analizza(default_prematch)
        assert 0.0 <= r.p_over_15 <= 1.0
        assert 0.0 <= r.p_under_15 <= 1.0

    def test_over_line_consistency_extreme(self, extreme_lines):
        """With total=5.5, over should still be sensible."""
        r = analizza(extreme_lines)
        assert 0.0 <= r.p_over_15 <= 1.0
        assert 0.0 <= r.p_over_25_ref <= 1.0
        # Over 5.5 is very high, but over 1.5 and 2.5 should be much higher
        if r.p_over_15 > 0:
            assert r.p_over_15 > r.p_over, (
                f"Over 1.5 ({r.p_over_15:.4f}) should be > Over 5.5 ({r.p_over:.4f})"
            )


# ---------------------------------------------------------------------------
# Tests for _enforce_probability_coherence (unit)
# ---------------------------------------------------------------------------

class TestEnforceProbabilityCoherenceUnit:
    """Unit tests for the coherence enforcement function."""

    def test_normalizes_1x2(self):
        from src.engine import _enforce_probability_coherence
        matrix = {(0, 0): 0.1, (1, 0): 0.3, (0, 1): 0.3, (1, 1): 0.2, (2, 1): 0.1}
        p1, px, p2, po, pu, pb = _enforce_probability_coherence(
            0.35, 0.30, 0.36, 0.50, 0.50, 0.55, matrix, 0, 0,
        )
        assert abs(p1 + px + p2 - 1.0) < 1e-9

    def test_normalizes_ou(self):
        from src.engine import _enforce_probability_coherence
        matrix = {(0, 0): 0.1, (1, 0): 0.3, (0, 1): 0.3, (1, 1): 0.2, (2, 1): 0.1}
        p1, px, p2, po, pu, pb = _enforce_probability_coherence(
            0.33, 0.33, 0.34, 0.53, 0.48, 0.50, matrix, 0, 0,
        )
        assert abs(po + pu - 1.0) < 1e-9

    def test_clamps_to_zero_one(self):
        from src.engine import _enforce_probability_coherence
        matrix = {(0, 0): 0.1, (1, 0): 0.3, (0, 1): 0.3, (1, 1): 0.2, (2, 1): 0.1}
        p1, px, p2, po, pu, pb = _enforce_probability_coherence(
            0.33, 0.34, 0.33, 0.50, 0.50, -0.05, matrix, 0, 0,
        )
        assert 0.0 <= pb <= 1.0

    def test_btts_correction_on_large_gap(self):
        """When BTTS gap > 0.5%, correction should move toward matrix value."""
        from src.engine import _enforce_probability_coherence
        # Matrix where both score is unlikely
        matrix = {
            (0, 0): 0.20, (1, 0): 0.35, (0, 1): 0.35, (1, 1): 0.05, (2, 0): 0.05,
        }
        # Matrix BTTS = 1 - P(a=0) - P(b=0) + P(0,0) = 1 - 0.60 - 0.55 + 0.20 = 0.05
        p1, px, p2, po, pu, pb = _enforce_probability_coherence(
            0.33, 0.34, 0.33, 0.50, 0.50, 0.40, matrix, 0, 0,
        )
        # Gap was 0.35 (0.40 - 0.05), so correction should apply
        # Corrected: 0.9 * 0.40 + 0.1 * 0.05 = 0.365
        assert pb < 0.40, f"BTTS should decrease: {pb:.4f}"
        assert pb > 0.04, f"BTTS should not overcorrect: {pb:.4f}"


# ---------------------------------------------------------------------------
# Tests for smooth_ensemble_weights
# ---------------------------------------------------------------------------

class TestSmoothEnsembleWeights:
    """Unit tests for weight smoothing."""

    def setup_method(self):
        """Reset global state before each test."""
        import src.models.ensemble_adaptive as ea
        ea._last_weights_history = []

    def test_no_smoothing_first_call(self):
        from src.models.ensemble_adaptive import smooth_ensemble_weights
        w = smooth_ensemble_weights(0.5, 0.3, 0.2)
        assert w == (0.5, 0.3, 0.2)

    def test_no_smoothing_small_change(self):
        from src.models.ensemble_adaptive import smooth_ensemble_weights
        smooth_ensemble_weights(0.5, 0.3, 0.2)
        w = smooth_ensemble_weights(0.52, 0.29, 0.19)
        # Change is only 0.02 < 0.15 threshold → no smoothing
        assert abs(w[0] - 0.52) < 1e-9

    def test_smoothing_large_change(self):
        from src.models.ensemble_adaptive import smooth_ensemble_weights
        smooth_ensemble_weights(0.5, 0.3, 0.2)
        # Large swing: bp from 0.5 to 0.8 (change = 0.3 > 0.15)
        w = smooth_ensemble_weights(0.8, 0.1, 0.1)
        # Should be smoothed toward 0.5: 0.6*0.5 + 0.4*0.8 = 0.62
        assert abs(w[0] - 0.62) < 1e-6
        # Should be normalized
        assert abs(sum(w) - 1.0) < 1e-9

    def test_smoothing_preserves_sum_one(self):
        from src.models.ensemble_adaptive import smooth_ensemble_weights
        smooth_ensemble_weights(0.4, 0.35, 0.25)
        w = smooth_ensemble_weights(0.7, 0.2, 0.1)
        assert abs(sum(w) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests for CalibrationTracker
# ---------------------------------------------------------------------------

class TestCalibrationTracker:
    """Unit tests for the calibration tracking infrastructure."""

    def test_record_and_count(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("test_mkt", 0.6, 1.0)
        t.record("test_mkt", 0.4, 0.0)
        assert t.count("test_mkt") == 2

    def test_brier_score(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("m", 0.8, 1.0)
        t.record("m", 0.2, 0.0)
        # Perfect: Brier = ((0.8-1)^2 + (0.2-0)^2) / 2 = (0.04 + 0.04)/2 = 0.04
        assert abs(t.brier_score("m") - 0.04) < 1e-9

    def test_hit_rate(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("m", 0.7, 1.0)  # hit
        t.record("m", 0.6, 0.0)  # miss
        t.record("m", 0.3, 0.0)  # not a pick (< threshold)
        hr, hits, total = t.hit_rate("m", threshold=0.50)
        assert hr == pytest.approx(0.5, abs=1e-9)
        assert hits == 1
        assert total == 2

    def test_summary(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("over", 0.55, 1.0)
        t.record("over", 0.55, 0.0)
        s = t.summary("over")
        assert "brier_score" in s["over"]
        assert s["over"]["n"] == 2

    def test_reliability_data(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        for _ in range(10):
            t.record("m", 0.5, 1.0)
        data = t.reliability_data("m", n_bins=5)
        assert len(data) > 0
        # Each entry should be (avg_pred, avg_actual, count)
        for avg_pred, avg_actual, count in data:
            assert 0.0 <= avg_pred <= 1.0
            assert 0.0 <= avg_actual <= 1.0
            assert count > 0

    def test_reset(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("m", 0.5, 1.0)
        t.reset("m")
        assert t.count("m") == 0

    def test_calibration_error(self):
        from src.models.calibration_tracking import CalibrationTracker
        t = CalibrationTracker()
        t.record("m", 0.8, 1.0)
        t.record("m", 0.2, 0.0)
        # MAE = (|0.8-1| + |0.2-0|) / 2 = 0.20
        assert abs(t.calibration_error("m") - 0.20) < 1e-9
