"""
test_markets.py — Test per i moduli di calcolo dei mercati.

Verifica:
  - Over/Under quarter lines
  - Asian Handicap
  - BTTS condizionale
"""

import pytest

from src.markets.asian_handicap import calcola_asian_handicap
from src.markets.btts import calcola_btts
from src.markets.over_under import calcola_over_under
from src.models.poisson import build_bivariate_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def full_matrix_eq():
    """Partita equilibrata 0-0 al 30'."""
    _, full, _ = build_bivariate_matrix(1.2, 1.2, 30, 2.4)
    return full


@pytest.fixture
def full_matrix_strong_home():
    """Casa fortemente favorita."""
    _, full, _ = build_bivariate_matrix(2.5, 0.5, 0, 3.0)
    return full


# ---------------------------------------------------------------------------
# Over/Under
# ---------------------------------------------------------------------------

class TestOverUnder:
    def test_sum_to_one(self, full_matrix_eq):
        p_u, p_o = calcola_over_under(full_matrix_eq, 0, 2.5)
        assert abs(p_u + p_o - 1.0) < 1e-9

    def test_over_when_goal_already_scored(self, full_matrix_eq):
        """Con 3 gol già segnati e linea 2.5, Over è già vinto."""
        # Con gol_attuali=3 >= linea=2.5, P(Under 2.5) = P(gol_rimanenti < -0.5) = 0
        p_u, p_o = calcola_over_under(full_matrix_eq, 3, 2.5)
        assert p_u == 0.0
        assert p_o == 1.0

    def test_quarter_line_between_half_lines(self, full_matrix_eq):
        """Under 2.75 deve essere tra Under 2.5 e Under 3.0."""
        p_u25, _ = calcola_over_under(full_matrix_eq, 0, 2.5)
        p_u275, _ = calcola_over_under(full_matrix_eq, 0, 2.75)
        p_u30, _ = calcola_over_under(full_matrix_eq, 0, 3.0)
        assert p_u25 < p_u275 < p_u30, (
            f"Quarter line non tra half-lines: U2.5={p_u25:.3f}, U2.75={p_u275:.3f}, U3.0={p_u30:.3f}"
        )

    def test_quarter_line_is_average_of_eff_probs(self, full_matrix_eq):
        """
        Under 2.75 = ½ × P_eff(U2.5) + ½ × P_eff(U3.0).

        P_eff(U2.5) = P(total ≤ 2)                    [half-line, no push]
        P_eff(U3.0) = P(total ≤ 2) + 0.5 × P(total=3) [whole line, push at 3]

        Under 2.75 = P(≤2) + 0.25 × P(=3)
        """
        p_u25, _ = calcola_over_under(full_matrix_eq, 0, 2.5)
        p_u275, _ = calcola_over_under(full_matrix_eq, 0, 2.75)
        p_u30, _ = calcola_over_under(full_matrix_eq, 0, 3.0)
        # Verifica la media degli effective probability
        expected = 0.5 * (p_u25 + p_u30)
        assert abs(p_u275 - expected) < 1e-9, f"U2.75={p_u275:.6f} ≠ avg(U2.5+U3.0)={expected:.6f}"

    def test_high_expected_goals_high_over(self, full_matrix_strong_home):
        """Con mu_h alto, Over 2.5 deve essere molto probabile."""
        _, p_o = calcola_over_under(full_matrix_strong_home, 0, 2.5)
        assert p_o > 0.5, f"Over 2.5 troppo bassa con mu_h=2.5: {p_o:.3f}"

    def test_monotone_increasing_over_with_line(self, full_matrix_eq):
        """P(Over X) deve diminuire all'aumentare di X."""
        lines = [1.5, 2.5, 3.5, 4.5]
        overs = [calcola_over_under(full_matrix_eq, 0, line)[1] for line in lines]
        for i in range(len(overs) - 1):
            assert overs[i] >= overs[i+1], f"P(Over) non monotona: {overs}"


# ---------------------------------------------------------------------------
# Asian Handicap
# ---------------------------------------------------------------------------

class TestAsianHandicap:
    def test_results_count(self, full_matrix_eq):
        """Deve restituire un risultato per ogni livello."""
        from src.config import UI
        results = calcola_asian_handicap(full_matrix_eq, UI.AH_LEVELS)
        assert len(results) == len(UI.AH_LEVELS)

    def test_probabilities_sum_to_one(self, full_matrix_eq):
        """P(win) + P(push) + P(lose) = 1 per ogni livello."""
        from src.config import UI
        results = calcola_asian_handicap(full_matrix_eq, UI.AH_LEVELS)
        for r in results:
            total = r["p_win"] + r["p_push"] + r["p_lose"]
            assert abs(total - 1.0) < 1e-9, f"AH {r['level']}: sum={total}"

    def test_p_eff_in_range(self, full_matrix_eq):
        """P_eff deve essere in [0, 1]."""
        from src.config import UI
        results = calcola_asian_handicap(full_matrix_eq, UI.AH_LEVELS)
        for r in results:
            assert 0.0 <= r["p_eff"] <= 1.0, f"p_eff fuori range: {r['p_eff']}"

    def test_ah0_symmetric(self, full_matrix_eq):
        """AH 0.0 in una partita simmetrica deve dare p_eff ≈ 0.5 + push/2."""
        results = calcola_asian_handicap(full_matrix_eq, [0.0])
        r = results[0]
        # In partita simmetrica: win ≈ lose, p_eff ≈ 0.5 + push*0.5 ≥ 0.5
        assert r["p_eff"] >= 0.45, f"AH 0.0 p_eff troppo bassa: {r['p_eff']}"

    def test_strong_home_ah_negative_high_coverage(self, full_matrix_strong_home):
        """Con casa molto forte, AH -0.5 deve avere alta copertura."""
        results = calcola_asian_handicap(full_matrix_strong_home, [-0.5])
        r = results[0]
        assert r["p_eff"] > 0.70, f"Casa forte ma AH -0.5 p_eff bassa: {r['p_eff']}"


# ---------------------------------------------------------------------------
# BTTS
# ---------------------------------------------------------------------------

class TestBTTS:
    def test_btts_both_scored_is_one(self, full_matrix_eq):
        p = calcola_btts(full_matrix_eq, 1, 1)
        assert p == 1.0

    def test_btts_home_only_scored(self, full_matrix_eq):
        """Con solo la casa a segno, BTTS = P(trasferta segna nel rimasto)."""
        p = calcola_btts(full_matrix_eq, 1, 0)
        p_trasf_scores = sum(prob for (a, b), prob in full_matrix_eq.items() if b > 0)
        assert abs(p - p_trasf_scores) < 1e-9

    def test_btts_none_scored(self, full_matrix_eq):
        """Senza gol, BTTS = P(entrambe segnano nel rimasto)."""
        p = calcola_btts(full_matrix_eq, 0, 0)
        p_both = sum(prob for (a, b), prob in full_matrix_eq.items() if a > 0 and b > 0)
        assert abs(p - p_both) < 1e-9

    def test_btts_in_range(self, full_matrix_eq):
        p = calcola_btts(full_matrix_eq, 0, 0)
        assert 0.0 <= p <= 1.0

    def test_btts_increases_with_xg(self):
        """BTTS deve essere più alta con xG più alti."""
        _, full_low, _ = build_bivariate_matrix(0.5, 0.5, 0, 1.0)
        _, full_high, _ = build_bivariate_matrix(2.0, 2.0, 0, 4.0)
        p_low = calcola_btts(full_low, 0, 0)
        p_high = calcola_btts(full_high, 0, 0)
        assert p_high > p_low, f"BTTS non aumenta con xG: {p_low:.3f} vs {p_high:.3f}"
