"""
test_kelly.py — Test per il modulo models/kelly.py.

Verifica:
  - Correttezza formula Kelly per back e lay
  - Gestione edge cases (edge zero, quote estreme, commissioni)
  - Calcolo EV
  - Calcolo edge netto
"""

import pytest

from src.models.kelly import (
    calcola_edge_back,
    calcola_edge_lay,
    calcola_ev_back,
    calcola_ev_lay,
    calcola_kelly_fraction,
    calcola_stake_kelly,
    calcola_stake_lay,
    quota_netta,
)


# ---------------------------------------------------------------------------
# quota_netta
# ---------------------------------------------------------------------------

class TestQuotaNetta:
    def test_zero_commission(self):
        """Con commissione 0, la quota netta = quota nominale."""
        assert quota_netta(3.0, 0.0) == 3.0

    def test_commission_reduces_odds(self):
        """La commissione deve ridurre la quota netta."""
        q = quota_netta(3.0, 0.025)
        assert q < 3.0
        assert q > 1.0

    def test_commission_formula(self):
        """Verifica la formula: 1 + (Q-1)*(1-c)."""
        q = quota_netta(3.0, 0.05)
        expected = 1.0 + (3.0 - 1.0) * (1.0 - 0.05)
        assert abs(q - expected) < 1e-10


# ---------------------------------------------------------------------------
# calcola_edge_back
# ---------------------------------------------------------------------------

class TestEdgeBack:
    def test_positive_edge_when_prob_above_implied(self):
        """Edge positivo quando il modello stima più del mercato."""
        # Modello: 55%, Mercato: 50% → edge = 0.05
        q_net = 2.0  # quota netta 2.0 → implied = 50%
        edge = calcola_edge_back(0.55, q_net)
        assert edge > 0

    def test_zero_edge_at_fair_price(self):
        """Edge zero quando il modello coincide con il mercato."""
        prob = 0.40
        q_net = 1.0 / prob
        edge = calcola_edge_back(prob, q_net)
        assert abs(edge) < 1e-10

    def test_negative_edge_when_below_implied(self):
        """Edge negativo quando il modello è sotto il mercato."""
        q_net = 2.0  # implied 50%
        edge = calcola_edge_back(0.45, q_net)
        assert edge < 0


# ---------------------------------------------------------------------------
# calcola_edge_lay
# ---------------------------------------------------------------------------

class TestEdgeLay:
    def test_positive_edge_when_market_overpriced(self):
        """Edge lay positivo quando il mercato sopravvaluta l'evento."""
        # Modello: 30%, Mercato: 40% implicita → il layer ha valore
        edge = calcola_edge_lay(0.30, 2.5, 0.0)  # Q=2.5 → implied=40%
        assert edge > 0

    def test_zero_edge_at_break_even(self):
        """Edge lay zero al break-even (senza commissione)."""
        prob = 0.40
        q = 1.0 / prob  # Q = 2.5 → p_BE = 40% (senza comm)
        edge = calcola_edge_lay(prob, q, 0.0)
        assert abs(edge) < 1e-10

    def test_commission_reduces_lay_edge(self):
        """La commissione riduce l'edge lay disponibile."""
        edge_no_comm = calcola_edge_lay(0.30, 2.5, 0.0)
        edge_with_comm = calcola_edge_lay(0.30, 2.5, 0.05)
        assert edge_no_comm > edge_with_comm


# ---------------------------------------------------------------------------
# calcola_stake_kelly (back)
# ---------------------------------------------------------------------------

class TestKellyBack:
    def test_zero_stake_at_zero_ev(self):
        """Kelly = 0 quando EV <= 0."""
        # prob * quota = 1.0 → EV = 0
        stake = calcola_stake_kelly(0.5, 2.0, 1000.0, 0.5)
        assert stake == 0.0

    def test_positive_stake_with_positive_edge(self):
        """Kelly > 0 con edge positivo."""
        stake = calcola_stake_kelly(0.6, 2.0, 1000.0, 0.5)
        assert stake > 0

    def test_stake_capped_at_max_pct(self):
        """La stake non deve superare il cap del bankroll."""
        from src.config import KELLY
        # Edge enorme: Kelly puro sarebbe molto alto
        stake = calcola_stake_kelly(0.99, 100.0, 1000.0, 1.0)
        assert stake <= 1000.0 * KELLY.KELLY_MAX_PCT + 0.01

    def test_stake_proportional_to_bankroll(self):
        """La stake deve scalare linearmente con il bankroll."""
        stake_1k = calcola_stake_kelly(0.6, 2.0, 1000.0, 0.5)
        stake_2k = calcola_stake_kelly(0.6, 2.0, 2000.0, 0.5)
        assert abs(stake_2k / stake_1k - 2.0) < 0.01

    def test_fraction_reduces_stake(self):
        """Frazione Kelly minore produce stake minore."""
        stake_half = calcola_stake_kelly(0.6, 2.0, 1000.0, 0.5)
        stake_quarter = calcola_stake_kelly(0.6, 2.0, 1000.0, 0.25)
        assert stake_half > stake_quarter


# ---------------------------------------------------------------------------
# calcola_stake_lay
# ---------------------------------------------------------------------------

class TestKellyLay:
    def test_none_when_no_edge(self):
        """Deve restituire None quando non c'è edge lay."""
        result = calcola_stake_lay(0.5, 2.0, 1000.0, 0.5, 0.0)
        assert result is None

    def test_returns_tuple_with_edge(self):
        """Deve restituire (stake, liability) con edge positivo."""
        # p_BE = 1.0/(2.5) = 0.40, edge = 0.40 - 0.30 = 0.10
        result = calcola_stake_lay(0.30, 2.5, 1000.0, 0.5, 0.0)
        assert result is not None
        stake, liability = result
        assert stake > 0
        assert liability > 0

    def test_liability_equals_stake_times_q_minus_1(self):
        """liability = stake * (Q - 1)."""
        result = calcola_stake_lay(0.30, 2.5, 1000.0, 0.5, 0.0)
        assert result is not None
        stake, liability = result
        expected_liability = stake * (2.5 - 1.0)
        assert abs(liability - expected_liability) < 0.01

    def test_none_for_very_low_odds(self):
        """Deve restituire None per quote sotto LAY_MIN_ODDS."""
        from src.config import KELLY
        result = calcola_stake_lay(0.10, KELLY.LAY_MIN_ODDS - 0.1, 1000.0, 0.5, 0.0)
        assert result is None

    def test_commission_adjusted_break_even(self):
        """Con commissione, il break-even lay è più alto → edge minore → stake minore."""
        # Usiamo prob e quota che NON colpiscano il cap Kelly (0.05)
        # f_liability senza comm = 1/Q - prob = 0.40 - 0.30 = 0.10 → cap 0.05 attivo
        # Per evitare il cap usiamo un'edge molto piccola
        # f_senza_comm = 1/3.0 - 0.30 = 0.333 - 0.30 = 0.033 < 0.05 (no cap)
        prob = 0.30
        q = 3.0
        # p_BE senza comm = 1/3.0 = 0.333, edge = 0.033
        result_no_comm = calcola_stake_lay(prob, q, 1000.0, 0.5, 0.0)
        # p_BE con comm 5%: (1-0.05)/(3.0-0.05) = 0.95/2.95 = 0.322, edge = 0.022
        result_with_comm = calcola_stake_lay(prob, q, 1000.0, 0.5, 0.05)
        assert result_no_comm is not None
        assert result_with_comm is not None
        # La stake (e liability) è ridotta dalla commissione (edge più basso)
        assert result_no_comm[0] > result_with_comm[0]


# ---------------------------------------------------------------------------
# calcola_kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_base_fraction_early_game(self):
        """La frazione base deve essere quella configurata."""
        from src.config import KELLY
        frac = calcola_kelly_fraction(0, 10)
        assert abs(frac - KELLY.KELLY_BASE_FRACTION) < 1e-10

    def test_reduced_late_game(self):
        """La frazione deve ridursi oltre il 75'."""
        frac_early = calcola_kelly_fraction(30, 10)
        frac_late = calcola_kelly_fraction(80, 10)
        assert frac_late < frac_early

    def test_reduced_without_shots(self):
        """La frazione deve ridursi senza dati tiri."""
        frac_with_shots = calcola_kelly_fraction(30, 10)
        frac_no_shots = calcola_kelly_fraction(30, 0)
        assert frac_no_shots < frac_with_shots

    def test_floor_not_breached(self):
        """La frazione non deve scendere sotto il floor."""
        from src.config import KELLY
        frac = calcola_kelly_fraction(90, 0)
        assert frac >= KELLY.KELLY_MIN_FRACTION


# ---------------------------------------------------------------------------
# EV calculations
# ---------------------------------------------------------------------------

class TestEV:
    def test_ev_back_positive_with_edge(self):
        """EV back deve essere positivo con edge."""
        ev = calcola_ev_back(100.0, 0.6, 2.0)  # EV = 100 * (0.6*2 - 1) = 20
        assert abs(ev - 20.0) < 0.01

    def test_ev_back_zero_at_fair(self):
        """EV back = 0 alla quota fair."""
        prob = 0.5
        q_net = 1.0 / prob
        ev = calcola_ev_back(100.0, prob, q_net)
        assert abs(ev) < 0.01

    def test_ev_lay_sign(self):
        """EV lay deve essere positivo quando il layer ha vantaggio."""
        # Prob reale 30%, quota 2.5 (40% implicita), senza comm
        ev = calcola_ev_lay(100.0, 0.30, 2.5, 0.0)
        # EV = 100 * ((1-0.3)*1.0 - 0.3*1.5) = 100 * (0.7 - 0.45) = 25
        assert ev > 0
