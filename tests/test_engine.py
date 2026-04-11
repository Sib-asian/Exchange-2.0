"""
test_engine.py — Test di integrazione per il motore completo.

Verifica la pipeline completa end-to-end:
  - Validazione input (MatchState)
  - Output coerenti (probabilità che sommano a 1, valori positivi)
  - Comportamento con e senza dati tiri
  - Edge cases (0-0 a 0', 5-3 a 80', cartellini rossi, ecc.)
"""

import pytest

from src.engine import ExchangeQuotes, MatchState, analizza

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stato_prematch():
    return MatchState(
        minuto=0,
        gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.25, tot_op=2.5,
        ah_cur=-0.25, tot_cur=2.5,
        linea_ou=2.5,
    )


@pytest.fixture
def stato_live_45():
    return MatchState(
        minuto=45,
        gol_casa=1, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.25, tot_op=2.5,
        ah_cur=-0.5, tot_cur=2.25,
        linea_ou=2.5,
        sot_h=5, soff_h=3,
        sot_a=2, soff_a=1,
    )


@pytest.fixture
def stato_late_game():
    return MatchState(
        minuto=75,
        gol_casa=2, gol_trasf=1,
        rossi_casa=0, rossi_trasf=1,
        ah_op=-0.5, tot_op=2.75,
        ah_cur=-1.0, tot_cur=1.5,
        linea_ou=3.5,
        sot_h=8, soff_h=4,
        sot_a=3, soff_a=2,
    )


# ---------------------------------------------------------------------------
# Test MatchState validation
# ---------------------------------------------------------------------------

class TestMatchStateValidation:
    def test_valid_state_no_exception(self, stato_prematch):
        """MatchState valido non deve sollevare eccezioni."""
        assert stato_prematch.minuto == 0

    def test_invalid_minuto_raises(self):
        # Fix #3.6: Ora usa ValueError invece di AssertionError
        with pytest.raises(ValueError):
            MatchState(
                minuto=91, gol_casa=0, gol_trasf=0,
                rossi_casa=0, rossi_trasf=0,
                ah_op=-0.25, tot_op=2.5,
                ah_cur=-0.25, tot_cur=2.5,
                linea_ou=2.5,
            )

    def test_negative_goals_raises(self):
        # Fix #3.6: Ora usa ValueError invece di AssertionError
        with pytest.raises(ValueError):
            MatchState(
                minuto=30, gol_casa=-1, gol_trasf=0,
                rossi_casa=0, rossi_trasf=0,
                ah_op=-0.25, tot_op=2.5,
                ah_cur=-0.25, tot_cur=2.5,
                linea_ou=2.5,
            )

    def test_invalid_bankroll_raises(self):
        # Fix #3.6: Ora usa ValueError invece di AssertionError
        with pytest.raises(ValueError):
            MatchState(
                minuto=0, gol_casa=0, gol_trasf=0,
                rossi_casa=0, rossi_trasf=0,
                ah_op=-0.25, tot_op=2.5,
                ah_cur=-0.25, tot_cur=2.5,
                linea_ou=2.5,
                bankroll=-100.0,
            )


# ---------------------------------------------------------------------------
# Test pipeline completa
# ---------------------------------------------------------------------------

class TestEngineOutput:
    def test_1x2_sum_to_one(self, stato_prematch):
        r = analizza(stato_prematch)
        total = r.p1 + r.px + r.p2
        assert abs(total - 1.0) < 1e-9, f"1X2 sum={total}"

    def test_over_under_sum_to_one(self, stato_prematch):
        r = analizza(stato_prematch)
        total = r.p_under + r.p_over
        assert abs(total - 1.0) < 1e-9, f"OU sum={total}"

    def test_btts_in_range(self, stato_prematch):
        r = analizza(stato_prematch)
        assert 0.0 <= r.p_btts <= 1.0

    def test_xg_positive(self, stato_prematch):
        r = analizza(stato_prematch)
        assert r.xg_h_final > 0
        assert r.xg_a_final > 0

    def test_rho_in_range(self, stato_prematch):
        from src.config import RHO
        r = analizza(stato_prematch)
        assert RHO.RHO_MIN <= r.rho <= RHO.BASE_MAX + 0.001

    def test_live_45_with_shots(self, stato_live_45):
        r = analizza(stato_live_45)
        assert abs(r.p1 + r.px + r.p2 - 1.0) < 1e-9
        assert abs(r.p_under + r.p_over - 1.0) < 1e-9

    def test_late_game_with_red_card(self, stato_late_game):
        r = analizza(stato_late_game)
        assert abs(r.p1 + r.px + r.p2 - 1.0) < 1e-9

    def test_home_favorite_state(self):
        """Con AH fortemente negativo, la casa deve essere favorita."""
        state = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-1.5, tot_op=3.0,
            ah_cur=-1.5, tot_cur=3.0,
            linea_ou=3.0,
        )
        r = analizza(state)
        assert r.p1 > r.p2, f"Casa non favorita: P1={r.p1:.3f}, P2={r.p2:.3f}"

    def test_away_favorite_state(self):
        """Con AH positivo, la trasferta deve essere favorita."""
        state = MatchState(
            minuto=0, gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=1.0, tot_op=2.5,
            ah_cur=1.0, tot_cur=2.5,
            linea_ou=2.5,
        )
        r = analizza(state)
        assert r.p2 > r.p1, f"Trasf non favorita: P1={r.p1:.3f}, P2={r.p2:.3f}"

    def test_btts_settled_both_scored(self):
        """Se entrambe hanno segnato, BTTS Sì = 1.0."""
        state = MatchState(
            minuto=60, gol_casa=1, gol_trasf=1,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-0.25, tot_op=2.5,
            ah_cur=0.0, tot_cur=1.5,
            linea_ou=2.5,
        )
        r = analizza(state)
        assert r.p_btts == pytest.approx(1.0, abs=1e-9)

    def test_over_already_won(self):
        """Se i gol totali superano la linea, Over = già vinto → P(Over) ≈ 1."""
        state = MatchState(
            minuto=60, gol_casa=2, gol_trasf=2,
            rossi_casa=0, rossi_trasf=0,
            ah_op=0.0, tot_op=2.5,
            ah_cur=0.0, tot_cur=1.5,
            linea_ou=2.5,
        )
        r = analizza(state)
        # Linea 2.5, gol_attuali = 4 >= 2.5 → P(over) = 1
        assert r.p_over > 0.999

    def test_top_cs_not_empty(self, stato_prematch):
        r = analizza(stato_prematch)
        assert len(r.top_cs) > 0

    def test_top_cs_ordered_by_probability(self, stato_prematch):
        r = analizza(stato_prematch)
        probs = [p for _, p in r.top_cs]
        assert probs == sorted(probs, reverse=True)

    def test_gol_tot_dist_sums_to_one(self, stato_prematch):
        r = analizza(stato_prematch)
        total = sum(r.gol_tot_dist.values())
        assert abs(total - 1.0) < 1e-9

    def test_momentum_zero_flat_lines(self, stato_prematch):
        """Con linee flat a minuto 0, il momentum deve essere 0."""
        r = analizza(stato_prematch)
        assert r.momentum == 0.0

    def test_market_divergence_zero_without_ocr(self, stato_prematch):
        """Senza quote OCR, market_divergence deve restare 0.0 (nessun dato mercato)."""
        r = analizza(stato_prematch)
        assert r.market_divergence == 0.0

    def test_market_divergence_nonzero_with_ocr_quotes(self):
        """Con quote OCR disponibili, market_divergence deve essere calcolata (> 0)."""
        state = MatchState(
            minuto=0,
            gol_casa=0, gol_trasf=0,
            rossi_casa=0, rossi_trasf=0,
            ah_op=-1.5, tot_op=3.0,
            ah_cur=-1.5, tot_cur=3.0,
            linea_ou=2.5,
            # Quote OCR che implicano una partita molto diversa dal modello
            ocr_quota_1=1.50, ocr_quota_x=4.0, ocr_quota_2=6.0,
            ocr_quota_over=1.90, ocr_quota_under=1.90,
        )
        r = analizza(state)
        # La divergenza è sempre ≥ 0 e dovrebbe essere positiva con quote presenti
        assert r.market_divergence >= 0.0
        # Non è necessariamente > 0 se le quote sono perfettamente allineate,
        # ma con questi input il modello e le quote divergono sicuramente.
        assert r.market_divergence > 0.0, (
            f"market_divergence dovrebbe essere > 0 con quote OCR: {r.market_divergence}"
        )


# ---------------------------------------------------------------------------
# Test ExchangeQuotes
# ---------------------------------------------------------------------------

class TestExchangeQuotes:
    def test_any_active_false_when_all_zero(self):
        q = ExchangeQuotes()
        assert not q.any_active

    def test_any_active_true_when_one_set(self):
        q = ExchangeQuotes(q_1=2.50)
        assert q.any_active

    def test_prob_implicita(self):
        from src.signals import Signal
        s = Signal(tipo="BACK", mercato="1", prob_mod=0.5, quota_fair=2.0, quota_exc=2.5)
        assert abs(s.prob_implicita - 0.4) < 1e-10
