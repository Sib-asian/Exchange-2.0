"""
test_time_decay.py — Test per il modulo models/time_decay.py.

Verifica:
  - Score effect residuale
  - Effetto cartellini rossi
  - Momentum di mercato
"""


from src.models.time_decay import calcola_momentum_mercato, time_decay_dinamico

# ---------------------------------------------------------------------------
# time_decay_dinamico
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_output_positive(self):
        """Gli xG devono sempre essere positivi."""
        xg_c, xg_t = time_decay_dinamico(1.0, 0.8, 30, 0, 0, 0, 0)
        assert xg_c > 0
        assert xg_t > 0

    def test_no_adjustment_with_0_0_no_cards(self):
        """Senza gol né cartellini, gli xG non devono cambiare."""
        xg_in = 1.2
        xg_c, xg_t = time_decay_dinamico(xg_in, xg_in, 30, 0, 0, 0, 0)
        assert abs(xg_c - xg_in) < 1e-9
        assert abs(xg_t - xg_in) < 1e-9

    def test_home_trailing_increases_home_xg(self):
        """La casa in svantaggio deve avere xG aumentato (pressing)."""
        xg_c, xg_t = time_decay_dinamico(1.0, 1.0, 30, 0, 2, 0, 0)
        assert xg_c > 1.0, f"xg_c={xg_c} non aumentato per casa in svantaggio"
        assert xg_t < 1.0, f"xg_t={xg_t} non ridotto per trasf in vantaggio"

    def test_home_leading_decreases_home_xg(self):
        """La casa in vantaggio deve avere xG ridotto (parking the bus)."""
        xg_c, xg_t = time_decay_dinamico(1.0, 1.0, 30, 2, 0, 0, 0)
        assert xg_c < 1.0
        assert xg_t > 1.0

    def test_score_effect_capped(self):
        """Lo score effect residuale non deve superare il cap (8%)."""
        # Vantaggio enorme (5-0): l'effetto è già saturato
        xg_in = 1.0
        xg_c, xg_t = time_decay_dinamico(xg_in, xg_in, 30, 0, 5, 0, 0)
        # Casa in svantaggio 0-5: xg_c aumentato, ma con cap
        max_boost = xg_in * (1.0 + 0.08 * 1.5)  # max teorico > cap reale
        assert xg_c <= max_boost

    def test_red_card_home_reduces_home_xg(self):
        """Un cartellino rosso alla casa deve ridurne gli xG."""
        xg_c, xg_t = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 1, 0)
        assert xg_c < 1.0, f"Rosso casa non riduce xg_c: {xg_c}"
        assert xg_t > 1.0, f"Rosso casa non aumenta xg_t: {xg_t}"

    def test_red_card_away_reduces_away_xg(self):
        """Un cartellino rosso alla trasferta deve ridurne gli xG."""
        xg_c, xg_t = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 0, 1)
        assert xg_t < 1.0
        assert xg_c > 1.0

    def test_two_reds_less_than_double_effect(self):
        """Due rossi devono avere effetto minore del doppio di uno (marginale decrescente)."""
        xg_c1, _ = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 1, 0)
        xg_c2, _ = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 2, 0)
        # xg_c2 deve essere > 0 (non azzerato) e > xg_c1 * (xg_c1/1.0)
        # i.e., il secondo rosso ha effetto marginale minore
        reduction_1 = 1.0 - xg_c1
        reduction_2 = 1.0 - xg_c2
        assert reduction_2 < 2 * reduction_1, "Secondo rosso non ha effetto marginale decrescente"

    def test_minute_90_returns_floor(self):
        """A minuto 90, gli xG devono essere al floor."""
        from src.config import DECAY
        xg_c, xg_t = time_decay_dinamico(1.0, 0.8, 90, 0, 0, 0, 0)
        assert xg_c == DECAY.XG_FLOOR
        assert xg_t == DECAY.XG_FLOOR

    def test_xg_floor_applied(self):
        """Gli xG non devono scendere sotto XG_FLOOR."""
        from src.config import DECAY
        # Rossi estremi
        xg_c, xg_t = time_decay_dinamico(0.001, 0.001, 30, 0, 0, 4, 4)
        assert xg_c >= DECAY.XG_FLOOR
        assert xg_t >= DECAY.XG_FLOOR

    def test_away_red_asymmetry(self):
        """Il rosso alla trasferta deve avere effetto leggermente maggiore."""
        # Stesse condizioni, red diverso
        xg_c_home_red, _ = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 1, 0)
        _, xg_t_away_red = time_decay_dinamico(1.0, 1.0, 30, 0, 0, 0, 1)
        # Con asimmetria, il rosso in trasferta riduce di più la trasf.
        reduction_home = 1.0 - xg_c_home_red
        reduction_away = 1.0 - xg_t_away_red
        assert reduction_away > reduction_home, "Asimmetria rosso in trasferta non applicata"


# ---------------------------------------------------------------------------
# calcola_momentum_mercato
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_zero_at_minute_zero(self):
        """Il momentum a minuto 0 deve essere 0."""
        m = calcola_momentum_mercato(0.5, 0.5, 0)
        assert m == 0.0

    def test_positive_with_movement(self):
        """Il momentum deve essere positivo con movimento di linea."""
        m = calcola_momentum_mercato(0.5, 0.5, 30)
        assert m > 0

    def test_capped_at_max(self):
        """Il momentum deve essere cappato al massimo."""
        from src.config import MOMENTUM
        m = calcola_momentum_mercato(10.0, 10.0, 10)
        assert m <= MOMENTUM.MOMENTUM_CAP

    def test_same_delta_higher_minute_lower_momentum(self):
        """Stesso delta a minuto più alto deve dare momentum inferiore (sqrt scaling)."""
        m_early = calcola_momentum_mercato(0.5, 0.5, 10)
        m_late = calcola_momentum_mercato(0.5, 0.5, 70)
        assert m_early > m_late, f"Momentum early={m_early:.3f} non > late={m_late:.3f}"

    def test_zero_delta_zero_momentum(self):
        """Con delta = 0, il momentum deve essere 0."""
        m = calcola_momentum_mercato(0.0, 0.0, 45)
        assert m == 0.0

    def test_ah_delta_more_impactful_than_tot_delta(self):
        """Il delta AH deve avere impatto maggiore del delta Total (peso 1 vs 0.5)."""
        m_ah = calcola_momentum_mercato(1.0, 0.0, 30)
        m_tot = calcola_momentum_mercato(0.0, 1.0, 30)
        assert m_ah > m_tot, f"AH delta non più impattante di TOT delta: {m_ah:.3f} vs {m_tot:.3f}"
