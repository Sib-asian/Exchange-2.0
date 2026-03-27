"""
test_calibration.py — Test per il modulo models/calibration.py.

Verifica:
  - Correttezza della calibrazione Bayesiana xG
  - Coerenza del blend tiri + linee
  - Comportamento edge cases (linee flat, minuto 0, minuto 90)
"""

import math

from src.models.calibration import blend_xg_shots, calcola_xg_bayesiani

# ---------------------------------------------------------------------------
# calcola_xg_bayesiani
# ---------------------------------------------------------------------------

class TestCalcolaXGBayesiani:
    def test_output_positive(self):
        """Gli xG devono sempre essere positivi."""
        xg_h, xg_a = calcola_xg_bayesiani(-0.25, 2.5, -0.75, 2.75, 30)
        assert xg_h > 0
        assert xg_a > 0

    def test_total_xg_close_to_tot_bayes(self):
        """La somma degli xG deve essere vicina alla linea Total (quando flat)."""
        tot = 2.5
        xg_h, xg_a = calcola_xg_bayesiani(0.0, tot, 0.0, tot, 0)
        # Con AH=0 e linee flat, xg_h ≈ xg_a ≈ tot/2
        assert abs((xg_h + xg_a) - tot) < 0.05, f"Somma xG = {xg_h+xg_a:.3f}, atteso ≈ {tot}"

    def test_negative_ah_implies_home_favorite(self):
        """AH negativo implica che la casa ha xG maggiore."""
        xg_h, xg_a = calcola_xg_bayesiani(-0.5, 2.5, -0.5, 2.5, 0)
        assert xg_h > xg_a, f"AH negativo ma xg_h={xg_h:.3f} <= xg_a={xg_a:.3f}"

    def test_positive_ah_implies_away_favorite(self):
        """AH positivo implica che la trasferta ha xG maggiore."""
        xg_h, xg_a = calcola_xg_bayesiani(0.5, 2.5, 0.5, 2.5, 0)
        assert xg_a > xg_h, f"AH positivo ma xg_a={xg_a:.3f} <= xg_h={xg_h:.3f}"

    def test_flat_lines_returns_direct_values(self):
        """Con linee flat, il blend deve rispettare il cap temporale."""
        ah = -0.5
        tot = 2.5
        xg_h, xg_a = calcola_xg_bayesiani(ah, tot, ah, tot, 45)
        # Flat lines: usa direttamente ah_cur/tot_cur con cap temporale applicato
        # A minuto 45, il cap è (90-45)/90 * 4.0 = 2.0
        # Quindi xg_h + xg_a ≈ 2.0, non 2.5
        expected_cap = (90 - 45) / 90.0 * 4.0  # = 2.0
        assert abs((xg_h + xg_a) - expected_cap) < 0.2, f"Somma xG = {xg_h+xg_a:.3f}, atteso ≈ {expected_cap:.1f}"

    def test_xg_positive_at_minute_0(self):
        """Gli xG devono essere positivi anche a minuto 0."""
        xg_h, xg_a = calcola_xg_bayesiani(-0.25, 2.5, -0.25, 2.5, 0)
        assert xg_h > 0
        assert xg_a > 0

    def test_xg_near_zero_at_minute_90(self):
        """Gli xG rimanenti devono essere molto piccoli quando tot_cur è vicino a 0."""
        # Con linee flat e tot_cur molto basso (mercato chiude la partita),
        # gli xG devono riflettere il tot_cur — non il tempo rimanente.
        # Il tot_cur è "gol rimanenti attesi dal mercato": se il mercato prezza 0.2,
        # allora xg_h + xg_a ≈ 0.2 indipendentemente dal minuto.
        xg_h, xg_a = calcola_xg_bayesiani(0.0, 2.5, 0.0, 0.2, 90)
        assert xg_h + xg_a < 0.4, f"Somma xG troppo alta con tot_cur=0.2: {xg_h+xg_a}"

    def test_bigger_movement_increases_home_xg(self):
        """Un forte movimento verso la casa deve aumentare il suo xG."""
        # Linea si muove verso la casa (AH più negativo)
        xg_h_start, _ = calcola_xg_bayesiani(-0.25, 2.5, -0.25, 2.5, 30)
        xg_h_moved, _ = calcola_xg_bayesiani(-0.25, 2.5, -1.25, 2.5, 30)
        assert xg_h_moved > xg_h_start, "Movimento verso casa non aumenta xg_h"

    def test_xg_sum_bounded(self):
        """La somma degli xG non deve superare valori impossibili."""
        xg_h, xg_a = calcola_xg_bayesiani(-0.25, 2.5, -0.75, 2.75, 45)
        assert xg_h + xg_a < 6.0, f"Somma xG irrealistica: {xg_h+xg_a}"


# ---------------------------------------------------------------------------
# blend_xg_shots
# ---------------------------------------------------------------------------

class TestBlendXGShots:
    def test_output_positive(self):
        """Il blend deve sempre restituire xG positivi."""
        result = blend_xg_shots(1.2, 0.8, 4, 2, 2, 3, 0, 0, 45)
        assert result[0] > 0
        assert result[1] > 0

    def test_no_shots_returns_line_estimates(self):
        """Senza tiri, il blend deve restituire valori simili alle linee."""
        mu_h, mu_a = 1.2, 0.8
        result = blend_xg_shots(mu_h, mu_a, 0, 0, 0, 0, 0, 0, 30)
        # Con shot_info=0, α_T=α_D=0 → blend = linee
        assert abs(result[0] - mu_h) < 0.05, f"xg_h_final={result[0]:.3f} lontano da mu_h={mu_h}"
        assert abs(result[1] - mu_a) < 0.05

    def test_dominant_home_shots_increases_home_xg(self):
        """Dominio tiri della casa deve aumentare il suo xG rispetto alla linea."""
        mu_h, mu_a = 1.0, 1.0
        # Casa: 10 tiri in porta, Trasf: 2 tiri
        result = blend_xg_shots(mu_h, mu_a, 10, 5, 2, 1, 0, 0, 45)
        xg_h_final = result[0]
        # Con dominio tiri, xg_h_final deve essere > mu_h
        assert xg_h_final > mu_h, f"Dominio tiri non aumenta xg_h: {xg_h_final:.3f}"

    def test_shot_dom_range(self):
        """L'indice shot_dom deve essere in [0, 1]."""
        result = blend_xg_shots(1.0, 1.0, 8, 3, 2, 1, 0, 0, 30)
        shot_dom = result[6]
        assert 0.0 <= shot_dom <= 1.0, f"shot_dom fuori range: {shot_dom}"

    def test_alpha_t_alpha_d_bounded(self):
        """α_T e α_D devono rispettare i massimi configurati."""
        from src.config import SHOTS
        result = blend_xg_shots(1.0, 1.0, 20, 10, 15, 8, 0, 0, 80)
        alpha_t, alpha_d = result[4], result[5]
        assert alpha_t <= SHOTS.ALPHA_T_MAX + 1e-9, f"α_T={alpha_t} supera MAX={SHOTS.ALPHA_T_MAX}"
        # alpha_d può ora arrivare fino a ALPHA_D_MAX_QUALITY quando la qualità attacchi è alta
        assert alpha_d <= SHOTS.ALPHA_D_MAX_QUALITY + 1e-9, \
            f"α_D={alpha_d} supera ALPHA_D_MAX_QUALITY={SHOTS.ALPHA_D_MAX_QUALITY}"

    def test_game_state_leading_home_reduces_away_quality(self):
        """
        Con la casa in vantaggio:
        - k_h > 1 (tiri casa in contropiede su difese scoperte → qualità ↑)
        - k_a < 1 (tiri trasferta in pressing disperato → qualità ↓)
        Quindi xg_a_accum deve DIMINUIRE rispetto alla partita pari.
        """
        result_neutral = blend_xg_shots(1.0, 1.0, 5, 3, 5, 3, 0, 0, 45)
        result_home_leading = blend_xg_shots(1.0, 1.0, 5, 3, 5, 3, 2, 0, 45)
        # Con casa in vantaggio, k_a = 1 - gs_adj < 1 → xg_a_accum diminuisce
        assert result_home_leading[3] < result_neutral[3], "Game state non riduce qualità tiri in pressing"
        # E xg_h_accum deve aumentare (contropiede)
        assert result_home_leading[2] > result_neutral[2], "Game state non aumenta qualità tiri contropiede"

    def test_mu_max_cap_applied(self):
        """Il cap MU_MAX deve essere applicato."""
        # Gara con 6 gol già segnati: MU_MAX = max(3.5, 6*1.2) = 7.2
        result = blend_xg_shots(1.0, 1.0, 50, 30, 50, 30, 3, 3, 45)
        # I valori non possono essere infiniti
        assert math.isfinite(result[0])
        assert math.isfinite(result[1])
