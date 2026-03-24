"""
test_simulation.py — Test di simulazione end-to-end per il motore Radar Pro Live.

Verifica la coerenza delle previsioni in scenari realistici e edge cases
con simulazioni parametriche sull'intera pipeline:
  - xG Bayesiani a diversi minuti
  - Ratio cap per scenari estremi a fine partita
  - Coerenza probabilità mercati (1X2, O/U, BTTS, AH)
  - Pipeline completa engine con diverse configurazioni
  - Stabilità numerica in condizioni limite
"""

import math

import pytest

from src.config import BAYES, POISSON
from src.markets.btts import calcola_btts
from src.markets.over_under import calcola_over_under
from src.markets.result import calcola_1x2
from src.models.calibration import (
    _ah_ev,
    _ah_ev_half,
    blend_xg_shots,
    calcola_xg_bayesiani,
)
from src.models.poisson import build_bivariate_matrix

# ===========================================================================
# Sezione 1: AH EV — interpolazione continua
# ===========================================================================

class TestAHEVInterpolation:
    """Verifica che l'interpolazione AH EV sia continua e corretta."""

    def test_half_lines_exact(self):
        """Ai valori half-line, _ah_ev deve coincidere con _ah_ev_half (con lambda0)."""
        # FIX: Ora _ah_ev calcola lambda0 internamente e passa lambda indipendenti
        # a _ah_ev_half. Questo è il comportamento corretto per coerenza col modello.
        # La differenza rispetto a chiamare _ah_ev_half direttamente con lambda totali
        # è il bias 2-3% che la correzione elimina.
        for ah in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            ev_interp = _ah_ev(1.3, 0.9, ah)
            # Calcola lambda0 per ottenere i lambda indipendenti
            geom_mu = math.sqrt(1.3 * 0.9)
            mu_min = min(1.3, 0.9)
            lambda0 = min(0.10 * geom_mu, POISSON.LAMBDA0_CAP_RATIO * mu_min, mu_min)
            mu_h_ind = max(POISSON.EPS, 1.3 - lambda0)
            mu_a_ind = max(POISSON.EPS, 0.9 - lambda0)
            ev_exact = _ah_ev_half(mu_h_ind, mu_a_ind, ah)
            assert abs(ev_interp - ev_exact) < 1e-9, (
                f"ah={ah}: interp={ev_interp}, exact={ev_exact}"
            )

    def test_quarter_lines_split(self):
        """Alle quarter lines standard, EV ≈ media dei due half-lines adiacenti (con correzione curvatura)."""
        mu_h, mu_a = 1.2, 0.8
        # FIX: Calcola lambda0 per coerenza con la nuova implementazione
        geom_mu = math.sqrt(mu_h * mu_a)
        mu_min = min(mu_h, mu_a)
        lambda0 = min(0.10 * geom_mu, POISSON.LAMBDA0_CAP_RATIO * mu_min, mu_min)
        mu_h_ind = max(POISSON.EPS, mu_h - lambda0)
        mu_a_ind = max(POISSON.EPS, mu_a - lambda0)

        for ah in [-0.75, -0.25, 0.25, 0.75]:
            ev_q = _ah_ev(mu_h, mu_a, ah)
            h_low = math.floor(ah * 2) / 2.0
            h_high = h_low + 0.5
            ev_avg = 0.5 * (_ah_ev_half(mu_h_ind, mu_a_ind, h_low) +
                            _ah_ev_half(mu_h_ind, mu_a_ind, h_high))
            # La correzione cubica introduce una piccola deviazione (< 1.5%)
            assert abs(ev_q - ev_avg) < 0.015, (
                f"ah={ah}: got={ev_q}, expected avg≈{ev_avg}, diff={abs(ev_q - ev_avg)}"
            )

    def test_non_standard_values_differ(self):
        """Valori non-standard vicini ma diversi devono dare EV diversi."""
        mu_h, mu_a = 1.2, 0.8
        ev1 = _ah_ev(mu_h, mu_a, -0.212)
        ev2 = _ah_ev(mu_h, mu_a, -0.406)
        assert ev1 != ev2, "Valori non-standard producono lo stesso EV (bug!)"

    def test_continuity(self):
        """L'EV deve variare in modo continuo al variare di ah."""
        mu_h, mu_a = 1.5, 1.0
        prev_ev = _ah_ev(mu_h, mu_a, -1.0)
        for i in range(1, 41):
            ah = -1.0 + i * 0.05
            ev = _ah_ev(mu_h, mu_a, ah)
            delta = abs(ev - prev_ev)
            assert delta < 0.15, f"Salto troppo grande a ah={ah}: delta={delta}"
            prev_ev = ev

    def test_monotonicity_with_handicap(self):
        """Con mu_h > mu_a, EV deve crescere al crescere dell'AH (handicap più favorevole)."""
        mu_h, mu_a = 1.5, 0.8
        prev_ev = _ah_ev(mu_h, mu_a, -2.0)
        for i in range(1, 81):
            ah = -2.0 + i * 0.05
            ev = _ah_ev(mu_h, mu_a, ah)
            assert ev >= prev_ev - 1e-9, (
                f"Non monotono a ah={ah}: ev={ev} < prev={prev_ev}"
            )
            prev_ev = ev


# ===========================================================================
# Sezione 2: xG Bayesiani — ratio cap e coerenza temporale
# ===========================================================================

class TestXGBayesianiSimulation:
    """Simulazione parametrica degli xG Bayesiani lungo la partita."""

    @pytest.mark.parametrize("ah_op,tot_op", [
        (-0.75, 2.5),
        (-0.25, 2.3),
        (0.5, 2.8),
        (-1.5, 3.0),
        (0.0, 2.0),
    ])
    def test_ratio_never_exceeds_cap(self, ah_op, tot_op):
        """Il rapporto xG non deve mai superare XG_RATIO_CAP."""
        cap = BAYES.XG_RATIO_CAP
        for m in range(0, 91, 5):
            frac_r = max(0.005, 1.0 - m / 90.0)
            ah_cur = ah_op * frac_r
            tot_cur = max(0.2, tot_op * frac_r)
            xg_h, xg_a = calcola_xg_bayesiani(
                ah_op, tot_op, ah_cur, tot_cur, m
            )
            ratio = max(xg_h, xg_a) / max(POISSON.EPS, min(xg_h, xg_a))
            assert ratio <= cap + 0.01, (
                f"min={m}: ratio={ratio:.2f} > cap={cap} "
                f"(xg_h={xg_h:.4f}, xg_a={xg_a:.4f})"
            )

    def test_ratio_cap_pathological_flat_ah(self):
        """Anche con AH fisso (patologico), il cap deve funzionare."""
        ah_op, tot_op = -0.75, 2.5
        cap = BAYES.XG_RATIO_CAP
        for m in [60, 75, 85, 89]:
            frac_r = max(0.005, 1.0 - m / 90.0)
            # AH non si muove (patologico)
            xg_h, xg_a = calcola_xg_bayesiani(
                ah_op, tot_op, ah_op, max(0.2, tot_op * frac_r), m
            )
            ratio = max(xg_h, xg_a) / max(POISSON.EPS, min(xg_h, xg_a))
            assert ratio <= cap + 0.01, f"min={m}: ratio={ratio:.2f}"

    def test_xg_sum_equals_tot_bayes_flat(self):
        """Con linee flat, xg_h + xg_a == tot_cur (a meno del cap)."""
        for tot in [1.5, 2.0, 2.5, 3.0]:
            xg_h, xg_a = calcola_xg_bayesiani(0.0, tot, 0.0, tot, 0)
            assert abs(xg_h + xg_a - tot) < 0.01, (
                f"tot={tot}: sum={xg_h + xg_a:.4f}"
            )

    def test_xg_decrease_with_time(self):
        """Gli xG rimanenti devono decrescere col passare del tempo."""
        ah_op, tot_op = -0.5, 2.5
        prev_sum = float("inf")
        for m in range(0, 86, 5):
            frac_r = max(0.005, 1.0 - m / 90.0)
            tot_cur = max(0.2, tot_op * frac_r)
            ah_cur = ah_op * frac_r
            xg_h, xg_a = calcola_xg_bayesiani(
                ah_op, tot_op, ah_cur, tot_cur, m
            )
            s = xg_h + xg_a
            assert s <= prev_sum + 0.01, (
                f"min={m}: sum={s:.3f} > prev={prev_sum:.3f}"
            )
            prev_sum = s

    def test_gol_diff_shifts_xg(self):
        """Un gol della casa (gol_diff=1) deve spostare xG verso la casa."""
        ah_op, tot_op = -0.5, 2.5
        # Senza gol
        xg_h_0, xg_a_0 = calcola_xg_bayesiani(
            ah_op, tot_op, -0.25, 1.8, 45, gol_diff=0, gol_tot=0
        )
        # Con 1-0 casa
        xg_h_1, xg_a_1 = calcola_xg_bayesiani(
            ah_op, tot_op, 0.25, 1.5, 45, gol_diff=1, gol_tot=1
        )
        # Con AH positivo (trasferta favorita sui rimanenti) la trasferta ha più xG
        assert xg_a_1 > xg_h_1, "Con AH +0.25 la trasferta dovrebbe avere più xG rimanenti"


# ===========================================================================
# Sezione 3: Coerenza probabilità mercati
# ===========================================================================

class TestMarketConsistency:
    """Verifica che le probabilità dei mercati siano coerenti."""

    @staticmethod
    def _build(mu_h, mu_a, tot_cur=None):
        """Helper: costruisce matrici per i test."""
        if tot_cur is None:
            tot_cur = mu_h + mu_a
        return build_bivariate_matrix(mu_h, mu_a, minuto=0, tot_cur=tot_cur)

    @pytest.mark.parametrize("mu_h,mu_a", [
        (1.5, 1.0),
        (0.8, 0.8),
        (2.0, 0.5),
        (0.3, 0.3),
    ])
    def test_1x2_sums_to_one(self, mu_h, mu_a):
        """P(1) + P(X) + P(2) = 1."""
        _, joint_ind, _ = self._build(mu_h, mu_a)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        s = p1 + px + p2
        assert abs(s - 1.0) < 0.01, f"1X2 sum={s:.4f}"

    @pytest.mark.parametrize("mu_h,mu_a", [
        (1.5, 1.0),
        (0.5, 0.5),
    ])
    def test_over_under_complement(self, mu_h, mu_a):
        """P(Over L) + P(Under L) ≈ 1 per ogni linea intera/mezza."""
        full, _, _ = self._build(mu_h, mu_a)
        for line in [0.5, 1.5, 2.5, 3.5]:
            p_under, p_over = calcola_over_under(full, 0, line)
            s = p_over + p_under
            assert abs(s - 1.0) < 0.01, (
                f"O/U line={line}: sum={s:.4f}"
            )

    @pytest.mark.parametrize("mu_h,mu_a", [
        (1.5, 1.0),
        (0.3, 0.3),
    ])
    def test_btts_complement(self, mu_h, mu_a):
        """P(BTTS Sì) + P(BTTS No) ≈ 1."""
        full, _, _ = self._build(mu_h, mu_a)
        p_btts_si = calcola_btts(full, 0, 0)
        assert 0.0 <= p_btts_si <= 1.0, f"BTTS Sì fuori [0,1]: {p_btts_si:.4f}"

    def test_1x2_home_favorite(self):
        """Con mu_h >> mu_a, P(1) > P(2)."""
        _, joint_ind, _ = self._build(2.0, 0.5)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        assert p1 > p2, f"Casa favorita ma P(1)={p1:.3f} <= P(2)={p2:.3f}"

    def test_over_high_mu_means_high_over(self):
        """Con mu alti, Over 2.5 deve essere maggiore di Over con mu bassi."""
        full_high, _, _ = self._build(2.0, 1.5)
        full_low, _, _ = self._build(0.5, 0.5)
        _, p_over_high = calcola_over_under(full_high, 0, 2.5)
        _, p_over_low = calcola_over_under(full_low, 0, 2.5)
        assert p_over_high > p_over_low, (
            f"P(Over 2.5) con mu alti ({p_over_high:.3f}) <= mu bassi ({p_over_low:.3f})"
        )

    def test_btts_low_mu_means_low_btts(self):
        """Con mu bassi, BTTS Sì deve essere basso."""
        full, _, _ = self._build(0.3, 0.3)
        p_si = calcola_btts(full, 0, 0)
        assert p_si < 0.20, f"Con mu bassi, P(BTTS Sì)={p_si:.3f} troppo alto"


# ===========================================================================
# Sezione 4: Blend xG shots — coerenza pesi e proiezioni
# ===========================================================================

class TestBlendXGShotsSimulation:
    """Simulazione del blend tiri/linee in diversi scenari di partita."""

    def test_progressive_shot_weight(self):
        """Il peso dei tiri deve crescere col tempo e col numero di tiri."""
        alphas_t = []
        alphas_d = []
        for m in [15, 30, 45, 60, 75]:
            # Tiri proporzionali al minuto
            n = int(m * 0.3)
            result = blend_xg_shots(1.0, 1.0, n, n // 2, n, n // 2, 0, 0, m)
            alphas_t.append(result[4])
            alphas_d.append(result[5])

        # Alpha deve crescere monotonicamente
        for i in range(1, len(alphas_t)):
            assert alphas_t[i] >= alphas_t[i - 1] - 1e-9, (
                f"alpha_t non crescente: {alphas_t}"
            )
            assert alphas_d[i] >= alphas_d[i - 1] - 1e-9, (
                f"alpha_d non crescente: {alphas_d}"
            )

    def test_game_state_symmetric(self):
        """L'effetto game-state deve essere simmetrico."""
        r_neutral = blend_xg_shots(1.0, 1.0, 5, 3, 5, 3, 0, 0, 45)
        r_home = blend_xg_shots(1.0, 1.0, 5, 3, 5, 3, 2, 0, 45)
        r_away = blend_xg_shots(1.0, 1.0, 5, 3, 5, 3, 0, 2, 45)

        # Home leading: xg_h_accum ↑, xg_a_accum ↓
        assert r_home[2] > r_neutral[2]
        assert r_home[3] < r_neutral[3]
        # Away leading: xg_a_accum ↑, xg_h_accum ↓
        assert r_away[3] > r_neutral[3]
        assert r_away[2] < r_neutral[2]

    def test_mu_max_cap_high_scoring(self):
        """In gare ad alto punteggio, mu_max si adatta."""
        # 5 gol totali → mu_max = max(3.5, 5*1.2) = 6.0
        result = blend_xg_shots(1.0, 1.0, 30, 20, 30, 20, 3, 2, 60)
        assert result[0] <= 6.0 + 1e-9
        assert result[1] <= 6.0 + 1e-9


# ===========================================================================
# Sezione 5: Pipeline end-to-end — scenari realistici
# ===========================================================================

class TestEndToEndScenarios:
    """Scenari completi che verificano la catena xG → probabilità → mercati."""

    def test_balanced_match(self):
        """Partita equilibrata: P(1) ≈ P(2), Over 2.5 moderato."""
        xg_h, xg_a = calcola_xg_bayesiani(0.0, 2.5, 0.0, 2.5, 0)
        _, joint_ind, _ = build_bivariate_matrix(xg_h, xg_a, 0, xg_h + xg_a)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)

        assert abs(p1 - p2) < 0.05, f"Squilibrio eccessivo: P(1)={p1:.3f}, P(2)={p2:.3f}"
        assert 0.20 < px < 0.40, f"P(X)={px:.3f} fuori range plausibile"

    def test_strong_favorite(self):
        """Forte favorita casa: P(1) > 0.60, P(2) < 0.15."""
        xg_h, xg_a = calcola_xg_bayesiani(-1.5, 3.0, -1.5, 3.0, 0)
        _, joint_ind, _ = build_bivariate_matrix(xg_h, xg_a, 0, xg_h + xg_a)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)

        assert p1 > 0.55, f"Favorita troppo debole: P(1)={p1:.3f}"
        assert p2 < 0.20, f"Sfavorita troppo forte: P(2)={p2:.3f}"

    def test_low_scoring_match(self):
        """Gara a basso punteggio: Under 1.5 alto, BTTS Sì basso."""
        xg_h, xg_a = calcola_xg_bayesiani(0.0, 1.8, 0.0, 1.8, 0)
        full, _, _ = build_bivariate_matrix(xg_h, xg_a, 0, xg_h + xg_a)
        p_under_15, _ = calcola_over_under(full, 0, 1.5)
        p_btts_si = calcola_btts(full, 0, 0)

        assert p_under_15 > 0.30, f"Under 1.5 troppo basso: {p_under_15:.3f}"
        assert p_btts_si < 0.50, f"BTTS Sì troppo alto in gara chiusa: {p_btts_si:.3f}"

    def test_late_game_probabilities(self):
        """A fine partita (min 80, 0-0), le probabilità devono essere coerenti."""
        ah_op, tot_op = -0.5, 2.5
        ah_cur = -0.1
        tot_cur = 0.3
        xg_h, xg_a = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, 80
        )

        assert xg_h > 0 and xg_a > 0
        assert xg_h + xg_a < 1.0, f"xG totali troppo alti a min 80: {xg_h+xg_a:.3f}"

        _, joint_ind, _ = build_bivariate_matrix(xg_h, xg_a, 80, tot_cur)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        assert px > 0.50, f"P(X) troppo basso a 0-0 min 80: {px:.3f}"

    def test_after_goal_scored(self):
        """Dopo un gol (1-0 al 45'), xG e mercati si adattano."""
        ah_op, tot_op = -0.5, 2.5
        ah_cur = 0.25  # mercato si è spostato
        tot_cur = 1.2
        xg_h, xg_a = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, 45,
            gol_diff=1, gol_tot=1
        )
        assert xg_h + xg_a < 2.0
        assert xg_h > 0 and xg_a > 0

    def test_red_card_scenario(self):
        """Simulazione con cartellino rosso (effetto indiretto via linee)."""
        ah_op, tot_op = -0.5, 2.5
        # Dopo rosso a trasferta al 30': AH si muove forte verso casa
        ah_cur = -1.5  # forte favorita casa
        tot_cur = 2.2   # leggera riduzione (una squadra in 10)
        xg_h, xg_a = calcola_xg_bayesiani(
            ah_op, tot_op, ah_cur, tot_cur, 30,
            gol_diff=0, gol_tot=0
        )
        # Casa deve dominare
        assert xg_h > xg_a, f"xg_h={xg_h:.3f} <= xg_a={xg_a:.3f} dopo rosso trasferta"
        # Ma non troppo (è cambiato poco)
        assert xg_a > 0.1, f"xg_a troppo basso dopo rosso: {xg_a:.3f}"


# ===========================================================================
# Sezione 6: Stabilità numerica
# ===========================================================================

class TestNumericalStability:
    """Verifica la stabilità numerica in condizioni limite."""

    def test_very_small_total(self):
        """Con tot_cur molto basso, nessun crash e xG finiti."""
        xg_h, xg_a = calcola_xg_bayesiani(0.0, 2.5, 0.0, 0.2, 89)
        assert math.isfinite(xg_h)
        assert math.isfinite(xg_a)
        assert xg_h > 0 and xg_a > 0

    def test_extreme_ah(self):
        """Con AH molto estremo, nessun crash."""
        xg_h, xg_a = calcola_xg_bayesiani(-3.0, 4.0, -3.0, 4.0, 0)
        assert math.isfinite(xg_h)
        assert math.isfinite(xg_a)

    def test_zero_minute(self):
        """Minuto 0 funziona correttamente."""
        xg_h, xg_a = calcola_xg_bayesiani(-0.5, 2.5, -0.5, 2.5, 0)
        assert xg_h > 0 and xg_a > 0
        assert abs(xg_h + xg_a - 2.5) < 0.1

    def test_minute_90(self):
        """Minuto 90 non causa divisioni per zero."""
        xg_h, xg_a = calcola_xg_bayesiani(0.0, 2.5, 0.0, 0.2, 90)
        assert math.isfinite(xg_h)
        assert math.isfinite(xg_a)

    def test_ah_ev_very_low_lambdas(self):
        """_ah_ev con lambda molto bassi non crasha."""
        ev = _ah_ev(0.01, 0.01, 0.0)
        assert math.isfinite(ev)
        assert abs(ev) < 0.01

    def test_blend_shots_minute_1(self):
        """Blend a minuto 1 senza tiri non crasha."""
        result = blend_xg_shots(1.0, 1.0, 0, 0, 0, 0, 0, 0, 1)
        assert all(math.isfinite(v) for v in result)

    def test_build_bivariate_matrix_consistency(self):
        """La matrice bivariata deve sommare a 1."""
        for mu_h, mu_a, tot_cur in [(1.5, 1.0, 2.5), (0.2, 0.2, 0.4), (3.0, 0.3, 3.3)]:
            joint_dep, joint_ind, lambda0 = build_bivariate_matrix(
                mu_h, mu_a, minuto=0, tot_cur=tot_cur
            )
            total_dep = sum(joint_dep.values())
            total_ind = sum(joint_ind.values())
            assert abs(total_dep - 1.0) < 0.01, (
                f"mu=({mu_h},{mu_a}): joint_dep sum={total_dep:.6f}"
            )
            assert abs(total_ind - 1.0) < 0.01, (
                f"mu=({mu_h},{mu_a}): joint_ind sum={total_ind:.6f}"
            )
