"""
test_poisson.py — Test per il modulo models/poisson.py.

Verifica:
  - Normalizzazione PMF di Poisson
  - Correzione Dixon-Coles (tau)
  - Correlazione dinamica rho
  - Matrice bivariata (normalizzazione, struttura)
"""


from src.models.poisson import (
    build_bivariate_matrix,
    dixon_coles_tau,
    poisson_pmf,
    rho_dinamico,
)

# ---------------------------------------------------------------------------
# PMF di Poisson
# ---------------------------------------------------------------------------

class TestPoissonPMF:
    def test_pmf_sums_to_one(self):
        for mu in [0.5, 1.0, 2.0, 3.5, 5.0, 8.0]:
            pmf = poisson_pmf(mu)
            total = sum(pmf)
            assert abs(total - 1.0) < 1e-9, f"PMF non normalizzata per mu={mu}: sum={total}"

    def test_pmf_zero_mu(self):
        pmf = poisson_pmf(0)
        assert pmf == [1.0]

    def test_pmf_negative_mu(self):
        pmf = poisson_pmf(-1.0)
        assert pmf == [1.0]

    def test_pmf_mode_is_floor_mu(self):
        """Il massimo della PMF di Poisson è floor(mu) o floor(mu)+1."""
        mu = 3.0
        pmf = poisson_pmf(mu)
        mode = pmf.index(max(pmf))
        assert mode in (int(mu), int(mu) - 1, int(mu) + 1), f"Modalità PMF inattesa: {mode}"

    def test_pmf_mean_close_to_mu(self):
        """La media della PMF deve essere vicina a mu."""
        for mu in [0.5, 1.5, 3.0, 5.0]:
            pmf = poisson_pmf(mu)
            mean = sum(k * p for k, p in enumerate(pmf))
            assert abs(mean - mu) < 0.01, f"Media PMF={mean:.4f} lontana da mu={mu}"

    def test_pmf_monotone_tail(self):
        """La PMF deve decresce nella coda (k > mu)."""
        mu = 2.0
        pmf = poisson_pmf(mu)
        k_start = int(mu) + 2
        if k_start + 2 < len(pmf):
            assert pmf[k_start] > pmf[k_start + 1] > pmf[k_start + 2]

    def test_pmf_large_mu(self):
        """PMF deve essere valida anche per mu grandi."""
        pmf = poisson_pmf(15.0)
        assert abs(sum(pmf) - 1.0) < 1e-9
        assert len(pmf) > 15


# ---------------------------------------------------------------------------
# Dixon-Coles tau
# ---------------------------------------------------------------------------

class TestDixonColes:
    def test_tau_identity_for_high_scores(self):
        """tau deve essere 1.0 per punteggi > 1-1."""
        for (i, j) in [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3)]:
            assert dixon_coles_tau(i, j, 1.0, 1.0) == 1.0, f"tau non 1.0 per ({i},{j})"

    def test_tau_00_with_negative_rho(self):
        """tau(0,0) con rho negativo deve essere > 1 (DC aumenta P(0-0))."""
        tau = dixon_coles_tau(0, 0, 1.0, 1.0, rho_dc=-0.13)
        assert tau > 1.0, f"tau(0,0) = {tau}"

    def test_tau_11_with_negative_rho(self):
        """tau(1,1) con rho negativo deve essere > 1 (DC aumenta P(1-1))."""
        tau = dixon_coles_tau(1, 1, 1.0, 1.0, rho_dc=-0.13)
        assert tau > 1.0, f"tau(1,1) = {tau}"

    def test_tau_10_with_negative_rho(self):
        """tau(1,0) con rho negativo deve essere < 1 (DC riduce P(1-0))."""
        tau = dixon_coles_tau(1, 0, 1.0, 1.0, rho_dc=-0.13)
        assert tau < 1.0, f"tau(1,0) = {tau}"

    def test_tau_01_with_negative_rho(self):
        """tau(0,1) con rho negativo deve essere < 1 (DC riduce P(0-1))."""
        tau = dixon_coles_tau(0, 1, 1.0, 1.0, rho_dc=-0.13)
        assert tau < 1.0, f"tau(0,1) = {tau}"

    def test_tau_clamp(self):
        """tau deve rimanere in [TAU_MIN, TAU_MAX]."""
        from src.config import DC
        # Valori estremi di rho_dc
        for rho in [-2.0, -1.0, 1.0, 2.0]:
            tau = dixon_coles_tau(0, 0, 1.0, 1.0, rho_dc=rho)
            assert DC.TAU_MIN <= tau <= DC.TAU_MAX, f"tau fuori range per rho={rho}: {tau}"

    def test_tau_zero_rho(self):
        """Con rho=0 tutti i tau devono essere 1.0."""
        for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            tau = dixon_coles_tau(i, j, 1.0, 1.0, rho_dc=0.0)
            assert abs(tau - 1.0) < 1e-12, f"tau non 1.0 con rho=0 per ({i},{j})"


# ---------------------------------------------------------------------------
# Rho dinamico
# ---------------------------------------------------------------------------

class TestRhoDinamico:
    def test_rho_decreases_with_time(self):
        """Rho deve diminuire man mano che la partita avanza."""
        r0 = rho_dinamico(2.5, 0, 0.0, 0)
        r45 = rho_dinamico(2.5, 45, 0.0, 0)
        r90 = rho_dinamico(2.5, 90, 0.0, 0)
        assert r0 > r45 > r90, f"Rho non decresce con il tempo: {r0:.3f} > {r45:.3f} > {r90:.3f}"

    def test_rho_decreases_with_high_total(self):
        """Rho deve diminuire con tot_cur più alto."""
        r_low = rho_dinamico(1.5, 0, 0.0, 0)
        r_high = rho_dinamico(4.0, 0, 0.0, 0)
        assert r_low > r_high

    def test_rho_decreases_with_shot_dominance(self):
        """Rho deve diminuire con dominio tiri."""
        r_eq = rho_dinamico(2.5, 30, 0.0, 0)
        r_dom = rho_dinamico(2.5, 30, 1.0, 0)
        assert r_eq > r_dom

    def test_rho_floor(self):
        """Rho non deve scendere sotto RHO_MIN."""
        from src.config import RHO
        r = rho_dinamico(10.0, 90, 1.0, 10)
        assert r >= RHO.RHO_MIN, f"Rho sotto il floor: {r}"

    def test_rho_ceiling(self):
        """Rho non deve superare BASE_MAX."""
        from src.config import RHO
        r = rho_dinamico(0.1, 0, 0.0, 0)
        assert r <= RHO.BASE_MAX + 0.001, f"Rho sopra il ceiling: {r}"

    def test_rho_decreases_with_goals(self):
        """Rho deve diminuire con più gol già segnati."""
        r0 = rho_dinamico(2.5, 30, 0.0, 0)
        r4 = rho_dinamico(2.5, 30, 0.0, 4)
        assert r0 > r4


# ---------------------------------------------------------------------------
# Matrice bivariata
# ---------------------------------------------------------------------------

class TestBivariateMatrix:
    def test_joint_ind_sums_to_one(self):
        """La matrice indipendente deve sommare a 1."""
        joint_ind, _, _ = build_bivariate_matrix(1.2, 0.8, 30, 2.0)
        total = sum(joint_ind.values())
        assert abs(total - 1.0) < 1e-9, f"joint_ind sum = {total}"

    def test_full_matrix_sums_to_one(self):
        """La matrice full deve sommare a 1."""
        _, full, _ = build_bivariate_matrix(1.2, 0.8, 30, 2.0)
        total = sum(full.values())
        assert abs(total - 1.0) < 1e-9, f"full sum = {total}"

    def test_1x2_probabilities_sum_to_one(self):
        """P(1) + P(X) + P(2) = 1."""
        from src.markets.result import calcola_1x2
        joint_ind, _, _ = build_bivariate_matrix(1.5, 0.9, 20, 2.4)
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        assert abs(p1 + px + p2 - 1.0) < 1e-9, f"1X2 sum = {p1+px+p2}"

    def test_over_under_sum_to_one(self):
        """P(Over) + P(Under) = 1."""
        from src.markets.over_under import calcola_over_under
        _, full, _ = build_bivariate_matrix(1.2, 0.8, 30, 2.5)
        p_under, p_over = calcola_over_under(full, 0, 2.5)
        assert abs(p_under + p_over - 1.0) < 1e-9, f"U+O sum = {p_under+p_over}"

    def test_btts_prob_in_range(self):
        """P(BTTS) deve essere in [0, 1]."""
        from src.markets.btts import calcola_btts
        _, full, _ = build_bivariate_matrix(1.0, 1.0, 45, 2.0)
        p = calcola_btts(full, 0, 0)
        assert 0.0 <= p <= 1.0, f"P(BTTS) fuori range: {p}"

    def test_btts_settled_both_scored(self):
        """Se entrambe le squadre hanno segnato, BTTS Sì = 1.0."""
        from src.markets.btts import calcola_btts
        _, full, _ = build_bivariate_matrix(1.0, 1.0, 45, 2.0)
        p = calcola_btts(full, 1, 1)
        assert p == 1.0

    def test_rho_returned(self):
        """Il rho restituito deve essere nel range valido."""
        from src.config import RHO
        _, _, rho = build_bivariate_matrix(1.2, 0.8, 30, 2.0)
        assert RHO.RHO_MIN <= rho <= RHO.BASE_MAX + 0.001

    def test_strong_home_team_has_high_p1(self):
        """Una squadra di casa molto forte deve avere P(1) > 0.70."""
        joint_ind, _, _ = build_bivariate_matrix(3.0, 0.3, 0, 3.3)
        from src.markets.result import calcola_1x2
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        assert p1 > 0.70, f"P(1) troppo bassa per squadra forte: {p1}"

    def test_symmetric_game_balanced_probs(self):
        """In una partita simmetrica P(1) ≈ P(2)."""
        joint_ind, _, _ = build_bivariate_matrix(1.2, 1.2, 0, 2.4)
        from src.markets.result import calcola_1x2
        p1, px, p2 = calcola_1x2(joint_ind, 0, 0)
        assert abs(p1 - p2) < 0.01, f"Asimmetria in partita simmetrica: P1={p1:.3f}, P2={p2:.3f}"


# ---------------------------------------------------------------------------
# Correct Score
# ---------------------------------------------------------------------------

class TestCorrectScore:
    def test_top_cs_probabilities_sum_leq_one(self):
        """La somma delle top-5 probabilità deve essere <= 1."""
        from src.markets.result import calcola_correct_score
        _, full, _ = build_bivariate_matrix(1.2, 0.8, 30, 2.0)
        top_cs, _ = calcola_correct_score(full, 0, 0)
        total = sum(p for _, p in top_cs)
        assert total <= 1.0 + 1e-9

    def test_gol_tot_dist_sums_to_one(self):
        """La distribuzione dei gol totali deve sommare a ~1."""
        from src.markets.result import calcola_correct_score
        _, full, _ = build_bivariate_matrix(1.2, 0.8, 30, 2.0)
        _, gol_dist = calcola_correct_score(full, 0, 0)
        total = sum(gol_dist.values())
        assert abs(total - 1.0) < 1e-9

    def test_correct_score_accounts_for_current_goals(self):
        """Il Correct Score deve includere i gol già segnati."""
        _, full, _ = build_bivariate_matrix(1.0, 1.0, 45, 2.0)
        from src.markets.result import calcola_correct_score
        # Con gol_casa=2, il punteggio finale minimo è 2-0
        top_cs, _ = calcola_correct_score(full, 2, 0)
        for (fc, ft), _ in top_cs:
            assert fc >= 2, f"Punteggio finale ({fc}-{ft}) ha meno gol della casa"
