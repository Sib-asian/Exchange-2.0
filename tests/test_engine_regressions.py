"""Regression tests for prematch consistency edge-cases."""

from src.engine import MatchState, analizza
from src.models.markov import markov_score_distribution


def _base_state(**kwargs) -> MatchState:
    payload = {
        "minuto": 0,
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
        "ah_op": 0.0,
        "tot_op": 2.5,
        "ah_cur": 0.0,
        "tot_cur": 2.5,
        "linea_ou": 2.75,
        # OCR signals disabled by default
        "ocr_imp_total": 0.0,
        "ocr_quota_1": 0.0,
        "ocr_quota_x": 0.0,
        "ocr_quota_2": 0.0,
        "ocr_quota_over": 0.0,
        "ocr_quota_under": 0.0,
        "h2h_over_pct": 60.0,
    }
    payload.update(kwargs)
    return MatchState(**payload)


def test_h2h_over_blend_applies_on_non_25_line_with_shift() -> None:
    """H2H over% (riferimento tipico 2.5) viene traslata verso linea_ou (qui 2.75) e blendata."""
    no_h2h = analizza(_base_state(h2h_over_pct=0.0))
    with_h2h = analizza(_base_state(h2h_over_pct=70.0))
    assert with_h2h.p_over > no_h2h.p_over


def test_h2h_over_blend_applies_on_25_line() -> None:
    """H2H over% still works when analyzed line is exactly 2.5."""
    no_h2h = analizza(_base_state(linea_ou=2.5, h2h_over_pct=0.0))
    with_h2h = analizza(_base_state(linea_ou=2.5, h2h_over_pct=70.0))
    assert with_h2h.p_over > no_h2h.p_over


def test_prematch_over25_ref_incorporates_selected_line_signal() -> None:
    """O/U 2.5 prematch should move with selected analyzed line signal."""
    low_line = analizza(_base_state(linea_ou=2.0, h2h_over_pct=0.0))
    high_line = analizza(_base_state(linea_ou=3.0, h2h_over_pct=0.0))
    assert low_line.p_over_25_ref > high_line.p_over_25_ref


def test_h2h_over_blend_scales_with_h2h_sample_size() -> None:
    """Stesso h2h_over_pct ma campione H2H più grande -> impatto maggiore su p_over."""
    base = analizza(_base_state(h2h_over_pct=0.0, h2h_matches_count=0))
    low_n = analizza(_base_state(h2h_over_pct=70.0, h2h_matches_count=2))
    high_n = analizza(_base_state(h2h_over_pct=70.0, h2h_matches_count=12))
    assert (high_n.p_over - base.p_over) > (low_n.p_over - base.p_over)


def test_prematch_ou_btts_coherence_guards() -> None:
    """Prematch guards keep O/U ladder and BTTS logically coherent."""
    r = analizza(
        _base_state(
            tot_op=1.75,
            tot_cur=1.75,
            linea_ou=2.5,
            ocr_quota_gg=1.40,
            ocr_quota_ng=2.80,
        )
    )
    assert r.p_over_15 >= r.p_over_25_ref - 1e-9
    assert r.p_btts <= r.p_over_15 + 1e-9


def test_prematch_tempo_mixture_moves_total_xg_with_signals() -> None:
    """Scenario-mixture should lift total xG when prematch tempo signals are high."""
    low = analizza(
        _base_state(
            tot_op=2.35,
            tot_cur=2.35,
            prev_over_pct_h=35.0,
            prev_over_pct_a=35.0,
            h2h_over_pct=35.0,
            fixture_historical_total=2.1,
        )
    )
    high = analizza(
        _base_state(
            tot_op=2.35,
            tot_cur=2.35,
            prev_over_pct_h=70.0,
            prev_over_pct_a=70.0,
            h2h_over_pct=72.0,
            fixture_historical_total=3.0,
        )
    )
    assert (high.xg_h_final + high.xg_a_final) > (low.xg_h_final + low.xg_a_final)


def test_engine_ocr_uses_extracted_ou_line_when_present(monkeypatch) -> None:
    """
    OCR O/U quotes should be interpreted on the same extracted line (ocr_imp_total)
    instead of the user-selected analysis line when available.
    """
    captured: dict[str, float] = {}

    def _fake_ocr_signal(
        quota_1: float,
        quota_x: float,
        quota_2: float,
        quota_over: float,
        quota_under: float,
        linea_ou: float,
    ) -> tuple[float, float]:
        captured["linea_ou"] = linea_ou
        return 0.0, 0.0

    monkeypatch.setattr("src.models.calibration.estrai_segnali_ocr_da_quote", _fake_ocr_signal)

    state = _base_state(
        linea_ou=2.75,       # linea analisi
        ocr_imp_total=2.25,  # linea OCR estratta (stessa fonte quote)
        ocr_quota_over=1.90,
        ocr_quota_under=1.90,
    )
    analizza(state)
    assert captured["linea_ou"] == 2.25


def test_engine_cache_key_contains_shot_dom_and_rho_dc(monkeypatch) -> None:
    """Engine should include shot_dom and rho_dc in cache keys."""
    captured_params: list[tuple] = []

    class _SpyCache:
        def get_or_compute(self, compute_fn, *params):
            captured_params.append(params)
            return compute_fn()

    monkeypatch.setattr("src.models.cache.get_matrix_cache", lambda: _SpyCache())

    state = MatchState(
        minuto=45,
        gol_casa=1,
        gol_trasf=0,
        rossi_casa=0,
        rossi_trasf=0,
        ah_op=-0.25,
        tot_op=2.5,
        ah_cur=-0.5,
        tot_cur=2.25,
        linea_ou=2.5,
        sot_h=7,
        soff_h=2,
        sot_a=1,
        soff_a=1,
        gialli_casa=3,
        gialli_trasf=2,
    )
    analizza(state)

    bivariate = next(p for p in captured_params if p and p[0] == "bivariate")
    markov = next(p for p in captured_params if p and p[0] == "markov")

    # bivariate: (..., gol_totali, shot_dom, rho_dc)
    assert isinstance(bivariate[-2], float)
    assert isinstance(bivariate[-1], float)
    assert bivariate[-2] > 0.0
    # markov: (..., gol_h, gol_a, rho_dc)
    assert isinstance(markov[-1], float)


def test_engine_new_premarket_fields_reach_calibration(monkeypatch) -> None:
    """Engine should forward new prematch signals to calcola_xg_bayesiani."""
    captured: dict[str, float] = {}

    def _fake_calcola_xg_bayesiani(
        ah_op: float,
        tot_op: float,
        ah_cur: float,
        tot_cur: float,
        minuto: int,
        gol_diff: int = 0,
        gol_tot: int = 0,
        **kwargs,
    ) -> tuple[float, float]:
        captured.update(kwargs)
        return 1.20, 1.10

    monkeypatch.setattr("src.models.calibration.calcola_xg_bayesiani", _fake_calcola_xg_bayesiani)

    state = MatchState(
        minuto=0,
        gol_casa=0,
        gol_trasf=0,
        rossi_casa=0,
        rossi_trasf=0,
        ah_op=-0.25,
        tot_op=2.5,
        ah_cur=-0.25,
        tot_cur=2.5,
        linea_ou=2.5,
        extraction_coverage=0.82,
        line_movement_ah_raw=-0.50,
        line_movement_total_raw=0.25,
        team_stats_home_shots=13.0,
        team_stats_away_shots=9.0,
        team_stats_home_corners=6.0,
        team_stats_away_corners=4.0,
        team_stats_home_possession=55.0,
        team_stats_away_possession=45.0,
    )

    analizza(state)
    assert captured["extraction_coverage"] == 0.82
    assert captured["line_movement_ah_raw"] == -0.50
    assert captured["line_movement_total_raw"] == 0.25


def test_markov_dc_negative_rho_increases_0_0() -> None:
    """
    Con rho_dc < 0, la correzione DC deve AUMENTARE P(0,0) rispetto a rho_dc=0.

    rho_dc negativo = correlazione negativa tra gol casa e trasferta →
    lo 0-0 è più probabile della Poisson indipendente.
    """
    dist_no_dc = markov_score_distribution(1.2, 1.0, 0, 0, 0, rho_dc=0.0)
    dist_neg_dc = markov_score_distribution(1.2, 1.0, 0, 0, 0, rho_dc=-0.13)

    p00_no_dc = dist_no_dc.get((0, 0), 0.0)
    p00_neg_dc = dist_neg_dc.get((0, 0), 0.0)

    assert p00_neg_dc > p00_no_dc, (
        f"rho_dc=-0.13 should increase P(0,0): "
        f"P(0,0|rho=0)={p00_no_dc:.6f} vs P(0,0|rho=-0.13)={p00_neg_dc:.6f}"
    )
