"""Regression tests for prematch consistency edge-cases."""

from src.engine import MatchState, analizza


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


def test_h2h_over_blend_disabled_for_non_25_line() -> None:
    """H2H over% should not shift O/U when analyzed line is not 2.5."""
    no_h2h = analizza(_base_state(h2h_over_pct=0.0))
    with_h2h = analizza(_base_state(h2h_over_pct=70.0))
    assert abs(no_h2h.p_over - with_h2h.p_over) < 1e-9


def test_h2h_over_blend_applies_on_25_line() -> None:
    """H2H over% still works when analyzed line is exactly 2.5."""
    no_h2h = analizza(_base_state(linea_ou=2.5, h2h_over_pct=0.0))
    with_h2h = analizza(_base_state(linea_ou=2.5, h2h_over_pct=70.0))
    assert with_h2h.p_over > no_h2h.p_over


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
