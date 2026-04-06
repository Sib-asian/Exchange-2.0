"""Test controlli euristici linee prematch."""

from src.models.line_sanity import prematch_line_quality


def test_prematch_line_quality_empty_when_live() -> None:
    assert prematch_line_quality(
        ah_op=0.0,
        ah_cur_raw=-0.5,
        tot_op=2.5,
        tot_cur_raw=2.5,
        linea_ou=2.5,
        gol_tot=1,
    ) == []


def test_warns_large_ah_move() -> None:
    msgs = prematch_line_quality(
        ah_op=0.0,
        ah_cur_raw=-2.5,
        tot_op=2.5,
        tot_cur_raw=2.5,
        linea_ou=2.5,
        gol_tot=0,
    )
    assert any("AH apertura" in m for m in msgs)


def test_warns_large_total_move() -> None:
    msgs = prematch_line_quality(
        ah_op=-0.25,
        ah_cur_raw=-0.25,
        tot_op=2.5,
        tot_cur_raw=4.25,
        linea_ou=2.5,
        gol_tot=0,
    )
    assert any("total" in m.lower() for m in msgs)


def test_no_warn_when_ou_line_equals_total() -> None:
    msgs = prematch_line_quality(
        ah_op=-0.25,
        ah_cur_raw=-0.25,
        tot_op=2.5,
        tot_cur_raw=2.5,
        linea_ou=2.5,
        gol_tot=0,
    )
    assert not any("sopra" in m for m in msgs)


def test_warns_ou_line_above_market_total() -> None:
    msgs = prematch_line_quality(
        ah_op=-0.25,
        ah_cur_raw=-0.25,
        tot_op=2.5,
        tot_cur_raw=2.5,
        linea_ou=3.0,
        gol_tot=0,
    )
    assert any("sopra" in m for m in msgs)


def test_warns_ou_line_below_market_total() -> None:
    msgs = prematch_line_quality(
        ah_op=-0.25,
        ah_cur_raw=-0.25,
        tot_op=2.75,
        tot_cur_raw=2.75,
        linea_ou=2.25,
        gol_tot=0,
    )
    assert any("sotto" in m for m in msgs)


def test_prediction_record_ou_line_default_from_dict() -> None:
    from src.tracking.prediction_log import record_from_dict

    r = record_from_dict(
        {
            "id": "x",
            "timestamp": "2026-01-01T00:00:00",
            "squadra_casa": "A",
            "squadra_trasf": "B",
            "p1": 0.4,
            "px": 0.3,
            "p2": 0.3,
        }
    )
    assert r.ou_line == 2.5
    r.gol_casa = 2
    r.gol_trasf = 1
    r.compute_derived_fields()
    assert r.over_25_hit is True
