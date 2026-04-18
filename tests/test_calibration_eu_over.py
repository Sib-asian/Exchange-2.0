"""Platt separato per Over 2.5 europeo vs linea analizzata."""

from src.models.calibration_curve import apply_calibration


def test_apply_calibration_eu_channel_when_map_present():
    maps = {
        "p1": (1.0, 0.0),
        "px": (1.0, 0.0),
        "p2": (1.0, 0.0),
        "p_over": (1.0, 0.0),
        "p_btts": (1.0, 0.0),
        "p_eu_over_25": (1.05, 0.02),
    }
    p1, px, p2, po, pu, pb, eu_o, eu_u = apply_calibration(
        0.4,
        0.3,
        0.3,
        0.55,
        0.45,
        0.5,
        maps,
        strength=1.0,
        p_eu_over_25=0.60,
    )
    assert abs((p1 + px + p2) - 1.0) < 1e-6
    assert eu_o is not None and eu_u is not None
    assert abs(eu_o + eu_u - 1.0) < 1e-6


def test_apply_calibration_eu_none_without_map():
    maps = {"p1": (1.0, 0.0), "px": (1.0, 0.0), "p2": (1.0, 0.0), "p_over": (1.0, 0.0), "p_btts": (1.0, 0.0)}
    *_, eu_o, eu_u = apply_calibration(
        0.4, 0.3, 0.3, 0.55, 0.45, 0.5, maps, p_eu_over_25=0.60,
    )
    assert eu_o is None and eu_u is None
