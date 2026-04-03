from src.models.prematch_history_calibration import calibrate_prematch_probs


def test_calibration_no_history_keeps_probabilities():
    p1, px, p2, po, pu, pb, sig = calibrate_prematch_probs(0.45, 0.27, 0.28, 0.54, 0.46, 0.52)
    assert abs((p1 + px + p2) - 1.0) < 1e-9
    assert abs((po + pu) - 1.0) < 1e-9
    assert 0.0 <= pb <= 1.0
    assert sig.samples >= 0
