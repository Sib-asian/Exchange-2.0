"""
confidence_calibration.py — Calibrazione del model_confidence dallo storico.

Il model_confidence attuale è calcolato con una formula euristica (radice quinta
del prodotto di 5 componenti). Non è calibrato: quando dice "0.70" non significa
che il modello è accurato il 70% delle volte.

Questo modulo costruisce una mappatura calibrata:
  confidence_raw → confidence_calibrated

usando regressione isotonica sui dati storici del prediction_log.

Approccio:
  1. Raggruppa le previsioni completate per bin di confidenza (0.1 di larghezza).
  2. Per ogni bin, calcola l'accuratezza reale (% previsioni corrette).
  3. Costruisci una mappa di calibrazione (isotonica = monotona non-decrescente).
  4. Applica la mappa alla confidenza raw per ottenere una confidenza calibrata.

Riferimenti:
  Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities"
  Zadrozny & Elkan (2002), "Transforming classifier scores into probability estimates"
"""

from __future__ import annotations

MIN_RECORDS_FOR_CONFIDENCE_CAL: int = 30
N_BINS: int = 10


def build_confidence_calibration_map() -> list[tuple[float, float]] | None:
    """
    Costruisce la mappa di calibrazione (confidence_raw → accuracy_reale).

    Usa i record completati dal prediction_log per costruire una curva
    monotona di calibrazione.

    Returns:
        Lista di (midpoint_confidence, observed_accuracy) ordinata,
        o None se non abbastanza dati.
    """
    try:
        from src.tracking.prediction_log import get_prediction_log
    except ImportError:
        return None

    log = get_prediction_log()
    completed = [
        r for r in log.get_completed()
        if r.is_prematch and r.risultato_1x2 in ("1", "X", "2")
    ]

    if len(completed) < MIN_RECORDS_FOR_CONFIDENCE_CAL:
        return None

    # Bin per confidence
    bins: dict[int, list[bool]] = {i: [] for i in range(N_BINS)}
    for r in completed:
        conf = max(0.0, min(1.0, r.model_confidence))
        bin_idx = min(N_BINS - 1, int(conf * N_BINS))

        # "Corretto" = il modello ha assegnato la probabilità più alta all'esito giusto
        probs = {"1": r.p1, "X": r.px, "2": r.p2}
        predicted = max(probs, key=lambda k: probs[k])
        correct = predicted == r.risultato_1x2
        bins[bin_idx].append(correct)

    # Costruisci la curva: (midpoint, accuracy)
    raw_curve: list[tuple[float, float]] = []
    for i in range(N_BINS):
        if bins[i]:
            midpoint = (i + 0.5) / N_BINS
            accuracy = sum(bins[i]) / len(bins[i])
            raw_curve.append((midpoint, accuracy))

    if len(raw_curve) < 3:
        return None

    # Enforce monotonicity (isotonic: accuracy deve essere non-decrescente)
    calibrated = _isotonic_regression(raw_curve)
    return calibrated


def apply_confidence_calibration(
    raw_confidence: float,
    cal_map: list[tuple[float, float]],
) -> float:
    """
    Applica la mappa di calibrazione alla confidenza raw.

    Usa interpolazione lineare tra i punti della mappa.

    Args:
        raw_confidence: Confidenza grezza dal motore.
        cal_map: Mappa (midpoint, calibrated_accuracy) dalla build.

    Returns:
        Confidenza calibrata in [0, 1].
    """
    if not cal_map:
        return raw_confidence

    conf = max(0.0, min(1.0, raw_confidence))

    # Sotto il primo punto: usa il valore del primo punto
    if conf <= cal_map[0][0]:
        return cal_map[0][1]
    # Sopra l'ultimo: usa l'ultimo
    if conf >= cal_map[-1][0]:
        return cal_map[-1][1]

    # Interpolazione lineare
    for i in range(len(cal_map) - 1):
        x0, y0 = cal_map[i]
        x1, y1 = cal_map[i + 1]
        if x0 <= conf <= x1:
            t = (conf - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)

    return raw_confidence


def _isotonic_regression(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Pool adjacent violators per enforcing monotonicity.

    Se accuracy[i] > accuracy[i+1], fonde i due bin in media pesata.
    Risultato: curva non-decrescente.
    """
    if not points:
        return []

    # Sort by x
    sorted_pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]
    weights = [1.0] * len(ys)

    # Pool adjacent violators
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            # Pool: merge i and i+1
            w_total = weights[i] + weights[i + 1]
            y_pooled = (weights[i] * ys[i] + weights[i + 1] * ys[i + 1]) / w_total
            x_pooled = (weights[i] * xs[i] + weights[i + 1] * xs[i + 1]) / w_total
            ys[i] = y_pooled
            xs[i] = x_pooled
            weights[i] = w_total
            ys.pop(i + 1)
            xs.pop(i + 1)
            weights.pop(i + 1)
            # Go back to check previous
            if i > 0:
                i -= 1
        else:
            i += 1

    return list(zip(xs, ys, strict=True))


__all__ = [
    "build_confidence_calibration_map",
    "apply_confidence_calibration",
]
