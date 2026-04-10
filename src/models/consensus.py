"""
consensus.py — Multi-model consensus e calibrazione isotonica.

Implementa:
  1. Consenso multi-modello: media pesata di 3 modelli diversi
     - Bivariate Poisson + Dixon-Coles (correlazione via Z + rho_DC)
     - CMP + Frank copula (overdispersion + copula archimedea)
     - Markov chain (correlazione via score-dependent rates)

  2. Calibrazione isotonica leggera: correzione di bias noti della Poisson
     - Draw overestimation (~2-3%)
     - Logistic sharpening per probabilità estreme

  3. Intervalli di credibilità: basati sullo spread tra modelli
     - Se i modelli concordano → CI stretto → alta fiducia
     - Se i modelli divergono → CI largo → bassa fiducia

Riferimenti:
  Karlis & Ntzoufras (2003), "Analysis of sports data by using bivariate Poisson"
  Genest & Neslehova (2007), "A primer on copulas for count data"
"""

from __future__ import annotations

import math

from src.config import CONSENSUS

# ---------------------------------------------------------------------------
# Probabilità dai mercati (riusa la stessa logica del modello principale)
# ---------------------------------------------------------------------------

def _probs_from_matrix(
    full: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
) -> dict[str, float]:
    """
    Calcola tutte le probabilità dai mercati da una matrice bivariata generica.

    Riproduce la logica di calcola_1x2, calcola_over_under, calcola_btts
    su una matrice arbitraria (non necessariamente dalla bivariate Poisson).
    """
    p1 = px = p2 = 0.0
    p_btts = 0.0
    gol_tot_probs: dict[int, float] = {}

    for (a, b), p in full.items():
        # 1X2 (sul risultato finale)
        diff = (gol_casa + a) - (gol_trasf + b)
        if diff > 0:
            p1 += p
        elif diff < 0:
            p2 += p
        else:
            px += p

        # BTTS
        final_h = gol_casa + a
        final_a = gol_trasf + b
        if final_h > 0 and final_a > 0:
            p_btts += p

        # Gol totali
        tot = gol_casa + gol_trasf + a + b
        gol_tot_probs[tot] = gol_tot_probs.get(tot, 0.0) + p

    # Normalizza 1X2
    sum_1x2 = p1 + px + p2
    if sum_1x2 > 0:
        p1 /= sum_1x2
        px /= sum_1x2
        p2 /= sum_1x2

    # Over/Under
    gol_attuali = gol_casa + gol_trasf
    line4 = round(linea_ou * 4)
    if line4 % 2 != 0:
        # Quarter line
        h_low = (line4 - 1) / 4.0
        h_high = (line4 + 1) / 4.0
        p_u_low = _p_under_from_matrix(full, gol_attuali, h_low)
        p_u_high = _p_under_from_matrix(full, gol_attuali, h_high)
        p_under = 0.5 * (p_u_low + p_u_high)
    else:
        p_under = _p_under_from_matrix(full, gol_attuali, linea_ou)

    p_under = min(max(p_under, 0.0), 1.0)
    p_over = 1.0 - p_under

    return {
        "p1": p1, "px": px, "p2": p2,
        "p_over": p_over, "p_under": p_under,
        "p_btts": min(1.0, max(0.0, p_btts)),
    }


def _p_under_from_matrix(
    full: dict[tuple[int, int], float],
    gol_attuali: int,
    line: float,
) -> float:
    """P(Under) per una singola linea da una matrice generica."""
    line4 = round(line * 4)
    if line4 % 4 == 0:
        int_line = int(line)
        p_win = sum(p for (a, b), p in full.items() if gol_attuali + a + b < int_line)
        p_push = sum(p for (a, b), p in full.items() if gol_attuali + a + b == int_line)
        return p_win + 0.5 * p_push
    else:
        return sum(p for (a, b), p in full.items() if gol_attuali + a + b < line)


# ---------------------------------------------------------------------------
# Consensus multi-modello
# ---------------------------------------------------------------------------

def compute_consensus(
    full_bp: dict[tuple[int, int], float],
    full_copula: dict[tuple[int, int], float],
    full_markov: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
    weights_ou: tuple[float, float, float] | None = None,
    weights_btts: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """
    Calcola le probabilità consensus come media pesata di 3 modelli.

    Args:
        full_bp: Matrice dal modello bivariate Poisson + DC.
        full_copula: Matrice dal modello CMP + Frank copula.
        full_markov: Matrice dal Markov chain score-state.
        gol_casa, gol_trasf: Gol attuali.
        linea_ou: Linea Over/Under.
        weights: Pesi dei 3 modelli per 1X2 (somma = 1).
        weights_ou: Pesi per Over/Under; se None → stessi di `weights`.
        weights_btts: Pesi per BTTS; se None → stessi di `weights`.

    Returns:
        Dict con probabilità consensus per tutti i mercati.
    """
    probs_bp = _probs_from_matrix(full_bp, gol_casa, gol_trasf, linea_ou)
    probs_copula = _probs_from_matrix(full_copula, gol_casa, gol_trasf, linea_ou)
    probs_markov = _probs_from_matrix(full_markov, gol_casa, gol_trasf, linea_ou)

    w_bp, w_cop, w_mk = weights
    w_ou = weights_ou if weights_ou is not None else weights
    w_bt = weights_btts if weights_btts is not None else weights
    consensus: dict[str, float] = {}

    for key in probs_bp:
        if key in ("p1", "px", "p2"):
            wb, wc, wm = w_bp, w_cop, w_mk
        elif key in ("p_over", "p_under"):
            wb, wc, wm = w_ou
        elif key == "p_btts":
            wb, wc, wm = w_bt
        else:
            wb, wc, wm = w_bp, w_cop, w_mk
        consensus[key] = (
            wb * probs_bp[key]
            + wc * probs_copula[key]
            + wm * probs_markov[key]
        )

    # Normalizza 1X2
    sum_1x2 = consensus["p1"] + consensus["px"] + consensus["p2"]
    if sum_1x2 > 0:
        consensus["p1"] /= sum_1x2
        consensus["px"] /= sum_1x2
        consensus["p2"] /= sum_1x2

    # Normalizza O/U
    sum_ou = consensus["p_over"] + consensus["p_under"]
    if sum_ou > 0:
        consensus["p_over"] /= sum_ou
        consensus["p_under"] /= sum_ou

    # Clamp BTTS (snap to exact 1.0/0.0 for settled markets)
    # Fix #6.2: Usa BTTS_CLAMP_EPSILON dal config invece di hardcoded 1e-12
    btts_raw = consensus["p_btts"]
    if btts_raw > 1.0 - CONSENSUS.BTTS_CLAMP_EPSILON:
        consensus["p_btts"] = 1.0
    elif btts_raw < CONSENSUS.BTTS_CLAMP_EPSILON:
        consensus["p_btts"] = 0.0

    return consensus


def per_model_market_probs(
    full_bp: dict[tuple[int, int], float],
    full_copula: dict[tuple[int, int], float],
    full_markov: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
) -> dict[str, dict[str, float]]:
    """
    Probabilità di mercato (1X2, O/U, BTTS) per ciascun modello prima del consensus.
    Usate per logging e per aggiornare i pesi ensemble da storico.
    """
    return {
        "bp": _probs_from_matrix(full_bp, gol_casa, gol_trasf, linea_ou),
        "copula": _probs_from_matrix(full_copula, gol_casa, gol_trasf, linea_ou),
        "markov": _probs_from_matrix(full_markov, gol_casa, gol_trasf, linea_ou),
    }


def agreement_1x2_from_per_raw(per_raw: dict[str, dict[str, float]]) -> float:
    """
    Accordo [0,1] tra i tre modelli sul 1X2, dalla media degli spread (max-min) su p1/px/p2.
    Usato prima della calibrazione isotonica per modulare il draw shrinkage.
    """
    spreads: list[float] = []
    for key in ("p1", "px", "p2"):
        v = [
            per_raw["bp"][key],
            per_raw["copula"][key],
            per_raw["markov"][key],
        ]
        spreads.append(max(v) - min(v))
    mean_sp = sum(spreads) / 3.0
    scale = CONSENSUS.AGREEMENT_1X2_SPREAD_SCALE
    return max(0.0, min(1.0, 1.0 - mean_sp * scale))


# ---------------------------------------------------------------------------
# Intervalli di credibilità (basati sullo spread tra modelli)
# ---------------------------------------------------------------------------

def compute_model_credible_intervals(
    full_bp: dict[tuple[int, int], float],
    full_copula: dict[tuple[int, int], float],
    full_markov: dict[tuple[int, int], float],
    gol_casa: int,
    gol_trasf: int,
    linea_ou: float,
) -> dict[str, tuple[float, float]]:
    """
    Calcola intervalli di credibilità basati sullo spread tra modelli.

    Lo spread naturale tra modelli diversi è un indicatore di incertezza:
    - Modelli concordi → CI stretto → alta fiducia
    - Modelli discordi → CI largo → bassa fiducia

    Returns:
        Dict {mercato: (ci_low, ci_high)} per ogni mercato.
    """
    probs_bp = _probs_from_matrix(full_bp, gol_casa, gol_trasf, linea_ou)
    probs_copula = _probs_from_matrix(full_copula, gol_casa, gol_trasf, linea_ou)
    probs_markov = _probs_from_matrix(full_markov, gol_casa, gol_trasf, linea_ou)

    # Credible interval via deviazione standard pesata dei 3 modelli.
    # Min/max sovrastima il CI perché un outlier sposta sempre un estremo.
    # Approccio migliorato: CI = consensus ± k*σ_pesata
    #   - k=1.5: copre ~87% di una distribuzione normale → conservativo ma non esagerato
    #   - I 3 modelli non sono indipendenti (stessi input) → non usare k=2 (troppo largo)
    # I pesi riflettono l'affidabilità relativa dei modelli.
    _W = [CONSENSUS.W_BP_MID, CONSENSUS.W_COP_MID, CONSENSUS.W_MK_MID]
    _K = 1.5   # fattore di copertura (0.87 prob)

    ci: dict[str, tuple[float, float]] = {}
    for key in probs_bp:
        vals = [probs_bp[key], probs_copula[key], probs_markov[key]]
        # Media pesata (≈ consensus)
        mu = sum(_W[i] * vals[i] for i in range(3))
        # Varianza pesata
        var = sum(_W[i] * (vals[i] - mu) ** 2 for i in range(3))
        sigma = math.sqrt(var)
        half = _K * sigma
        ci[key] = (max(0.0, mu - half), min(1.0, mu + half))

    return ci


# ---------------------------------------------------------------------------
# Calibrazione isotonica leggera
# ---------------------------------------------------------------------------

def calibrate_probabilities(
    p1: float,
    px: float,
    p2: float,
    p_over: float,
    p_under: float,
    p_btts: float,
    draw_shrinkage: float = 0.97,
) -> tuple[float, float, float, float, float, float]:
    """
    Calibrazione leggera basata su bias noti del modello Poisson.

    1. Draw shrinkage: la Poisson sovrastima P(X) del ~2-3% perché non modella
       il comportamento attivo delle squadre per evitare il pareggio (pressing
       tardivo, sostituzioni offensive). Riduzione del 3%.

    2. Logistic sharpening: le probabilità estreme (>0.85 o <0.15) del modello
       sono tipicamente sottostimate nella loro certezza. L'isotonic regression
       su dati reali mostra che un modello Poisson con P>0.85 ha calibrazione
       vera ~P+1-2%. Applichiamo un leggero sharpening logistico.

    Args:
        p1, px, p2: Probabilità 1X2.
        p_over, p_under: Probabilità Over/Under.
        p_btts: Probabilità BTTS Sì.
        draw_shrinkage: Fattore di riduzione del pareggio (0.97 = -3%).

    Returns:
        Tuple (p1, px, p2, p_over, p_under, p_btts) calibrate.
    """
    # 1. Draw shrinkage — redistribuzione proporzionale.
    # Il surplus sottratto al pareggio viene redistribuito proporzionalmente
    # a casa e trasferta (non 50/50). Se casa è favorita 60/40, il surplus
    # va 60% alla casa e 40% alla trasferta.
    px_cal = px * draw_shrinkage
    surplus = px * (1.0 - draw_shrinkage)
    p1p2 = p1 + p2
    if p1p2 > 1e-9:
        p1_cal = p1 + surplus * (p1 / p1p2)
        p2_cal = p2 + surplus * (p2 / p1p2)
    else:
        p1_cal = p1 + surplus * 0.5
        p2_cal = p2 + surplus * 0.5

    # Normalizza 1X2
    sum_1x2 = p1_cal + px_cal + p2_cal
    if sum_1x2 > 0:
        p1_cal /= sum_1x2
        px_cal /= sum_1x2
        p2_cal /= sum_1x2

    # 2. Logistic sharpening leggero per O/U e BTTS
    # Fix #1.10: Usa α dal config invece di hardcoded
    # Preserva valori esatti 0.0 / 1.0 per mercati già settled
    p_over_cal = p_over if p_over in (0.0, 1.0) else logistic_sharpen_over(p_over)
    p_under_cal = 1.0 - p_over_cal
    p_btts_cal = p_btts if p_btts in (0.0, 1.0) else logistic_sharpen_btts(p_btts)

    return p1_cal, px_cal, p2_cal, p_over_cal, p_under_cal, p_btts_cal


def _logistic_sharpen(p: float, alpha: float = 1.03) -> float:
    """
    Logistic sharpening: p_cal = sigmoid(α × logit(p)).

    Con α > 1, le probabilità estreme diventano più estreme.
    Con α = 1.03, una P=0.60 diventa ~0.604, una P=0.85 diventa ~0.854.
    L'effetto è molto conservativo per evitare overconfidence.
    """
    p = max(1e-9, min(1.0 - 1e-9, p))
    if p <= 0.001 or p >= 0.999:
        return p
    logit = math.log(p / (1.0 - p))
    cal_logit = alpha * logit
    return 1.0 / (1.0 + math.exp(-cal_logit))


def _logistic_sharpen_tails(
    p: float,
    *,
    alpha_mid: float,
    alpha_high: float,
    alpha_low: float,
    p_threshold_high: float,
    p_threshold_low: float,
) -> float:
    """α più alto vicino a 1, più basso vicino a 0 (calibrazione empirica code)."""
    if p in (0.0, 1.0):
        return p
    if p >= p_threshold_high:
        w = min(1.0, (p - p_threshold_high) / max(1e-9, 1.0 - p_threshold_high))
        alpha = alpha_mid + w * (alpha_high - alpha_mid)
    elif p <= p_threshold_low:
        w = min(1.0, (p_threshold_low - p) / max(1e-9, p_threshold_low))
        alpha = alpha_mid + w * (alpha_low - alpha_mid)
    else:
        alpha = alpha_mid
    return _logistic_sharpen(p, alpha=alpha)


def logistic_sharpen_over(p: float) -> float:
    """Sharpening O/U: code alta più aggressive, bassa più soft."""
    return _logistic_sharpen_tails(
        p,
        alpha_mid=CONSENSUS.LOGISTIC_ALPHA_OVER,
        alpha_high=CONSENSUS.LOGISTIC_ALPHA_OVER_HIGH,
        alpha_low=CONSENSUS.LOGISTIC_ALPHA_OVER_LOW,
        p_threshold_high=CONSENSUS.LOGISTIC_EXTREME_HIGH,
        p_threshold_low=CONSENSUS.LOGISTIC_EXTREME_LOW,
    )


def logistic_sharpen_btts(p: float) -> float:
    """Sharpening BTTS con stessa logica asimmetrica."""
    return _logistic_sharpen_tails(
        p,
        alpha_mid=CONSENSUS.LOGISTIC_ALPHA_BTTS,
        alpha_high=CONSENSUS.LOGISTIC_ALPHA_BTTS_HIGH,
        alpha_low=CONSENSUS.LOGISTIC_ALPHA_BTTS_LOW,
        p_threshold_high=CONSENSUS.LOGISTIC_EXTREME_HIGH,
        p_threshold_low=CONSENSUS.LOGISTIC_EXTREME_LOW,
    )


# ---------------------------------------------------------------------------
# Divergenza modello-mercato (proxy Brier score)
# ---------------------------------------------------------------------------

def compute_model_market_divergence(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
) -> float:
    """
    Calcola la divergenza media tra modello e mercato (proxy Brier score).

    Un valore basso (~0.01) indica buon allineamento.
    Un valore alto (~0.10+) indica potenziale miscalibrazione.

    Args:
        model_probs: {mercato: prob_modello}
        market_probs: {mercato: prob_implicita} (0 se non disponibile)

    Returns:
        Divergenza media quadratica in [0, 1].
    """
    diffs = []
    for key in model_probs:
        p_market = market_probs.get(key, 0.0)
        if p_market > 0:
            diffs.append((model_probs[key] - p_market) ** 2)

    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)
