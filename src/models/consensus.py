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
) -> dict[str, float]:
    """
    Calcola le probabilità consensus come media pesata di 3 modelli.

    Args:
        full_bp: Matrice dal modello bivariate Poisson + DC.
        full_copula: Matrice dal modello CMP + Frank copula.
        full_markov: Matrice dal Markov chain score-state.
        gol_casa, gol_trasf: Gol attuali.
        linea_ou: Linea Over/Under.
        weights: Pesi dei 3 modelli (somma = 1).

    Returns:
        Dict con probabilità consensus per tutti i mercati.
    """
    probs_bp = _probs_from_matrix(full_bp, gol_casa, gol_trasf, linea_ou)
    probs_copula = _probs_from_matrix(full_copula, gol_casa, gol_trasf, linea_ou)
    probs_markov = _probs_from_matrix(full_markov, gol_casa, gol_trasf, linea_ou)

    w_bp, w_cop, w_mk = weights
    consensus: dict[str, float] = {}

    for key in probs_bp:
        consensus[key] = (
            w_bp * probs_bp[key]
            + w_cop * probs_copula[key]
            + w_mk * probs_markov[key]
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

    ci: dict[str, tuple[float, float]] = {}
    for key in probs_bp:
        vals = [probs_bp[key], probs_copula[key], probs_markov[key]]
        lo = min(vals)
        hi = max(vals)
        ci[key] = (max(0.0, lo), min(1.0, hi))

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
    # 1. Draw shrinkage
    px_cal = px * draw_shrinkage
    redistrib = px * (1.0 - draw_shrinkage) * 0.5
    p1_cal = p1 + redistrib
    p2_cal = p2 + redistrib

    # Normalizza 1X2
    sum_1x2 = p1_cal + px_cal + p2_cal
    if sum_1x2 > 0:
        p1_cal /= sum_1x2
        px_cal /= sum_1x2
        p2_cal /= sum_1x2

    # 2. Logistic sharpening leggero per O/U e BTTS
    # Fix #1.10: Usa α dal config invece di hardcoded
    # Preserva valori esatti 0.0 / 1.0 per mercati già settled
    p_over_cal = p_over if p_over in (0.0, 1.0) else _logistic_sharpen(p_over, alpha=CONSENSUS.LOGISTIC_ALPHA_OVER)
    p_under_cal = 1.0 - p_over_cal
    p_btts_cal = p_btts if p_btts in (0.0, 1.0) else _logistic_sharpen(p_btts, alpha=CONSENSUS.LOGISTIC_ALPHA_BTTS)

    return p1_cal, px_cal, p2_cal, p_over_cal, p_under_cal, p_btts_cal


def _logistic_sharpen(p: float, alpha: float = 1.03) -> float:
    """
    Logistic sharpening: p_cal = sigmoid(α × logit(p)).

    Con α > 1, le probabilità estreme diventano più estreme.
    Con α = 1.03, una P=0.60 diventa ~0.604, una P=0.85 diventa ~0.854.
    L'effetto è molto conservativo per evitare overconfidence.
    """
    if p <= 0.001 or p >= 0.999:
        return p
    logit = math.log(p / (1.0 - p))
    cal_logit = alpha * logit
    return 1.0 / (1.0 + math.exp(-cal_logit))


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
