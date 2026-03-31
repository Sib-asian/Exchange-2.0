"""
ai_adjustments.py — Moltiplicatori xG da dati AI (assenze + forma).

Converte i dati qualitativi trovati da Gemini (assenze squadre, forma recente)
in moltiplicatori quantitativi conservativi da applicare agli xG del modello.

Principi chiave:
  - ABSENCE_MARKET_ALPHA = 0.40: il mercato ha già prezzato ~60% dell'impatto;
    noi aggiungiamo solo il 40% residuo non catturato.
  - I moltiplicatori sono sempre clampati: non si scende mai sotto 0.82 per la
    propria squadra, non si sale oltre 1.12 per il portiere avversario.
  - La forma è centrata su 0: W=+1, D=0, L=-1 con decay temporale.
    Effetto massimo ±8% con una forma perfetta/disastrosa.

Riferimenti:
  Frick & Simmons (2008): market typically prices ~50-70% of known injuries.
  Goddard & Asimakopoulos (2004): injury news betting efficiency in UK football.
"""

from __future__ import annotations

import re

from src.config import AI_ADJ

# ---------------------------------------------------------------------------
# Role / status token sets per matching robusto
# ---------------------------------------------------------------------------

_STRIKER_TOKENS = {"striker", "forward", "attaccante", "centravanti", "st", "cf", "ss", "fw"}
_GK_TOKENS = {"goalkeeper", "portiere", "keeper", "gk"}
_MID_TOKENS = {"midfielder", "centrocampista", "mediano", "trequartista", "mezzala",
               "regista", "cm", "am", "dm", "mf", "ala"}
_DEF_TOKENS = {"defender", "difensore", "terzino", "stopper", "centrale",
               "cb", "lb", "rb", "wb", "libero"}

_CONFIRMED_TOKENS = {"confermato", "confirmed", "out", "assente", "infortunato",
                     "squalificato", "suspended", "injured", "indisponibile"}
_PROBABLE_TOKENS = {"probabile", "probable", "doubtful", "dubbio", "incerto"}


# ---------------------------------------------------------------------------
# Parser stringa assenza
# ---------------------------------------------------------------------------

def _parse_player_absence(text: str) -> tuple[str, str]:
    """
    Estrae (ruolo, status) da una stringa di assenza usando word-boundary regex.

    Esempi:
      "Salah (ST, CONFERMATO)" → ("striker", "confirmed")
      "Alisson (Portiere, Dubbio)" → ("gk", "probable")
      "Henderson" → ("mid", "probable")  ← default conservativo

    Returns:
        (role, status) con default ("mid", "probable") per ambiguità.
    """
    text_lower = text.lower()
    # Estrai token alfanumerici (parole incluse abbreviazioni come "st", "gk")
    tokens = set(re.findall(r'\b[a-z]+\b', text_lower))

    # Priorità: striker > gk > def > mid (default)
    if tokens & _STRIKER_TOKENS:
        role = "striker"
    elif tokens & _GK_TOKENS:
        role = "gk"
    elif tokens & _DEF_TOKENS:
        role = "def"
    elif tokens & _MID_TOKENS:
        role = "mid"
    else:
        role = "mid"  # default conservativo

    # Status
    if tokens & _CONFIRMED_TOKENS:
        status = "confirmed"
    elif tokens & _PROBABLE_TOKENS:
        status = "probable"
    else:
        status = "probable"  # default conservativo

    return role, status


# ---------------------------------------------------------------------------
# Assenze → moltiplicatore xG
# ---------------------------------------------------------------------------

def calcola_assenze_mult(
    assenze: list[str],
    per_avversario: bool = False,
) -> float:
    """
    Calcola il moltiplicatore xG per una squadra data la lista di assenze.

    Args:
        assenze: Lista di stringhe come ["Salah (ST, CONFERMATO)", "Alisson (GK, Dubbio)"].
        per_avversario: Se True, calcola l'effetto sull'avversario (GK assente nella
                        squadra indicata → avversario segna di più). Default False = effetto
                        sulla propria squadra (proprie assenze riducono il proprio xG).

    Returns:
        Moltiplicatore float clampato:
          - per_avversario=False: [AI_ADJ.ABSENCE_MULT_MIN, 1.0]
          - per_avversario=True: [1.0, AI_ADJ.ABSENCE_MULT_MAX_GK]
          1.0 = nessun effetto.
    """
    if not assenze:
        return 1.0

    # Coefficienti base (impatto sulla propria squadra): < 1.0 = riduzione xG
    _own_coeff: dict[tuple[str, str], float] = {
        ("striker", "confirmed"): AI_ADJ.STRIKER_CONFIRMED_MULT,
        ("striker", "probable"):  AI_ADJ.STRIKER_PROBABLE_MULT,
        ("gk",      "confirmed"): 1.0,   # GK mancante → l'effetto è sull'avversario
        ("gk",      "probable"):  1.0,
        ("mid",     "confirmed"): AI_ADJ.MID_CONFIRMED_MULT,
        ("mid",     "probable"):  AI_ADJ.MID_PROBABLE_MULT,
        ("def",     "confirmed"): AI_ADJ.DEF_CONFIRMED_MULT,
        ("def",     "probable"):  AI_ADJ.DEF_PROBABLE_MULT,
    }

    # Coefficienti effetto sull'AVVERSARIO per portiere assente: > 1.0 = aumento xG avversario
    _opp_gk_coeff: dict[tuple[str, str], float] = {
        ("gk", "confirmed"): AI_ADJ.GK_OPP_CONFIRMED_MULT,
        ("gk", "probable"):  AI_ADJ.GK_OPP_PROBABLE_MULT,
    }

    total_impact = 0.0

    for assenza in assenze:
        role, status = _parse_player_absence(assenza)

        if per_avversario:
            coeff = _opp_gk_coeff.get((role, status), 1.0)
        else:
            # fallback: usa il probable se il confirmed non è in tabella
            coeff = _own_coeff.get((role, status),
                                   _own_coeff.get((role, "probable"), 1.0))

        if coeff != 1.0:
            # Applica ABSENCE_MARKET_ALPHA: cattura solo la parte non prezzata dal mercato
            total_impact += (coeff - 1.0) * AI_ADJ.ABSENCE_MARKET_ALPHA

    raw_mult = 1.0 + total_impact

    if per_avversario:
        return max(1.0, min(AI_ADJ.ABSENCE_MULT_MAX_GK, raw_mult))
    else:
        return max(AI_ADJ.ABSENCE_MULT_MIN, min(1.0, raw_mult))


# ---------------------------------------------------------------------------
# Forma → moltiplicatore xG
# ---------------------------------------------------------------------------

def calcola_forma_mult(forma: str) -> float:
    """
    Calcola il moltiplicatore xG dalla forma recente (es. "WDLWW").

    W=+1, D=0, L=-1 con pesi decrescenti dal risultato più recente (primo).
    L'effetto è conservativo: max ±FORMA_MAX_EFFECT (default 8%).

    Args:
        forma: Stringa di 1-5 caratteri "W"/"D"/"L" (più recente = primo).
               Caratteri non riconosciuti sono ignorati.

    Returns:
        Moltiplicatore float in [1 - FORMA_MAX_EFFECT, 1 + FORMA_MAX_EFFECT].
        > 1.0 = forma positiva → più xG
        < 1.0 = forma negativa → meno xG
        1.0 = forma neutra (es. "WDLWD")
    """
    if not forma:
        return 1.0

    weights = AI_ADJ.FORMA_WEIGHTS  # (0.35, 0.25, 0.20, 0.12, 0.08)
    forma_upper = forma.upper()

    score = 0.0
    for i, char in enumerate(forma_upper[:5]):
        if i >= len(weights):
            break
        w = weights[i]
        if char == "W":
            score += w * 1.0
        elif char == "L":
            score += w * (-1.0)
        # D = 0, altri caratteri = 0

    # score ∈ [-1, +1] teoricamente (somma pesi = 1.0)
    # Mappa a moltiplicatore: score × MAX_EFFECT
    raw_effect = score * AI_ADJ.FORMA_MAX_EFFECT
    max_e = AI_ADJ.FORMA_MAX_EFFECT

    return max(1.0 - max_e, min(1.0 + max_e, 1.0 + raw_effect))
