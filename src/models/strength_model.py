"""
strength_model.py — Modello di forza indipendente dal mercato.

Produce xG stimati basati SOLO su statistiche storiche della squadra
(gol segnati/subiti, prestazioni casa/trasferta, forma recente),
senza usare le linee asiatiche.

Questo rompe la dipendenza circolare: quando il modello di forza dice 2.5
e il mercato dice 2.0, il sistema ha un segnale indipendente di valore.

Riferimenti:
  Maher (1982), "Modelling association football scores"
  Dixon & Coles (1997), "Modelling Association Football Scores"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Costanti del modello di forza
# ---------------------------------------------------------------------------

# Media gol/partita top-5 leghe europee (2018-2024): ~2.68 totali, ~1.34/squadra
LEAGUE_AVG_GOALS_PER_TEAM: float = 1.34
HOME_ADVANTAGE: float = 1.12   # casa segna ~12% in più della media
AWAY_DISADVANTAGE: float = 0.90  # trasferta segna ~10% in meno

# Peso del modello di forza nel blend con xG da linee
STRENGTH_BLEND_ALPHA: float = 0.12  # conservativo: 12% forza, 88% mercato

# Peso dei dati specifici casa/trasferta vs. media generale
HOME_AWAY_WEIGHT: float = 0.40  # 40% home/away specifico, 60% media generale

# Clamp per evitare xG estremi dal modello di forza
XG_STRENGTH_MIN: float = 0.30
XG_STRENGTH_MAX: float = 3.50


# ---------------------------------------------------------------------------
# Calcolo xG indipendenti
# ---------------------------------------------------------------------------

def compute_strength_xg(
    prev_avg_scored_h: float,
    prev_avg_conceded_h: float,
    prev_avg_scored_a: float,
    prev_avg_conceded_a: float,
    home_gf_h: float = 0.0,
    home_ga_h: float = 0.0,
    away_gf_a: float = 0.0,
    away_ga_a: float = 0.0,
    last6_gf_h: float = 0.0,
    last6_ga_h: float = 0.0,
    last6_gf_a: float = 0.0,
    last6_ga_a: float = 0.0,
) -> tuple[float, float] | None:
    """
    Calcola xG indipendenti dal mercato usando il modello Maher/Dixon-Coles.

    Modello: xG_h = league_avg * att_h * def_a * home_advantage
    dove:
      att_h = gol segnati dalla casa / media lega (forza offensiva relativa)
      def_a = gol subiti dalla trasferta / media lega (debolezza difensiva relativa)

    Il modello usa 3 livelli di dati (pesati):
      1. Media generale (prev_avg_scored/conceded) — 60%
      2. Specifica casa/trasferta (home_gf/away_gf) — 40%
      3. Forma recente (last6) — usata come correttivo

    Args:
        prev_avg_scored_h: Gol medi segnati dalla casa (ultimi 10)
        prev_avg_conceded_h: Gol medi subiti dalla casa (ultimi 10)
        prev_avg_scored_a: Gol medi segnati dalla trasferta (ultimi 10)
        prev_avg_conceded_a: Gol medi subiti dalla trasferta (ultimi 10)
        home_gf_h: Gol medi segnati in casa (prestazione casalinga)
        home_ga_h: Gol medi subiti in casa
        away_gf_a: Gol medi segnati in trasferta
        away_ga_a: Gol medi subiti in trasferta
        last6_gf_h: Gol segnati nelle ultime 6 (casa)
        last6_ga_h: Gol subiti nelle ultime 6 (casa)
        last6_gf_a: Gol segnati nelle ultime 6 (trasferta)
        last6_ga_a: Gol subiti nelle ultime 6 (trasferta)

    Returns:
        (xg_h_strength, xg_a_strength) oppure None se dati insufficienti.
    """
    # Servono almeno i dati base (prev_avg) per calcolare
    if prev_avg_scored_h <= 0 or prev_avg_conceded_a <= 0:
        return None
    if prev_avg_scored_a <= 0 or prev_avg_conceded_h <= 0:
        return None

    avg = LEAGUE_AVG_GOALS_PER_TEAM

    # Rating offensivo/difensivo dalla media generale
    att_h_gen = prev_avg_scored_h / avg
    def_h_gen = prev_avg_conceded_h / avg
    att_a_gen = prev_avg_scored_a / avg
    def_a_gen = prev_avg_conceded_a / avg

    # Rating dalla prestazione specifica casa/trasferta (se disponibile)
    if home_gf_h > 0 and away_ga_a > 0:
        att_h_specific = home_gf_h / avg
        def_a_specific = away_ga_a / avg
        att_h = (1.0 - HOME_AWAY_WEIGHT) * att_h_gen + HOME_AWAY_WEIGHT * att_h_specific
        def_a = (1.0 - HOME_AWAY_WEIGHT) * def_a_gen + HOME_AWAY_WEIGHT * def_a_specific
    else:
        att_h = att_h_gen
        def_a = def_a_gen

    if away_gf_a > 0 and home_ga_h > 0:
        att_a_specific = away_gf_a / avg
        def_h_specific = home_ga_h / avg
        att_a = (1.0 - HOME_AWAY_WEIGHT) * att_a_gen + HOME_AWAY_WEIGHT * att_a_specific
        def_h = (1.0 - HOME_AWAY_WEIGHT) * def_h_gen + HOME_AWAY_WEIGHT * def_h_specific
    else:
        att_a = att_a_gen
        def_h = def_h_gen

    # Correttivo forma recente: se la forma delle ultime 6 devia significativamente
    # dalla media, aggiusta i rating. Peso leggero (20%) per evitare overfitting.
    if last6_gf_h > 0 and last6_ga_a > 0:
        _form_att_h = (last6_gf_h / 6.0) / avg
        _form_def_a = (last6_ga_a / 6.0) / avg
        att_h = 0.80 * att_h + 0.20 * _form_att_h
        def_a = 0.80 * def_a + 0.20 * _form_def_a

    if last6_gf_a > 0 and last6_ga_h > 0:
        _form_att_a = (last6_gf_a / 6.0) / avg
        _form_def_h = (last6_ga_h / 6.0) / avg
        att_a = 0.80 * att_a + 0.20 * _form_att_a
        def_h = 0.80 * def_h + 0.20 * _form_def_h

    # xG dal modello di forza (Maher 1982)
    xg_h = avg * att_h * def_a * HOME_ADVANTAGE
    xg_a = avg * att_a * def_h * AWAY_DISADVANTAGE

    xg_h = max(XG_STRENGTH_MIN, min(XG_STRENGTH_MAX, xg_h))
    xg_a = max(XG_STRENGTH_MIN, min(XG_STRENGTH_MAX, xg_a))

    return xg_h, xg_a


def blend_strength_with_market(
    xg_h_market: float,
    xg_a_market: float,
    xg_h_strength: float,
    xg_a_strength: float,
    alpha: float = STRENGTH_BLEND_ALPHA,
) -> tuple[float, float]:
    """
    Blenda xG da mercato con xG dal modello di forza.

    Formula: xG_blend = (1 - alpha) * xG_market + alpha * xG_strength

    Args:
        xg_h_market: xG casa dal mercato (Bayesian calibration).
        xg_a_market: xG trasferta dal mercato.
        xg_h_strength: xG casa dal modello di forza.
        xg_a_strength: xG trasferta dal modello di forza.
        alpha: Peso del modello di forza [0, 1].

    Returns:
        (xg_h_blended, xg_a_blended).
    """
    xg_h = (1.0 - alpha) * xg_h_market + alpha * xg_h_strength
    xg_a = (1.0 - alpha) * xg_a_market + alpha * xg_a_strength
    return xg_h, xg_a


__all__ = ["compute_strength_xg", "blend_strength_with_market"]
