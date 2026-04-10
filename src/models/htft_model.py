"""
htft_model.py — Modello predittivo HT/FT basato su transizioni storiche.

Nowgoal estrae 18 campi HTFT (9 per squadra) che descrivono la distribuzione
storica delle transizioni primo-tempo → risultato-finale:
  - HTW→FTW (vantaggio mantenuto, ~75%)
  - HTW→FTD (crollo, ~10%)
  - HTW→FTL (collasso, ~4%)
  - HTD→FTW (rimonta da pari, ~30%)
  - HTD→FTD (stallo, ~40%)
  - HTD→FTL (sconfitta dal pari, ~30%)
  - HTL→FTW (grande rimonta, ~15-20%)
  - HTL→FTD (recupero parziale, ~20%)
  - HTL→FTL (sconfitta confermata, ~55-65%)

Questi pattern sono specifici per squadra e contengono informazione
indipendente dalla distribuzione Poisson dei gol.

Uso:
  1. Prematch: calibra p1/px/p2 con i tassi di transizione medi
  2. Live (dopo HT): condiziona p1/px/p2 sullo stato HT reale
"""

from __future__ import annotations

# Peso blend prematch (conservativo: i dati HTFT sono su ~10 partite)
HTFT_PREMATCH_BLEND: float = 0.06

# Peso blend live (dopo HT: dati molto informativi)
HTFT_LIVE_BLEND: float = 0.12

# Minimo partite effettive per usare il modello
MIN_MATCHES_HTFT: int = 5


def compute_htft_adjustment(
    p1: float,
    px: float,
    p2: float,
    *,
    htft_home_htw_ftw: int = 0,
    htft_home_htw_ftd: int = 0,
    htft_home_htw_ftl: int = 0,
    htft_home_htd_ftw: int = 0,
    htft_home_htd_ftd: int = 0,
    htft_home_htd_ftl: int = 0,
    htft_home_htl_ftw: int = 0,
    htft_home_htl_ftd: int = 0,
    htft_home_htl_ftl: int = 0,
    htft_away_htw_ftw: int = 0,
    htft_away_htw_ftd: int = 0,
    htft_away_htw_ftl: int = 0,
    htft_away_htd_ftw: int = 0,
    htft_away_htd_ftd: int = 0,
    htft_away_htd_ftl: int = 0,
    htft_away_htl_ftw: int = 0,
    htft_away_htl_ftd: int = 0,
    htft_away_htl_ftl: int = 0,
    minuto: int = 0,
    ht_result: str = "",
) -> tuple[float, float, float]:
    """
    Aggiusta le probabilità 1X2 usando i pattern HT/FT storici.

    In prematch: usa la distribuzione aggregata per calibrare.
    In live (dopo HT): condiziona sullo stato HT osservato.

    Args:
        p1, px, p2: Probabilità 1X2 correnti.
        htft_*: Contatori transizioni HT→FT (da Nowgoal).
        minuto: Minuto attuale.
        ht_result: Stato HT ("W"=home winning, "D"=draw, "L"=home losing).
                   Vuoto se prematch o prima metà.

    Returns:
        (p1_adj, px_adj, p2_adj) aggiustate.
    """
    # Calcola i totali per la casa
    h_total = (htft_home_htw_ftw + htft_home_htw_ftd + htft_home_htw_ftl
               + htft_home_htd_ftw + htft_home_htd_ftd + htft_home_htd_ftl
               + htft_home_htl_ftw + htft_home_htl_ftd + htft_home_htl_ftl)

    a_total = (htft_away_htw_ftw + htft_away_htw_ftd + htft_away_htw_ftl
               + htft_away_htd_ftw + htft_away_htd_ftd + htft_away_htd_ftl
               + htft_away_htl_ftw + htft_away_htl_ftd + htft_away_htl_ftl)

    if h_total < MIN_MATCHES_HTFT and a_total < MIN_MATCHES_HTFT:
        return p1, px, p2

    if minuto >= 45 and ht_result in ("W", "D", "L"):
        return _live_ht_conditioned(
            p1, px, p2, ht_result,
            h_total, a_total,
            htft_home_htw_ftw, htft_home_htw_ftd, htft_home_htw_ftl,
            htft_home_htd_ftw, htft_home_htd_ftd, htft_home_htd_ftl,
            htft_home_htl_ftw, htft_home_htl_ftd, htft_home_htl_ftl,
            htft_away_htw_ftw, htft_away_htw_ftd, htft_away_htw_ftl,
            htft_away_htd_ftw, htft_away_htd_ftd, htft_away_htd_ftl,
            htft_away_htl_ftw, htft_away_htl_ftd, htft_away_htl_ftl,
        )

    return _prematch_aggregate(
        p1, px, p2,
        h_total, a_total,
        htft_home_htw_ftw, htft_home_htw_ftd, htft_home_htw_ftl,
        htft_home_htd_ftw, htft_home_htd_ftd, htft_home_htd_ftl,
        htft_home_htl_ftw, htft_home_htl_ftd, htft_home_htl_ftl,
        htft_away_htw_ftw, htft_away_htw_ftd, htft_away_htw_ftl,
        htft_away_htd_ftw, htft_away_htd_ftd, htft_away_htd_ftl,
        htft_away_htl_ftw, htft_away_htl_ftd, htft_away_htl_ftl,
    )


def _prematch_aggregate(
    p1: float, px: float, p2: float,
    h_total: int, a_total: int,
    h_htw_ftw: int, h_htw_ftd: int, h_htw_ftl: int,
    h_htd_ftw: int, h_htd_ftd: int, h_htd_ftl: int,
    h_htl_ftw: int, h_htl_ftd: int, h_htl_ftl: int,
    a_htw_ftw: int, a_htw_ftd: int, a_htw_ftl: int,
    a_htd_ftw: int, a_htd_ftd: int, a_htd_ftl: int,
    a_htl_ftw: int, a_htl_ftd: int, a_htl_ftl: int,
) -> tuple[float, float, float]:
    """Prematch: usa distribuzione aggregata (media casa+trasf)."""
    # Home: FT win rate from HTFT
    h_ft_win = h_htw_ftw + h_htd_ftw + h_htl_ftw
    h_ft_draw = h_htw_ftd + h_htd_ftd + h_htl_ftd
    h_ft_loss = h_htw_ftl + h_htd_ftl + h_htl_ftl

    # Away: FT win rate from HTFT (NB: away wins = home losses in context)
    a_ft_win = a_htw_ftw + a_htd_ftw + a_htl_ftw
    a_ft_draw = a_htw_ftd + a_htd_ftd + a_htl_ftd
    a_ft_loss = a_htw_ftl + a_htd_ftl + a_htl_ftl

    htft_p1 = htft_px = htft_p2 = 0.0
    count = 0

    if h_total >= MIN_MATCHES_HTFT:
        htft_p1 += h_ft_win / h_total
        htft_px += h_ft_draw / h_total
        htft_p2 += h_ft_loss / h_total
        count += 1

    if a_total >= MIN_MATCHES_HTFT:
        # Away perspective: their wins are home losses
        htft_p1 += a_ft_loss / a_total  # Away losing = home winning
        htft_px += a_ft_draw / a_total
        htft_p2 += a_ft_win / a_total   # Away winning = home losing
        count += 1

    if count == 0:
        return p1, px, p2

    htft_p1 /= count
    htft_px /= count
    htft_p2 /= count

    # Blend conservativo
    alpha = HTFT_PREMATCH_BLEND
    r1 = (1.0 - alpha) * p1 + alpha * htft_p1
    rx = (1.0 - alpha) * px + alpha * htft_px
    r2 = (1.0 - alpha) * p2 + alpha * htft_p2

    # Normalizza
    s = r1 + rx + r2
    if s > 0:
        return r1 / s, rx / s, r2 / s
    return p1, px, p2


def _live_ht_conditioned(
    p1: float, px: float, p2: float,
    ht_result: str,
    h_total: int, a_total: int,
    h_htw_ftw: int, h_htw_ftd: int, h_htw_ftl: int,
    h_htd_ftw: int, h_htd_ftd: int, h_htd_ftl: int,
    h_htl_ftw: int, h_htl_ftd: int, h_htl_ftl: int,
    a_htw_ftw: int, a_htw_ftd: int, a_htw_ftl: int,
    a_htd_ftw: int, a_htd_ftd: int, a_htd_ftl: int,
    a_htl_ftw: int, a_htl_ftd: int, a_htl_ftl: int,
) -> tuple[float, float, float]:
    """Live post-HT: condiziona sullo stato HT osservato."""
    cond_p1 = cond_px = cond_p2 = 0.0
    count = 0

    # Select the right row based on HT result
    if ht_result == "W":
        # Home is winning at HT
        if h_total >= MIN_MATCHES_HTFT:
            h_row = h_htw_ftw + h_htw_ftd + h_htw_ftl
            if h_row > 0:
                cond_p1 += h_htw_ftw / h_row
                cond_px += h_htw_ftd / h_row
                cond_p2 += h_htw_ftl / h_row
                count += 1
        if a_total >= MIN_MATCHES_HTFT:
            a_row = a_htl_ftw + a_htl_ftd + a_htl_ftl  # Away losing at HT = Home winning
            if a_row > 0:
                cond_p1 += a_htl_ftl / a_row  # Away FT loss = Home FT win
                cond_px += a_htl_ftd / a_row
                cond_p2 += a_htl_ftw / a_row  # Away FT win = Home FT loss
                count += 1

    elif ht_result == "D":
        if h_total >= MIN_MATCHES_HTFT:
            h_row = h_htd_ftw + h_htd_ftd + h_htd_ftl
            if h_row > 0:
                cond_p1 += h_htd_ftw / h_row
                cond_px += h_htd_ftd / h_row
                cond_p2 += h_htd_ftl / h_row
                count += 1
        if a_total >= MIN_MATCHES_HTFT:
            a_row = a_htd_ftw + a_htd_ftd + a_htd_ftl
            if a_row > 0:
                cond_p1 += a_htd_ftl / a_row
                cond_px += a_htd_ftd / a_row
                cond_p2 += a_htd_ftw / a_row
                count += 1

    elif ht_result == "L":
        if h_total >= MIN_MATCHES_HTFT:
            h_row = h_htl_ftw + h_htl_ftd + h_htl_ftl
            if h_row > 0:
                cond_p1 += h_htl_ftw / h_row
                cond_px += h_htl_ftd / h_row
                cond_p2 += h_htl_ftl / h_row
                count += 1
        if a_total >= MIN_MATCHES_HTFT:
            a_row = a_htw_ftw + a_htw_ftd + a_htw_ftl  # Away winning at HT = Home losing
            if a_row > 0:
                cond_p1 += a_htw_ftl / a_row
                cond_px += a_htw_ftd / a_row
                cond_p2 += a_htw_ftw / a_row
                count += 1

    if count == 0:
        return p1, px, p2

    cond_p1 /= count
    cond_px /= count
    cond_p2 /= count

    # Blend più aggressivo in live (dati condizionali molto informativi)
    alpha = HTFT_LIVE_BLEND
    r1 = (1.0 - alpha) * p1 + alpha * cond_p1
    rx = (1.0 - alpha) * px + alpha * cond_px
    r2 = (1.0 - alpha) * p2 + alpha * cond_p2

    s = r1 + rx + r2
    if s > 0:
        return r1 / s, rx / s, r2 / s
    return p1, px, p2


__all__ = ["compute_htft_adjustment"]
