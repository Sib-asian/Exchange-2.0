"""
Scanner Partite Autonomo — pagina separata, non tocca l'analisi principale.

Flusso:
  1. Incolla N link Nowgoal H2H (uno per riga)
  2. Il software estrae i dati da ogni pagina in parallelo
  3. Costruisce un modello di forza indipendente (senza AH/Total)
  4. Mostra la classifica partite per segnale di confidenza
"""

from __future__ import annotations

import concurrent.futures
import math
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Scanner Partite",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Modello di forza indipendente
# ---------------------------------------------------------------------------

_HOME_ADV = 1.12          # vantaggio casalingo di default (se no strength data)
_LEAGUE_AVG_FALLBACK = 1.30  # media gol se non ci sono dati
_FETCH_TIMEOUT = 30       # secondi massimi per URL


def _calcola_forza(pa: Any) -> dict | None:
    """
    Stima le probabilità di una partita usando solo i dati H2H/standings.
    Nessuna linea di mercato richiesta.

    Restituisce dict con p1, px, p2, p_over25, p_under25, p_btts, xg_h, xg_a,
    confidence — oppure None se i dati sono insufficienti.
    """
    # ── Filtro dati minimi ────────────────────────────────────────────────────
    hm_tot = pa.home_matches or 0
    am_tot = pa.away_matches or 0
    if hm_tot < 2 and am_tot < 2:
        return None  # Troppo pochi dati per entrambe le squadre

    # ── Medie gol per partita (attack/defense) ────────────────────────────────
    # Priorità: stats home-specifiche > prev_avg > totali / partite
    h_games_home = max(1, (pa.home_home_win or 0) + (pa.home_home_draw or 0) + (pa.home_home_lose or 0))
    a_games_away = max(1, (pa.away_away_win or 0) + (pa.away_away_draw or 0) + (pa.away_away_lose or 0))

    h_attack  = (pa.home_home_scored or 0) / h_games_home if (pa.home_home_scored or 0) > 0 else 0.0
    h_defense = (pa.home_home_conceded or 0) / h_games_home if (pa.home_home_conceded or 0) > 0 else 0.0
    a_attack  = (pa.away_away_scored or 0) / a_games_away if (pa.away_away_scored or 0) > 0 else 0.0
    a_defense = (pa.away_away_conceded or 0) / a_games_away if (pa.away_away_conceded or 0) > 0 else 0.0

    # Fallback: Previous Scores averages
    if h_attack == 0 and (pa.home_prev_avg_scored or 0) > 0:
        h_attack  = pa.home_prev_avg_scored
        h_defense = pa.home_prev_avg_conceded or 0.0
    if a_attack == 0 and (pa.away_prev_avg_scored or 0) > 0:
        a_attack  = pa.away_prev_avg_scored
        a_defense = pa.away_prev_avg_conceded or 0.0

    # Fallback totale / partite giocate
    if h_attack == 0:
        hm = max(1, hm_tot)
        h_attack  = (pa.home_scored or 0) / hm
        h_defense = (pa.home_conceded or 0) / hm
    if a_attack == 0:
        am = max(1, am_tot)
        a_attack  = (pa.away_scored or 0) / am
        a_defense = (pa.away_conceded or 0) / am

    # Ultimo fallback: media campionato generica
    if h_attack == 0:
        h_attack = h_defense = a_attack = a_defense = _LEAGUE_AVG_FALLBACK

    # ── Dixon-Coles semplificato ──────────────────────────────────────────────
    league_avg = max(0.6, (h_attack + h_defense + a_attack + a_defense) / 4)

    # Cap valori al 2× league_avg per evitare outlier irrealistici
    cap = league_avg * 2.0
    h_attack  = min(h_attack,  cap)
    h_defense = min(h_defense, cap)
    a_attack  = min(a_attack,  cap)
    a_defense = min(a_defense, cap)

    # Home advantage dinamico da strength rating (range 1.05–1.20)
    sh = pa.strength_home or 0
    sa = pa.strength_away or 0
    if sh > 0 and sa > 0:
        avg_s = (sh + sa) / 2.0
        home_adv = 1.08 + 0.04 * (avg_s / 75.0)
        home_adv = max(1.05, min(1.20, home_adv))
    else:
        home_adv = _HOME_ADV

    xg_h = (h_attack / league_avg) * (a_defense / league_avg) * league_avg * home_adv
    xg_a = (a_attack / league_avg) * (h_defense / league_avg) * league_avg

    # Correzione forma — None check esplicito (evita 0.0 → 1.0)
    fm_h = pa.forma_mult_h if pa.forma_mult_h is not None else 1.0
    fm_a = pa.forma_mult_a if pa.forma_mult_a is not None else 1.0
    xg_h *= max(0.7, min(1.4, fm_h))
    xg_a *= max(0.7, min(1.4, fm_a))

    # Correzione Nowgoal strength rating — aggiusta anche quando solo un team ha il dato
    if sh > 0 and sa > 0:
        ratio = sh / sa
        xg_h *= ratio ** 0.15
        xg_a *= (1 / ratio) ** 0.15
    elif sh > 0:
        xg_h *= (sh / 75.0) ** 0.10  # solo casa ha rating
    elif sa > 0:
        xg_a *= (sa / 75.0) ** 0.10  # solo trasferrata ha rating

    # Blend con H2H avg goals se disponibile
    if (pa.fixture_historical_total or 0) > 0.5:
        tot_model = xg_h + xg_a
        tot_blend = 0.75 * tot_model + 0.25 * pa.fixture_historical_total
        if tot_model > 0:
            scale = tot_blend / tot_model
            xg_h *= scale
            xg_a *= scale

    xg_h = max(0.25, min(3.5, xg_h))
    xg_a = max(0.25, min(3.0, xg_a))

    # ── Poisson → 1X2, O/U, BTTS ─────────────────────────────────────────────
    def _pp(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    p1 = px = p2 = p_btts = 0.0
    gol_dist: dict[int, float] = {}
    for h in range(9):
        for a in range(9):
            p = _pp(h, xg_h) * _pp(a, xg_a)
            if h > a:
                p1 += p
            elif h == a:
                px += p
            else:
                p2 += p
            if h > 0 and a > 0:
                p_btts += p
            gol_dist[h + a] = gol_dist.get(h + a, 0) + p

    p_over25 = sum(v for k, v in gol_dist.items() if k >= 3)
    p_under25 = 1.0 - p_over25

    # ── Blend H2H 1X2 (20%) ──────────────────────────────────────────────────
    tot_h2h = (pa.h2h_home_win_pct or 0) + (pa.h2h_draw_pct or 0) + (pa.h2h_away_win_pct or 0)
    if tot_h2h > 0.5:
        alpha = 0.20
        h2h_p1 = pa.h2h_home_win_pct / tot_h2h
        h2h_px = pa.h2h_draw_pct / tot_h2h
        h2h_p2 = pa.h2h_away_win_pct / tot_h2h
        p1 = (1 - alpha) * p1 + alpha * h2h_p1
        px = (1 - alpha) * px + alpha * h2h_px
        p2 = (1 - alpha) * p2 + alpha * h2h_p2
        tot = p1 + px + p2
        if tot > 0:
            p1 /= tot
            px /= tot
            p2 /= tot

    # ── Blend quote iniziali mercato (35%) se disponibili ────────────────────
    m1 = pa.mkt_init_1 or 0.0
    mx = pa.mkt_init_x or 0.0
    m2 = pa.mkt_init_2 or 0.0
    if m1 > 1.0 and mx > 1.0 and m2 > 1.0:
        inv1 = 1 / m1
        invx = 1 / mx
        inv2 = 1 / m2
        tot_mkt = inv1 + invx + inv2
        mkt_p1 = inv1 / tot_mkt
        mkt_px = invx / tot_mkt
        mkt_p2 = inv2 / tot_mkt
        alpha_mkt = 0.35
        p1 = (1 - alpha_mkt) * p1 + alpha_mkt * mkt_p1
        px = (1 - alpha_mkt) * px + alpha_mkt * mkt_px
        p2 = (1 - alpha_mkt) * p2 + alpha_mkt * mkt_p2
        tot = p1 + px + p2
        if tot > 0:
            p1 /= tot
            px /= tot
            p2 /= tot

    # ── Blend O/U: H2H over% (25%) + prev over% casa/trasf. (10% cadauno) ────
    h2h_over    = pa.h2h_over_pct or 0.0
    prev_h_over = getattr(pa, "prev_home_over_pct", None) or 0.0
    prev_a_over = getattr(pa, "prev_away_over_pct", None) or 0.0

    if h2h_over > 0.5:
        p_over25 = 0.75 * p_over25 + 0.25 * (h2h_over / 100.0)
        p_under25 = 1.0 - p_over25

    for prev_pct, w in [(prev_h_over, 0.10), (prev_a_over, 0.10)]:
        if prev_pct > 0.5:
            p_over25 = (1 - w) * p_over25 + w * (prev_pct / 100.0)
            p_under25 = 1.0 - p_over25

    # ── Confidenza ───────────────────────────────────────────────────────────
    pts = 0
    if hm_tot >= 10:
        pts += 2
    elif hm_tot >= 5:
        pts += 1
    if am_tot >= 10:
        pts += 2
    elif am_tot >= 5:
        pts += 1
    if tot_h2h > 0.5:
        pts += 1
    if (pa.fixture_historical_total or 0) > 0.5:
        pts += 1
    if fm_h != 1.0 or fm_a != 1.0:
        pts += 1
    if m1 > 1.0:
        pts += 2  # quote iniziali mercato = forte segnale
    confidence = min(1.0, pts / 9)

    return {
        "p1":        round(p1, 4),
        "px":        round(px, 4),
        "p2":        round(p2, 4),
        "p_over25":  round(p_over25, 4),
        "p_under25": round(p_under25, 4),
        "p_btts":    round(p_btts, 4),
        "xg_h":      round(xg_h, 2),
        "xg_a":      round(xg_a, 2),
        "confidence": round(confidence, 2),
    }


def _calcola_ht_da_xg(xg_h: float, xg_a: float) -> tuple[float, float, float, float]:
    """
    Stima rapida 1° tempo usando scaling 46% da xG FT (proporzione tipica).
    Restituisce (p_ht1, p_htx, p_ht2, p_ht_over05).
    """
    lam_h = xg_h * 0.46
    lam_a = xg_a * 0.46

    def _pp(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    p_ht1 = p_htx = p_ht2 = 0.0
    for h in range(6):
        for a in range(6):
            p = _pp(h, lam_h) * _pp(a, lam_a)
            if h > a:
                p_ht1 += p
            elif h == a:
                p_htx += p
            else:
                p_ht2 += p
    p_ht_over05 = 1.0 - _pp(0, lam_h) * _pp(0, lam_a)
    return p_ht1, p_htx, p_ht2, p_ht_over05


def _top_segnali(probs: dict, confidence: float) -> list[tuple[str, float, int]]:
    """
    Restituisce i top-3 segnali ordinati per distanza da 50%.
    Ogni elemento: (mercato, prob, stelle).
    """
    gg_no = 1.0 - probs["p_btts"]
    candidati = [
        ("1 Casa",     probs["p1"]),
        ("X Pareggio", probs["px"]),
        ("2 Trasf.",   probs["p2"]),
        ("Over 2.5",   probs["p_over25"]),
        ("Under 2.5",  probs["p_under25"]),
        ("GG Sì",      probs["p_btts"]),
        ("GG No",      gg_no),
    ]
    candidati.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)

    result = []
    for mercato, prob in candidati[:3]:
        forza = prob * confidence
        if forza >= 0.55:
            stelle = 3
        elif forza >= 0.45:
            stelle = 2
        elif forza >= 0.35:
            stelle = 1
        else:
            stelle = 0
        result.append((mercato, prob, stelle))
    return result


# ---------------------------------------------------------------------------
# Fetch + analisi di una singola partita
# ---------------------------------------------------------------------------

def _analizza_url(url: str) -> dict | None:
    """Fetcha e analizza un singolo URL Nowgoal. Ritorna dict con error o risultati."""
    try:
        from src.ocr import extract_prematch_analysis_from_url
        pa = extract_prematch_analysis_from_url(url)
        if not pa or not pa.extraction_success:
            err = getattr(pa, "error_message", "Estrazione fallita") if pa else "Nessun risultato"
            return {"url": url, "error": err}

        probs = _calcola_forza(pa)
        if probs is None:
            return None

        segnali = _top_segnali(probs, probs["confidence"])
        mercato, prob, stelle = segnali[0]

        home   = pa.home_team   or "Casa"
        away   = pa.away_team   or "Trasf."
        league = pa.league_name or ""
        date   = pa.match_date  or ""

        return {
            "url":     url,
            "home":    home,
            "away":    away,
            "league":  league,
            "date":    date,
            "probs":   probs,
            "segnali": segnali,   # top-3
            "mercato": mercato,   # top segnale
            "prob":    prob,
            "stelle":  stelle,
            "pa":      pa,
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🔍 Scanner Partite")
st.caption(
    "Analisi autonoma senza linee di mercato — incolla i link Nowgoal H2H, "
    "il software calcola le probabilità e ordina per segnale."
)

st.divider()

urls_raw = st.text_area(
    "Link Nowgoal H2H (uno per riga)",
    placeholder=(
        "https://www.nowgoal.com/match/h2h-2965240\n"
        "https://www.nowgoal.com/match/h2h-2964843\n"
        "..."
    ),
    height=160,
    help="Vai su Nowgoal → partita → tab H2H → copia URL dalla barra del browser",
)

col_btn, col_info = st.columns([2, 3])
with col_btn:
    avvia = st.button("🔍 SCANSIONA", type="primary", use_container_width=True)
with col_info:
    st.caption(
        "Il modello usa: medie gol, forma recente, H2H storico, "
        "strength rating e (se presenti) quote iniziali mercato."
    )

if avvia:
    urls = [u.strip() for u in urls_raw.strip().splitlines() if u.strip()]
    if not urls:
        st.warning("Incolla almeno un link Nowgoal.")
        st.stop()

    st.divider()
    progress_bar = st.progress(0.0, text="Avvio analisi...")
    status_area  = st.empty()
    risultati: list[dict] = []
    errori:    list[str]  = []

    completate = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_analizza_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            completate += 1
            progress_bar.progress(
                completate / len(urls),
                text=f"Analizzate {completate}/{len(urls)} partite...",
            )
            try:
                res = future.result(timeout=_FETCH_TIMEOUT)
            except concurrent.futures.TimeoutError:
                errori.append(f"{futures[future]} — Timeout dopo {_FETCH_TIMEOUT}s")
                continue
            if res is None:
                errori.append(futures[future])
            elif "error" in res:
                errori.append(f"{res['url']} — {res['error']}")
            else:
                risultati.append(res)

    progress_bar.empty()
    status_area.empty()

    if not risultati:
        st.error("Nessuna partita analizzata con successo. Verifica i link.")
        if errori:
            with st.expander("Errori"):
                for e in errori:
                    st.code(e)
        st.stop()

    # Ordina per segnale: stelle desc, poi forza (prob × confidence) desc
    risultati.sort(
        key=lambda r: (r["stelle"], r["prob"] * r["probs"]["confidence"]),
        reverse=True,
    )

    # ── Header risultati ──────────────────────────────────────────────────────
    n_ok  = len(risultati)
    n_err = len(errori)
    st.success(
        f"**{n_ok} partite analizzate**"
        + (f" · {n_err} errori" if n_err else "")
    )

    st.divider()

    # ── Classifica segnali ────────────────────────────────────────────────────
    st.subheader("Classifica segnali")

    for r in risultati:
        probs    = r["probs"]
        segnali  = r["segnali"]
        stelle   = "⭐" * r["stelle"] if r["stelle"] > 0 else "·"
        conf_pct = int(probs["confidence"] * 100)

        # Badge colore
        if r["stelle"] >= 3:
            badge = "🟢"
        elif r["stelle"] == 2:
            badge = "🟡"
        else:
            badge = "⚪"

        # Titolo card con top-3 segnali inline
        top3_str = "  ·  ".join(f"**{m}** {p:.0%}" for m, p, _ in segnali)
        titolo = f"{badge} {stelle}  **{r['home']} vs {r['away']}**"
        if r["league"]:
            titolo += f"  ·  _{r['league']}_"
        if r["date"]:
            titolo += f"  ·  {r['date']}"

        with st.expander(
            f"{titolo}  →  {top3_str}  (conf. {conf_pct}%)",
            expanded=(r["stelle"] >= 3),
        ):
            # ── 1X2 ──────────────────────────────────────────────────────────
            c1, cx, c2 = st.columns(3)
            c1.metric("1 Casa",     f"{probs['p1']:.0%}")
            cx.metric("X Pareggio", f"{probs['px']:.0%}")
            c2.metric("2 Trasf.",   f"{probs['p2']:.0%}")

            # ── O/U + BTTS ────────────────────────────────────────────────────
            co, cu, cgg, cng = st.columns(4)
            co.metric("Over 2.5",  f"{probs['p_over25']:.0%}")
            cu.metric("Under 2.5", f"{probs['p_under25']:.0%}")
            cgg.metric("GG Sì",    f"{probs['p_btts']:.0%}")
            cng.metric("GG No",    f"{1 - probs['p_btts']:.0%}")

            st.caption(
                f"xG Casa: **{probs['xg_h']:.2f}** · "
                f"xG Trasf.: **{probs['xg_a']:.2f}** · "
                f"Confidenza dati: **{conf_pct}%**"
            )

            # ── Primo Tempo (stima da xG) ─────────────────────────────────────
            p_ht1, p_htx, p_ht2, p_ht_o05 = _calcola_ht_da_xg(
                probs["xg_h"], probs["xg_a"]
            )
            st.caption("**Primo Tempo (stima da xG FT)**")
            cht1, chtx, cht2, chto = st.columns(4)
            cht1.metric("1T Casa",     f"{p_ht1:.0%}")
            chtx.metric("1T Pareggio", f"{p_htx:.0%}")
            cht2.metric("1T Trasf.",   f"{p_ht2:.0%}")
            if p_ht_o05 > 0.01:
                chto.metric("1T Over 0.5", f"{p_ht_o05:.0%}")

            # ── Dati grezzi con tabs ──────────────────────────────────────────
            pa = r["pa"]
            with st.expander("Dati grezzi estratti", expanded=False):
                tab_h, tab_a, tab_h2h = st.tabs(
                    [f"⚽ {r['home']}", f"✈️ {r['away']}", "📊 H2H & Mercato"]
                )

                with tab_h:
                    hg = max(1, (pa.home_home_win or 0) + (pa.home_home_draw or 0) + (pa.home_home_lose or 0))
                    st.write(f"**Partite totali:** {pa.home_matches or '—'}  ·  **Rank:** {pa.home_rank or '—'}")
                    sh_val = pa.strength_home or 0
                    if sh_val:
                        st.write(f"**Strength Nowgoal:** {sh_val}/100")
                    hl6w = pa.home_last6_win or 0
                    hl6d = pa.home_last6_draw or 0
                    hl6l = pa.home_last6_lose or 0
                    fm_h = pa.forma_mult_h if pa.forma_mult_h is not None else 1.0
                    st.write(f"**Forma Last 6:** {hl6w}V {hl6d}P {hl6l}S  ·  ×{fm_h:.2f}")
                    if pa.home_home_scored:
                        st.write(
                            f"**Media gol in casa:** "
                            f"{pa.home_home_scored/hg:.2f} fatti · "
                            f"{(pa.home_home_conceded or 0)/hg:.2f} subiti"
                        )
                    pho = getattr(pa, "prev_home_over_pct", None) or 0
                    if pho > 0:
                        st.write(f"**Over% in casa (prev):** {pho:.0f}%")

                with tab_a:
                    ag = max(1, (pa.away_away_win or 0) + (pa.away_away_draw or 0) + (pa.away_away_lose or 0))
                    st.write(f"**Partite totali:** {pa.away_matches or '—'}  ·  **Rank:** {pa.away_rank or '—'}")
                    sa_val = pa.strength_away or 0
                    if sa_val:
                        st.write(f"**Strength Nowgoal:** {sa_val}/100")
                    al6w = pa.away_last6_win or 0
                    al6d = pa.away_last6_draw or 0
                    al6l = pa.away_last6_lose or 0
                    fm_a = pa.forma_mult_a if pa.forma_mult_a is not None else 1.0
                    st.write(f"**Forma Last 6:** {al6w}V {al6d}P {al6l}S  ·  ×{fm_a:.2f}")
                    if pa.away_away_scored:
                        st.write(
                            f"**Media gol in trasferta:** "
                            f"{pa.away_away_scored/ag:.2f} fatti · "
                            f"{(pa.away_away_conceded or 0)/ag:.2f} subiti"
                        )
                    pao = getattr(pa, "prev_away_over_pct", None) or 0
                    if pao > 0:
                        st.write(f"**Over% in trasferta (prev):** {pao:.0f}%")

                with tab_h2h:
                    st.write(
                        f"**1X2 storico:**  "
                        f"Casa {pa.h2h_home_win_pct:.0f}%  ·  "
                        f"X {pa.h2h_draw_pct:.0f}%  ·  "
                        f"Trasf. {pa.h2h_away_win_pct:.0f}%"
                    )
                    st.write(f"**Over 2.5 storico:** {pa.h2h_over_pct:.0f}%")
                    if (pa.fixture_historical_total or 0) > 0:
                        st.write(f"**Media gol H2H:** {pa.fixture_historical_total:.2f}")
                    if (pa.mkt_init_1 or 0) > 1.0:
                        st.write(
                            f"**Quote iniziali mercato:** "
                            f"1 @{pa.mkt_init_1:.2f}  ·  "
                            f"X @{pa.mkt_init_x:.2f}  ·  "
                            f"2 @{pa.mkt_init_2:.2f}"
                        )

            st.caption(f"[Apri su Nowgoal]({r['url']})")

    # ── Errori ───────────────────────────────────────────────────────────────
    if errori:
        with st.expander(f"⚠️ {len(errori)} link non analizzati"):
            for e in errori:
                st.code(e)
