"""
signals.py — Generazione dei segnali di betting (back/lay).

Implementa la logica di valutazione dei segnali sia rapidi (senza quote exchange)
che avanzati (con quote exchange e Kelly criterion).

Tutto il flusso è ora senza stato globale: le funzioni restituiscono
liste di Signal invece di usare flag globali.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from src.config import KELLY, SIGNALS

_LOG = logging.getLogger("exchange.signals")
from src.engine import ExchangeQuotes  # noqa: F401 — used in type annotations
from src.models.kelly import (
    calcola_edge_back,
    calcola_edge_lay,
    calcola_ev_back,
    calcola_ev_lay,
    calcola_kelly_fraction,
    calcola_stake_kelly,
    calcola_stake_lay,
    quota_netta,
)

# ---------------------------------------------------------------------------
# Dataclass per un singolo segnale
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Rappresenta un singolo segnale di betting."""

    tipo: str            # "BACK" | "LAY" | "INFO_BACK" | "INFO_LAY"
    mercato: str         # Etichetta del mercato (es. "1 CASA", "Over 2.5")
    prob_mod: float      # Probabilità del modello
    quota_fair: float    # Quota fair del modello

    # Campi opzionali (presenti solo con quote exchange)
    quota_exc: float = 0.0
    quota_netta: float = 0.0
    edge: float = 0.0
    stake: float = 0.0
    liability: float = 0.0   # Solo per LAY
    ev_euro: float = 0.0
    riduzioni: list[str] = field(default_factory=list)
    kelly_raw: float = 0.0   # Stake Kelly lordo (prima di fraction e momentum)

    @property
    def prob_implicita(self) -> float:
        return 1.0 / self.quota_exc if self.quota_exc > 0 else 0.0


# ---------------------------------------------------------------------------
# Calcolo soglie dinamiche
# ---------------------------------------------------------------------------

def calcola_soglie(
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
    model_agreement: float = 1.0,
    *,
    n_shots_tot: int = 0,
    model_confidence: float = 1.0,
    momentum: float = 0.0,
) -> dict[str, float]:
    """
    Calcola le soglie di probabilità dinamiche per tutti i mercati.

    Le soglie crescono col tempo per compensare la riduzione di incertezza
    e i costi di spread più elevati nelle fasi avanzate della partita.

    IMPORTANTE: Ogni mercato ha soglie DIVERSE per evitare raccomandazioni
    identiche. Le soglie sono diversificate per:
    - 1X2: soglia base
    - BTTS Sì: +5% rispetto a 1X2 (mercato più volatile)
    - BTTS No: +3% rispetto a 1X2
    - Over/Under: soglia base OU + offset dedicato

    Quando i modelli sono in disaccordo (model_agreement < MODEL_AGREEMENT_LOW),
    le soglie vengono alzate fino a MODEL_AGREEMENT_PENALTY_MAX per ridurre i
    falsi positivi: se tre modelli indipendenti danno stime molto diverse,
    la stima consensus è meno affidabile.

    Args:
        minuto: Minuto attuale [0, 90].
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.
        model_agreement: Grado di accordo tra i 3 modelli [0, 1].

    Returns:
        Dict con le soglie per ogni mercato.
    """
    frac = minuto / 90.0
    frac_sqrt = math.sqrt(frac) if frac > 0 else 0.0
    base_1x2 = max(SIGNALS.SOGLIA_BACK_MIN, SIGNALS.SOGLIA_BACK_BASE + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt)

    # FIX: Soglie OU con offset dedicato per diversificare da 1X2
    base_ou = max(SIGNALS.OVER_BASE_MIN,
                  SIGNALS.OVER_BASE_THRESHOLD + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt + SIGNALS.SOGLIA_OU_OFFSET)

    # Penalità disaccordo modelli: se agreement < LOW, le soglie salgono linearmente.
    # Formula: penalty = max(0, (LOW - agreement) / LOW) × PENALTY_MAX
    # Esempio: agreement=0.40, LOW=0.60 → penalty = (0.20/0.60) × 0.08 = 2.7%
    agreement_penalty = 0.0
    if model_agreement < SIGNALS.MODEL_AGREEMENT_LOW:
        agreement_penalty = (
            (SIGNALS.MODEL_AGREEMENT_LOW - model_agreement)
            / SIGNALS.MODEL_AGREEMENT_LOW
            * SIGNALS.MODEL_AGREEMENT_PENALTY_MAX
        )

    # Upgrade 8-6: Aggiustamento adattivo basato sulla ricchezza informativa.
    # Alta ricchezza (molti tiri, alta confidenza) → riduce soglie → più segnali.
    # Bassa ricchezza (pochi dati) → alza soglie → meno segnali.
    _info_adj = 0.0
    try:
        from src.models.adaptive_thresholds import compute_threshold_adjustment
        _info_adj = compute_threshold_adjustment(
            minuto, n_shots_tot, model_confidence, momentum,
        )
    except Exception as _ate:
        _LOG.debug("adaptive threshold computation skipped: %s", _ate)
    base_1x2 = max(SIGNALS.SOGLIA_BACK_MIN, base_1x2 + _info_adj)
    base_ou = max(SIGNALS.OVER_BASE_MIN, base_ou + _info_adj)

    gol_mancanti = max(0.0, linea_ou - gol_attuali)
    ou_gol_bonus = min(SIGNALS.OVER_GOL_BONUS_CAP, max(0.0, (gol_mancanti - 1.0) * SIGNALS.OVER_GOL_BONUS_RATE))

    # FIX: Soglie DIVERSIFICATE per mercato con offset configurabili
    # BTTS Sì richiede che ENTRAMBE le squadre segnino → più difficile → soglia +5%
    # BTTS No è più probabile ma anche più soggetto a varianza → soglia +3%
    # Over/Under hanno il loro offset dedicato (SOGLIA_OU_OFFSET)
    return {
        "1x2":      base_1x2 + agreement_penalty + SIGNALS.SOGLIA_1X2_OFFSET,
        "btts_si":  base_1x2 + SIGNALS.SOGLIA_BTTS_OFFSET + agreement_penalty,
        "btts_no":  max(SIGNALS.SOGLIA_BTTS_NO_MIN,
                        SIGNALS.SOGLIA_BTTS_NO_BASE + SIGNALS.SOGLIA_BACK_SLOPE * frac_sqrt
                        + 0.03 + agreement_penalty),
        "ou_over":  base_ou + ou_gol_bonus + agreement_penalty,
        "ou_under": base_ou + agreement_penalty,
        "gol_mancanti": gol_mancanti,
    }


# ---------------------------------------------------------------------------
# Segnali rapidi (senza quote exchange)
# ---------------------------------------------------------------------------

def genera_segnali_rapidi(
    prob_1: float,
    prob_x: float,
    prob_2: float,
    prob_over: float,
    prob_under: float,
    prob_btts: float,
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
    model_confidence: float = 1.0,
    model_agreement: float = 1.0,
    gol_casa: int = 0,
    gol_trasf: int = 0,
    top_cs: list[tuple[tuple[int, int], float]] | None = None,
    signals_blocked: bool = False,
) -> list[Signal]:
    """
    Genera segnali rapidi basati sulle probabilità del modello, senza quote exchange.

    Richiede solo la probabilità del modello e la confronta con le soglie dinamiche.
    Utile per utenti senza accesso diretto alle quote dell'exchange.

    Args:
        prob_*: Probabilità del modello per ogni mercato.
        minuto: Minuto attuale.
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.
        model_confidence: Score di confidenza del modello [0, 1].
            Se < SIGNALS.MIN_CONFIDENCE_FOR_SIGNALS, nessun segnale viene emesso.
        model_agreement: Grado di accordo tra i 3 modelli [0, 1].
            Usato per scalare le soglie verso l'alto quando i modelli divergono.

    Returns:
        Lista di Signal di tipo INFO_BACK o INFO_LAY. Lista vuota se la
        confidenza del modello è sotto soglia (linee stantie, dati mancanti, ecc.).
    """
    # Quality firewall: blocco operativo deciso dalla pipeline.
    if signals_blocked:
        return []

    # Gate confidenza: se il modello non è affidabile, non emettere segnali.
    # Con linee stantie, assenza di dati sui tiri o modelli in forte disaccordo,
    # qualsiasi segnale sarebbe rumore — meglio il silenzio di un consiglio errato.
    if model_confidence < SIGNALS.MIN_CONFIDENCE_FOR_SIGNALS:
        return []

    soglie = calcola_soglie(minuto, linea_ou, gol_attuali, model_agreement)

    # Floor live per segnali RAPIDI (senza quote exchange):
    # in partita il mercato ha già prezzato gli eventi recenti → servono
    # probabilità più nette. Per i segnali avanzati (con quote) l'edge è
    # il filtro principale e il floor non si applica.
    # Nota: X (draw) usa un floor più basso perché non soffre del problema
    # "segnale ovvio" (es. BACK 1 al 1-0) che il floor 0.63 vuole evitare.
    _s1    = max(SIGNALS.SOGLIA_LIVE_BACK_MIN, soglie["1x2"]) if minuto > 0 else soglie["1x2"]
    _sx    = max(SIGNALS.SOGLIA_LIVE_DRAW_MIN, soglie["1x2"]) if minuto > 0 else soglie["1x2"]
    _s2    = max(SIGNALS.SOGLIA_LIVE_BACK_MIN, soglie["1x2"]) if minuto > 0 else soglie["1x2"]
    _sou_o = max(SIGNALS.SOGLIA_LIVE_OU_MIN,  soglie["ou_over"]) if minuto > 0 else soglie["ou_over"]
    _sou_u = max(SIGNALS.SOGLIA_LIVE_OU_MIN,  soglie["ou_under"]) if minuto > 0 else soglie["ou_under"]
    _sbtts = max(SIGNALS.SOGLIA_LIVE_BACK_MIN, soglie["btts_si"]) if minuto > 0 else soglie["btts_si"]
    _sbtts_no = max(SIGNALS.SOGLIA_LIVE_BACK_MIN, soglie["btts_no"]) if minuto > 0 else soglie["btts_no"]

    segnali: list[Signal] = []

    # ── DNB + Double Chance ───────────────────────────────────────────────────
    _eps = 1e-8
    _s_dnb = _s1  # DNB: stesso floor di BACK 1/2
    _s_dc  = max(SIGNALS.SOGLIA_DC_LIVE_MIN, SIGNALS.SOGLIA_DC) if minuto > 0 else SIGNALS.SOGLIA_DC

    p_dnb_h = prob_1 / (prob_1 + prob_2 + _eps)
    p_dnb_a = prob_2 / (prob_1 + prob_2 + _eps)

    mercati: list[tuple[str, float, float]] = [
        ("1 Casa",         prob_1,              _s1),
        ("X Pareggio",     prob_x,              _sx),
        ("2 Trasf.",       prob_2,              _s2),
        (f"Over {linea_ou}",  prob_over,         _sou_o),
        (f"Under {linea_ou}", prob_under,        _sou_u),
        ("BTTS Sì",        prob_btts,           _sbtts),
        ("BTTS No",        1.0 - prob_btts,     _sbtts_no),
        ("DNB Casa",       p_dnb_h,             _s_dnb),
        ("DNB Trasf.",     p_dnb_a,             _s_dnb),
    ]
    # DC: solo se entrambi i componenti del mercato sono significativi
    if prob_1 > 0.20 and prob_x > 0.15:
        mercati.append(("DC 1X", prob_1 + prob_x, _s_dc))
    if prob_2 > 0.20 and prob_x > 0.15:
        mercati.append(("DC X2", prob_x + prob_2, _s_dc))
    if prob_1 > 0.20 and prob_2 > 0.15:
        mercati.append(("DC 12", prob_1 + prob_2, _s_dc))

    for etichetta, prob, soglia_back in mercati:
        q_fair = 1.0 / prob if prob > SIGNALS.MIN_PROB_FOR_QUOTE else SIGNALS.MAX_QUOTE_FALLBACK

        # Skip eventi quasi certi
        if q_fair < SIGNALS.QUICK_SIGNAL_MIN_FAIR_Q:
            continue

        q_min_back = q_fair * (1.0 + SIGNALS.MARGINE_RAPIDO)
        q_max_lay = q_fair / (1.0 + SIGNALS.MARGINE_RAPIDO)

        # Penalità "vantaggio ovvio": si applica a 1 Casa, 2 Trasf. e ai
        # relativi DNB (che includono solo quella squadra).
        # La penalità decade col tempo: 1-0 al 10' è rumore, 1-0 al 65' è probabile.
        # Formula: penalty_raw × decay dove decay = max(0.30, 1 - minuto/90)
        #   → al minuto 0: decay=1.0 (mai attivo, minuto>0 richiesto)
        #   → al minuto 30: decay=0.67 (penalità ridotta a 67%)
        #   → al minuto 60: decay=0.33 (penalità a 33%)
        #   → al minuto 70+: non applicata (cutoff)
        effective_soglia = soglia_back
        if minuto > 0 and minuto < SIGNALS.LEAD_SOGLIA_MINUTE_CUTOFF:
            _lead_decay = max(0.30, 1.0 - minuto / 90.0)
            if etichetta in ("1 Casa", "DNB Casa") and gol_casa > gol_trasf:
                _lead = gol_casa - gol_trasf
                _raw_penalty = min(SIGNALS.LEAD_SOGLIA_PENALTY_CAP,
                                   _lead * SIGNALS.LEAD_SOGLIA_PENALTY_RATE)
                effective_soglia += _raw_penalty * _lead_decay
            elif etichetta in ("2 Trasf.", "DNB Trasf.") and gol_trasf > gol_casa:
                _lead = gol_trasf - gol_casa
                _raw_penalty = min(SIGNALS.LEAD_SOGLIA_PENALTY_CAP,
                                   _lead * SIGNALS.LEAD_SOGLIA_PENALTY_RATE)
                effective_soglia += _raw_penalty * _lead_decay

        if prob >= effective_soglia:
            segnali.append(Signal(
                tipo="INFO_BACK",
                mercato=etichetta,
                prob_mod=prob,
                quota_fair=q_fair,
                quota_exc=q_min_back,
            ))
        elif prob <= SIGNALS.SOGLIA_LAY_MAX and q_fair >= SIGNALS.LAY_MIN_FAIR_Q:
            segnali.append(Signal(
                tipo="INFO_LAY",
                mercato=etichetta,
                prob_mod=prob,
                quota_fair=q_fair,
                quota_exc=q_max_lay,
            ))

    # ── Correct Score ─────────────────────────────────────────────────────────
    if top_cs:
        cs_score, cs_prob = top_cs[0]
        if cs_prob >= SIGNALS.SOGLIA_CS_MIN:
            q_fair_cs = 1.0 / cs_prob
            if q_fair_cs >= SIGNALS.QUICK_SIGNAL_MIN_FAIR_Q:
                segnali.append(Signal(
                    tipo="INFO_BACK",
                    mercato=f"CS {cs_score[0]}-{cs_score[1]}",
                    prob_mod=cs_prob,
                    quota_fair=q_fair_cs,
                    quota_exc=q_fair_cs * (1.0 + SIGNALS.MARGINE_RAPIDO),
                ))

    return _filtra_segnali_coerenti(segnali)


# ---------------------------------------------------------------------------
# Filtro coerenza cross-mercato
# ---------------------------------------------------------------------------

# Gruppi di mercati mutualmente esclusivi: un solo BACK è ammesso per gruppo.
_GRUPPO_1X2 = {"1 Casa", "X Pareggio", "2 Trasf."}
_GRUPPO_DNB = {"DNB Casa", "DNB Trasf."}
_GRUPPO_DC  = {"DC 1X", "DC X2", "DC 12"}
_TIPO_BACK = {"BACK", "INFO_BACK"}
_TIPO_LAY  = {"LAY", "INFO_LAY"}


def _filtra_segnali_coerenti(segnali: list[Signal]) -> list[Signal]:
    """
    Applica quattro livelli di filtro per eliminare segnali contraddittori o ridondanti.

    Regola 1 — Esclusività mutua 1X2:
        Solo il miglior segnale BACK tra 1/X/2 sopravvive.
        Ragionamento: scommettere su due esiti diversi dello stesso mercato
        è matematicamente incoerente (non importa chi vince, perdi commissioni).

    Regola 2 — Esclusività mutua O/U e BTTS:
        Solo una direzione per mercato (Over O Under, BTTS Sì O No).
        Stesso principio: non puoi guadagnare backando entrambi i lati.

    Regola 3 — Coerenza cross-mercato:
        - BACK Over + BACK BTTS No → incoerente (gol multipli senza BTTS?
          raro, ma sopra 2.5 è implausibile). Rimuove il segnale con edge minore.
        - BACK Under + BACK BTTS Sì → incoerente (BTTS richiede ≥2 gol,
          difficile compatibile con Under 2.5 o meno). Rimuove il più debole.

    Regola 4 — LAY X ridondante con BACK 1 o BACK 2:
        LAY X (banca contro il pareggio) è direzionalmente identico a BACK 1/2.

    Regola 5 — Distanza minima tra quote (FIX):
        Se due segnali hanno quote fair troppo simili (< MIN_QUOTE_DISTANCE),
        tieni solo quello con edge/probabilità maggiore.

    Ordine finale: segnali ordinati per edge decrescente (migliori prima),
    limitati a MAX_SEGNALI_RAPIDI o MAX_SEGNALI_AVANZATI.
    """
    if not segnali:
        return segnali

    def _forza(s: Signal) -> float:
        """Forza del segnale: edge se disponibile, altrimenti prob_mod."""
        return s.edge if s.edge > 0 else s.prob_mod

    # ── Regola 1: 1X2 — un solo BACK ────────────────────────────────────────
    back_1x2 = [s for s in segnali if s.mercato in _GRUPPO_1X2 and s.tipo in _TIPO_BACK]
    if len(back_1x2) > 1:
        best = max(back_1x2, key=_forza)
        segnali = [s for s in segnali if s not in back_1x2 or s is best]

    # I LAY su 1X2 sono direzionalmente compatibili con BACK su altro esito,
    # ma più di uno LAY su 1X2 è confuso → teniamo solo il migliore.
    lay_1x2 = [s for s in segnali if s.mercato in _GRUPPO_1X2 and s.tipo in _TIPO_LAY]
    if len(lay_1x2) > 1:
        best_lay = max(lay_1x2, key=_forza)
        segnali = [s for s in segnali if s not in lay_1x2 or s is best_lay]

    # ── Regola 1b: DNB — una sola direzione (Casa o Trasf.) ──────────────────
    back_dnb = [s for s in segnali if s.mercato in _GRUPPO_DNB and s.tipo in _TIPO_BACK]
    if len(back_dnb) > 1:
        best_dnb = max(back_dnb, key=_forza)
        segnali = [s for s in segnali if s not in back_dnb or s is best_dnb]

    # ── Regola 1c: DC — una sola variante (1X, X2 o 12) ─────────────────────
    back_dc = [s for s in segnali if s.mercato in _GRUPPO_DC and s.tipo in _TIPO_BACK]
    if len(back_dc) > 1:
        best_dc = max(back_dc, key=_forza)
        segnali = [s for s in segnali if s not in back_dc or s is best_dc]

    # ── Regola 2: O/U — una sola direzione ──────────────────────────────────
    back_over = [s for s in segnali if "Over" in s.mercato and s.tipo in _TIPO_BACK]
    back_under = [s for s in segnali if "Under" in s.mercato and s.tipo in _TIPO_BACK]
    if back_over and back_under:
        # Tieni solo il più forte tra le due direzioni
        tutti_ou = back_over + back_under
        best_ou = max(tutti_ou, key=_forza)
        segnali = [s for s in segnali if s not in tutti_ou or s is best_ou]

    # ── Regola 2b: BTTS — una sola direzione ────────────────────────────────
    back_btts_si = [s for s in segnali if s.mercato == "BTTS Sì" and s.tipo in _TIPO_BACK]
    back_btts_no = [s for s in segnali if s.mercato == "BTTS No" and s.tipo in _TIPO_BACK]
    if back_btts_si and back_btts_no:
        tutti_btts = back_btts_si + back_btts_no
        best_btts = max(tutti_btts, key=_forza)
        segnali = [s for s in segnali if s not in tutti_btts or s is best_btts]

    # ── Regola 3: coerenza cross-mercato ────────────────────────────────────
    has_back_over = any("Over" in s.mercato and s.tipo in _TIPO_BACK for s in segnali)
    has_back_under = any("Under" in s.mercato and s.tipo in _TIPO_BACK for s in segnali)
    has_btts_si = any(s.mercato == "BTTS Sì" and s.tipo in _TIPO_BACK for s in segnali)
    has_btts_no = any(s.mercato == "BTTS No" and s.tipo in _TIPO_BACK for s in segnali)

    if has_back_over and has_btts_no:
        # Over alto + BTTS No → gol multipli da UNA squadra: raro, incoerente
        over_s = next(s for s in segnali if "Over" in s.mercato and s.tipo in _TIPO_BACK)
        no_s = next(s for s in segnali if s.mercato == "BTTS No" and s.tipo in _TIPO_BACK)
        rimuovi = no_s if _forza(over_s) >= _forza(no_s) else over_s
        segnali = [s for s in segnali if s is not rimuovi]

    if has_back_under and has_btts_si:
        # Under + BTTS Sì → impossibile su linee ≤ 1.5, incoerente ≤ 2.5
        under_s = next(s for s in segnali if "Under" in s.mercato and s.tipo in _TIPO_BACK)
        si_s = next(s for s in segnali if s.mercato == "BTTS Sì" and s.tipo in _TIPO_BACK)
        rimuovi = si_s if _forza(under_s) >= _forza(si_s) else under_s
        segnali = [s for s in segnali if s is not rimuovi]

    # ── Regola 4: LAY X ridondante con BACK 1 o BACK 2 ──────────────────────
    # LAY X (banca contro il pareggio) è direzionalmente identico a BACK 1/2:
    # entrambi guadagnano se la squadra mantiene/conquista il vantaggio.
    # Mostrare entrambi confonde l'utente senza aggiungere informazione.
    # Eccezione: il LAY X da solo è un segnale valido quando non c'è un BACK 1X2,
    # perché comunica "non credo al pareggio" senza prendere posizione su chi vince.
    has_back_1 = any(s.mercato == "1 Casa" and s.tipo in _TIPO_BACK for s in segnali)
    has_back_2 = any(s.mercato == "2 Trasf." and s.tipo in _TIPO_BACK for s in segnali)
    if has_back_1 or has_back_2:
        segnali = [s for s in segnali
                   if not (s.mercato == "X Pareggio" and s.tipo in _TIPO_LAY)]

    # ── Regola 5 (FIX): Distanza minima tra quote fair ──────────────────────
    # Se due segnali hanno quote fair troppo vicine, uno è ridondante.
    segnali_sorted = sorted(segnali, key=_forza, reverse=True)
    filtered: list[Signal] = []
    for s in segnali_sorted:
        is_close = False
        for existing in filtered:
            if abs(s.quota_fair - existing.quota_fair) < SIGNALS.MIN_QUOTE_DISTANCE:
                # Quote troppo vicine → salta il più debole (s)
                is_close = True
                break
        if not is_close:
            filtered.append(s)
    segnali = filtered

    # ── Regola 6: Clustering detection ─────────────────────────────────────
    # Segnali altamente correlati vengono annotati (non rimossi) con un avviso
    # per ricordare che le scommesse si muovono insieme.
    _mercati_segnali = {s.mercato for s in segnali if s.tipo in _TIPO_BACK}
    for s in segnali:
        if s.tipo not in _TIPO_BACK:
            continue
        correlati = []
        if s.mercato == "1 Casa" and "DNB Casa" in _mercati_segnali:
            correlati.append("≈DNB Casa")
        if s.mercato == "DNB Casa" and "1 Casa" in _mercati_segnali:
            correlati.append("≈BACK 1")
        if s.mercato == "2 Trasf." and "DNB Trasf." in _mercati_segnali:
            correlati.append("≈DNB Trasf.")
        if s.mercato == "DNB Trasf." and "2 Trasf." in _mercati_segnali:
            correlati.append("≈BACK 2")
        if "Over" in s.mercato and "BTTS Sì" in _mercati_segnali:
            correlati.append("≈BTTS Sì")
        if s.mercato == "BTTS Sì":
            over_mkt = next((m for m in _mercati_segnali if "Over" in m), None)
            if over_mkt:
                correlati.append(f"≈{over_mkt}")
        if correlati:
            s.riduzioni.append("cluster: " + ", ".join(correlati))

    # Ordina: BACK/LAY prima, poi per forza decrescente
    ordine_tipo = {"BACK": 0, "LAY": 1, "INFO_BACK": 2, "INFO_LAY": 3}
    segnali.sort(key=lambda s: (ordine_tipo.get(s.tipo, 9), -_forza(s)))

    # FIX: Limita il numero di segnali
    # Conta quanti sono rapidi (INFO_*) vs avanzati (BACK/LAY)
    rapidi = [s for s in segnali if s.tipo in ("INFO_BACK", "INFO_LAY")]
    avanzati = [s for s in segnali if s.tipo in ("BACK", "LAY")]

    # Se ci sono solo rapidi, limita a MAX_SEGNALI_RAPIDI
    if rapidi and not avanzati:
        return rapidi[:SIGNALS.MAX_SEGNALI_RAPIDI]
    # Se ci sono avanzati, limita a MAX_SEGNALI_AVANZATI
    elif avanzati:
        return segnali[:SIGNALS.MAX_SEGNALI_AVANZATI]
    else:
        return segnali


# ---------------------------------------------------------------------------
# Segnali avanzati (con quote exchange)
# ---------------------------------------------------------------------------

def valuta_mercato(
    etichetta: str,
    prob_mod: float,
    q_exc: float,
    soglia_back: float,
    bankroll: float,
    comm_rate: float,
    kelly_frac: float,
    momentum_factor: float,
    back_only: bool = False,
    minuto: int = 0,
    kelly_frac_base: float = KELLY.KELLY_BASE_FRACTION,
    model_confidence: float = 1.0,
) -> Signal | None:
    """
    Valuta un singolo mercato con quota exchange.

    Calcola l'edge netto (dopo commissione) e genera un segnale
    BACK o LAY se l'edge supera la soglia minima.

    Args:
        etichetta: Nome del mercato.
        prob_mod: Probabilità del modello.
        q_exc: Quota sull'exchange (0 = non inserita).
        soglia_back: Soglia minima di probabilità per il back.
        bankroll: Capitale disponibile.
        comm_rate: Commissione exchange in [0, 1).
        kelly_frac: Frazione Kelly.
        momentum_factor: Fattore riduzione stake per momentum.
        back_only: Se True, non valuta il LAY.

    Returns:
        Signal se trovato valore, None altrimenti.
    """
    q_fair = 1.0 / prob_mod if prob_mod > SIGNALS.MIN_PROB_FOR_QUOTE else SIGNALS.MAX_QUOTE_FALLBACK

    # Skip eventi quasi certi
    if q_fair < SIGNALS.MIN_FAIR_Q:
        return None

    # Senza quota exchange: indicazione qualitativa con soglia adattiva al tempo
    if q_exc <= 1.0:
        frac_giocata = minuto / 90.0
        frac_sqrt = math.sqrt(frac_giocata) if frac_giocata > 0 else 0.0
        soglia_q = max(SIGNALS.SOGLIA_QUALITATIVA_MIN,
                       SIGNALS.SOGLIA_QUALITATIVA_BASE + SIGNALS.SOGLIA_QUALITATIVA_SLOPE * frac_sqrt)
        if prob_mod >= soglia_q:
            return Signal(
                tipo="INFO_BACK",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
            )
        if prob_mod <= SIGNALS.SOGLIA_LAY_MAX and not back_only:
            return Signal(
                tipo="INFO_LAY",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
            )
        return None

    # Con quota exchange: calcolo edge preciso
    q_net = quota_netta(q_exc, comm_rate)
    edge_back = calcola_edge_back(prob_mod, q_net)
    edge_lay = calcola_edge_lay(prob_mod, q_exc, comm_rate)

    # Edge minimo dinamico: con bassa confidenza richiediamo un vantaggio più netto.
    # Tra MIN_CONFIDENCE_FOR_SIGNALS e CONF_EDGE_BOOST_HIGH il boost scala linearmente.
    # Razionale: se il modello è incerto (linee stantie, pochi tiri, modelli discordi),
    # solo un'opportunità molto chiara giustifica l'operazione.
    if model_confidence < SIGNALS.CONF_EDGE_BOOST_HIGH:
        t = max(0.0, (SIGNALS.CONF_EDGE_BOOST_HIGH - model_confidence)
                     / (SIGNALS.CONF_EDGE_BOOST_HIGH - SIGNALS.MIN_CONFIDENCE_FOR_SIGNALS))
        edge_boost = t * SIGNALS.CONF_EDGE_BOOST_MAX
    else:
        edge_boost = 0.0
    min_edge_back = SIGNALS.MIN_EDGE_BACK + edge_boost
    min_edge_lay = SIGNALS.MIN_EDGE_LAY + edge_boost

    # BACK
    if edge_back >= min_edge_back and prob_mod >= soglia_back:
        # Modulazione momentum × edge: edge forte → meno riduzione, edge debole → più riduzione.
        if edge_back >= SIGNALS.MOMENTUM_EDGE_STRONG:
            adj_momentum = max(SIGNALS.MOMENTUM_STAKE_FLOOR,
                               1.0 - (1.0 - momentum_factor) * (1.0 - SIGNALS.MOMENTUM_EDGE_DAMPEN))
        elif edge_back < SIGNALS.MIN_EDGE_BACK * 1.5:
            adj_momentum = max(SIGNALS.MOMENTUM_STAKE_FLOOR,
                               1.0 - (1.0 - momentum_factor) * (1.0 + SIGNALS.MOMENTUM_EDGE_AMPLIFY))
        else:
            adj_momentum = momentum_factor
        # kelly_raw: stake al 100% Kelly senza fraction/momentum (per trasparenza breakdown)
        kelly_raw = calcola_stake_kelly(prob_mod, q_net, bankroll, 1.0, edge_back)
        stake_raw = calcola_stake_kelly(prob_mod, q_net, bankroll, kelly_frac, edge_back)
        stake = stake_raw * adj_momentum

        if stake > 0:
            ev = calcola_ev_back(stake, prob_mod, q_net)
            riduzioni = _build_riduzioni(comm_rate, momentum_factor, kelly_frac, q_net)
            return Signal(
                tipo="BACK",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
                quota_exc=q_exc,
                quota_netta=q_net,
                edge=edge_back,
                stake=stake,
                ev_euro=ev,
                riduzioni=riduzioni,
                kelly_raw=kelly_raw,
            )

    # LAY
    if not back_only and edge_lay >= min_edge_lay and q_exc >= KELLY.LAY_MIN_ODDS:
        result = calcola_stake_lay(prob_mod, q_exc, bankroll, kelly_frac, comm_rate)
        if result is not None:
            stake_lay, liab_lay = result
            # Modulazione momentum × edge (lay)
            if edge_lay >= SIGNALS.MOMENTUM_EDGE_STRONG:
                adj_momentum_lay = max(SIGNALS.MOMENTUM_STAKE_FLOOR,
                                       1.0 - (1.0 - momentum_factor) * (1.0 - SIGNALS.MOMENTUM_EDGE_DAMPEN))
            elif edge_lay < SIGNALS.MIN_EDGE_LAY * 1.5:
                adj_momentum_lay = max(SIGNALS.MOMENTUM_STAKE_FLOOR,
                                       1.0 - (1.0 - momentum_factor) * (1.0 + SIGNALS.MOMENTUM_EDGE_AMPLIFY))
            else:
                adj_momentum_lay = momentum_factor
            stake_lay *= adj_momentum_lay
            liab_lay *= adj_momentum_lay
            ev = calcola_ev_lay(stake_lay, prob_mod, q_exc, comm_rate)
            riduzioni = _build_riduzioni(0.0, momentum_factor, kelly_frac, None)
            return Signal(
                tipo="LAY",
                mercato=etichetta,
                prob_mod=prob_mod,
                quota_fair=q_fair,
                quota_exc=q_exc,
                edge=edge_lay,
                stake=stake_lay,
                liability=liab_lay,
                ev_euro=ev,
                riduzioni=riduzioni,
            )

    return None


def _build_riduzioni(
    comm_rate: float,
    momentum_factor: float,
    kelly_frac: float,
    q_net: float | None,
) -> list[str]:
    """Costruisce la lista testuale delle riduzioni applicate alla stake."""
    riduzioni = []
    if comm_rate > 0 and q_net is not None:
        riduzioni.append(f"comm {comm_rate*100:.1f}% → @{q_net:.3f} netto")
    if momentum_factor < 1.0:
        riduzioni.append(f"momentum ×{momentum_factor:.2f}")
    if kelly_frac < KELLY.KELLY_BASE_FRACTION:
        riduzioni.append(f"Kelly ×{kelly_frac:.2f}")
    return riduzioni


def genera_segnali_avanzati(
    prob_1: float,
    prob_x: float,
    prob_2: float,
    prob_over: float,
    prob_under: float,
    prob_btts: float,
    quotes: ExchangeQuotes,
    minuto: int,
    linea_ou: float,
    gol_attuali: int,
    bankroll: float,
    comm_rate: float,
    n_shots_tot: int,
    momentum: float,
    model_confidence: float = 1.0,
    model_agreement: float = 1.0,
    gol_casa: int = 0,
    gol_trasf: int = 0,
    signals_blocked: bool = False,
    ci_tightness: float = 0.55,
    credible_intervals: dict[str, tuple[float, float]] | None = None,
) -> list[Signal]:
    """
    Genera segnali avanzati con quote exchange, Kelly criterion e EV.

    Args:
        prob_*: Probabilità del modello.
        quotes: Quote dall'exchange.
        minuto: Minuto attuale.
        linea_ou: Linea Over/Under selezionata.
        gol_attuali: Gol totali già segnati.
        bankroll: Capitale disponibile.
        comm_rate: Commissione exchange.
        n_shots_tot: Numero tiri totali inseriti.
        momentum: Indice momentum di mercato.
        model_confidence: Score di confidenza del modello [0, 1].
        ci_tightness: [0,1] da intervalli modelli → scala frazione Kelly base.
        credible_intervals: se presente, sconto stake per incertezza (kelly_uncertainty).

    Returns:
        Lista di Signal con calcoli Kelly/EV completi.
    """
    if signals_blocked:
        return []

    # Gate confidenza: sopprimi segnali avanzati se il modello non è affidabile.
    # Stessa soglia dei rapidi: linee stantie, dati assenti, modelli discordi
    # rendono anche l'edge misurato non affidabile (il modello è fuori calibrazione).
    if model_confidence < SIGNALS.MIN_CONFIDENCE_FOR_SIGNALS:
        return []

    soglie = calcola_soglie(
        minuto,
        linea_ou,
        gol_attuali,
        model_agreement,
        n_shots_tot=n_shots_tot,
        model_confidence=model_confidence,
        momentum=momentum,
    )
    kelly_frac = calcola_kelly_fraction(
        minuto, n_shots_tot, model_confidence, ci_tightness=ci_tightness
    )

    # #4: Scala il momentum effettivo per model_agreement.
    # Quando i modelli divergono, la stima xG è già incerta → non amplificare con momentum.
    _agree_scale = max(SIGNALS.MOMENTUM_AGREE_FLOOR, model_agreement)
    _effective_momentum = momentum * _agree_scale

    # #8: Momentum factor differenziato per mercato.
    # 1X2 è meno sensibile al momentum (forma pre-partita domina).
    # BTTS Sì è molto sensibile (richiede gol multipli = alta volatilità).
    def _mf(market_mult: float) -> float:
        return max(
            SIGNALS.MOMENTUM_STAKE_FLOOR,
            1.0 - SIGNALS.MOMENTUM_STAKE_REDUCTION_RATE
                  * max(0.0, _effective_momentum - SIGNALS.MOMENTUM_STAKE_THRESHOLD)
                  * market_mult,
        )

    segnali: list[Signal] = []

    # Upgrade 8-3: Mappa etichetta → chiave credible interval per Kelly uncertainty.
    _label_to_ci_key: dict[str, str] = {
        "1 Casa": "p1", "X Pareggio": "px", "2 Trasf.": "p2",
        "BTTS Sì": "p_btts", "BTTS No": "p_btts",
    }

    def _valuta(etichetta: str, prob: float, q_exc: float, soglia: float,
                back_only: bool = False, mf: float | None = None) -> None:
        s = valuta_mercato(
            etichetta, prob, q_exc, soglia,
            bankroll, comm_rate, kelly_frac,
            mf if mf is not None else _mf(SIGNALS.MOMENTUM_MKT_OVER),
            back_only,
            minuto=minuto,
            model_confidence=model_confidence,
        )
        if s is not None:
            # Upgrade 8-3: Kelly uncertainty discount.
            # Riduce lo stake se l'intervallo credibile è largo (edge incerto).
            if credible_intervals and s.stake > 0 and s.edge > 0:
                try:
                    from src.models.kelly_uncertainty import compute_edge_uncertainty_discount
                    _ci_key = _label_to_ci_key.get(etichetta)
                    if _ci_key is None and "Over" in etichetta:
                        _ci_key = "p_over"
                    elif _ci_key is None and "Under" in etichetta:
                        _ci_key = "p_under"
                    if _ci_key and _ci_key in credible_intervals:
                        _ci = credible_intervals[_ci_key]
                        _discount = compute_edge_uncertainty_discount(s.edge, _ci[1] - _ci[0])
                        s.stake *= _discount
                        s.ev_euro *= _discount
                except Exception as _cde:
                    _LOG.debug("credible interval edge discount skipped: %s", _cde)
            segnali.append(s)

    # 1X2 — soglia adattiva per penalizzare il BACK sulla squadra già vincente
    def _soglia_1x2_con_lead(etichetta: str) -> float:
        s = soglie["1x2"]
        if minuto > 0 and minuto < SIGNALS.LEAD_SOGLIA_MINUTE_CUTOFF:
            if etichetta == "1 Casa" and gol_casa > gol_trasf:
                s += min(SIGNALS.LEAD_SOGLIA_PENALTY_CAP,
                         (gol_casa - gol_trasf) * SIGNALS.LEAD_SOGLIA_PENALTY_RATE)
            elif etichetta == "2 Trasf." and gol_trasf > gol_casa:
                s += min(SIGNALS.LEAD_SOGLIA_PENALTY_CAP,
                         (gol_trasf - gol_casa) * SIGNALS.LEAD_SOGLIA_PENALTY_RATE)
        return s

    _valuta("1 Casa", prob_1, quotes.q_1, _soglia_1x2_con_lead("1 Casa"), mf=_mf(SIGNALS.MOMENTUM_MKT_1X2))
    _valuta("2 Trasf.", prob_2, quotes.q_2, _soglia_1x2_con_lead("2 Trasf."), mf=_mf(SIGNALS.MOMENTUM_MKT_1X2))
    if quotes.q_x > 1.0:
        _valuta("X Pareggio", prob_x, quotes.q_x, soglie["1x2"], mf=_mf(SIGNALS.MOMENTUM_MKT_1X2))

    # Over/Under
    # LAY Over = scommettere che non arriveranno abbastanza gol = equivalente a BACK Under.
    # Per evitare segnali doppi (LAY Over + BACK Under per lo stesso scenario), il LAY
    # Over è disabilitato in condizioni normali. Eccezione: late game (>75') con ≥2 gol
    # mancanti, dove la liquidità del mercato Over è superiore e il LAY è più efficiente.
    gol_mancanti = soglie["gol_mancanti"]
    if minuto >= SIGNALS.LATE_GAME_LAY_OVER_MINUTE and gol_mancanti >= SIGNALS.LATE_GAME_LAY_OVER_GOALS:
        _valuta(f"Over {linea_ou}", prob_over, quotes.q_over, soglie["ou_over"],
                back_only=False, mf=_mf(SIGNALS.MOMENTUM_MKT_OVER))
    else:
        _valuta(f"Over {linea_ou}", prob_over, quotes.q_over, soglie["ou_over"],
                back_only=True, mf=_mf(SIGNALS.MOMENTUM_MKT_OVER))
    _valuta(f"Under {linea_ou}", prob_under, quotes.q_under, soglie["ou_under"],
            mf=_mf(SIGNALS.MOMENTUM_MKT_UNDER))

    # BTTS
    _valuta("BTTS Sì", prob_btts, quotes.q_btts_si, soglie["btts_si"],
            mf=_mf(SIGNALS.MOMENTUM_MKT_BTTS_SI))
    _valuta("BTTS No", 1.0 - prob_btts, quotes.q_btts_no, soglie["btts_no"],
            mf=_mf(SIGNALS.MOMENTUM_MKT_BTTS_NO))

    return _filtra_segnali_coerenti(segnali)
