"""
research.py — Ricerca autonoma contesto partita via Gemini + Google Search.

Usa Gemini 2.0 Flash con il tool google_search per cercare autonomamente:
  - Formazioni probabili e infortuni/squalifiche
  - Forma recente delle squadre (ultimi 5 risultati)
  - Head-to-head recente
  - Contesto della partita (derby, coppa, lotta salvezza, ecc.)

Restituisce aggiustamenti conservativi sui prior del modello (tot_op, ah_op)
basati sulle informazioni trovate.

Nota: richiede GEMINI_API_KEY (gratuito, 1500 req/giorno).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclass risultato ricerca
# ---------------------------------------------------------------------------

@dataclass
class RicercaPartita:
    """Risultato della ricerca autonoma sul contesto della partita."""

    squadra_casa: str = ""
    squadra_trasf: str = ""
    competizione: str = ""

    # Infortuni / squalifiche trovati
    assenze_casa: list[str] = field(default_factory=list)
    assenze_trasf: list[str] = field(default_factory=list)

    # Forma recente (es. "WDLWW" — W=vittoria, D=pareggio, L=sconfitta)
    forma_casa: str = ""
    forma_trasf: str = ""

    # Head-to-head
    h2h_sommario: str = ""        # es. "Casa 3 vittorie, 2 pareggi, Trasf 1"
    h2h_media_gol: float = 0.0    # media gol negli ultimi H2H

    # Contesto libero
    contesto: str = ""

    # Aggiustamenti suggeriti sui prior del modello
    adj_tot: float = 0.0   # aggiustamento su tot_op (es. -0.25 se striker out)
    adj_ah: float = 0.0    # aggiustamento su ah_op (es. +0.15 se casa in crisi)

    # Qualità della ricerca
    affidabilita: str = "bassa"   # "alta" / "media" / "bassa"
    note_aggiustamento: str = ""  # spiegazione degli adj

    # Fonti trovate da Gemini
    fonti: list[str] = field(default_factory=list)

    # Meta
    success: bool = False
    error: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# API key helper (riusa stesso pattern di ocr.py)
# ---------------------------------------------------------------------------

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_SEARCH_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]


def _get_gemini_api_key() -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Prompt — approccio a 2 stadi
# ---------------------------------------------------------------------------

def _build_prompt_raccolta(squadra_casa: str, squadra_trasf: str, competizione: str) -> str:
    """Stadio 1: cerca fatti in prosa (con google_search). Nessun vincolo JSON."""
    comp_str = f" ({competizione})" if competizione else ""
    return f"""Sei un analista sportivo. Cerca informazioni AGGIORNATE sulla partita:

{squadra_casa} vs {squadra_trasf}{comp_str}

Usa Google per trovare:
1. Infortuni, squalifiche, assenze confermate OGGI per entrambe le squadre
2. Forma recente (ultimi 5 risultati) di entrambe le squadre — indica W/D/L per ogni partita
3. Head-to-head recente (ultime 3-5 partite dirette): risultati, gol segnati
4. Contesto speciale: derby, lotta salvezza/titolo, campo neutro, meteo avverso

Rispondi liberamente con tutte le informazioni che trovi. Sii preciso e conciso.
"""


def _build_prompt_formato(
    fatti: str,
    squadra_casa: str,
    squadra_trasf: str,
    competizione: str,
) -> str:
    """Stadio 2: formatta i fatti come JSON (senza google_search)."""
    comp_str = f" ({competizione})" if competizione else ""
    # Tronca i fatti a 1500 chars per lasciare spazio all'output
    return f"""Dati partita {squadra_casa} vs {squadra_trasf}{comp_str}:
{fatti[:1500]}

Rispondi con SOLO questo JSON (nessun altro testo, nessun markdown):
{{"assenze_casa":[],"assenze_trasf":[],"forma_casa":"","forma_trasf":"","h2h_sommario":"","h2h_media_gol":0.0,"contesto":"","adj_tot":0.0,"adj_ah":0.0,"affidabilita":"bassa","note_aggiustamento":""}}

Compila i campi con i dati trovati. Regole:
- adj_tot: range -0.50/+0.30. Attaccante out=-0.25, derby=-0.15, nessuna info=0.0
- adj_ah: range -0.25/+0.25. Casa in crisi=+0.15, nessuna info=0.0
- affidabilita: "alta" se info recenti, "media" se generiche, "bassa" se poche
- Campi senza dati: usa [] o "" o 0.0
- Output SOLO il JSON raw, NO markdown, NO backtick
"""


# ---------------------------------------------------------------------------
# Chiamata API Gemini con Google Search
# ---------------------------------------------------------------------------

def _chiama_gemini_con_ricerca(prompt: str) -> tuple[str, list[str]]:
    """
    Stadio 1: chiama Gemini con google_search attivo.
    Restituisce (testo_prosa, lista_url_fonti).
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY non configurata")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }

    last_error = ""
    for model in _GEMINI_SEARCH_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            candidates = result.get("candidates", [])
            if not candidates:
                last_error = f"{model}: nessun candidato"
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            testo = "".join(p.get("text", "") for p in parts)

            fonti: list[str] = []
            grounding = candidates[0].get("groundingMetadata", {})
            for chunk in grounding.get("groundingChunks", []):
                uri = chunk.get("web", {}).get("uri", "")
                if uri:
                    fonti.append(uri)

            return testo, fonti

        except urllib.error.HTTPError as e:
            last_error = f"{model}: HTTP {e.code}"
            if e.code == 429:
                import time
                time.sleep(2)
            continue
        except Exception as e:
            last_error = f"{model}: {e}"
            continue

    raise RuntimeError(f"Tutti i modelli falliti. Ultimo errore: {last_error}")


def _chiama_gemini_solo_testo(prompt: str) -> str:
    """
    Stadio 2: chiama Gemini SENZA google_search per formattare in JSON.
    Senza grounding Gemini segue le istruzioni di formato molto più fedelmente.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY non configurata")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,      # quasi deterministico (0.0 non supportato da tutti i modelli)
            "maxOutputTokens": 4096,
        },
    }

    last_error = ""
    for model in _GEMINI_SEARCH_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            candidates = result.get("candidates", [])
            if not candidates:
                last_error = f"{model}: nessun candidato"
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

        except urllib.error.HTTPError as e:
            last_error = f"{model}: HTTP {e.code}"
            if e.code == 429:
                import time
                time.sleep(2)
            continue
        except Exception as e:
            last_error = f"{model}: {e}"
            continue

    raise RuntimeError(f"Tutti i modelli falliti. Ultimo errore: {last_error}")


# ---------------------------------------------------------------------------
# Helper: estrazione JSON robusta
# ---------------------------------------------------------------------------

def _estrai_json(testo: str) -> dict | None:
    """
    Estrae il primo oggetto JSON valido da una stringa di testo.

    Gemini con google_search spesso restituisce:
    - Testo libero prima del JSON
    - JSON dentro blocchi markdown (```json ... ```)
    - JSON valido ma preceduto da spiegazioni
    - Risposta completamente priva di JSON

    Strategia: prova in ordine
    1. Blocco markdown ```json ... ```
    2. Blocco markdown ``` ... ```
    3. Regex greedy {.*} sull'intera stringa
    4. Cerca tutti i candidati {.*} e prova a parsarli dal più lungo
    """
    if not testo:
        return None

    # 1. Blocco markdown ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)```", testo)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Blocco markdown generico ``` ... ```
    m = re.search(r"```\s*([\s\S]*?)```", testo)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Regex greedy: dal primo { all'ultimo }
    m = re.search(r"\{[\s\S]*\}", testo)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 4. Trova tutte le occorrenze di { e prova a parsare dal più lungo candidato
    starts = [i for i, c in enumerate(testo) if c == "{"]
    ends   = [i for i, c in enumerate(testo) if c == "}"]
    candidati = sorted(
        [(s, e) for s in starts for e in ends if e > s],
        key=lambda se: se[1] - se[0],
        reverse=True,
    )
    for s, e in candidati[:10]:  # prova i 10 candidati più lunghi
        try:
            return json.loads(testo[s:e+1])
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Parser risposta JSON
# ---------------------------------------------------------------------------

def _parse_risposta(testo: str, fonti: list[str], squadra_casa: str, squadra_trasf: str, competizione: str) -> RicercaPartita:
    """Estrae il JSON dalla risposta di Gemini e costruisce RicercaPartita."""

    dati = _estrai_json(testo)
    if dati is None:
        return RicercaPartita(
            squadra_casa=squadra_casa,
            squadra_trasf=squadra_trasf,
            competizione=competizione,
            success=False,
            error="Risposta non contiene JSON valido",
            raw_response=testo[:500],
        )

    # Clamp aggiustamenti entro range sicuri
    adj_tot = max(-0.50, min(0.30, float(dati.get("adj_tot", 0.0))))
    adj_ah = max(-0.25, min(0.25, float(dati.get("adj_ah", 0.0))))

    return RicercaPartita(
        squadra_casa=squadra_casa,
        squadra_trasf=squadra_trasf,
        competizione=competizione,
        assenze_casa=dati.get("assenze_casa", []),
        assenze_trasf=dati.get("assenze_trasf", []),
        forma_casa=dati.get("forma_casa", ""),
        forma_trasf=dati.get("forma_trasf", ""),
        h2h_sommario=dati.get("h2h_sommario", ""),
        h2h_media_gol=float(dati.get("h2h_media_gol", 0.0)),
        contesto=dati.get("contesto", ""),
        adj_tot=adj_tot,
        adj_ah=adj_ah,
        affidabilita=dati.get("affidabilita", "bassa"),
        note_aggiustamento=dati.get("note_aggiustamento", ""),
        fonti=fonti[:5],  # max 5 fonti
        success=True,
        raw_response=testo[:1000],
    )


# ---------------------------------------------------------------------------
# Funzione pubblica
# ---------------------------------------------------------------------------

def ricerca_contesto_partita(
    squadra_casa: str,
    squadra_trasf: str,
    competizione: str = "",
) -> RicercaPartita:
    """
    Esegue ricerca autonoma sul contesto della partita via Gemini + Google Search.

    Args:
        squadra_casa: Nome squadra di casa (es. "Arsenal").
        squadra_trasf: Nome squadra trasferta (es. "Chelsea").
        competizione: Competizione opzionale (es. "Premier League").

    Returns:
        RicercaPartita con tutte le informazioni trovate e gli aggiustamenti suggeriti.
    """
    if not squadra_casa.strip() or not squadra_trasf.strip():
        return RicercaPartita(
            success=False,
            error="Inserisci i nomi di entrambe le squadre.",
        )

    api_key = _get_gemini_api_key()
    if not api_key:
        return RicercaPartita(
            success=False,
            error="GEMINI_API_KEY non configurata. Aggiungila nei secrets di Streamlit.",
        )

    sc = squadra_casa.strip()
    st_t = squadra_trasf.strip()
    comp = competizione.strip()
    _raw_stage1 = ""
    _raw_stage2 = ""

    try:
        # Stadio 1: raccolta fatti con google_search (risposta in prosa libera)
        prompt_raccolta = _build_prompt_raccolta(sc, st_t, comp)
        _raw_stage1, fonti = _chiama_gemini_con_ricerca(prompt_raccolta)

        # Stadio 2: formattazione JSON senza google_search (Gemini segue le istruzioni)
        prompt_formato = _build_prompt_formato(_raw_stage1, sc, st_t, comp)
        _raw_stage2 = _chiama_gemini_solo_testo(prompt_formato)

        return _parse_risposta(_raw_stage2, fonti, sc, st_t, comp)
    except Exception as e:
        return RicercaPartita(
            squadra_casa=squadra_casa,
            squadra_trasf=squadra_trasf,
            competizione=competizione,
            success=False,
            error=str(e),
            raw_response=f"[STADIO 1]\n{_raw_stage1[:500]}\n\n[STADIO 2]\n{_raw_stage2[:500]}",
        )


# ---------------------------------------------------------------------------
# Feature A: Validatore quote OCR
# ---------------------------------------------------------------------------

def valida_quote_ocr(
    squadra_casa: str,
    squadra_trasf: str,
    quota_1: float,
    quota_x: float,
    quota_2: float,
    quota_over: float,
    quota_under: float,
    linea_ou: float = 2.5,
) -> dict:
    """
    Valida le quote estratte da OCR chiedendo a Gemini di cercare le quote
    di mercato attuali per questa partita e confrontarle con quelle estratte.

    Se le quote OCR si discostano significativamente dal mercato reale,
    restituisce un confidence_scale ridotto che abbassa il peso dell'OCR
    nel blend Bayesiano.

    Returns:
        Dict con:
          - confidence_scale: float [0.70, 1.0]. 1.0 = quote verificate.
          - flags: list[str]. Descrizione delle anomalie trovate.
    """
    default = {"confidence_scale": 1.0, "flags": []}

    if not squadra_casa.strip() or not squadra_trasf.strip():
        return default

    api_key = _get_gemini_api_key()
    if not api_key:
        return default

    quote_str = ""
    if quota_1 > 1.0:
        quote_str += f"1X2: {quota_1:.2f} / {quota_x:.2f} / {quota_2:.2f}. "
    if quota_over > 1.0:
        quote_str += f"O/U {linea_ou:.1f}: {quota_over:.2f} / {quota_under:.2f}. "

    if not quote_str:
        return default  # niente da validare

    prompt_cerca = (
        f"Cerca le quote bookmaker attuali (Bet365, Pinnacle, William Hill) per la partita "
        f"{squadra_casa} vs {squadra_trasf}. "
        f"Quote da verificare — {quote_str} "
        f"Descrivi le quote che trovi online e confrontale con quelle da verificare."
    )
    prompt_json = f"""Hai trovato queste informazioni sulle quote per {squadra_casa} vs {squadra_trasf}:

---
{{FATTI}}
---

Quote originali da verificare: {quote_str}

Compila SOLO questo JSON:
{{"confidence_scale": 0.95, "flags": ["anomalia se presente"], "quote_trovate": "descrizione"}}

confidence_scale: 1.0=concordi(<5%), 0.90=diff 5-10%, 0.80=diff 10-20%, 0.70=molto diverse o non trovate.
Se non trovate usa confidence_scale=0.90 e flags=[]. Output SOLO il JSON raw, NO markdown, NO backtick.
"""

    try:
        fatti, _ = _chiama_gemini_con_ricerca(prompt_cerca)
        testo = _chiama_gemini_solo_testo(prompt_json.replace("{FATTI}", fatti[:2000]))
        dati = _estrai_json(testo)
        if dati is None:
            return default
        cs = float(dati.get("confidence_scale", 1.0))
        cs = max(0.70, min(1.0, cs))
        flags = [str(f) for f in dati.get("flags", []) if str(f).strip()]
        return {"confidence_scale": cs, "flags": flags}
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Feature B: Prior storico H2H
# ---------------------------------------------------------------------------

def cerca_prior_storico(
    squadra_casa: str,
    squadra_trasf: str,
    competizione: str = "",
) -> float:
    """
    Cerca la media storica di gol nelle partite dirette tra le due squadre
    tramite Gemini + Google Search.

    Usata come prior aggiuntivo (10%) nel blend del totale atteso in prematch.

    Returns:
        float: Media gol negli ultimi H2H (0.0 se non trovata o non affidabile).
    """
    if not squadra_casa.strip() or not squadra_trasf.strip():
        return 0.0

    api_key = _get_gemini_api_key()
    if not api_key:
        return 0.0

    comp_str = f" in {competizione}" if competizione else ""
    prompt_cerca = (
        f"Cerca i risultati delle ultime 5-10 partite dirette (head-to-head) tra "
        f"{squadra_casa} e {squadra_trasf}{comp_str}. "
        f"Per ogni partita indica il risultato e i gol totali segnati."
    )
    prompt_json = f"""Hai trovato questi dati H2H per {squadra_casa} vs {squadra_trasf}:

---
{{FATTI}}
---

Calcola la media gol e compila SOLO questo JSON:
{{"media_gol": 2.4, "n_partite": 6, "affidabilita": "alta", "note": "descrizione"}}

affidabilita: "alta"=5+ partite recenti, "media"=3-4, "bassa"=meno di 3 o dati vecchi.
Se non hai dati usa media_gol=0.0 e n_partite=0. Output SOLO il JSON raw, NO markdown, NO backtick.
"""

    try:
        fatti, _ = _chiama_gemini_con_ricerca(prompt_cerca)
        testo = _chiama_gemini_solo_testo(prompt_json.replace("{FATTI}", fatti[:2000]))
        dati = _estrai_json(testo)
        if dati is None:
            return 0.0
        media = float(dati.get("media_gol", 0.0))
        n_partite = int(dati.get("n_partite", 0))
        affidabilita = str(dati.get("affidabilita", "bassa"))
        if n_partite < 3 or affidabilita == "bassa" or media <= 0.5:
            return 0.0
        return max(0.5, min(6.0, media))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Feature C: Interpretazione movimento linee
# ---------------------------------------------------------------------------

def interpreta_movimento_linee(
    squadra_casa: str,
    squadra_trasf: str,
    delta_ah: float,
    delta_tot: float,
    competizione: str = "",
) -> float:
    """
    Chiede a Gemini di cercare il motivo del movimento delle linee (AH/Total)
    per questa partita e restituisce un moltiplicatore di qualità per w_cur.

    Il moltiplicatore indica quanto fidarsi del movimento osservato:
    - > 1.0: movimento affidabile (sharp money, notizie ufficiali) → w_cur aumenta
    - < 1.0: movimento inaffidabile (liquidità bassa, public betting) → w_cur ridotto
    - = 1.0: nessuna informazione trovata o movimento ambiguo

    Args:
        delta_ah: Variazione pura AH (full-game corrente - apertura).
        delta_tot: Variazione pura Total (full-game corrente - apertura).

    Returns:
        float: Moltiplicatore qualità [0.80, 1.30]. Default 1.0.
    """
    if not squadra_casa.strip() or not squadra_trasf.strip():
        return 1.0

    api_key = _get_gemini_api_key()
    if not api_key:
        return 1.0

    # Movimento minimo per essere significativo
    if abs(delta_ah) < 0.15 and abs(delta_tot) < 0.15:
        return 1.0  # nessun movimento rilevante, nessuna call API

    comp_str = f" ({competizione})" if competizione else ""
    movimento_str = ""
    if abs(delta_ah) >= 0.15:
        direzione = "verso casa" if delta_ah < 0 else "verso trasferta"
        movimento_str += f"AH si è mosso di {delta_ah:+.2f} ({direzione}). "
    if abs(delta_tot) >= 0.15:
        direzione = "al rialzo" if delta_tot > 0 else "al ribasso"
        movimento_str += f"Total si è mosso di {delta_tot:+.2f} ({direzione}). "

    prompt_cerca = (
        f"Cerca notizie recenti su {squadra_casa} vs {squadra_trasf}{comp_str}: "
        f"infortuni, squalifiche, comunicati ufficiali. "
        f"Movimento linee osservato: {movimento_str} "
        f"Spiega se ci sono notizie che giustificherebbero questo movimento."
    )
    prompt_json = f"""Per la partita {squadra_casa} vs {squadra_trasf}, movimento {movimento_str}

Notizie trovate:
---
{{FATTI}}
---

Compila SOLO questo JSON:
{{"movement_quality": 1.0, "tipo": "incerto", "motivo": "spiegazione"}}

movement_quality: sharp/news confermati=1.15-1.30, parziali=1.05-1.15, ambiguo=0.95-1.05,
public/nessuna info=0.85-0.95, nessuna info trovata=1.0. Output SOLO il JSON raw, NO markdown, NO backtick.
"""

    try:
        fatti, _ = _chiama_gemini_con_ricerca(prompt_cerca)
        testo = _chiama_gemini_solo_testo(prompt_json.replace("{FATTI}", fatti[:2000]))
        dati = _estrai_json(testo)
        if dati is None:
            return 1.0
        mq = float(dati.get("movement_quality", 1.0))
        return max(0.80, min(1.30, mq))
    except Exception:
        return 1.0
