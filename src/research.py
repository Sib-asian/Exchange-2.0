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
_GEMINI_SEARCH_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash"]


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
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(squadra_casa: str, squadra_trasf: str, competizione: str) -> str:
    comp_str = f" ({competizione})" if competizione else ""
    return f"""Sei un analista sportivo. Cerca informazioni AGGIORNATE sulla partita:

{squadra_casa} vs {squadra_trasf}{comp_str}

Cerca autonomamente queste informazioni usando Google:
1. Infortuni, squalifiche, assenze confermate per entrambe le squadre OGGI
2. Forma recente (ultimi 5 risultati) di entrambe le squadre
3. Head-to-head recente (ultime 3-5 partite dirette): risultati e gol
4. Eventuale contesto speciale (derby, finale, lotta salvezza/titolo, campo neutro, meteo avverso)

Dopo aver cercato, restituisci ESCLUSIVAMENTE un JSON valido con questa struttura esatta:

{{
  "assenze_casa": ["Nome giocatore (ruolo) - motivo", ...],
  "assenze_trasf": ["Nome giocatore (ruolo) - motivo", ...],
  "forma_casa": "WDLWW",
  "forma_trasf": "LLWDW",
  "h2h_sommario": "Descrizione breve degli ultimi H2H",
  "h2h_media_gol": 2.4,
  "contesto": "Descrizione contesto partita in 1-2 frasi",
  "adj_tot": -0.20,
  "adj_ah": 0.10,
  "affidabilita": "alta",
  "note_aggiustamento": "Spiegazione in italiano del perché questi aggiustamenti"
}}

REGOLE PER GLI AGGIUSTAMENTI (adj_tot e adj_ah):
- adj_tot: quanto modificare il totale atteso. Range: -0.50 a +0.30
  * Centravanti titolare out → -0.20 a -0.30
  * Entrambi gli attaccanti principali out → -0.35 a -0.50
  * Derby / partita tesa → -0.10 a -0.20
  * Squadra che pressa alto vs difesa passiva → +0.10 a +0.20
  * Se nessuna info rilevante → 0.0
- adj_ah: quanto modificare l'handicap asiatico apertura. Range: -0.25 a +0.25
  * Casa in crisi di risultati (3+ sconfitte) → +0.15 (favorisce meno la casa)
  * Trasferta senza vittorie in 5 → -0.10 (favorisce più la casa)
  * Se nessuna info rilevante → 0.0
- affidabilita: "alta" se hai trovato info fresche (max 48h), "media" se generiche, "bassa" se poco trovato

IMPORTANTE:
- Se non trovi informazioni per un campo, usa lista vuota [] o stringa vuota ""
- h2h_media_gol = 0.0 se non trovi dati H2H
- Non inventare nulla — usa solo ciò che hai trovato realmente
- Restituisci SOLO il JSON, niente altro testo prima o dopo
"""


# ---------------------------------------------------------------------------
# Chiamata API Gemini con Google Search
# ---------------------------------------------------------------------------

def _chiama_gemini_con_ricerca(prompt: str) -> tuple[str, list[str]]:
    """
    Chiama Gemini con tool google_search attivo.
    Restituisce (testo_risposta, lista_url_fonti).
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY non configurata")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.1,    # bassa: vogliamo fatti, non creatività
            "maxOutputTokens": 1024,
        },
    }

    last_error = ""
    for model in _GEMINI_SEARCH_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Estrai testo risposta
            candidates = result.get("candidates", [])
            if not candidates:
                last_error = f"{model}: nessun candidato nella risposta"
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            testo = "".join(p.get("text", "") for p in parts)

            # Estrai fonti dal grounding metadata
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


# ---------------------------------------------------------------------------
# Parser risposta JSON
# ---------------------------------------------------------------------------

def _parse_risposta(testo: str, fonti: list[str], squadra_casa: str, squadra_trasf: str, competizione: str) -> RicercaPartita:
    """Estrae il JSON dalla risposta di Gemini e costruisce RicercaPartita."""

    # Cerca blocco JSON nella risposta (può esserci testo prima/dopo)
    json_match = re.search(r"\{[\s\S]*\}", testo)
    if not json_match:
        return RicercaPartita(
            squadra_casa=squadra_casa,
            squadra_trasf=squadra_trasf,
            competizione=competizione,
            success=False,
            error="Risposta non contiene JSON valido",
            raw_response=testo[:500],
        )

    try:
        dati = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return RicercaPartita(
            squadra_casa=squadra_casa,
            squadra_trasf=squadra_trasf,
            competizione=competizione,
            success=False,
            error=f"JSON malformato: {e}",
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

    try:
        prompt = _build_prompt(squadra_casa.strip(), squadra_trasf.strip(), competizione.strip())
        testo, fonti = _chiama_gemini_con_ricerca(prompt)
        return _parse_risposta(testo, fonti, squadra_casa.strip(), squadra_trasf.strip(), competizione.strip())
    except Exception as e:
        return RicercaPartita(
            squadra_casa=squadra_casa,
            squadra_trasf=squadra_trasf,
            competizione=competizione,
            success=False,
            error=str(e),
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

    prompt = f"""Sei un validatore di quote. Cerca le quote bookmaker ATTUALI per la partita:

{squadra_casa} vs {squadra_trasf}

Quote estratte da screenshot (da validare):
{quote_str}

Cerca su Google le quote correnti di almeno un bookmaker affidabile (Bet365, Pinnacle, William Hill, Betfair).
Confronta le quote trovate con quelle estratte.

Restituisci SOLO un JSON valido:
{{
  "confidence_scale": 0.95,
  "flags": ["Lista di anomalie, vuota se tutto ok"],
  "quote_trovate": "Descrizione breve delle quote trovate online"
}}

REGOLE per confidence_scale:
- 1.0: Quote concordi (differenza < 5%)
- 0.90: Differenza 5-10% su almeno una quota
- 0.80: Differenza 10-20% o quote sospette
- 0.70: Quote molto diverse o non trovate

Se non riesci a trovare le quote online, usa confidence_scale=0.90 e flags=[].
Restituisci SOLO il JSON, nient'altro.
"""

    try:
        testo, _ = _chiama_gemini_con_ricerca(prompt)
        match = re.search(r"\{[\s\S]*\}", testo)
        if not match:
            return default
        dati = json.loads(match.group())
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
    prompt = f"""Cerca la media storica di gol nelle partite dirette (head-to-head) tra:

{squadra_casa} vs {squadra_trasf}{comp_str}

Cerca le ultime 5-10 partite dirette tra queste squadre.
Calcola la media di gol totali (home + away) per partita.

Restituisci SOLO un JSON valido:
{{
  "media_gol": 2.4,
  "n_partite": 6,
  "affidabilita": "alta",
  "note": "Breve descrizione dei dati trovati"
}}

REGOLE:
- media_gol: media gol totali per partita (0.0 se non trovata)
- n_partite: numero di partite analizzate (0 se dati non trovati)
- affidabilita: "alta" (≥5 partite recenti), "media" (3-4), "bassa" (<3 o dati vecchi)
- Se non trovi dati affidabili, usa media_gol=0.0 e n_partite=0
- Restituisci SOLO il JSON, nient'altro
"""

    try:
        testo, _ = _chiama_gemini_con_ricerca(prompt)
        match = re.search(r"\{[\s\S]*\}", testo)
        if not match:
            return 0.0
        dati = json.loads(match.group())
        media = float(dati.get("media_gol", 0.0))
        n_partite = int(dati.get("n_partite", 0))
        affidabilita = str(dati.get("affidabilita", "bassa"))
        # Restituisce 0.0 se i dati sono insufficienti o inaffidabili
        if n_partite < 3 or affidabilita == "bassa" or media <= 0.5:
            return 0.0
        return max(0.5, min(6.0, media))  # clamp a range plausibile
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

    prompt = f"""Sei un analista di movimenti di mercato scommesse. Analizza il movimento delle linee per:

{squadra_casa} vs {squadra_trasf}{comp_str}

Movimento osservato (dalla quota di apertura all'attuale):
{movimento_str}

Cerca su Google le notizie RECENTI su questa partita:
- Infortuni/squalifiche che spiegherebbero il movimento
- Comunicati ufficiali delle squadre
- Analisi di esperti o tipster di fama

Determina se il movimento è:
- "sharp": causato da scommettitori professionali o notizie fondamentali (alta qualità)
- "public": causato da scommesse del grande pubblico (bassa qualità segnale)
- "news": causato da notizie ufficiali verificabili (alta qualità)
- "incerto": motivo non chiaro

Restituisci SOLO un JSON valido:
{{
  "movement_quality": 1.10,
  "tipo": "sharp",
  "motivo": "Breve spiegazione in italiano",
  "confidence": "alta"
}}

REGOLE per movement_quality:
- Movimento sharp confermato o notizie ufficiali: 1.15-1.30
- Notizie riportate ma non confermate: 1.05-1.15
- Movimento ambiguo o misto: 0.95-1.05
- Probabile public betting senza motivazione fondamentale: 0.85-0.95
- Nessuna informazione trovata → usa 1.0

Restituisci SOLO il JSON, nient'altro.
"""

    try:
        testo, _ = _chiama_gemini_con_ricerca(prompt)
        match = re.search(r"\{[\s\S]*\}", testo)
        if not match:
            return 1.0
        dati = json.loads(match.group())
        mq = float(dati.get("movement_quality", 1.0))
        return max(0.80, min(1.30, mq))
    except Exception:
        return 1.0
