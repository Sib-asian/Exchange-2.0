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
