"""
ocr.py — Estrazione automatica dati da screenshot di siti scommesse.

Supporta multipli backend per l'analisi immagini (in ordine di preferenza):
  1. z-ai CLI (gratuito, locale)
  2. Google Gemini API (gratuito 1500 req/giorno)
  3. OpenAI Vision API (a pagamento, fallback finale)

Utilizza il VLM (Vision Language Model) per leggere immagini da:
  - Siti di scommesse (Bet365, Betfair, Pinnacle, ecc.)
  - App mobili
  - Desktop screenshot

Estrae automaticamente:
  - Nomi delle squadre
  - Quote 1X2
  - Quote Over/Under
  - Quote BTTS (GG/NG)

Configurazione:
  - GEMINI_API_KEY: via st.secrets o variabile d'ambiente (GRATUITO)
  - OPENAI_API_KEY: via st.secrets o variabile d'ambiente (a pagamento)

Il modulo restituisce i dati in formato strutturato per l'uso nell'UI Streamlit.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# OpenAI import con fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class ExtractedData:
    """Dati estratti da uno screenshot di scommesse."""

    # Squadre
    squadra_casa: str = ""
    squadra_trasf: str = ""

    # Quote 1X2
    quota_1: float = 0.0
    quota_x: float = 0.0
    quota_2: float = 0.0

    # Quote Over/Under
    linea_ou: float = 0.0
    quota_over: float = 0.0
    quota_under: float = 0.0

    # Quote BTTS (Goal/No Goal)
    quota_gg: float = 0.0  # Goal Goal (BTTS Sì)
    quota_ng: float = 0.0  # No Goal (BTTS No)

    # Metadati estrazione
    raw_response: str = ""
    extraction_success: bool = False
    error_message: str = ""
    confidence: str = "medium"  # low, medium, high
    backend_used: str = ""  # zai_cli, gemini, openai

    def to_dict(self) -> dict[str, Any]:
        """Converte in dizionario per compatibilità Streamlit."""
        return {
            "squadra_casa": self.squadra_casa,
            "squadra_trasf": self.squadra_trasf,
            "quota_1": self.quota_1,
            "quota_x": self.quota_x,
            "quota_2": self.quota_2,
            "linea_ou": self.linea_ou,
            "quota_over": self.quota_over,
            "quota_under": self.quota_under,
            "quota_gg": self.quota_gg,
            "quota_ng": self.quota_ng,
            "extraction_success": self.extraction_success,
            "error_message": self.error_message,
            "confidence": self.confidence,
            "backend_used": self.backend_used,
        }

@dataclass
class LiveStatsExtracted:
    """Statistiche live estratte da screenshot Nowgoal/simili."""

    # Stato partita
    minuto: int = 0
    gol_casa: int = 0
    gol_trasf: int = 0

    # Cartellini
    rossi_casa: int = 0
    rossi_trasf: int = 0
    gialli_casa: int = 0
    gialli_trasf: int = 0

    # Tiri
    tiri_porta_casa: int = 0  # Shots on target
    tiri_porta_trasf: int = 0
    tiri_fuori_casa: int = 0  # Shots off target
    tiri_fuori_trasf: int = 0
    tiri_bloccati_casa: int = 0  # Blocked shots
    tiri_bloccati_trasf: int = 0

    # Corner
    corner_casa: int = 0
    corner_trasf: int = 0

    # Possesso
    possesso_casa: float = 0.0  # Percentuale (0-100)
    possesso_trasf: float = 0.0

    # Attacchi
    attacchi_casa: int = 0
    attacchi_trasf: int = 0
    attacchi_pericolosi_casa: int = 0  # Dangerous attacks
    attacchi_pericolosi_trasf: int = 0

    # Falli
    falli_casa: int = 0
    falli_trasf: int = 0

    # Cronologia eventi estratta dalla pagina
    # Ogni elemento: {"t": "goal|yellow|red|sub", "sq": "h|a", "pl": "nome", "min": int}
    eventi: list = field(default_factory=list)

    # Metadati
    raw_response: str = ""
    extraction_success: bool = False
    error_message: str = ""
    confidence: str = "medium"
    backend_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converte in dizionario."""
        return {
            "minuto": self.minuto,
            "gol_casa": self.gol_casa,
            "gol_trasf": self.gol_trasf,
            "rossi_casa": self.rossi_casa,
            "rossi_trasf": self.rossi_trasf,
            "gialli_casa": self.gialli_casa,
            "gialli_trasf": self.gialli_trasf,
            "tiri_porta_casa": self.tiri_porta_casa,
            "tiri_porta_trasf": self.tiri_porta_trasf,
            "tiri_fuori_casa": self.tiri_fuori_casa,
            "tiri_fuori_trasf": self.tiri_fuori_trasf,
            "tiri_bloccati_casa": self.tiri_bloccati_casa,
            "tiri_bloccati_trasf": self.tiri_bloccati_trasf,
            "corner_casa": self.corner_casa,
            "corner_trasf": self.corner_trasf,
            "possesso_casa": self.possesso_casa,
            "possesso_trasf": self.possesso_trasf,
            "attacchi_casa": self.attacchi_casa,
            "attacchi_trasf": self.attacchi_trasf,
            "attacchi_pericolosi_casa": self.attacchi_pericolosi_casa,
            "attacchi_pericolosi_trasf": self.attacchi_pericolosi_trasf,
            "falli_casa": self.falli_casa,
            "falli_trasf": self.falli_trasf,
            "extraction_success": self.extraction_success,
            "confidence": self.confidence,
            "backend_used": self.backend_used,
        }


# Prompt per l'estrazione dati dallo screenshot
EXTRACTION_PROMPT = """Analizza questo screenshot di un sito di scommesse o app betting.

Estrai i seguenti dati e restituiscili in formato JSON esattamente come mostrato:

{
    "squadra_casa": "Nome squadra casa",
    "squadra_trasf": "Nome squadra trasferta",
    "quota_1": 0.00,
    "quota_x": 0.00,
    "quota_2": 0.00,
    "linea_ou": 0.0,
    "quota_over": 0.00,
    "quota_under": 0.00,
    "quota_gg": 0.00,
    "quota_ng": 0.00,
    "confidence": "high/medium/low"
}

REGOLE DI ESTRAZIONE:

1. NOMI SQUADRE:
   - Cerca i nomi delle due squadre che si affrontano
   - Solitamente mostrati come "Squadra A vs Squadra B" o "Squadra A - Squadra B"
   - La prima è la squadra di CASA, la seconda è la TRASFERTA

2. QUOTE 1X2:
   - 1 = vittoria casa
   - X = pareggio
   - 2 = vittoria trasferta
   - Cerca valori tipici tra 1.01 e 50.00

3. QUOTE OVER/UNDER:
   - Cerca la linea (solitamente 0.5, 1.5, 2.5, 3.5)
   - Over = più gol della linea
   - Under = meno gol della linea
   - Se ci sono più linee, scegli la principale (solitamente 2.5)

4. QUOTE BTTS (GG/NG):
   - GG = Goal Goal = entrambe segnano = BTTS Sì
   - NG = No Goal = almeno una non segna = BTTS No
   - Può anche essere chiamato "Both Teams to Score" o "Entrambe segnano"

5. CONFIDENCE:
   - "high" = tutti i dati chiaramente visibili e leggibili
   - "medium" = alcuni dati parziali o poco chiari
   - "low" = immagine sfocata, tagliata o dati molto incerti

SE UN DATO NON È PRESENTE O NON È LEGGIBILE:
- Imposta il valore numerico a 0.0
- Imposta le stringhe a ""
- Abbassa la confidence di conseguenza

IMPORTANTE: Restituisci SOLO il JSON, nessun altro testo prima o dopo."""

# ============================================================================
# Backend: z-ai CLI
# ============================================================================

def _get_env_with_path() -> dict[str, str]:
    """Restituisce environment con PATH aggiornato."""
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    extra_paths = [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/home/z/.bun/bin",
        os.path.expanduser("~/.bun/bin"),
    ]
    for p in extra_paths:
        if p not in current_path:
            current_path = f"{p}:{current_path}"
    env["PATH"] = current_path
    return env

def _find_zai_command() -> tuple[str | None, list[str] | None]:
    """Trova il modo di eseguire z-ai CLI."""
    zai_path = shutil.which("z-ai")
    if zai_path:
        return zai_path, None

    absolute_paths = [
        "/usr/local/bin/z-ai",
        "/usr/bin/z-ai",
        os.path.expanduser("~/.bun/bin/z-ai"),
    ]
    for path in absolute_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path, None

    bun_path = shutil.which("bun")
    if bun_path:
        cli_path = os.path.expanduser("~/.bun/install/global/node_modules/z-ai-web-dev-sdk/dist/cli.js")
        if os.path.isfile(cli_path):
            return bun_path, [cli_path]

    node_path = shutil.which("node")
    if node_path:
        cli_path = os.path.expanduser("~/.bun/install/global/node_modules/z-ai-web-dev-sdk/dist/cli.js")
        if os.path.isfile(cli_path):
            return node_path, [cli_path]

    return None, None

def _check_zai_available() -> bool:
    """Verifica se z-ai è disponibile."""
    executable, _ = _find_zai_command()
    return executable is not None

def _extract_with_zai_cli(image_path: Path) -> ExtractedData:
    """Estrae dati usando z-ai CLI."""
    executable, extra_args = _find_zai_command()
    if executable is None:
        return ExtractedData(extraction_success=False, error_message="z-ai CLI non disponibile")

    if extra_args:
        cmd = [executable] + extra_args + ["vision", "-p", EXTRACTION_PROMPT, "-i", str(image_path)]
    else:
        cmd = [executable, "vision", "-p", EXTRACTION_PROMPT, "-i", str(image_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90, env=_get_env_with_path())
        if result.returncode != 0:
            return ExtractedData(extraction_success=False, error_message=f"z-ai CLI: {result.stderr or 'Errore'}")
        parsed = _parse_vlm_response(result.stdout)
        parsed.backend_used = "zai_cli"
        return parsed
    except subprocess.TimeoutExpired:
        return ExtractedData(extraction_success=False, error_message="z-ai CLI: timeout")
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"z-ai CLI: {e}")

# ============================================================================
# Backend: Google Gemini API
# ============================================================================

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",      # quota separata da 2.0-flash
    "gemini-1.5-flash-latest",    # fallback legacy
]
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

def _get_gemini_api_key() -> str | None:
    """Ottiene la API key di Gemini da environment o Streamlit secrets."""
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

def _check_gemini_available() -> bool:
    """Verifica se Gemini API è disponibile."""
    return _get_gemini_api_key() is not None

def _extract_with_gemini(image_path: Path) -> ExtractedData:
    """Estrae dati usando Google Gemini API con retry e fallback tra modelli."""
    import time

    api_key = _get_gemini_api_key()
    if not api_key:
        return ExtractedData(extraction_success=False, error_message="Gemini: API key non configurata")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"Gemini: errore lettura file: {e}")

    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    request_body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                {"text": EXTRACTION_PROMPT},
            ],
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }
    payload = json.dumps(request_body).encode("utf-8")

    last_error = ""
    for model in _GEMINI_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        # Retry con backoff per rate limit (429)
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    response_data = json.loads(response.read().decode("utf-8"))

                if "candidates" in response_data and response_data["candidates"]:
                    parts = response_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts:
                        text_response = parts[0].get("text", "")
                        if text_response:
                            parsed = _parse_vlm_response(text_response)
                            parsed.backend_used = "gemini"
                            return parsed
                last_error = f"Gemini ({model}): risposta vuota"
                break  # Risposta vuota, prova prossimo modello
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s
                    continue
                # Leggi dettagli errore per diagnostica
                error_detail = ""
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_json = json.loads(error_body)
                    error_detail = error_json.get("error", {}).get("message", "")
                except Exception:
                    pass
                detail_suffix = f" - {error_detail}" if error_detail else ""
                last_error = f"Gemini ({model}): HTTP {e.code}{detail_suffix}"
                break  # Prova il prossimo modello
            except Exception as e:
                last_error = f"Gemini ({model}): {e}"
                break

    return ExtractedData(extraction_success=False, error_message=last_error)

# ============================================================================
# Backend: OpenAI Vision API
# ============================================================================

_DEFAULT_MODEL = "gpt-4o-mini"

def _get_openai_api_key() -> str | None:
    """Ottiene la API key OpenAI da environment o Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except ImportError:
        pass
    return None

def _get_openai_model() -> str:
    """Ottiene il modello OpenAI configurato."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
            return st.secrets["OPENAI_MODEL"]
    except ImportError:
        pass
    return os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)

def _check_openai_available() -> bool:
    """Verifica se OpenAI è disponibile."""
    return OPENAI_AVAILABLE and _get_openai_api_key() is not None

def _extract_with_openai(image_path: Path) -> ExtractedData:
    """Estrae dati usando OpenAI Vision API."""
    if not OPENAI_AVAILABLE:
        return ExtractedData(extraction_success=False, error_message="OpenAI: libreria non installata")

    api_key = _get_openai_api_key()
    if not api_key:
        return ExtractedData(extraction_success=False, error_message="OpenAI: API key non configurata")

    try:
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/png")
        image_b64 = base64.b64encode(image_bytes).decode()

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=_get_openai_model(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}", "detail": "high"}}
                ]
            }],
            max_tokens=1000,
            temperature=0.1
        )
        text_response = response.choices[0].message.content or ""
        parsed = _parse_vlm_response(text_response)
        parsed.backend_used = "openai"
        return parsed
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"OpenAI: {e}")

# ============================================================================
# Live Stats Extraction (Nowgoal/simili)
# ============================================================================

LIVE_STATS_PROMPT = """Analizza questo screenshot di partita di calcio live (Nowgoal o sito simile).
Sinistra=casa(h), destra=trasferta(a).

REGOLE:
- min: minuto attuale. Se "FT"/"Full Time"→90. Se "HT"/"Half Time"→45. Se "32'"→32. Se "1st Half - 32"→32.
- g_h/g_a: punteggio attuale (es. "1 - 0" → g_h=1, g_a=0)
- r_h/r_a: conta i cartellini rossi nella cronologia eventi (rosso=espulsione diretta o 2° giallo)
- y_h/y_a: conta i cartellini gialli nella cronologia eventi (NON contare il 2° giallo che ha causato rosso)
- Shots on Goal/Target=sot, Shots off Goal/Target=soff, Blocked=blk
- Se vedi solo "Shots" totali (non distinti): soff=shots-sot
- pos_h/pos_a: Possession % (es. "70%" → 70)
- att_h/att_a: Total Attacks; datt_h/datt_a: Dangerous Attacks
- ev: cronologia eventi visibile nella pagina. Per ogni evento rilevante includi:
  {"t":"goal|yellow|red|sub","sq":"h|a","pl":"nome giocatore","min":numero}
  t=goal (gol), t=yellow (giallo), t=red (rosso/espulsione), t=sub (sostituzione)
  sq="h" se evento a sinistra/casa, sq="a" se a destra/trasferta
  Se non c'è cronologia visibile: ev=[]

Rispondi SOLO con JSON valido (0 se dato non visibile):
{"min":0,"g_h":0,"g_a":0,"r_h":0,"r_a":0,"y_h":0,"y_a":0,"sot_h":0,"sot_a":0,"soff_h":0,"soff_a":0,"blk_h":0,"blk_a":0,"cor_h":0,"cor_a":0,"pos_h":0,"pos_a":0,"att_h":0,"att_a":0,"datt_h":0,"datt_a":0,"fou_h":0,"fou_a":0,"ev":[]}"""


def _extract_live_stats_with_gemini(image_path: Path) -> LiveStatsExtracted:
    """Estrae statistiche live usando Google Gemini API."""
    import time

    api_key = _get_gemini_api_key()
    if not api_key:
        return LiveStatsExtracted(
            extraction_success=False,
            error_message="Gemini: API key non configurata",
        )

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        return LiveStatsExtracted(
            extraction_success=False,
            error_message=f"Gemini: errore lettura file: {e}",
        )

    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    request_body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                {"text": LIVE_STATS_PROMPT},
            ],
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "response_mime_type": "application/json",
        },
    }
    payload = json.dumps(request_body).encode("utf-8")

    last_error = ""
    for model in _GEMINI_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    response_data = json.loads(response.read().decode("utf-8"))

                if "candidates" in response_data and response_data["candidates"]:
                    parts = response_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts:
                        text_response = parts[0].get("text", "")
                        if text_response:
                            parsed = _parse_live_stats_response(text_response)
                            parsed.backend_used = "gemini"
                            return parsed
                last_error = f"Gemini ({model}): risposta vuota"
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                error_detail = ""
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_json = json.loads(error_body)
                    error_detail = error_json.get("error", {}).get("message", "")
                except Exception:
                    pass
                detail_suffix = f" - {error_detail}" if error_detail else ""
                last_error = f"Gemini ({model}): HTTP {e.code}{detail_suffix}"
                break
            except Exception as e:
                last_error = f"Gemini ({model}): {e}"
                break

    return LiveStatsExtracted(extraction_success=False, error_message=last_error)


# Mapping flessibile: chiave alternativa → chiave standard
# Gemini può restituire nomi in inglese, abbreviati o con varianti
_LIVE_STATS_KEY_ALIASES: dict[str, str] = {
    # Minuto
    "minute": "minuto", "min": "minuto", "time": "minuto",
    # Gol
    "goals_home": "gol_casa", "gol_home": "gol_casa", "home_goals": "gol_casa",
    "score_home": "gol_casa", "home_score": "gol_casa",
    "goals_away": "gol_trasf", "gol_away": "gol_trasf", "away_goals": "gol_trasf",
    "score_away": "gol_trasf", "away_score": "gol_trasf",
    # Rossi
    "red_cards_home": "rossi_casa", "red_home": "rossi_casa", "home_red_cards": "rossi_casa",
    "red_cards_away": "rossi_trasf", "red_away": "rossi_trasf", "away_red_cards": "rossi_trasf",
    # Gialli
    "yellow_cards_home": "gialli_casa", "yellow_home": "gialli_casa",
    "home_yellow_cards": "gialli_casa",
    "yellow_cards_away": "gialli_trasf", "yellow_away": "gialli_trasf",
    "away_yellow_cards": "gialli_trasf",
    # Tiri in porta
    "shots_on_target_home": "tiri_porta_casa", "sot_home": "tiri_porta_casa",
    "shots_on_goal_home": "tiri_porta_casa", "home_shots_on_target": "tiri_porta_casa",
    "home_shots_on_goal": "tiri_porta_casa",
    "shots_on_target_away": "tiri_porta_trasf", "sot_away": "tiri_porta_trasf",
    "shots_on_goal_away": "tiri_porta_trasf", "away_shots_on_target": "tiri_porta_trasf",
    "away_shots_on_goal": "tiri_porta_trasf",
    # Tiri fuori
    "shots_off_target_home": "tiri_fuori_casa", "home_shots_off_target": "tiri_fuori_casa",
    "shots_off_goal_home": "tiri_fuori_casa", "home_shots_off_goal": "tiri_fuori_casa",
    "shots_off_target_away": "tiri_fuori_trasf", "away_shots_off_target": "tiri_fuori_trasf",
    "shots_off_goal_away": "tiri_fuori_trasf", "away_shots_off_goal": "tiri_fuori_trasf",
    # Tiri bloccati
    "blocked_shots_home": "tiri_bloccati_casa", "blocked_home": "tiri_bloccati_casa",
    "home_blocked_shots": "tiri_bloccati_casa", "home_blocked": "tiri_bloccati_casa",
    "blocked_shots_away": "tiri_bloccati_trasf", "blocked_away": "tiri_bloccati_trasf",
    "away_blocked_shots": "tiri_bloccati_trasf", "away_blocked": "tiri_bloccati_trasf",
    # Corner
    "corners_home": "corner_casa", "home_corners": "corner_casa",
    "corner_kicks_home": "corner_casa", "home_corner_kicks": "corner_casa",
    "corners_away": "corner_trasf", "away_corners": "corner_trasf",
    "corner_kicks_away": "corner_trasf", "away_corner_kicks": "corner_trasf",
    # Possesso
    "possession_home": "possesso_casa", "home_possession": "possesso_casa",
    "possession_away": "possesso_trasf", "away_possession": "possesso_trasf",
    # Attacchi
    "attacks_home": "attacchi_casa", "home_attacks": "attacchi_casa",
    "attacks_away": "attacchi_trasf", "away_attacks": "attacchi_trasf",
    "dangerous_attacks_home": "attacchi_pericolosi_casa",
    "home_dangerous_attacks": "attacchi_pericolosi_casa",
    "dangerous_attacks_away": "attacchi_pericolosi_trasf",
    "away_dangerous_attacks": "attacchi_pericolosi_trasf",
    # Falli
    "fouls_home": "falli_casa", "home_fouls": "falli_casa",
    "fouls_away": "falli_trasf", "away_fouls": "falli_trasf",
    "free_kicks_home": "falli_casa", "home_free_kicks": "falli_casa",
    "free_kicks_away": "falli_trasf", "away_free_kicks": "falli_trasf",
    # Tiri totali (Nowgoal "Shots" = tiri totali, non solo in porta)
    "shots_home": "tiri_totali_casa", "home_shots": "tiri_totali_casa",
    "shots_away": "tiri_totali_trasf", "away_shots": "tiri_totali_trasf",
    # ── Chiavi compatte dal prompt ridotto ──
    "g_h": "gol_casa", "g_a": "gol_trasf",
    "r_h": "rossi_casa", "r_a": "rossi_trasf",
    "y_h": "gialli_casa", "y_a": "gialli_trasf",
    "sot_h": "tiri_porta_casa", "sot_a": "tiri_porta_trasf",
    "soff_h": "tiri_fuori_casa", "soff_a": "tiri_fuori_trasf",
    "blk_h": "tiri_bloccati_casa", "blk_a": "tiri_bloccati_trasf",
    "cor_h": "corner_casa", "cor_a": "corner_trasf",
    "pos_h": "possesso_casa", "pos_a": "possesso_trasf",
    "att_h": "attacchi_casa", "att_a": "attacchi_trasf",
    "datt_h": "attacchi_pericolosi_casa", "datt_a": "attacchi_pericolosi_trasf",
    "fou_h": "falli_casa", "fou_a": "falli_trasf",
}


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Appiattisce un dict annidato in chiavi piatte.

    Gestisce strutture come:
      {"shots_on_goal": {"home": 4, "away": 2}}
      → {"shots_on_goal_home": 4, "shots_on_goal_away": 2}

      {"corner_kicks": [6, 2]}
      → {"corner_kicks_home": 6, "corner_kicks_away": 2}
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            # Annida: {"home": 4, "away": 2} → key_home=4, key_away=2
            for sub_key, sub_val in value.items():
                sub_lower = sub_key.lower()
                if sub_lower in ("home", "casa", "h", "left"):
                    flat[f"{full_key}_home"] = sub_val
                elif sub_lower in ("away", "trasf", "trasferta", "a", "right"):
                    flat[f"{full_key}_away"] = sub_val
                else:
                    flat[f"{full_key}_{sub_key}"] = sub_val
        elif isinstance(value, list) and len(value) == 2:
            # Lista [home, away]
            flat[f"{full_key}_home"] = value[0]
            flat[f"{full_key}_away"] = value[1]
        else:
            flat[full_key] = value
    return flat


def _normalize_live_stats_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Normalizza le chiavi del JSON usando gli alias noti.

    Gestisce:
    1. Chiavi piatte italiane (tiri_porta_casa) → passano direttamente
    2. Chiavi piatte inglesi (shots_on_target_home) → mappate via alias
    3. Strutture annidate ({"shots": {"home": 4}}) → appiattite e mappate
    """
    # Step 1: appiattisci strutture annidate
    flat = _flatten_dict(data)

    # Step 2: mappa le chiavi tramite alias
    normalized: dict[str, Any] = {}
    for key, value in flat.items():
        canonical = _LIVE_STATS_KEY_ALIASES.get(key.lower(), key)
        normalized[canonical] = value
    return normalized


def _parse_live_stats_response(response: str) -> LiveStatsExtracted:
    """Parsa la risposta del VLM per le statistiche live."""
    if not response or not response.strip():
        return LiveStatsExtracted(
            extraction_success=False, error_message="Risposta vuota",
        )

    try:
        json_str = response.strip()

        # Rimuovi markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 2:
                json_str = parts[1]

        json_str = json_str.strip()

        # Estrai il JSON tra il primo { e l'ultimo } (ignora testo prima/dopo)
        first_brace = json_str.find("{")
        last_brace  = json_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = json_str[first_brace : last_brace + 1]
        elif first_brace != -1:
            json_str = json_str[first_brace:]

        json_str = json_str.strip()

        # Tentativo di riparare JSON troncato (output troppo lungo → tagliato)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = json_str.rstrip().rstrip(",")

            # 1. Chiudi array aperti (es. "ev":[...])
            open_brackets = repaired.count("[") - repaired.count("]")
            if open_brackets > 0:
                # Rimuovi l'ultimo elemento incompleto dell'array se necessario
                last_open = repaired.rfind("{", repaired.rfind("["))
                last_close = repaired.rfind("}")
                if last_open > last_close:
                    repaired = repaired[:last_open].rstrip().rstrip(",")
                repaired += "]" * open_brackets

            # 2. Se ci sono oggetti non chiusi, rimuovi l'ultimo entry incompleto.
            # Esempio: '{"a":1,"b":2,"c_incomp' → taglia all'ultima virgola → '{"a":1,"b":2'
            if repaired.count("{") > repaired.count("}"):
                last_comma = repaired.rfind(",")
                if last_comma != -1:
                    # Verifica che dopo la virgola ci sia un entry incompleto
                    # (non vogliamo troncare un array come "ev":[...],)
                    after_comma = repaired[last_comma + 1:].strip()
                    if after_comma and not after_comma.startswith("{") and not after_comma.startswith("["):
                        repaired = repaired[:last_comma]

            # 3. Chiudi oggetti ancora aperti
            open_braces = repaired.count("{") - repaired.count("}")
            if open_braces > 0:
                repaired = repaired.rstrip().rstrip(",")
                repaired += "}" * open_braces

            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                # Nessun tentativo di repair ha funzionato: rilancia l'eccezione
                # originale così che il blocco esterno restituisca extraction_success=False.
                raise

        # Normalizza chiavi: Gemini potrebbe usare nomi inglesi o varianti
        data = _normalize_live_stats_keys(data)

        # Deriva tiri fuori = tiri totali - tiri in porta (se mancano)
        if not data.get("tiri_fuori_casa") and data.get("tiri_totali_casa"):
            tot = _safe_int(data["tiri_totali_casa"])
            on = _safe_int(data.get("tiri_porta_casa", 0))
            data["tiri_fuori_casa"] = max(0, tot - on)
        if not data.get("tiri_fuori_trasf") and data.get("tiri_totali_trasf"):
            tot = _safe_int(data["tiri_totali_trasf"])
            on = _safe_int(data.get("tiri_porta_trasf", 0))
            data["tiri_fuori_trasf"] = max(0, tot - on)

        # Parsing eventi: valida che sia una lista di dict con i campi attesi
        raw_ev = data.get("ev", data.get("eventi", []))
        eventi: list = []
        if isinstance(raw_ev, list):
            for ev in raw_ev:
                if not isinstance(ev, dict):
                    continue
                t   = str(ev.get("t", "")).lower()
                sq  = str(ev.get("sq", "")).lower()
                pl  = str(ev.get("pl", ev.get("player", "")))
                mn  = _safe_int(ev.get("min", ev.get("minute", 0)))
                if t in ("goal", "yellow", "red", "sub") and sq in ("h", "a"):
                    eventi.append({"t": t, "sq": sq, "pl": pl, "min": mn})

        # Ricava rossi/gialli dagli eventi se il conteggio diretto è 0
        # (fallback robusto: conta eventi nella cronologia)
        r_h = _safe_int(data.get("rossi_casa"))
        r_a = _safe_int(data.get("rossi_trasf"))
        y_h = _safe_int(data.get("gialli_casa"))
        y_a = _safe_int(data.get("gialli_trasf"))
        if eventi and (r_h == 0 and r_a == 0 and y_h == 0 and y_a == 0):
            r_h = sum(1 for e in eventi if e["t"] == "red"    and e["sq"] == "h")
            r_a = sum(1 for e in eventi if e["t"] == "red"    and e["sq"] == "a")
            y_h = sum(1 for e in eventi if e["t"] == "yellow" and e["sq"] == "h")
            y_a = sum(1 for e in eventi if e["t"] == "yellow" and e["sq"] == "a")

        return LiveStatsExtracted(
            minuto=_safe_int(data.get("minuto")),
            gol_casa=_safe_int(data.get("gol_casa")),
            gol_trasf=_safe_int(data.get("gol_trasf")),
            rossi_casa=r_h,
            rossi_trasf=r_a,
            gialli_casa=y_h,
            gialli_trasf=y_a,
            tiri_porta_casa=_safe_int(data.get("tiri_porta_casa")),
            tiri_porta_trasf=_safe_int(data.get("tiri_porta_trasf")),
            tiri_fuori_casa=_safe_int(data.get("tiri_fuori_casa")),
            tiri_fuori_trasf=_safe_int(data.get("tiri_fuori_trasf")),
            tiri_bloccati_casa=_safe_int(data.get("tiri_bloccati_casa")),
            tiri_bloccati_trasf=_safe_int(data.get("tiri_bloccati_trasf")),
            corner_casa=_safe_int(data.get("corner_casa")),
            corner_trasf=_safe_int(data.get("corner_trasf")),
            possesso_casa=_safe_float(data.get("possesso_casa")),
            possesso_trasf=_safe_float(data.get("possesso_trasf")),
            attacchi_casa=_safe_int(data.get("attacchi_casa")),
            attacchi_trasf=_safe_int(data.get("attacchi_trasf")),
            attacchi_pericolosi_casa=_safe_int(data.get("attacchi_pericolosi_casa")),
            attacchi_pericolosi_trasf=_safe_int(data.get("attacchi_pericolosi_trasf")),
            falli_casa=_safe_int(data.get("falli_casa")),
            falli_trasf=_safe_int(data.get("falli_trasf")),
            eventi=eventi,
            confidence=str(data.get("confidence", "medium")).lower(),
            raw_response=response,
            extraction_success=True,
        )
    except json.JSONDecodeError as e:
        return LiveStatsExtracted(
            extraction_success=False,
            error_message=f"JSON error: {e}",
            raw_response=response,
        )
    except Exception as e:
        return LiveStatsExtracted(
            extraction_success=False,
            error_message=f"Parse error: {e}",
            raw_response=response,
        )


def extract_live_stats_from_bytes(
    image_bytes: bytes, extension: str = ".png",
) -> LiveStatsExtracted:
    """Estrae statistiche live da bytes di un'immagine."""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, prefix="live_ocr_",
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            return _extract_live_stats_with_gemini(Path(tmp_path))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
    except Exception as e:
        return LiveStatsExtracted(
            extraction_success=False,
            error_message=f"Errore temp file: {e}",
        )


# ============================================================================
# Prematch Analysis Extraction (Nowgoal tab "Analysis")
# ============================================================================

@dataclass
class PrematchAnalysisExtracted:
    """Dati estratti dallo screen Analysis di Nowgoal (pre-partita)."""

    extraction_success: bool = False
    error_message: str = ""
    raw_response: str = ""

    # H2H
    h2h_home_win_pct: float = 0.0    # 0-100
    h2h_draw_pct: float = 0.0
    h2h_away_win_pct: float = 0.0
    h2h_avg_goals_home: float = 0.0  # gol medi segnati dalla casa in H2H
    h2h_avg_goals_away: float = 0.0  # gol medi segnati dalla trasferta in H2H
    # H2H trend scommesse (da grafici O/U e AH nella sezione H2H)
    h2h_over_pct: float = 0.0        # % partite H2H andate Over (0-100)
    h2h_ah_home_cover_pct: float = 0.0  # % partite H2H in cui la casa ha coperto l'AH

    # Standings casa — riga Total
    home_rank: int = 0
    home_matches: int = 0
    home_win: int = 0
    home_draw: int = 0
    home_lose: int = 0
    home_scored: int = 0
    home_conceded: int = 0
    home_win_rate: float = 0.0       # Rate % dalla tabella classifica
    home_last6_win: int = 0
    home_last6_draw: int = 0
    home_last6_lose: int = 0
    # Riga Home (performance specificamente in casa)
    home_home_win: int = 0
    home_home_draw: int = 0
    home_home_lose: int = 0
    home_home_scored: float = 0.0    # gol segnati in casa
    home_home_conceded: float = 0.0  # gol subiti in casa
    # Riga HT (half-time standings)
    home_ht_win: int = 0
    home_ht_draw: int = 0
    home_ht_lose: int = 0

    # Standings trasferta — riga Total
    away_rank: int = 0
    away_matches: int = 0
    away_win: int = 0
    away_draw: int = 0
    away_lose: int = 0
    away_scored: int = 0
    away_conceded: int = 0
    away_win_rate: float = 0.0
    away_last6_win: int = 0
    away_last6_draw: int = 0
    away_last6_lose: int = 0
    # Riga Away (performance specificamente in trasferta)
    away_away_win: int = 0
    away_away_draw: int = 0
    away_away_lose: int = 0
    away_away_scored: float = 0.0
    away_away_conceded: float = 0.0
    # Riga HT (half-time standings)
    away_ht_win: int = 0
    away_ht_draw: int = 0
    away_ht_lose: int = 0

    # Previous Scores Statistics (form della squadra di casa come host, trasferta come ospite)
    home_prev_win_pct: float = 0.0    # % vittorie nelle ultime 10 partite in casa (0-100)
    home_prev_avg_scored: float = 0.0 # media gol segnati
    home_prev_avg_conceded: float = 0.0
    home_prev_over_pct: float = 0.0   # % partite Over nelle ultime 10 (0-100)
    away_prev_win_pct: float = 0.0
    away_prev_avg_scored: float = 0.0
    away_prev_avg_conceded: float = 0.0
    away_prev_over_pct: float = 0.0

    # Quote 1X2 iniziali consensus (media bookmaker da odds comparison — solo via URL)
    mkt_init_1: float = 0.0   # quota 1 iniziale media
    mkt_init_x: float = 0.0   # quota X iniziale media
    mkt_init_2: float = 0.0   # quota 2 iniziale media

    # HT statistics dai precedenti H2H
    h2h_ht_home_win_pct: float = 0.0
    h2h_ht_draw_pct: float = 0.0
    h2h_ht_away_win_pct: float = 0.0

    # Goal timing corrente stagione (gol per partita nel 1° e 2° tempo)
    home_goals_1h: float = 0.0
    home_goals_2h: float = 0.0
    away_goals_1h: float = 0.0
    away_goals_2h: float = 0.0

    # Line movement: linee iniziali (solo via screenshot — non estratte da URL)
    initial_ah_line: float = 0.0      # linea AH iniziale del bookmaker
    initial_total_line: float = 0.0   # linea Total iniziale
    strength_home: int = 0            # punteggio forza casa (0-100, da Nowgoal)
    strength_away: int = 0            # punteggio forza trasferta

    # Metadati partita (estratti automaticamente via URL)
    home_team: str = ""
    away_team: str = ""
    league_name: str = ""
    match_date: str = ""

    # Parametri derivati → entrano direttamente nel MatchState
    fixture_historical_total: float = 0.0  # media gol H2H totali
    forma_mult_h: float = 1.0              # moltiplicatore xG casa
    forma_mult_a: float = 1.0             # moltiplicatore xG trasferta


PREMATCH_ANALYSIS_PROMPT = """Sei un assistente che legge screenshot della pagina "Analysis" di Nowgoal.

=== STRUTTURA DELLA PAGINA (importante per non confondere le sezioni) ===

La pagina ha questo ordine DALL'ALTO verso il BASSO:
1. BARRA QUOTE in cima (numeri piccoli, righe "Low/Initial/Live")
2. STRENGTH COMPARISON (numero grande a sinistra = casa, numero grande a destra = trasferta)
3. H2H COMPARISON (barra % con Win/Draw/Lose, poi goal per game)
4. WHO WILL WIN (voting, mostra "AH: X")
5. STANDINGS (tabella con due colori: arancione=casa, blu=trasferta)
   - PRIMA metà della tabella = FT (Full Time): righe Total / Home / Away / Last 6
   - SECONDA metà della tabella = HT (Half Time): righe Total / Home / Away / Last 6
   - Le due sezioni FT e HT sono SEPARATE da un'intestazione "HT"
6. HEAD TO HEAD STATISTICS (con grafici Asian Handicap Odds e Over/Under Odds)
7. PREVIOUS SCORES STATISTICS (statistiche recenti della squadra di casa)

=== ESTRAZIONE ===

**SEZIONE 1 — STRENGTH COMPARISON**
I due numeri grandi ai lati: casa=sinistra, trasferta=destra.

**SEZIONE 2 — H2H COMPARISON**
- "Win X (Y%)" a sinistra → home_win_pct = Y
- "Draw X (Y%)" al centro → draw_pct = Y
- "Lose X (Y%)" a destra → away_win_pct = Y
- "X goals   Goal Score/Loss per Game   Y goals" → avg_goals_home=X, avg_goals_away=Y
  (il primo numero è la media gol della squadra di CASA, il secondo della TRASFERTA)

**SEZIONE 3 — HEAD TO HEAD STATISTICS (i grafici)**
Due grafici: "Asian Handicap Odds" e "Over/Under Odds"
- Nel grafico AH: percentuale "Home XX%" → ah_home_cover_pct = XX
- Nel grafico O/U: percentuale "Over XX%" → over_pct = XX

**SEZIONE 4 — STANDINGS — Squadra di CASA (tabella arancione, a sinistra)**
Il titolo mostra [LEGA-RANK] es. "[SPA D2-3]" → rank=3. Se il testo non è leggibile,
cerca il numero dopo l'ultimo "-" nel titolo es. "[ARG D1-17]" → rank=17.

PARTE FT (Full Time) — righe nell'ORDINE: Total, Home, Away, Last 6
Le colonne sono ESATTAMENTE in quest'ordine (contale da sinistra, posizione 1→9):
  1=Matches | 2=Win | 3=Draw | 4=Lose | 5=Scored | 6=Conceded | 7=Pts | 8=Rank | 9=Rate%

⚠️ ERRORE COMUNE: non confondere Draw (col.3) con Lose (col.4). Sono colonne SEPARATE.
   Esempio corretto: se vedi "6  1  2  3  4  5" → Matches=6, Win=1, Draw=2, Lose=3, Scored=4, Conceded=5

Leggi:
- Riga "Total": tutte le colonne → matches, win, draw, lose, scored, conceded, win_rate=Rate%
- Riga "Home" (FT, NON HT): win, draw, lose, scored, conceded → home_win, home_draw, home_lose, home_scored, home_conceded
- Riga "Last 6": win, draw, lose → last6_win, last6_draw, last6_lose
  VINCOLO: last6_win + last6_draw + last6_lose DEVE essere uguale a 6. Se la somma non è 6,
  rileggila contando attentamente le colonne 2, 3, 4.

PARTE HT (Half Time) — SEPARATA dalla parte FT, inizia dopo l'intestazione "HT"
Leggi:
- Riga "Total" nella sezione HT: win, draw, lose → ht_win, ht_draw, ht_lose

**SEZIONE 5 — STANDINGS — Squadra TRASFERTA (tabella blu, a destra)**
Stessa struttura. Rank: leggi dal titolo es. "[ARG D1-5]" → rank=5.
Se il pannello destro è parzialmente tagliato, leggi solo ciò che è visibile e usa 0 per il resto.

PARTE FT:
- Riga "Total": matches, win, draw, lose, scored, conceded, win_rate
- Riga "Away" (FT): win, draw, lose, scored, conceded → away_win, away_draw, away_lose, away_scored, away_conceded
  (NON la riga "Home" della trasferta — ci interessa la performance IN TRASFERTA)
- Riga "Last 6": last6_win, last6_draw, last6_lose
  VINCOLO: last6_win + last6_draw + last6_lose DEVE essere uguale a 6.

PARTE HT:
- Riga "Total": ht_win, ht_draw, ht_lose

**SEZIONE 6 — PREVIOUS SCORES STATISTICS**
Questa sezione appare spesso solo per la squadra di casa. Ha un filtro che dice es.
"Almeria • Home • Same League • Last 10".
- Riga di riepilogo: "Win X (Y%)" → win_pct=Y, "Draw X (Z%)" → ignora, "Lose X (W%)" → ignora
- "X.X goals  Goal Score/Loss per Game  Y.Y goals" → avg_scored=X.X, avg_conceded=Y.Y
  (il PRIMO numero = gol segnati dalla squadra di casa, SECONDO = gol subiti)
- Nei grafici O/U: "Over XX%" → over_pct=XX

Se è visibile anche la sezione per la trasferta, estrai gli stessi campi in "prev_away".
Se NON è visibile, usa 0 per tutti i campi "prev_away".

**SEZIONE 7 — BARRA QUOTE IN CIMA**
In cima allo screenshot c'è una riga con label "Initial" (o "Low") con piccoli numeri.
Cerca i valori della linea AH iniziale (es. -0.5, -1, 0, 1) e Total iniziale (es. 2.5, 3).
Spesso si vedono valori come "2.5/3" che indicano il Total iniziale.
Se non sono leggibili, usa 0.

=== OUTPUT ===
Rispondi SOLO con JSON valido, nessun testo fuori dal JSON. Usa 0 per valori non visibili.

{
  "h2h": {
    "home_win_pct": 67,
    "draw_pct": 33,
    "away_win_pct": 0,
    "avg_goals_home": 2.3,
    "avg_goals_away": 1.0,
    "over_pct": 67,
    "ah_home_cover_pct": 67
  },
  "strength": {
    "home": 60,
    "away": 40
  },
  "home": {
    "rank": 3,
    "matches": 31,
    "win": 16,
    "draw": 7,
    "lose": 8,
    "scored": 59,
    "conceded": 43,
    "win_rate": 51.6,
    "home_win": 10,
    "home_draw": 2,
    "home_lose": 2,
    "home_scored": 36,
    "home_conceded": 22,
    "ht_win": 12,
    "ht_draw": 16,
    "ht_lose": 3,
    "last6_win": 4,
    "last6_draw": 1,
    "last6_lose": 1
  },
  "away": {
    "rank": 13,
    "matches": 31,
    "win": 11,
    "draw": 7,
    "lose": 13,
    "scored": 44,
    "conceded": 43,
    "win_rate": 35.5,
    "away_win": 4,
    "away_draw": 2,
    "away_lose": 9,
    "away_scored": 19,
    "away_conceded": 23,
    "ht_win": 7,
    "ht_draw": 19,
    "ht_lose": 5,
    "last6_win": 4,
    "last6_draw": 0,
    "last6_lose": 2
  },
  "prev_home": {
    "win_pct": 80,
    "avg_scored": 1.9,
    "avg_conceded": 1.3,
    "over_pct": 60
  },
  "prev_away": {
    "win_pct": 0,
    "avg_scored": 0,
    "avg_conceded": 0,
    "over_pct": 0
  },
  "lines": {
    "initial_ah": -0.5,
    "initial_total": 2.5
  }
}"""


def _forma_mult_from_standings(
    win: int, draw: int, lose: int,
    last6_win: int, last6_draw: int, last6_lose: int,
    scored: int, conceded: int, matches: int,
) -> float:
    """
    Calcola forma_mult [0.92, 1.08] da dati standings.

    Fonti di segnale (ponderate):
    - Last 6 (60%): forma recente, più predittiva
    - Overall win rate (25%): qualità assoluta della squadra
    - Goal difference rate (15%): efficienza offensiva/difensiva
    """
    if matches <= 0:
        return 1.0

    # Last 6: punti su massimo possibile
    last6_tot = last6_win + last6_draw + last6_lose
    if last6_tot > 0:
        last6_pts_rate = (last6_win * 3 + last6_draw) / (last6_tot * 3)
        last6_score = (last6_pts_rate - 0.50) * 2   # [-1, +1]
    else:
        last6_score = 0.0

    # Overall win rate (baseline ~38% per squadra media)
    win_rate = (win * 3 + draw) / (matches * 3)
    overall_score = (win_rate - 0.38) / 0.38        # [-1, +1] circa

    # Goal difference per partita
    gd_rate = (scored - conceded) / matches
    goal_score = max(-1.0, min(1.0, gd_rate / 1.5))  # [-1, +1]

    combined = 0.60 * last6_score + 0.25 * overall_score + 0.15 * goal_score
    combined = max(-1.0, min(1.0, combined))

    # Max ±8% (stesso di FORMA_MAX_EFFECT in config)
    return max(0.92, min(1.08, 1.0 + combined * 0.08))


def _parse_prematch_analysis_response(response: str) -> PrematchAnalysisExtracted:
    """Parsa la risposta di Gemini per l'analisi prematch."""
    if not response or not response.strip():
        return PrematchAnalysisExtracted(
            extraction_success=False, error_message="Risposta vuota",
        )

    try:
        json_str = response.strip()

        # Rimuovi markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 2:
                json_str = parts[1]

        # Estrai tra primo { e ultimo }
        first_brace = json_str.find("{")
        last_brace  = json_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = json_str[first_brace : last_brace + 1]
        elif first_brace != -1:
            json_str = json_str[first_brace:]

        json_str = json_str.strip()

        # Parse con repair
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = json_str.rstrip().rstrip(",")
            if repaired.count("{") > repaired.count("}"):
                last_comma = repaired.rfind(",")
                if last_comma != -1:
                    after = repaired[last_comma + 1:].strip()
                    if after and not after.startswith("{") and not after.startswith("["):
                        repaired = repaired[:last_comma]
            open_braces = repaired.count("{") - repaired.count("}")
            if open_braces > 0:
                repaired = repaired.rstrip().rstrip(",") + "}" * open_braces
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                raise

        def _f(val: Any, default: float = 0.0) -> float:
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        def _i(val: Any, default: int = 0) -> int:
            try:
                return int(float(val)) if val is not None else default
            except (TypeError, ValueError):
                return default

        match_info = data.get("match", {})
        h2h      = data.get("h2h", {})
        home     = data.get("home", {})
        away     = data.get("away", {})
        strength = data.get("strength", {})
        prev_h   = data.get("prev_home", {})
        prev_a   = data.get("prev_away", {})
        odds     = data.get("odds", {})
        lines    = data.get("lines", {})

        # H2H base
        h2h_home_win = _f(h2h.get("home_win_pct"))
        h2h_draw     = _f(h2h.get("draw_pct"))
        h2h_away_win = _f(h2h.get("away_win_pct"))
        h2h_avg_h    = _f(h2h.get("avg_goals_home"))
        h2h_avg_a    = _f(h2h.get("avg_goals_away"))
        h2h_over     = _f(h2h.get("over_pct"))
        h2h_ah_home  = _f(h2h.get("ah_home_cover_pct"))
        # H2H halftime
        h2h_ht_home  = _f(h2h.get("ht_home_win_pct"))
        h2h_ht_draw  = _f(h2h.get("ht_draw_pct"))
        h2h_ht_away  = _f(h2h.get("ht_away_win_pct"))

        # Standings casa
        hm   = _i(home.get("matches"))
        hw   = _i(home.get("win"))
        hd   = _i(home.get("draw"))
        hl   = _i(home.get("lose"))
        hsc  = _i(home.get("scored"))
        hco  = _i(home.get("conceded"))
        hwr  = _f(home.get("win_rate"))
        hl6w = _i(home.get("last6_win"))
        hl6d = _i(home.get("last6_draw"))
        hl6l = _i(home.get("last6_lose"))
        # Sanity check: Last 6 must sum to 6; if not, try to correct using Total row
        if hl6w + hl6d + hl6l != 6 and hl6w + hl6d + hl6l > 0:
            _total = hl6w + hl6d + hl6l
            if _total > 0:
                hl6w = round(hl6w * 6 / _total)
                hl6d = round(hl6d * 6 / _total)
                hl6l = 6 - hl6w - hl6d
        # Home-specific row
        hhw  = _i(home.get("home_win"))
        hhd  = _i(home.get("home_draw"))
        hhl  = _i(home.get("home_lose"))
        hhsc = _f(home.get("home_scored"))
        hhco = _f(home.get("home_conceded"))
        # HT row
        hhtw = _i(home.get("ht_win"))
        hhtd = _i(home.get("ht_draw"))
        hhtl = _i(home.get("ht_lose"))
        # Goal timing
        hg1h = _f(home.get("goals_1h"))
        hg2h = _f(home.get("goals_2h"))

        # Standings trasferta
        am   = _i(away.get("matches"))
        aw   = _i(away.get("win"))
        ad   = _i(away.get("draw"))
        al   = _i(away.get("lose"))
        asc  = _i(away.get("scored"))
        aco  = _i(away.get("conceded"))
        awr  = _f(away.get("win_rate"))
        al6w = _i(away.get("last6_win"))
        al6d = _i(away.get("last6_draw"))
        al6l = _i(away.get("last6_lose"))
        # Sanity check: Last 6 must sum to 6
        if al6w + al6d + al6l != 6 and al6w + al6d + al6l > 0:
            _total = al6w + al6d + al6l
            if _total > 0:
                al6w = round(al6w * 6 / _total)
                al6d = round(al6d * 6 / _total)
                al6l = 6 - al6w - al6d
        # Away-specific row
        aaw  = _i(away.get("away_win"))
        aad  = _i(away.get("away_draw"))
        aal  = _i(away.get("away_lose"))
        aasc = _f(away.get("away_scored"))
        aaco = _f(away.get("away_conceded"))
        # HT row
        ahtw = _i(away.get("ht_win"))
        ahtd = _i(away.get("ht_draw"))
        ahtl = _i(away.get("ht_lose"))
        # Goal timing
        ag1h = _f(away.get("goals_1h"))
        ag2h = _f(away.get("goals_2h"))

        # Parametri derivati
        # Usa forma basata su riga Home-specific se disponibile, altrimenti Total
        _hw_eff  = hhw  if hhw  > 0 else hw
        _hd_eff  = hhd  if hhd  > 0 else hd
        _hl_eff  = hhl  if hhl  > 0 else hl
        _hsc_eff = int(hhsc) if hhsc > 0 else hsc
        _hco_eff = int(hhco) if hhco > 0 else hco
        _hm_eff  = _hw_eff + _hd_eff + _hl_eff if (_hw_eff + _hd_eff + _hl_eff) > 0 else hm

        _aw_eff  = aaw  if aaw  > 0 else aw
        _ad_eff  = aad  if aad  > 0 else ad
        _al_eff  = aal  if aal  > 0 else al
        _asc_eff = int(aasc) if aasc > 0 else asc
        _aco_eff = int(aaco) if aaco > 0 else aco
        _am_eff  = _aw_eff + _ad_eff + _al_eff if (_aw_eff + _ad_eff + _al_eff) > 0 else am

        # fixture_historical_total: media gol attesi per questa specifica partita.
        # Blend di H2H (60%) e stima da form recente (40%) se disponibile.
        # Form estimate: media gol segnati da casa + gol subiti da trasferta (e viceversa).
        h2h_total = h2h_avg_h + h2h_avg_a
        prev_h_scored    = _f(prev_h.get("avg_scored"))
        prev_h_conceded  = _f(prev_h.get("avg_conceded"))
        prev_a_scored    = _f(prev_a.get("avg_scored"))
        prev_a_conceded  = _f(prev_a.get("avg_conceded"))
        prev_h_over      = _f(prev_h.get("over_pct"))
        prev_a_over      = _f(prev_a.get("over_pct"))

        form_total = 0.0
        if prev_h_scored > 0 and prev_a_conceded > 0:
            form_total += (prev_h_scored + prev_a_conceded) / 2.0
        if prev_a_scored > 0 and prev_h_conceded > 0:
            form_total += (prev_a_scored + prev_h_conceded) / 2.0

        if h2h_total > 0.1 and form_total > 0.1:
            fixture_total = 0.60 * h2h_total + 0.40 * form_total
        elif h2h_total > 0.1:
            fixture_total = h2h_total
        else:
            fixture_total = form_total  # fallback se H2H non disponibile

        # Correzione Over% — segnale indipendente dai gol medi.
        # Se le partite di queste squadre tendono a Over, spingi leggermente il totale
        # verso l'alto (e viceversa). Baseline media: ~55%. Max correzione: ±0.20 gol.
        if fixture_total > 0.1:
            _over_signals = []
            if h2h_over > 0:
                _over_signals.append(h2h_over)
            if prev_h_over > 0:
                _over_signals.append(prev_h_over)
            if prev_a_over > 0:
                _over_signals.append(prev_a_over)
            if _over_signals:
                _over_avg = sum(_over_signals) / len(_over_signals)
                _over_corr = (_over_avg - 55.0) / 100.0  # [-0.55, +0.45]
                _over_corr = max(-0.20, min(0.20, _over_corr * 0.40))  # max ±0.20 gol
                fixture_total = max(0.5, fixture_total + _over_corr)

        forma_h = _forma_mult_from_standings(
            _hw_eff, _hd_eff, _hl_eff, hl6w, hl6d, hl6l, _hsc_eff, _hco_eff, _hm_eff,
        )
        forma_a = _forma_mult_from_standings(
            _aw_eff, _ad_eff, _al_eff, al6w, al6d, al6l, _asc_eff, _aco_eff, _am_eff,
        )

        return PrematchAnalysisExtracted(
            extraction_success=True,
            raw_response=response,
            # H2H
            h2h_home_win_pct=h2h_home_win,
            h2h_draw_pct=h2h_draw,
            h2h_away_win_pct=h2h_away_win,
            h2h_avg_goals_home=h2h_avg_h,
            h2h_avg_goals_away=h2h_avg_a,
            h2h_over_pct=h2h_over,
            h2h_ah_home_cover_pct=h2h_ah_home,
            h2h_ht_home_win_pct=h2h_ht_home,
            h2h_ht_draw_pct=h2h_ht_draw,
            h2h_ht_away_win_pct=h2h_ht_away,
            # Standings casa
            home_rank=_i(home.get("rank")),
            home_matches=hm, home_win=hw, home_draw=hd, home_lose=hl,
            home_scored=hsc, home_conceded=hco, home_win_rate=hwr,
            home_last6_win=hl6w, home_last6_draw=hl6d, home_last6_lose=hl6l,
            home_home_win=hhw, home_home_draw=hhd, home_home_lose=hhl,
            home_home_scored=hhsc, home_home_conceded=hhco,
            home_ht_win=hhtw, home_ht_draw=hhtd, home_ht_lose=hhtl,
            home_goals_1h=hg1h, home_goals_2h=hg2h,
            # Standings trasferta
            away_rank=_i(away.get("rank")),
            away_matches=am, away_win=aw, away_draw=ad, away_lose=al,
            away_scored=asc, away_conceded=aco, away_win_rate=awr,
            away_last6_win=al6w, away_last6_draw=al6d, away_last6_lose=al6l,
            away_away_win=aaw, away_away_draw=aad, away_away_lose=aal,
            away_away_scored=aasc, away_away_conceded=aaco,
            away_ht_win=ahtw, away_ht_draw=ahtd, away_ht_lose=ahtl,
            away_goals_1h=ag1h, away_goals_2h=ag2h,
            # Previous Scores
            home_prev_win_pct=_f(prev_h.get("win_pct")),
            home_prev_avg_scored=_f(prev_h.get("avg_scored")),
            home_prev_avg_conceded=_f(prev_h.get("avg_conceded")),
            home_prev_over_pct=_f(prev_h.get("over_pct")),
            away_prev_win_pct=_f(prev_a.get("win_pct")),
            away_prev_avg_scored=_f(prev_a.get("avg_scored")),
            away_prev_avg_conceded=_f(prev_a.get("avg_conceded")),
            away_prev_over_pct=_f(prev_a.get("over_pct")),
            # Market initial odds (only from URL extraction)
            mkt_init_1=_f(odds.get("init_1")),
            mkt_init_x=_f(odds.get("init_x")),
            mkt_init_2=_f(odds.get("init_2")),
            # Lines (from screenshot only — not extracted via URL)
            initial_ah_line=_f(lines.get("initial_ah")),
            initial_total_line=_f(lines.get("initial_total")),
            # Strength
            strength_home=_i(strength.get("home")),
            strength_away=_i(strength.get("away")),
            # Match metadata
            home_team=str(match_info.get("home_team", "") or ""),
            away_team=str(match_info.get("away_team", "") or ""),
            league_name=str(match_info.get("league", "") or ""),
            match_date=str(match_info.get("date", "") or ""),
            # Derived
            fixture_historical_total=fixture_total,
            forma_mult_h=forma_h,
            forma_mult_a=forma_a,
        )

    except json.JSONDecodeError as e:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message=f"JSON error: {e}",
            raw_response=response,
        )
    except Exception as e:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message=f"Parse error: {e}",
            raw_response=response,
        )


def _extract_prematch_analysis_with_gemini(image_paths: list[Path]) -> PrematchAnalysisExtracted:
    """Estrae analisi prematch da uno o due screenshot usando Gemini.

    Accetta una lista di 1-2 immagini (per schermate lunghe).
    Entrambe le immagini vengono inviate nello stesso prompt.
    """
    import time

    api_key = _get_gemini_api_key()
    if not api_key:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message="Gemini: API key non configurata",
        )

    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }

    parts: list[dict] = []
    for path in image_paths[:2]:  # max 2 immagini
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = mime_map.get(path.suffix.lower(), "image/jpeg")
            parts.append({"inline_data": {"mime_type": mime, "data": img_b64}})
        except Exception as e:
            return PrematchAnalysisExtracted(
                extraction_success=False,
                error_message=f"Gemini: errore lettura file {path.name}: {e}",
            )

    parts.append({"text": PREMATCH_ANALYSIS_PROMPT})

    request_body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "response_mime_type": "application/json",
        },
    }
    payload = json.dumps(request_body).encode("utf-8")

    last_error = ""
    for model in _GEMINI_MODELS:
        url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))

                if "candidates" in resp_data and resp_data["candidates"]:
                    parts_resp = resp_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts_resp:
                        text = parts_resp[0].get("text", "")
                        if text:
                            return _parse_prematch_analysis_response(text)
                last_error = f"Gemini ({model}): risposta vuota"
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                last_error = f"Gemini ({model}): HTTP {e.code}"
                break
            except Exception as e:
                last_error = f"Gemini ({model}): {e}"
                break

    return PrematchAnalysisExtracted(extraction_success=False, error_message=last_error)


def extract_prematch_analysis_from_bytes(
    images: list[tuple[bytes, str]],
) -> PrematchAnalysisExtracted:
    """
    Estrae analisi prematch da uno o due screenshot (bytes).

    Args:
        images: lista di (bytes_immagine, extension) — max 2 elementi.
                Extension es. ".png", ".jpg".
    """
    tmp_paths: list[Path] = []
    try:
        for img_bytes, ext in images[:2]:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext, prefix="prematch_ocr_",
            ) as tmp:
                tmp.write(img_bytes)
                tmp_paths.append(Path(tmp.name))

        return _extract_prematch_analysis_with_gemini(tmp_paths)
    except Exception as e:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message=f"Errore temp file: {e}",
        )
    finally:
        for p in tmp_paths:
            with contextlib.suppress(OSError):
                os.unlink(p)


# ============================================================================
# URL-based extraction (Jina Reader + Gemini text)
# ============================================================================

_JINA_BASE_URL = "https://r.jina.ai/"
_NOWGOAL_DOMAINS = ("nowgoal.com", "nowgoal.net", "nowgoal.info", "nowgoal26.com")

PREMATCH_ANALYSIS_TEXT_PROMPT = """Sei un assistente che estrae dati statistici da testo di una pagina Nowgoal Analysis/H2H.

Estrai i dati indicati e rispondi SOLO con JSON valido. Usa 0 per valori non trovati.

DATI DA ESTRARRE:

0. match — Informazioni base della partita:
   - home_team: nome squadra di casa
   - away_team: nome squadra trasferta
   - league: nome della lega/competizione
   - date: data della partita (formato YYYY-MM-DD se disponibile)

1. h2h — Head to Head Statistics (scontri diretti precedenti):
   - home_win_pct: % vittorie casa (es. "Win 2 (20%)" → 20)
   - draw_pct: % pareggi
   - away_win_pct: % vittorie trasferta
   - avg_goals_home: media gol segnati dalla casa nei precedenti H2H
   - avg_goals_away: media gol segnati dalla trasferta nei precedenti H2H
   - over_pct: % Over nei precedenti H2H
   - ah_home_cover_pct: % AH copertura casa nei precedenti H2H
   - ht_home_win_pct: % precedenti H2H con casa in vantaggio ALL'INTERVALLO (HT)
   - ht_draw_pct: % precedenti H2H con pareggio all'intervallo
   - ht_away_win_pct: % precedenti H2H con trasferta in vantaggio all'intervallo

2. strength — Strength Comparison (due numeri grandi):
   - home: numero squadra di casa
   - away: numero squadra trasferta

3. odds — Live Odds Comparison, riga Initial SOLO 1X2 (NON estrarre AH o O/U lines):
   Media delle quote Initial 1X2 dei bookmaker presenti (Bet365, Sbobet, 188bet o altri):
   - init_1: media quota casa iniziale
   - init_x: media quota pareggio iniziale
   - init_2: media quota trasferta iniziale

4. home — standings squadra di CASA:
   - rank: dal titolo [LEGA-RANK] es. "[ARG D1-17]" → 17
   - matches, win, draw, lose, scored, conceded, win_rate: riga Total (FT)
   - home_win, home_draw, home_lose, home_scored, home_conceded: riga Home (FT)
   - last6_win, last6_draw, last6_lose: riga Last 6 — DEVONO sommare a 6
   - ht_win, ht_draw, ht_lose: riga Total sezione HT
   - goals_1h: totale gol segnati nel 1° tempo (dalla stagione corrente)
   - goals_2h: totale gol segnati nel 2° tempo

5. away — standings squadra TRASFERTA:
   - rank, matches, win, draw, lose, scored, conceded, win_rate: riga Total (FT)
   - away_win, away_draw, away_lose, away_scored, away_conceded: riga Away (FT)
     ⚠ NON la riga Home della trasferta — serve la performance IN TRASFERTA
   - last6_win, last6_draw, last6_lose: riga Last 6 — DEVONO sommare a 6
   - ht_win, ht_draw, ht_lose: riga Total sezione HT
   - goals_1h: totale gol segnati nel 1° tempo
   - goals_2h: totale gol segnati nel 2° tempo

6. prev_home — Previous Scores Statistics sezione squadra di casa:
   - win_pct, avg_scored, avg_conceded, over_pct

7. prev_away — Previous Scores Statistics sezione squadra trasferta (partite in trasferta):
   - win_pct, avg_scored, avg_conceded, over_pct (usa 0 se la sezione non è presente)

{
  "match": {"home_team": "", "away_team": "", "league": "", "date": ""},
  "h2h": {"home_win_pct": 0, "draw_pct": 0, "away_win_pct": 0,
          "avg_goals_home": 0.0, "avg_goals_away": 0.0, "over_pct": 0, "ah_home_cover_pct": 0,
          "ht_home_win_pct": 0, "ht_draw_pct": 0, "ht_away_win_pct": 0},
  "strength": {"home": 0, "away": 0},
  "odds": {"init_1": 0.0, "init_x": 0.0, "init_2": 0.0},
  "home": {"rank": 0, "matches": 0, "win": 0, "draw": 0, "lose": 0,
           "scored": 0, "conceded": 0, "win_rate": 0.0,
           "home_win": 0, "home_draw": 0, "home_lose": 0, "home_scored": 0, "home_conceded": 0,
           "last6_win": 0, "last6_draw": 0, "last6_lose": 0,
           "ht_win": 0, "ht_draw": 0, "ht_lose": 0,
           "goals_1h": 0, "goals_2h": 0},
  "away": {"rank": 0, "matches": 0, "win": 0, "draw": 0, "lose": 0,
           "scored": 0, "conceded": 0, "win_rate": 0.0,
           "away_win": 0, "away_draw": 0, "away_lose": 0, "away_scored": 0, "away_conceded": 0,
           "last6_win": 0, "last6_draw": 0, "last6_lose": 0,
           "ht_win": 0, "ht_draw": 0, "ht_lose": 0,
           "goals_1h": 0, "goals_2h": 0},
  "prev_home": {"win_pct": 0, "avg_scored": 0.0, "avg_conceded": 0.0, "over_pct": 0},
  "prev_away": {"win_pct": 0, "avg_scored": 0.0, "avg_conceded": 0.0, "over_pct": 0}
}"""


def _is_valid_nowgoal_url(url: str) -> bool:
    """Verifica che l'URL sia una pagina Analysis/H2H di Nowgoal."""
    url_lower = url.lower()
    is_nowgoal = any(d in url_lower for d in _NOWGOAL_DOMAINS)
    has_analysis = any(x in url_lower for x in ("h2h", "analysis"))
    return is_nowgoal and has_analysis


def _fetch_jina_reader(url: str, timeout: int = 30) -> str:
    """Recupera il testo di una pagina via Jina Reader (r.jina.ai)."""
    jina_url = _JINA_BASE_URL + url
    req = urllib.request.Request(
        jina_url,
        headers={
            "Accept": "text/plain",
            "User-Agent": "Mozilla/5.0 (compatible; Exchange-Bot/1.0)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_prematch_analysis_from_text(page_text: str) -> PrematchAnalysisExtracted:
    """Invia il testo della pagina a Gemini (text-only) e parsa la risposta."""
    import time

    api_key = _get_gemini_api_key()
    if not api_key:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message="Gemini: API key non configurata",
        )

    # Limita a ~10000 caratteri per non eccedere il contesto (la pagina è verbosa)
    text_truncated = page_text[:10000]
    full_prompt = f"TESTO PAGINA NOWGOAL:\n\n{text_truncated}\n\n---\n\n{PREMATCH_ANALYSIS_TEXT_PROMPT}"

    request_body = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "response_mime_type": "application/json",
        },
    }
    payload = json.dumps(request_body).encode("utf-8")

    last_error = ""
    for model in _GEMINI_MODELS:
        api_url = f"{_GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    api_url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))

                if "candidates" in resp_data and resp_data["candidates"]:
                    parts_resp = resp_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts_resp:
                        text = parts_resp[0].get("text", "")
                        if text:
                            return _parse_prematch_analysis_response(text)
                last_error = f"Gemini ({model}): risposta vuota"
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                last_error = f"Gemini ({model}): HTTP {e.code}"
                break
            except Exception as e:
                last_error = f"Gemini ({model}): {e}"
                break

    return PrematchAnalysisExtracted(extraction_success=False, error_message=last_error)


def extract_prematch_analysis_from_url(url: str) -> PrematchAnalysisExtracted:
    """
    Estrae analisi prematch da un URL Nowgoal via Jina Reader + Gemini.

    Args:
        url: URL della pagina Analysis di Nowgoal
             (es. https://www.nowgoal.com/match/h2h-2800452)

    Returns:
        PrematchAnalysisExtracted con i dati estratti, o errore se fallisce.
    """
    url = url.strip()
    if not url:
        return PrematchAnalysisExtracted(
            extraction_success=False, error_message="URL vuoto",
        )

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not _is_valid_nowgoal_url(url):
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message=(
                "URL non riconosciuto. Usa un link Nowgoal Analysis "
                "(es. nowgoal.com/match/h2h-XXXXXX)"
            ),
        )

    try:
        page_text = _fetch_jina_reader(url)
    except Exception as e:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message=f"Errore lettura pagina: {e}",
        )

    if not page_text or len(page_text) < 200:
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message="Pagina vuota o non leggibile tramite Jina Reader",
        )

    return _extract_prematch_analysis_from_text(page_text)


# ============================================================================
# Main Extraction Functions
# ============================================================================

def extract_from_image_file(image_path: str | Path) -> ExtractedData:
    """
    Estrae i dati da un file immagine usando il VLM.

    Prova i backend in ordine:
    1. z-ai CLI (se disponibile, gratuito)
    2. Google Gemini API (se configurato, gratuito)
    3. OpenAI Vision API (se configurato, a pagamento)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return ExtractedData(extraction_success=False, error_message=f"File non trovato: {image_path}")

    errors = []

    # 1. z-ai CLI (gratuito, locale)
    if _check_zai_available():
        result = _extract_with_zai_cli(image_path)
        if result.extraction_success:
            return result
        errors.append(result.error_message)

    # 2. Gemini (gratuito)
    if _check_gemini_available():
        result = _extract_with_gemini(image_path)
        if result.extraction_success:
            return result
        errors.append(result.error_message)

    # 3. OpenAI (a pagamento)
    if _check_openai_available():
        result = _extract_with_openai(image_path)
        if result.extraction_success:
            return result
        errors.append(result.error_message)

    if not errors:
        return ExtractedData(
            extraction_success=False,
            error_message="Nessun backend VLM configurato. Imposta GEMINI_API_KEY (gratuito) o OPENAI_API_KEY."
        )

    return ExtractedData(extraction_success=False, error_message=" | ".join(errors))

def extract_from_bytes(image_bytes: bytes, extension: str = ".png") -> ExtractedData:
    """Estrae i dati da bytes di un'immagine."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension, prefix="ocr_") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            return extract_from_image_file(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"Errore temp file: {e}")

def extract_from_base64(base64_data: str, mime_type: str = "image/png") -> ExtractedData:
    """Estrae i dati da una stringa base64."""
    try:
        mime_to_ext = {".png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        extension = mime_to_ext.get(mime_type, ".png")
        image_bytes = base64.b64decode(base64_data)
        return extract_from_bytes(image_bytes, extension)
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"Errore base64: {e}")

# ============================================================================
# Response Parsing
# ============================================================================

def _parse_vlm_response(response: str) -> ExtractedData:
    """Parsa la risposta del VLM e estrae i dati strutturati."""
    if not response or not response.strip():
        return ExtractedData(extraction_success=False, error_message="Risposta vuota")

    try:
        json_str = response.strip()

        # Rimuovi markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 2:
                json_str = parts[1]

        json_str = json_str.strip()

        # Trova inizio JSON
        lines = json_str.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_str = "\n".join(lines[i:])
                break

        if "```" in json_str:
            json_str = json_str.split("```")[0]

        data = json.loads(json_str.strip())

        # Gestisci formato API response
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            if content and content.strip().startswith("{"):
                data = json.loads(content)

        return ExtractedData(
            squadra_casa=str(data.get("squadra_casa", "")).strip(),
            squadra_trasf=str(data.get("squadra_trasf", "")).strip(),
            quota_1=_safe_float(data.get("quota_1")),
            quota_x=_safe_float(data.get("quota_x")),
            quota_2=_safe_float(data.get("quota_2")),
            linea_ou=_safe_float(data.get("linea_ou")),
            quota_over=_safe_float(data.get("quota_over")),
            quota_under=_safe_float(data.get("quota_under")),
            quota_gg=_safe_float(data.get("quota_gg")),
            quota_ng=_safe_float(data.get("quota_ng")),
            confidence=str(data.get("confidence", "medium")).lower(),
            raw_response=response,
            extraction_success=True,
        )
    except json.JSONDecodeError as e:
        return _fallback_extraction(response, str(e))
    except Exception as e:
        return ExtractedData(extraction_success=False, error_message=f"Parse error: {e}", raw_response=response)

def _fallback_extraction(response: str, original_error: str) -> ExtractedData:
    """Fallback per estrarre dati quando il JSON parsing fallisce."""
    data = ExtractedData(extraction_success=False, error_message=f"JSON error: {original_error}", raw_response=response)

    try:
        match = re.search(r"([A-Za-zÀ-ÿ\s]+)\s+(?:vs|-|–)\s+([A-Za-zÀ-ÿ\s]+)", response, re.IGNORECASE)
        if match:
            data.squadra_casa = match.group(1).strip()
            data.squadra_trasf = match.group(2).strip()

        quotes = re.findall(r"(\d+[.,]\d{2,3})", response)
        if len(quotes) >= 3:
            data.quota_1 = _safe_float(quotes[0])
            data.quota_x = _safe_float(quotes[1])
            data.quota_2 = _safe_float(quotes[2])

        if data.squadra_casa or data.quota_1 > 0:
            data.extraction_success = True
            data.confidence = "low"
    except Exception:
        pass

    return data

def _safe_int(value: Any) -> int:
    """Converte un valore in int in modo sicuro."""
    if value is None:
        return 0
    try:
        return int(float(str(value).replace(",", ".")))
    except (ValueError, TypeError):
        return 0


def _safe_float(value: Any) -> float:
    """Converte un valore in float in modo sicuro."""
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return 0.0

def validate_extracted_data(data: ExtractedData) -> tuple[bool, list[str]]:
    """Valida i dati estratti e restituisce eventuali problemi."""
    warnings = []

    if not data.squadra_casa:
        warnings.append("Squadra casa non rilevata")
    if not data.squadra_trasf:
        warnings.append("Squadra trasferta non rilevata")

    if data.quota_1 <= 0 or data.quota_x <= 0 or data.quota_2 <= 0:
        warnings.append("Quote 1X2 incomplete")
    else:
        for q, name in [(data.quota_1, "1"), (data.quota_x, "X"), (data.quota_2, "2")]:
            if q < 1.01 or q > 50.0:
                warnings.append(f"Quota {name} fuori range")

    if data.quota_over <= 0 and data.quota_under <= 0:
        warnings.append("Quote O/U non rilevate")
    if data.quota_gg <= 0 and data.quota_ng <= 0:
        warnings.append("Quote BTTS non rilevate")

    return data.quota_1 > 0 and data.quota_x > 0 and data.quota_2 > 0, warnings
