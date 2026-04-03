"""
ocr.py — Estrazione automatica dati da screenshot di siti scommesse e URL Nowgoal.

METODO PRIMARIO: REGEX (GRATUITO, senza API key, senza limiti)
==============================================================
L'estrazione da URL Nowgoal usa principalmente regex per:
  - H2H (Win/Draw/Lose %, Over %, media gol)
  - Strength Comparison
  - Standings (Total, Home, Away, Last 6)
  - Previous Scores Statistics
  - Quote iniziali 1X2

Supporta molteplici formati H2H:
  - "Win 3 (30%) Draw 3 (30%) Lose 4 (40%)"
  - "1W 3D 2L"
  - "Win 30% Draw 30% Lose 40%"
  - "1-3-2"

Backend per analisi IMMAGINI (screenshot):
  1. z-ai CLI (gratuito, locale)
  2. Google Gemini API (gratuito 1500 req/giorno)
  3. OpenAI Vision API (a pagamento, fallback finale)

Configurazione OPZIONALE:
  - GEMINI_API_KEY: via st.secrets o variabile d'ambiente (fallback)
  - OPENAI_API_KEY: via st.secrets o variabile d'ambiente (fallback)

Il modulo funziona completamente SENZA configurare API key per l'estrazione da URL.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import modulo weather per API OpenWeather
# Prova diversi modi di import per compatibilità con Streamlit
WEATHER_MODULE_AVAILABLE = False
_get_weather_for_match = None
_get_weather_for_city = None

try:
    # Prova import relativo
    from src.weather import get_weather_for_match, get_weather_for_city
    WEATHER_MODULE_AVAILABLE = True
    _get_weather_for_match = get_weather_for_match
    _get_weather_for_city = get_weather_for_city
except ImportError:
    try:
        # Prova import diretto del file
        import importlib.util
        import os as _os
        _weather_path = _os.path.join(_os.path.dirname(__file__), 'weather.py')
        if _os.path.exists(_weather_path):
            _spec = importlib.util.spec_from_file_location("weather", _weather_path)
            _weather_module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_weather_module)
            _get_weather_for_match = _weather_module.get_weather_for_match
            _get_weather_for_city = _weather_module.get_weather_for_city
            WEATHER_MODULE_AVAILABLE = True
    except Exception as e:
        print(f"[WEATHER] Import fallito: {e}")
        WEATHER_MODULE_AVAILABLE = False

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
    h2h_btts_pct: float = 0.0        # % partite H2H con entrambe a segno
    h2h_matches_count: int = 0       # numero match H2H usati per statistiche

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
    # Riga Last 6 FT: gol fatti / subiti nelle ultime 6 (colonne Scored, Conceded su Nowgoal)
    home_last6_scored: int = 0
    home_last6_conceded: int = 0
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
    away_last6_scored: int = 0
    away_last6_conceded: int = 0
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
    league_source: str = "unknown"   # nowgoal | mirror | external | unknown
    match_date: str = ""

    # Parametri derivati → entrano direttamente nel MatchState
    fixture_historical_total: float = 0.0  # media gol H2H totali
    forma_mult_h: float = 1.0              # moltiplicatore xG casa
    forma_mult_a: float = 1.0             # moltiplicatore xG trasferta

    # === NUOVI CAMPI: Partite recenti (h_data / a_data) ===
    # Lista di tuple (gol_fatti, gol_subiti) per ultime N partite
    # Calcolato da h_data/a_data nel JS di Nowgoal
    home_recent_results: list = field(default_factory=list)  # [(gf, gs), ...] ultime partite casa
    away_recent_results: list = field(default_factory=list)  # [(gf, gs), ...] ultime partite trasferta
    home_form_trend: float = 0.0      # Trend forma: positivo = in miglioramento
    away_form_trend: float = 0.0      # Trend forma: negativo = in peggioramento
    home_xg_from_recent: float = 0.0  # xG stimato da partite recenti
    away_xg_from_recent: float = 0.0  # xG stimato da partite recenti
    scoring_streak_h: int = 0         # partite consecutive con gol segnato (casa)
    scoring_streak_a: int = 0         # partite consecutive con gol segnato (trasferta)
    clean_sheet_streak_h: int = 0     # partite consecutive senza subire (casa)
    clean_sheet_streak_a: int = 0     # partite consecutive senza subire (trasferta)

    # === NUOVI CAMPI: Quote multi-bookmaker (Vs_hOdds) ===
    # Usati principalmente dallo Scanner (pagina principale ha input manuale)
    ah_line_open: float = 0.0         # Linea AH apertura (media bookmaker)
    ah_line_close: float = 0.0        # Linea AH chiusura (più recente)
    ah_home_odds_open: float = 0.0    # Quota AH casa apertura
    ah_away_odds_open: float = 0.0    # Quota AH trasferta apertura
    total_line_open: float = 0.0      # Linea Total apertura
    total_line_close: float = 0.0     # Linea Total chiusura
    total_over_odds_open: float = 0.0 # Quota Over apertura
    total_under_odds_open: float = 0.0 # Quota Under apertura
    line_movement_ah: float = 0.0     # Movimento linea AH (close - open)
    line_movement_total: float = 0.0  # Movimento linea Total (close - open)
    odds_sharp_signal: float = 0.0    # Segnale sharp: |movement| * confidenza

    # === NUOVI CAMPI: Punti classifica (motivazione) ===
    home_points: int = 0              # Punti in classifica casa
    away_points: int = 0              # Punti in classifica trasferta
    home_motivation: str = "normal"   # "high" / "normal" / "low"
    away_motivation: str = "normal"   # "high" = lotta titolo/salvezza, "low" = salvo
    home_absences_count: int = 0      # numero assenze/infortuni/squalifiche casa
    away_absences_count: int = 0      # numero assenze/infortuni/squalifiche trasferta
    # Righe per calcola_assenze_mult, es. "Nome (CM, injured)" — da Nowgoal o bullet list
    home_absences_players: list[str] = field(default_factory=list)
    away_absences_players: list[str] = field(default_factory=list)

    # === NUOVI CAMPI: ID squadre ===
    home_id: int = 0                  # ID squadra casa (da h2h_home)
    away_id: int = 0                  # ID squadra trasferta (da h2h_away)

    # === NUOVI CAMPI: Dati dalla pagina LIVE (estratti automaticamente) ===
    # Meteo
    weather_condition: str = ""       # es. "Partly cloudy", "Rain", "Clear"
    weather_temp: int = 0             # temperatura in °C
    weather_impact: float = 0.0       # -0.05 per pioggia, -0.03 per vento, etc.
    
    # HT/FT Statistics (da pagina LIVE)
    htft_home_htw_ftw: int = 0        # HT win → FT win per casa
    htft_home_htd_ftw: int = 0        # HT draw → FT win per casa
    htft_home_htl_ftw: int = 0        # HT lose → FT win per casa
    htft_home_htw_ftd: int = 0        # HT win → FT draw per casa
    htft_home_htd_ftd: int = 0        # HT draw → FT draw per casa
    htft_home_htl_ftd: int = 0        # HT lose → FT draw per casa
    htft_home_htw_ftl: int = 0        # HT win → FT lose per casa
    htft_home_htd_ftl: int = 0        # HT draw → FT lose per casa
    htft_home_htl_ftl: int = 0        # HT lose → FT lose per casa
    htft_away_htw_ftw: int = 0        # stesso per trasferta
    htft_away_htd_ftw: int = 0
    htft_away_htl_ftw: int = 0
    htft_away_htw_ftd: int = 0
    htft_away_htd_ftd: int = 0
    htft_away_htl_ftd: int = 0
    htft_away_htw_ftl: int = 0
    htft_away_htd_ftl: int = 0
    htft_away_htl_ftl: int = 0
    
    # Team Statistics extra (da pagina LIVE, ultimi 10 match)
    team_stats_home_goals: float = 0.0      # media gol casa
    team_stats_home_conceded: float = 0.0   # media gol subiti casa
    team_stats_home_shots: float = 0.0      # media tiri subiti casa
    team_stats_home_corners: float = 0.0    # media corner casa
    team_stats_home_yellows: float = 0.0    # media gialli casa
    team_stats_home_fouls: float = 0.0      # media falli casa
    team_stats_home_possession: float = 0.0 # media possesso casa
    team_stats_away_goals: float = 0.0      # stessi per trasferta
    team_stats_away_conceded: float = 0.0
    team_stats_away_shots: float = 0.0
    team_stats_away_corners: float = 0.0
    team_stats_away_yellows: float = 0.0
    team_stats_away_fouls: float = 0.0
    team_stats_away_possession: float = 0.0
    
    # Quote LIVE (informativo, non sostituisce input manuale)
    live_ah_line: float = 0.0         # linea AH live
    live_ah_home_odds: float = 0.0    # quota AH casa live
    live_ah_away_odds: float = 0.0    # quota AH trasferta live
    live_total_line: float = 0.0      # linea Total live
    live_over_odds: float = 0.0       # quota Over live
    live_under_odds: float = 0.0      # quota Under live
    # Quality report sintetico per capire affidabilita' estrazione URL.
    extraction_coverage: float = 0.0  # [0,1] percentuale sezioni chiave trovate
    extraction_notes: list[str] = field(default_factory=list)
    extraction_section_scores: dict[str, float] = field(default_factory=dict)  # 0/1 per sezione


_NOISY_TEAM_SUFFIXES = (
    " - football",
    " football",
    " - live",
    " live score",
    " live and analysis",
)
_INVALID_LEAGUE_TOKENS = {"statistics", "analysis", "football", "live score"}


def _clean_team_name(value: str) -> str:
    name = (value or "").strip()
    name = re.sub(r"\s+", " ", name)
    low = name.lower()
    for suffix in _NOISY_TEAM_SUFFIXES:
        if low.endswith(suffix):
            name = name[: -len(suffix)].strip()
            low = name.lower()
    # Evita catture "lunghe" con testo descrittivo attaccato.
    name = re.sub(r"\s+-\s+.*$", "", name).strip()
    return name


def _clean_league_name(value: str) -> str:
    league = (value or "").strip()
    league = re.sub(r"\s+", " ", league)
    if league.lower() in _INVALID_LEAGUE_TOKENS:
        return ""
    # Rimuove porzioni descrittive non di lega.
    league = re.sub(r"\s*[-:]\s*(round|statistics|analysis).*$", "", league, flags=re.IGNORECASE)
    return league.strip()


def _extract_match_identity_from_text(text: str) -> tuple[str, str, str]:
    """Parser dedicato per home/away/league, separato dal parser principale."""
    home = ""
    away = ""
    league = ""
    title = re.search(
        r"Title:\s*([A-Za-z][A-Za-z\s\.'\-]{1,40}?)\s+(?:VS|vs)\s+([A-Za-z][A-Za-z\s\.'\-]{1,40}?)(?:\s+-|\s+Live|\s+Match|\s+Analysis|$)",
        text,
        re.IGNORECASE,
    )
    if title:
        home = _clean_team_name(title.group(1))
        away = _clean_team_name(title.group(2))
    breadcrumb = re.search(r"Football>\s*([A-Za-z][A-Za-z0-9\s\-\.\(\)]+?)>\s*$", text, re.MULTILINE)
    if breadcrumb:
        league = _clean_league_name(breadcrumb.group(1))
    # Fallback: blocco "Football> ... >" anche con trailing testo.
    if not league:
        breadcrumb2 = re.search(r"Football>\s*([A-Za-z][A-Za-z0-9\s\-\.\(\)]+?)\s*>\s*", text, re.IGNORECASE)
        if breadcrumb2:
            league = _clean_league_name(breadcrumb2.group(1))
    # Fallback: riga sotto titolo partita "Australia A-League · Round 23".
    if not league:
        comp_line = re.search(r"\n\s*([A-Za-z][A-Za-z0-9\-\s]+League|[A-Za-z][A-Za-z0-9\-\s]+Cup)\s*[·\|]", text)
        if comp_line:
            league = _clean_league_name(comp_line.group(1))
    # Fallback da codifica lega tipo [AUS D1-4] -> Australia A-League
    if not league:
        code = re.search(r"\[?([A-Z]{2,3}\s*D\d)-\d+\]?", text)
        if code:
            map_code = {
                "AUS D1": "Australia A-League",
                "ENG D1": "England Premier League",
                "ENG D2": "England Championship",
                "ITA D1": "Italy Serie A",
                "ESP D1": "Spain LaLiga",
                "SPA D2": "Spain Segunda",
                "GER D1": "Germany Bundesliga",
                "FRA D1": "France Ligue 1",
            }
            league = map_code.get(code.group(1).strip(), "")
    # Fallback: codifica lega senza rank (es. "AUS D1")
    if not league:
        code2 = re.search(r"\b([A-Z]{2,3}\s*D\d)\b", text)
        if code2:
            map_code = {
                "AUS D1": "Australia A-League",
                "ENG D1": "England Premier League",
                "ENG D2": "England Championship",
                "ITA D1": "Italy Serie A",
                "ESP D1": "Spain LaLiga",
                "SPA D2": "Spain Segunda",
                "GER D1": "Germany Bundesliga",
                "FRA D1": "France Ligue 1",
            }
            league = map_code.get(code2.group(1).strip(), "")
    return home, away, league


def _compute_streaks_from_results(results: list[tuple[int, int]]) -> tuple[int, int]:
    """Ritorna (scoring_streak, clean_sheet_streak) su risultati recenti."""
    scoring = 0
    clean_sheet = 0
    for gf, gs in results:
        if gf > 0:
            scoring += 1
        else:
            break
    for gf, gs in results:
        if gs == 0:
            clean_sheet += 1
        else:
            break
    return scoring, clean_sheet


def _fill_last6_goals_from_recent_results(r: PrematchAnalysisExtracted) -> None:
    """Se Last 6 non ha colonne Scored/Conceded, somma i primi 6 risultati da h_data/a_data."""
    if r.home_last6_scored == 0 and r.home_last6_conceded == 0 and len(r.home_recent_results) >= 6:
        r.home_last6_scored = sum(g for g, s in r.home_recent_results[:6])
        r.home_last6_conceded = sum(s for g, s in r.home_recent_results[:6])
    if r.away_last6_scored == 0 and r.away_last6_conceded == 0 and len(r.away_recent_results) >= 6:
        r.away_last6_scored = sum(g for g, s in r.away_recent_results[:6])
        r.away_last6_conceded = sum(s for g, s in r.away_recent_results[:6])


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
- Riga "Last 6": matches, win, draw, lose, scored, conceded → last6_win, last6_draw, last6_lose, last6_scored, last6_conceded
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
- Riga "Last 6": last6_win, last6_draw, last6_lose, last6_scored, last6_conceded (Scored/Conceded dalla tabella)
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
    "last6_lose": 1,
    "last6_scored": 12,
    "last6_conceded": 9
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
    "last6_lose": 2,
    "last6_scored": 11,
    "last6_conceded": 8
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
        hl6sc = _i(home.get("last6_scored"))
        hl6co = _i(home.get("last6_conceded"))
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
        al6sc = _i(away.get("last6_scored"))
        al6co = _i(away.get("last6_conceded"))
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
            home_last6_scored=hl6sc, home_last6_conceded=hl6co,
            home_home_win=hhw, home_home_draw=hhd, home_home_lose=hhl,
            home_home_scored=hhsc, home_home_conceded=hhco,
            home_ht_win=hhtw, home_ht_draw=hhtd, home_ht_lose=hhtl,
            home_goals_1h=hg1h, home_goals_2h=hg2h,
            # Standings trasferta
            away_rank=_i(away.get("rank")),
            away_matches=am, away_win=aw, away_draw=ad, away_lose=al,
            away_scored=asc, away_conceded=aco, away_win_rate=awr,
            away_last6_win=al6w, away_last6_draw=al6d, away_last6_lose=al6l,
            away_last6_scored=al6sc, away_last6_conceded=al6co,
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
# L'HTML Nowgoal ha spesso H2H/infortuni oltre i primi 100k caratteri (es. injury ~290k).
_RAW_HTML_APPEND_MAX = 400_000
_LEAGUE_CACHE_PATH = Path("data/league_fallback_cache.json")
_LEAGUE_CACHE_TTL_SEC = 24 * 3600
_NOWGOAL_DOMAINS = (
    "nowgoal.com", "nowgoal.net", "nowgoal.info",
    "nowgoal26.com", "nowgoal8.com",
    "live5.nowgoal26.com",  # Nuovo dominio funzionante
)

_LEAGUE_ALIAS_MAP = {
    "australian a-league": "Australia A-League",
    "a-league": "Australia A-League",
    "english premier league": "England Premier League",
    "premier league": "England Premier League",
    "italian serie a": "Italy Serie A",
    "serie a": "Italy Serie A",
    "spanish la liga": "Spain LaLiga",
    "laliga": "Spain LaLiga",
    "german bundesliga": "Germany Bundesliga",
    "french ligue 1": "France Ligue 1",
}

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
   - last6_win, last6_draw, last6_lose, last6_scored, last6_conceded: riga Last 6 FT — W+D+L = 6; scored/conceded dalle colonne omonime
   - ht_win, ht_draw, ht_lose: riga Total sezione HT
   - goals_1h: totale gol segnati nel 1° tempo (dalla stagione corrente)
   - goals_2h: totale gol segnati nel 2° tempo

5. away — standings squadra TRASFERTA:
   - rank, matches, win, draw, lose, scored, conceded, win_rate: riga Total (FT)
   - away_win, away_draw, away_lose, away_scored, away_conceded: riga Away (FT)
     ⚠ NON la riga Home della trasferta — serve la performance IN TRASFERTA
   - last6_win, last6_draw, last6_lose, last6_scored, last6_conceded: riga Last 6 FT — W+D+L = 6; scored/conceded dalle colonne omonime
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
           "last6_win": 0, "last6_draw": 0, "last6_lose": 0, "last6_scored": 0, "last6_conceded": 0,
           "ht_win": 0, "ht_draw": 0, "ht_lose": 0,
           "goals_1h": 0, "goals_2h": 0},
  "away": {"rank": 0, "matches": 0, "win": 0, "draw": 0, "lose": 0,
           "scored": 0, "conceded": 0, "win_rate": 0.0,
           "away_win": 0, "away_draw": 0, "away_lose": 0, "away_scored": 0, "away_conceded": 0,
           "last6_win": 0, "last6_draw": 0, "last6_lose": 0, "last6_scored": 0, "last6_conceded": 0,
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


def _normalize_league_name_external(name: str) -> str:
    cleaned = _clean_league_name(name)
    if not cleaned:
        return ""
    key = cleaned.strip().lower()
    return _LEAGUE_ALIAS_MAP.get(key, cleaned)


def _league_cache_key(home_team: str, away_team: str) -> str:
    return f"{home_team.strip().lower()}|{away_team.strip().lower()}"


def _load_league_cache() -> dict[str, dict[str, Any]]:
    try:
        if _LEAGUE_CACHE_PATH.exists():
            raw = json.loads(_LEAGUE_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return {}


def _save_league_cache(cache: dict[str, dict[str, Any]]) -> None:
    try:
        _LEAGUE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LEAGUE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _lookup_league_external_sportsdb(home_team: str, away_team: str) -> str:
    """Lookup esterno lega via TheSportsDB (solo fallback finale)."""
    if not home_team or not away_team:
        return ""
    q = urllib.parse.quote_plus(f"{home_team} vs {away_team}")
    url = f"https://www.thesportsdb.com/api/v1/json/3/searchevents.php?e={q}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return ""
    events = data.get("event") or []
    home_low = home_team.lower()
    away_low = away_team.lower()
    for ev in events:
        eh = str(ev.get("strHomeTeam", "")).lower()
        ea = str(ev.get("strAwayTeam", "")).lower()
        lg = str(ev.get("strLeague", ""))
        if lg and ((home_low in eh and away_low in ea) or (home_low in ea and away_low in eh)):
            return _normalize_league_name_external(lg)
    if events:
        lg = str(events[0].get("strLeague", ""))
        return _normalize_league_name_external(lg)
    return ""


def _fallback_league_external(home_team: str, away_team: str) -> str:
    key = _league_cache_key(home_team, away_team)
    cache = _load_league_cache()
    now_ts = int(time.time())
    hit = cache.get(key)
    if isinstance(hit, dict):
        ts = int(hit.get("ts", 0) or 0)
        lg = str(hit.get("league", "") or "")
        if lg and (now_ts - ts) <= _LEAGUE_CACHE_TTL_SEC:
            return lg
    league = _lookup_league_external_sportsdb(home_team, away_team)
    if league:
        cache[key] = {"league": league, "ts": now_ts}
        _save_league_cache(cache)
    return league


def _fallback_league_from_nowgoal_mirrors(match_id: str, original_domain: str) -> str:
    """Prova mirror Nowgoal alternativi per recuperare solo la lega."""
    mirrors = [d for d in _NOWGOAL_DOMAINS if d != original_domain]
    for domain in mirrors:
        probe_url = f"https://{domain}/match/h2h-{match_id}"
        try:
            txt = _fetch_jina_reader(probe_url, timeout=12)
        except Exception:
            continue
        if not txt or len(txt) < 20:
            continue
        _, _, league = _extract_match_identity_from_text(txt)
        if league:
            return league
    return ""


def _fetch_raw_html(url: str, timeout: int = 30) -> str:
    """
    Recupera l'HTML grezzo di una pagina usando z-ai CLI o Jina Reader.
    
    Prova:
    1. z-ai CLI page_reader (restituisce HTML completo con JavaScript)
    2. Jina Reader con formato raw
    3. Richiesta diretta (fallisce se protezione anti-bot)
    """
    # Metodo 1: Usa z-ai CLI page_reader
    try:
        executable, extra_args = _find_zai_command()
        if executable:
            if extra_args:
                cmd = [executable] + extra_args + ["function", "-n", "page_reader", "-a", f'{{"url": "{url}"}}']
            else:
                cmd = [executable, "function", "-n", "page_reader", "-a", f'{{"url": "{url}"}}']
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env=_get_env_with_path()
            )
            if result.returncode == 0:
                # Rimuovi log messages e estrai JSON
                stdout = result.stdout.strip()
                # Trova il JSON tra i log messages
                json_start = stdout.find('{')
                json_end = stdout.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = stdout[json_start:json_end]
                    data = json.loads(json_str)
                    if data.get("code") == 200 and data.get("data", {}).get("html"):
                        return data["data"]["html"]
    except Exception:
        pass
    
    # Metodo 2: Jina Reader raw
    try:
        # https://r.jina.ai/http://URL restituisce HTML
        raw_url = url
        if raw_url.startswith("https://"):
            raw_url = "http://" + raw_url[8:]
        jina_url = _JINA_BASE_URL + raw_url
        req = urllib.request.Request(
            jina_url,
            headers={"Accept": "text/html"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            if len(html) > 1000:
                return html
    except Exception:
        pass
    
    # Metodo 3: Richiesta diretta (probabilmente fallisce con anti-bot)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _text_before_previous_scores_statistics(text: str) -> str:
    """Esclude la sezione Previous Scores: stesso formato W/D/L ma non è H2H."""
    low = text.lower()
    idx = low.find("previous scores statistics")
    return text[:idx] if idx >= 0 else text


_LEAGUE_CODE_PREFIX_RE = re.compile(r"^([A-Z]{2,3}\s+D\d+\s+)+", re.IGNORECASE)


def _strip_league_code_prefix(left_chunk: str) -> str:
    s = (left_chunk or "").strip()
    s = _LEAGUE_CODE_PREFIX_RE.sub("", s)
    return s.strip()


def _teams_name_match(a: str, b: str) -> bool:
    ca = _clean_team_name(a).lower()
    cb = _clean_team_name(b).lower()
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    if len(ca) >= 4 and len(cb) >= 4 and (ca in cb or cb in ca):
        return True
    return False


def _h2h_goals_for_fixture_teams(
    hist_home: str, hist_away: str, gh: int, ga: int, fixture_home: str, fixture_away: str
) -> tuple[int, int] | None:
    """
    (gol squadra che nella partita in programma è in casa, gol ospite programmato)
    per uno scontro storico dove hist_home ha giocato in casa con punteggio gh-ga.
    """
    if _teams_name_match(hist_home, fixture_home) and _teams_name_match(hist_away, fixture_away):
        return gh, ga
    if _teams_name_match(hist_home, fixture_away) and _teams_name_match(hist_away, fixture_home):
        return ga, gh
    return None


def _head_to_head_table_slice(text: str) -> str:
    """Dopo l'ultima intestazione Head to Head Statistics, fino a Previous Scores."""
    low = text.lower()
    starts = list(
        re.finditer(r"(?:^|\n)\s*(?:#+\s*)?(?:head\s*to\s*head|h2h)\s+statistics", low, re.IGNORECASE)
    )
    if not starts:
        return ""
    start = starts[-1].end()
    end = low.find("previous scores statistics", start)
    return text[start:] if end < 0 else text[start:end]


def _parse_h2h_score_rows(table_slice: str) -> list[tuple[str, str, int, int]]:
    """
    Righe stile Nowgoal: 'AUS D1 Auckland FC 2-1(1-1) Adelaide United'
    -> (casa_storica, trasferta_storica, gol_casa, gol_trasferta).
    """
    rows: list[tuple[str, str, int, int]] = []
    for line in table_slice.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r"^(.+?)\s+(\d+)-(\d+)\(([^)]*)\)\s*(.+)$", line)
        if not m:
            continue
        left_raw, gh_s, ga_s, right_raw = m.group(1), m.group(2), m.group(3), m.group(5)
        gh, ga = int(gh_s), int(ga_s)
        th = _strip_league_code_prefix(left_raw)
        ta = right_raw.strip()
        if th and ta:
            rows.append((th, ta, gh, ga))
    return rows


def _derive_h2h_from_score_table(
    parsed_rows: list[tuple[str, str, int, int]], fixture_home: str, fixture_away: str
) -> dict | None:
    """
    W/D/L % e medie gol dalla prospettiva fixture_home vs fixture_away.
    BTTS % su tutte le righe parsate (stesso campione del conteggio partite in tabella).
    """
    if not fixture_home or not fixture_away or not parsed_rows:
        return None
    goals_fh: list[int] = []
    goals_fa: list[int] = []
    for th, ta, gh, ga in parsed_rows:
        pair = _h2h_goals_for_fixture_teams(th, ta, gh, ga, fixture_home, fixture_away)
        if pair is None:
            continue
        gfh, gfa = pair
        goals_fh.append(gfh)
        goals_fa.append(gfa)
    n = len(goals_fh)
    if n == 0:
        return None
    w = sum(1 for gfh, gfa in zip(goals_fh, goals_fa) if gfh > gfa)
    d = sum(1 for gfh, gfa in zip(goals_fh, goals_fa) if gfh == gfa)
    l = sum(1 for gfh, gfa in zip(goals_fh, goals_fa) if gfh < gfa)
    btts_hits = sum(1 for _th, _ta, gh, ga in parsed_rows if gh > 0 and ga > 0)
    return {
        "n": n,
        "h2h_home_win_pct": round(w / n * 100, 1),
        "h2h_draw_pct": round(d / n * 100, 1),
        "h2h_away_win_pct": round(l / n * 100, 1),
        "h2h_avg_goals_home": round(sum(goals_fh) / n, 2),
        "h2h_avg_goals_away": round(sum(goals_fa) / n, 2),
        "h2h_btts_pct": round(btts_hits * 100.0 / len(parsed_rows), 1),
        "h2h_matches_count": len(parsed_rows),
    }


def _extract_h2h_with_regex(text: str) -> dict:
    """
    Estrae dati H2H dal testo usando regex (GRATUITO, senza API key).

    Supporta molteplici formati:
    - Formato completo: "Win 3 (30%) Draw 3 (30%) Lose 4 (40%)"
    - Formato compatto: "1W 3D 2L"
    - Formato percentuale: "Win 30% Draw 30% Lose 40%"
    - Formato misto: "1W 3D 2L" per casa e "2W 3D 1L" per trasferta

    Usa solo il testo prima di "Previous Scores Statistics" e, se c'è l'intestazione
    tabella H2H, il formato compatto XW YD ZL solo dentro quella sezione (evita
    Last 6 e altre righe nella pagina).
    """
    result = {
        "h2h_home_win_pct": 0.0,
        "h2h_draw_pct": 0.0,
        "h2h_away_win_pct": 0.0,
    }

    work = _text_before_previous_scores_statistics(text)
    text_lower = work.lower()

    if re.search(r"no\s*data!?[\s\S]{0,100}(?:previous\s*scores|league/cup)", text_lower, re.IGNORECASE):
        return result

    pattern_full = (
        r"Win\s+(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*Draw\s+(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)\s*"
        r"Lose\s+(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)"
    )
    pattern_compact = r"(\d+)W\s+(\d+)D\s+(\d+)L"
    pattern_pct = r"Win\s*(\d+(?:\.\d+)?)\s*%\s*Draw\s*(\d+(?:\.\d+)?)\s*%\s*Lose\s*(\d+(?:\.\d+)?)\s*%"

    h2h_hdr = re.search(
        r"(?:^|\n)\s*(?:#+\s*)?(?:head\s*to\s*head|h2h)\s+statistics", text_lower, re.IGNORECASE
    )

    pre_table = work[: h2h_hdr.start()] if h2h_hdr else work
    match = re.search(pattern_full, pre_table, re.IGNORECASE)
    if match:
        result["h2h_home_win_pct"] = float(match.group(2))
        result["h2h_draw_pct"] = float(match.group(4))
        result["h2h_away_win_pct"] = float(match.group(6))
        return result

    if h2h_hdr:
        h2h_start = h2h_hdr.end()
        rel = re.search(r"previous\s*score", text_lower[h2h_start:])
        h2h_end = h2h_start + rel.start() if rel else len(work)
        h2h_text = work[h2h_start:h2h_end]

        match = re.search(pattern_full, h2h_text, re.IGNORECASE)
        if match:
            result["h2h_home_win_pct"] = float(match.group(2))
            result["h2h_draw_pct"] = float(match.group(4))
            result["h2h_away_win_pct"] = float(match.group(6))
            return result

        match = re.search(pattern_compact, h2h_text, re.IGNORECASE)
        if match:
            w, d, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
            total = w + d + l
            if total > 0:
                result["h2h_home_win_pct"] = round(w / total * 100, 1)
                result["h2h_draw_pct"] = round(d / total * 100, 1)
                result["h2h_away_win_pct"] = round(l / total * 100, 1)
                return result

        match = re.search(pattern_pct, h2h_text, re.IGNORECASE)
        if match:
            result["h2h_home_win_pct"] = float(match.group(1))
            result["h2h_draw_pct"] = float(match.group(2))
            result["h2h_away_win_pct"] = float(match.group(3))
            return result

        return result

    match = re.search(pattern_full, work, re.IGNORECASE)
    if match:
        result["h2h_home_win_pct"] = float(match.group(2))
        result["h2h_draw_pct"] = float(match.group(4))
        result["h2h_away_win_pct"] = float(match.group(6))
        return result

    matches = re.findall(pattern_compact, work, re.IGNORECASE)
    if matches:
        w, d, l = matches[0]
        w, d, l = int(w), int(d), int(l)
        total = w + d + l
        if total > 0:
            result["h2h_home_win_pct"] = round(w / total * 100, 1)
            result["h2h_draw_pct"] = round(d / total * 100, 1)
            result["h2h_away_win_pct"] = round(l / total * 100, 1)
            return result

    match = re.search(pattern_pct, work, re.IGNORECASE)
    if match:
        result["h2h_home_win_pct"] = float(match.group(1))
        result["h2h_draw_pct"] = float(match.group(2))
        result["h2h_away_win_pct"] = float(match.group(3))
        return result

    pattern_slash = r"(\d+)W\s+(\d+)D\s+(\d+)L\s*/\s*(\d+)W\s+(\d+)D\s+(\d+)L"
    match = re.search(pattern_slash, work, re.IGNORECASE)
    if match:
        w1, d1, l1 = int(match.group(1)), int(match.group(2)), int(match.group(3))
        total1 = w1 + d1 + l1
        if total1 > 0:
            result["h2h_home_win_pct"] = round(w1 / total1 * 100, 1)
            result["h2h_draw_pct"] = round(d1 / total1 * 100, 1)
            result["h2h_away_win_pct"] = round(l1 / total1 * 100, 1)
            return result

    pattern_dash = r"(\d+)-(\d+)-(\d+)"
    match = re.search(pattern_dash, work)
    if match:
        w, d, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
        total = w + d + l
        if total > 0 and total <= 50:
            result["h2h_home_win_pct"] = round(w / total * 100, 1)
            result["h2h_draw_pct"] = round(d / total * 100, 1)
            result["h2h_away_win_pct"] = round(l / total * 100, 1)
            return result

    pattern_text = r"(?:Home\s+)?(?:Team\s+)?won\s+(\d+).*?drew?\s+(\d+).*?lost?\s+(\d+)"
    match = re.search(pattern_text, work, re.IGNORECASE)
    if match:
        w, d, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
        total = w + d + l
        if total > 0:
            result["h2h_home_win_pct"] = round(w / total * 100, 1)
            result["h2h_draw_pct"] = round(d / total * 100, 1)
            result["h2h_away_win_pct"] = round(l / total * 100, 1)
            return result

    return result


def convert_nowgoal_line_to_software(line_str: str | float, invert_sign: bool = True) -> float:
    """
    Converte una linea AH o Total dal formato Nowgoal al formato del software.
    
    CONVERSIONI:
    1. Quarter Lines: "-0.5/1" → -0.75 (media dei due valori)
    2. Inversione Segno: Nowgoal 0.5 → Software -0.5 (quando casa è favorita)
    
    Args:
        line_str: La linea nel formato Nowgoal (può essere stringa o numero)
        invert_sign: Se True, inverte il segno (per AH lines).
                    Per Total lines, usare False.
    
    Returns:
        La linea convertita nel formato del software.
    
    Examples:
        >>> convert_nowgoal_line_to_software("0.5")      # Casa favorita su Nowgoal
        -0.5                                            # Software: casa favorita = negativo
        >>> convert_nowgoal_line_to_software("-0.5/1")   # Quarter line
        0.75                                            # Media di -0.5 e 1, poi segno invertito
        >>> convert_nowgoal_line_to_software("0.5/1", invert_sign=False)  # Total line
        0.75
    """
    # Se è già un numero, converti direttamente
    if isinstance(line_str, (int, float)):
        nowgoal_value = float(line_str)
        return -nowgoal_value if invert_sign else nowgoal_value
    
    # Converti in stringa e pulisci
    line_str = str(line_str).strip().replace(' ', '')
    
    if not line_str:
        return 0.0
    
    try:
        # Quarter line: "-0.5/1" o "0.5/1"
        if '/' in line_str:
            parts = line_str.split('/')
            if len(parts) == 2:
                val1 = float(parts[0])
                val2 = float(parts[1])
                nowgoal_value = (val1 + val2) / 2.0
            else:
                # Formato inatteso, prova a parsare come singolo valore
                nowgoal_value = float(line_str.replace('/', ''))
        else:
            nowgoal_value = float(line_str)
        
        # Inverti il segno per AH (Nowgoal: positivo = casa favorita)
        # Software: negativo = casa favorita
        if invert_sign:
            return -nowgoal_value
        else:
            return nowgoal_value
            
    except (ValueError, TypeError):
        return 0.0


def _format_absence_line_for_mult(role_code: str, player_name: str) -> str:
    """Stringa compatibile con calcola_assenze_mult / _parse_player_absence."""
    r = role_code.upper().strip()
    name = re.sub(r"\s+", " ", (player_name or "").strip())
    if not name or name.lower().startswith("no data"):
        return ""
    if r == "GK":
        tag = "GK"
    elif r in ("CB", "LB", "RB", "WB", "LWB", "RWB"):
        tag = "CB"
    elif r in ("CM", "DM", "AM", "LM", "RM", "MC"):
        tag = "CM"
    elif r in ("LW", "RW"):
        tag = r
    elif r in ("CF", "ST", "SS", "FW"):
        tag = "CF"
    else:
        tag = "CM"
    return f"{name} ({tag}, injured)"


def _trim_nowgoal_injury_body(body: str) -> str:
    """Ferma il blocco prima di Last Match Lineups / formazioni (stesso blocco Injury su Nowgoal)."""
    out_lines: list[str] = []
    for line in body.splitlines():
        if re.search(r"Last Match Lineups", line, re.IGNORECASE):
            break
        if re.match(r"^\s*Lineups\s*\(", line.strip(), re.IGNORECASE):
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def _parse_nowgoal_injury_block(body: str) -> list[str]:
    """Estrae righe **CM** 8 Nome Cognome dalla sezione Injury/Suspension Nowgoal."""
    body = _trim_nowgoal_injury_body(body)
    out: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        m = re.match(r"^\*{0,2}([A-Z]{2,4})\*{0,2}\s+\d+\s+(.+)$", s)
        if not m:
            continue
        role, rest = m.group(1), m.group(2).strip()
        # Jina: nomi spesso come markdown link `[Nome](url)`
        link_m = re.match(r"^\[([^\]]+)\]\([^)]*\)\s*$", rest)
        if link_m:
            rest = link_m.group(1).strip()
        else:
            rest = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", rest).strip()
        formatted = _format_absence_line_for_mult(role, rest)
        if formatted:
            out.append(formatted)
    return out


def _parse_nowgoal_injury_column_html(fragment: str) -> list[str]:
    """Colonna injuryH / injuryG nell'HTML Nowgoal (player-row con <b>RUOLO</b>)."""
    if not fragment or not fragment.strip():
        return []
    if "player-row" not in fragment.lower():
        return []
    rows = re.findall(
        r'<div[^>]*\bclass\s*=\s*["\'][^"\']*player-row[^"\']*["\'][^>]*>\s*'
        r'<b>\s*([A-Z]{2,4})\s*</b>\s*'
        r'<span>\s*(\d+)\s*</span>\s*'
        r'<a[^>]*>\s*([^<]*?)\s*</a>',
        fragment,
        re.DOTALL | re.IGNORECASE,
    )
    out: list[str] = []
    for role, _num, name in rows:
        line = _format_absence_line_for_mult(role.strip().upper(), name.strip())
        if line:
            out.append(line)
    return out[:8]


def _extract_injury_player_lists_from_nowgoal_html(html: str) -> tuple[list[str], list[str]]:
    """
    Estrae infortuni dalle colonne id=injuryH (casa) e id=injuryG (trasferta).
    Necessario perché Jina spesso tronca o omette i nomi; il markdown usa <b>CF</b> non **CF**.
    """
    if not html or ("injuryH" not in html and "injuryG" not in html):
        return [], []
    home_m = re.search(
        r'<\s*div[^>]*\bid\s*=\s*["\']injuryH["\'][^>]*>(.*?)'
        r'(?=<\s*div[^>]*\bid\s*=\s*["\']injuryG["\'])',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    away_m = re.search(
        r'<\s*div[^>]*\bid\s*=\s*["\']injuryG["\'][^>]*>(.*?)'
        r'(?=<\s*div[^>]*\bid\s*=\s*["\']lineupH["\']|'
        r'<\s*div[^>]*\bid\s*=\s*["\']lineupG["\']|'
        r'Last Match Lineups)',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    hl = _parse_nowgoal_injury_column_html(home_m.group(1)) if home_m else []
    al = _parse_nowgoal_injury_column_html(away_m.group(1)) if away_m else []
    return hl, al


def _split_nowgoal_injury_markdown_scope(rest: str) -> tuple[list[str], list[str]]:
    """Parte dopo 'Injury and Suspension' (o equivalente): split su righe 'Injury' isolate."""
    end_m = re.search(r"\bLast Match Lineups\b", rest, re.IGNORECASE)
    if not end_m:
        end_m = re.search(r"(?m)^\s*Lineups\s*\(", rest, re.IGNORECASE)
    if not end_m:
        end_m = re.search(r"\n(?:#{0,3}\s*)?Fixture\s*\(", rest, re.IGNORECASE)
    scope = rest[: end_m.start()] if end_m else rest[:12000]
    parts = re.split(r"(?m)^\s*Injury\s*$", scope)
    if len(parts) < 2:
        return [], []
    home_list = _parse_nowgoal_injury_block(parts[1])
    away_list: list[str] = []
    if len(parts) >= 3:
        away_list = _parse_nowgoal_injury_block(parts[2])
    return home_list, away_list


def _extract_nowgoal_injury_player_lists(text: str) -> tuple[list[str], list[str]]:
    """
    Formato pagina Analysis Nowgoal: due blocchi separati da una riga 'Injury' (team casa / team ospite).
    """
    m = re.search(r"Injur(?:y|ies)\s+and\s+Suspensions?", text, re.IGNORECASE)
    if m:
        return _split_nowgoal_injury_markdown_scope(text[m.end() :])
    # Jina omette spesso il titolo: cerca un blocco dove dopo "Injury" isolata compaiono righe **RUOLO**
    for inj in re.finditer(r"(?m)^\s*Injury\s*$", text):
        tail = text[inj.end() :]
        if not re.search(r"\*\*[A-Z]{2,4}\*\*", tail[:8000]):
            continue
        home_l, away_l = _split_nowgoal_injury_markdown_scope(tail)
        if home_l or away_l:
            return home_l, away_l
    return [], []


def _bullet_absence_player_lists(text: str) -> tuple[list[str], list[str]]:
    """Home/Away con elenco puntato (snapshot test / markdown semplice)."""
    section = re.search(
        r"(?:^|\n)#{0,3}\s*Injur(?:y|ies)\s*(?:&|and)\s*Suspensions?\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    scope = section.group(1) if section else ""
    if not scope.strip():
        m2 = re.search(
            r"Injur(?:y|ies)\s+and\s+Suspensions?\s*\n(.*?)(?=\n##\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        scope = m2.group(1) if m2 else ""

    def _lines_from_block(block: str) -> list[str]:
        acc: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            bm = re.match(r"[-*]\s+(.+)", line)
            if not bm:
                continue
            raw = bm.group(1).strip()
            name = re.split(r"\s*\(", raw, maxsplit=1)[0].strip()
            if not name:
                continue
            rl = raw.lower()
            role_tag = "CM"
            if "gk" in rl or "goalkeeper" in rl:
                role_tag = "GK"
            elif re.search(r"\b(st|cf|fw|forward|striker)\b", rl):
                role_tag = "CF"
            elif re.search(r"\b(cb|lb|rb|def)\b", rl):
                role_tag = "CB"
            elif re.search(r"\b(cm|dm|am|mid)\b", rl):
                role_tag = "CM"
            acc.append(f"{name} ({role_tag}, injured)")
        return acc

    home_m = re.search(r"(?is)Home\s*\n(.*?)(?=^\s*Away\s*$)", scope, re.MULTILINE)
    away_m = re.search(r"(?is)Away\s*\n(.*?)(?=^\s*Home\s*$|\Z)", scope, re.MULTILINE)
    hl = _lines_from_block(home_m.group(1)) if home_m else []
    al = _lines_from_block(away_m.group(1)) if away_m else []
    return hl, al


def _extract_absence_counts_fallback(text: str) -> tuple[int, int]:
    """Solo conteggi quando non ci sono liste strutturate (numeri espliciti o bullet count)."""
    section = re.search(
        r"(Injuries(?:\s*&\s*Suspensions)?|Injuries and Suspensions)(.*?)(?:^##\s|\Z)",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    scope = section.group(2) if section else text
    home = away = 0

    m_home = re.search(
        r"(Home|Casa)[^\n]{0,40}(?:injur|suspend|assenz|indisponib)[^\n]*?(\d+)",
        scope,
        re.IGNORECASE,
    )
    m_away = re.search(
        r"(Away|Trasf|Ospit)[^\n]{0,40}(?:injur|suspend|assenz|indisponib)[^\n]*?(\d+)",
        scope,
        re.IGNORECASE,
    )
    if m_home:
        home = int(m_home.group(2))
    if m_away:
        away = int(m_away.group(2))

    if home == 0 or away == 0:
        home_block = re.search(r"(Home|Casa)(.*?)(?:Away|Trasf|Ospit|$)", scope, re.IGNORECASE | re.DOTALL)
        away_block = re.search(r"(Away|Trasf|Ospit)(.*?)(?:Home|Casa|$)", scope, re.IGNORECASE | re.DOTALL)
        if home == 0 and home_block:
            lines = [ln.strip() for ln in home_block.group(2).splitlines()]
            home = sum(
                1
                for ln in lines
                if re.search(
                    r"(^[-*]\s+)|\b(out|injur|suspend|assente|infortun|doubtful)\b",
                    ln,
                    re.IGNORECASE,
                )
            )
        if away == 0 and away_block:
            lines = [ln.strip() for ln in away_block.group(2).splitlines()]
            away = sum(
                1
                for ln in lines
                if re.search(
                    r"(^[-*]\s+)|\b(out|injur|suspend|assente|infortun|doubtful)\b",
                    ln,
                    re.IGNORECASE,
                )
            )

    return max(0, min(8, home)), max(0, min(8, away))


def _extract_absences_full(text: str) -> tuple[int, int, list[str], list[str]]:
    """
    Conteggi + liste giocatore per calcola_assenze_mult.
    Ordine: HTML injuryH/G (RAW) → Nowgoal markdown → bullet Home/Away → fallback numeri.
    """
    marker = "=== RAW HTML ==="
    if marker in text:
        html_part = text.split(marker, 1)[1].strip()
        h_html, a_html = _extract_injury_player_lists_from_nowgoal_html(html_part)
        if h_html or a_html:
            h_html, a_html = h_html[:8], a_html[:8]
            return len(h_html), len(a_html), h_html, a_html

    home_l, away_l = _extract_nowgoal_injury_player_lists(text)
    if home_l or away_l:
        home_l = home_l[:8]
        away_l = away_l[:8]
        return len(home_l), len(away_l), home_l, away_l

    hb, ab = _bullet_absence_player_lists(text)
    if hb or ab:
        hb, ab = hb[:8], ab[:8]
        return len(hb), len(ab), hb, ab

    h, a = _extract_absence_counts_fallback(text)
    return h, a, [], []


def _extract_absence_counts(text: str) -> tuple[int, int]:
    """Solo conteggi (compat live e test)."""
    h, a, _, _ = _extract_absences_full(text)
    return h, a


def _extract_identity_from_url(url: str) -> tuple[str, str, str]:
    """Fallback identity parser directly from URL path/query."""
    home = away = league = ""
    # Path style: /match/h2h-2871782/adelaide-united-vs-auckland-fc
    slug = re.search(r"/match/(?:h2h|live|detail)-\d+/([a-z0-9\-]+)", url, re.IGNORECASE)
    if slug:
        parts = slug.group(1).split("-vs-")
        if len(parts) == 2:
            home = _clean_team_name(parts[0].replace("-", " ").title())
            away = _clean_team_name(parts[1].replace("-", " ").title())

    # Query hint: ?league=AUS%20D1 or competition=AUS D1-4
    q_league = re.search(r"(?:league|competition)=([A-Za-z0-9%+\-\s]+)", url, re.IGNORECASE)
    if q_league:
        raw = urllib.parse.unquote_plus(q_league.group(1))
        _, _, league = _extract_match_identity_from_text(raw)
        if not league:
            league = _clean_league_name(raw)

    return home, away, league


def _extract_all_with_regex(text: str) -> dict:
    """
    Estrae TUTTI i dati prematch dal testo usando solo regex.
    Metodo PRIMARIO - GRATUITO, senza API key, senza limiti.
    
    Estrae:
    - H2H (win/draw/lose %, over %, media gol)
    - Strength comparison
    - Standings casa/trasferta
    - Previous scores
    - Quote iniziali (se presenti)
    """
    result = {
        # H2H
        "h2h_home_win_pct": 0.0,
        "h2h_draw_pct": 0.0,
        "h2h_away_win_pct": 0.0,
        "h2h_over_pct": 0.0,
        "h2h_btts_pct": 0.0,
        "h2h_matches_count": 0,
        "h2h_avg_goals_home": 0.0,
        "h2h_avg_goals_away": 0.0,
        # Strength
        "strength_home": 0,
        "strength_away": 0,
        # Standings casa
        "home_rank": 0,
        "home_matches": 0,
        "home_win": 0,
        "home_draw": 0,
        "home_lose": 0,
        "home_scored": 0,
        "home_conceded": 0,
        "home_win_rate": 0.0,
        "home_home_win": 0,
        "home_home_draw": 0,
        "home_home_lose": 0,
        "home_home_scored": 0,
        "home_home_conceded": 0,
        "home_last6_win": 0,
        "home_last6_draw": 0,
        "home_last6_lose": 0,
        "home_last6_scored": 0,
        "home_last6_conceded": 0,
        # HT standings casa
        "home_ht_win": 0,
        "home_ht_draw": 0,
        "home_ht_lose": 0,
        # Standings trasferta
        "away_rank": 0,
        "away_matches": 0,
        "away_win": 0,
        "away_draw": 0,
        "away_lose": 0,
        "away_scored": 0,
        "away_conceded": 0,
        "away_win_rate": 0.0,
        "away_away_win": 0,
        "away_away_draw": 0,
        "away_away_lose": 0,
        "away_away_scored": 0,
        "away_away_conceded": 0,
        "away_last6_win": 0,
        "away_last6_draw": 0,
        "away_last6_lose": 0,
        "away_last6_scored": 0,
        "away_last6_conceded": 0,
        # HT standings trasferta
        "away_ht_win": 0,
        "away_ht_draw": 0,
        "away_ht_lose": 0,
        # Previous scores
        "prev_home_win_pct": 0.0,
        "prev_home_avg_scored": 0.0,
        "prev_home_avg_conceded": 0.0,
        "prev_home_over_pct": 0.0,
        "prev_away_win_pct": 0.0,
        "prev_away_avg_scored": 0.0,
        "prev_away_avg_conceded": 0.0,
        "prev_away_over_pct": 0.0,
        # Quote iniziali
        "mkt_init_1": 0.0,
        "mkt_init_x": 0.0,
        "mkt_init_2": 0.0,
        # Info partita
        "home_team": "",
        "away_team": "",
        "league_name": "",
        "match_date": "",
        # === NUOVI CAMPI ===
        # Partite recenti (da h_data/a_data JS)
        "home_recent_results": [],      # [(gf, gs), ...]
        "away_recent_results": [],      # [(gf, gs), ...]
        "home_form_trend": 0.0,
        "away_form_trend": 0.0,
        "home_xg_from_recent": 0.0,
        "away_xg_from_recent": 0.0,
        "scoring_streak_h": 0,
        "scoring_streak_a": 0,
        "clean_sheet_streak_h": 0,
        "clean_sheet_streak_a": 0,
        # Quote multi-bookmaker (da Vs_hOdds JS)
        "ah_line_open": 0.0,
        "ah_line_close": 0.0,
        "ah_home_odds_open": 0.0,
        "ah_away_odds_open": 0.0,
        "total_line_open": 0.0,
        "total_line_close": 0.0,
        "total_over_odds_open": 0.0,
        "total_under_odds_open": 0.0,
        "line_movement_ah": 0.0,
        "line_movement_total": 0.0,
        "odds_sharp_signal": 0.0,
        # Punti classifica
        "home_points": 0,
        "away_points": 0,
        "home_motivation": "normal",
        "away_motivation": "normal",
        "home_absences_count": 0,
        "away_absences_count": 0,
        "home_absences_players": [],
        "away_absences_players": [],
        "home_id": 0,
        "away_id": 0,
        # Team Statistics (ultimi 10 match)
        "team_stats_home_goals": 0.0,
        "team_stats_home_conceded": 0.0,
        "team_stats_home_shots": 0.0,
        "team_stats_home_corners": 0.0,
        "team_stats_home_yellows": 0.0,
        "team_stats_home_fouls": 0.0,
        "team_stats_home_possession": 0.0,
        "team_stats_away_goals": 0.0,
        "team_stats_away_conceded": 0.0,
        "team_stats_away_shots": 0.0,
        "team_stats_away_corners": 0.0,
        "team_stats_away_yellows": 0.0,
        "team_stats_away_fouls": 0.0,
        "team_stats_away_possession": 0.0,
        # Qualita' estrazione
        "extraction_notes": [],
        "extraction_section_scores": {},
    }

    # === H2H ===
    pre_ps = _text_before_previous_scores_statistics(text)
    id_h2h_home, id_h2h_away, _ = _extract_match_identity_from_text(text)

    h2h = _extract_h2h_with_regex(text)
    result["h2h_home_win_pct"] = h2h["h2h_home_win_pct"]
    result["h2h_draw_pct"] = h2h["h2h_draw_pct"]
    result["h2h_away_win_pct"] = h2h["h2h_away_win_pct"]

    h2h_table_body = _head_to_head_table_slice(text)
    parsed_h2h_rows = _parse_h2h_score_rows(h2h_table_body)
    derived_h2h = (
        _derive_h2h_from_score_table(parsed_h2h_rows, id_h2h_home, id_h2h_away)
        if parsed_h2h_rows and id_h2h_home and id_h2h_away
        else None
    )
    if derived_h2h:
        result["h2h_home_win_pct"] = derived_h2h["h2h_home_win_pct"]
        result["h2h_draw_pct"] = derived_h2h["h2h_draw_pct"]
        result["h2h_away_win_pct"] = derived_h2h["h2h_away_win_pct"]
        result["h2h_avg_goals_home"] = derived_h2h["h2h_avg_goals_home"]
        result["h2h_avg_goals_away"] = derived_h2h["h2h_avg_goals_away"]
        result["h2h_btts_pct"] = derived_h2h["h2h_btts_pct"]
        result["h2h_matches_count"] = derived_h2h["h2h_matches_count"]
    elif parsed_h2h_rows:
        btts_hits = sum(1 for _th, _ta, gh, ga in parsed_h2h_rows if gh > 0 and ga > 0)
        result["h2h_btts_pct"] = round(btts_hits * 100.0 / len(parsed_h2h_rows), 1)
        result["h2h_matches_count"] = len(parsed_h2h_rows)

    # H2H Over % — preferisci la sezione tabella H2H, non la prima "Over" della pagina.
    over_scope = h2h_table_body if h2h_table_body.strip() else pre_ps
    over_match = re.search(r"(\d+(?:\.\d+)?)%\s*Over", over_scope, re.IGNORECASE)
    if over_match:
        result["h2h_over_pct"] = float(over_match.group(1))
    else:
        over_match2 = re.search(r"Over\s*(\d+(?:\.\d+)?)\s*%?", over_scope, re.IGNORECASE)
        if over_match2:
            result["h2h_over_pct"] = float(over_match2.group(1))

    # H2H media gol da testo solo se non ricavate dalla tabella punteggi.
    if not derived_h2h and (
        result["h2h_home_win_pct"] > 0
        or result["h2h_draw_pct"] > 0
        or result["h2h_away_win_pct"] > 0
    ):
        h2h_goals_pattern = r"(\d+[.,]\d+)\s*goals?\s*Goal\s*Score/Loss\s*per\s*Game\s*(\d+[.,]\d+)\s*goals?"
        h2h_goals_match = re.search(h2h_goals_pattern, pre_ps, re.IGNORECASE)
        if h2h_goals_match:
            result["h2h_avg_goals_home"] = float(h2h_goals_match.group(1).replace(",", "."))
            result["h2h_avg_goals_away"] = float(h2h_goals_match.group(2).replace(",", "."))
        else:
            h2h_section = re.search(
                r"(?:head\s*to\s*head|h2h)\s*statistics?(.*?)(?:previous\s*score|who\s*will\s*win|$)",
                pre_ps,
                re.IGNORECASE | re.DOTALL,
            )
            if h2h_section:
                h2h_text = h2h_section.group(1)
                goals_pattern = r"(\d+[.,]\d+)\s*goals?"
                goals_matches = re.findall(goals_pattern, h2h_text, re.IGNORECASE)
                if len(goals_matches) >= 2:
                    result["h2h_avg_goals_home"] = float(goals_matches[0].replace(",", "."))
                    result["h2h_avg_goals_away"] = float(goals_matches[1].replace(",", "."))

    # === STRENGTH ===
    # Nowgoal mostra la Strength come due numeri su righe isolate, seguiti dalle etichette:
    #   62
    #
    #   38
    #
    #   *   H2H
    #   *   State
    #   *   Attack
    # NOTA: NON usare "/" come separatore perché matcherebbe erroneamente "2/2" da "2/2.5" nelle quote!
    strength_found = False

    # Pattern 1: Due numeri su righe isolate seguiti da H2H/State/Attack (formato Nowgoal via Jina)
    # Cerca: numero \n\n numero \n\n seguito da H2H o State entro 200 caratteri
    strength_nowgoal = re.search(
        r"\n\s*(\d{1,3})\s*\n+\s*(\d{1,3})\s*\n.{0,200}?(?:H2H|State|Attack|Defence)",
        text, re.IGNORECASE | re.DOTALL
    )
    if strength_nowgoal:
        s1, s2 = int(strength_nowgoal.group(1)), int(strength_nowgoal.group(2))
        # Validazione: entrambi devono essere > 0 e somma ragionevole (non sono quote o piccoli numeri)
        if s1 > 0 and s2 > 0 and s1 + s2 > 20:
            result["strength_home"] = s1
            result["strength_away"] = s2
            strength_found = True

    # Pattern 2: "Strength: 60 vs 40" o "Strength Comparison: 60 - 40"
    if not strength_found:
        strength_labeled = re.search(
            r"Strength[^:]*?:\s*(\d{1,3})\s*(?:vs|VS|-|–)\s*(\d{1,3})",
            text, re.IGNORECASE
        )
        if strength_labeled:
            s1, s2 = int(strength_labeled.group(1)), int(strength_labeled.group(2))
            if 0 <= s1 <= 100 and 0 <= s2 <= 100:
                result["strength_home"] = s1
                result["strength_away"] = s2
                strength_found = True

    # Pattern 3: "% H2H Comparison %" - es: "71%H2H Comparison 29%"
    # Questi sono i percentuali di forza relativa
    if not strength_found:
        h2h_comp_match = re.search(r"(\d{1,3})%\s*H2H\s*Comparison\s*(\d{1,3})%", text, re.IGNORECASE)
        if h2h_comp_match:
            s1, s2 = int(h2h_comp_match.group(1)), int(h2h_comp_match.group(2))
            # Questi sono percentuali, non valori assoluti. Convertiamo in punteggio 0-100
            if s1 + s2 == 100:
                result["strength_home"] = s1
                result["strength_away"] = s2
                strength_found = True
    
    # === STANDINGS ===
    # Cerca tabelle con righe Total/Home/Away/Last 6
    # Nowgoal ha struttura: [Rank] NomeSquadra → FT section → HT section per ogni squadra
    
    # Trova le posizioni delle sezioni squadra (es. "[JPN D1-1] Vissel Kobe")
    team_markers = list(re.finditer(r'\[[^\]]+\]\s*([A-Za-z][A-Za-z\s\-\'\.]+)', text))
    
    # Estrai le sezioni FT e HT per ogni squadra
    ft_home_section = ""
    ht_home_section = ""
    ft_away_section = ""
    ht_away_section = ""
    
    if len(team_markers) >= 2:
        # Prima squadra (casa): dal match 0 al match 1 (o fine testo)
        home_start = team_markers[0].start()
        home_end = team_markers[1].start() if len(team_markers) > 1 else len(text)
        home_team_text = text[home_start:home_end]
        
        # Seconda squadra (trasferta): dal match 1 alla fine (o prossima squadra)
        away_start = team_markers[1].start()
        away_end = team_markers[2].start() if len(team_markers) > 2 else len(text)
        away_team_text = text[away_start:away_end]
        
        # Separa FT e HT per casa (HT inizia con "HT Matches" o "HT " seguito da tabella)
        ht_split_home = re.split(r'\n\s*HT\s+Matches', home_team_text, flags=re.IGNORECASE)
        ft_home_section = ht_split_home[0] if len(ht_split_home) >= 1 else ""
        ht_home_section = "HT Matches" + ht_split_home[1] if len(ht_split_home) >= 2 else ""
        
        # Separa FT e HT per trasferta
        ht_split_away = re.split(r'\n\s*HT\s+Matches', away_team_text, flags=re.IGNORECASE)
        ft_away_section = ht_split_away[0] if len(ht_split_away) >= 1 else ""
        ht_away_section = "HT Matches" + ht_split_away[1] if len(ht_split_away) >= 2 else ""
    
    # === FT DATA ===
    # Riga Total FT: Matches, Win, Draw, Lose, Scored, Conceded
    total_pattern = r"Total\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    
    # FT Casa
    ft_home_total = re.search(total_pattern, ft_home_section, re.IGNORECASE)
    if ft_home_total:
        t = ft_home_total.groups()
        result["home_matches"] = int(t[0])
        result["home_win"] = int(t[1])
        result["home_draw"] = int(t[2])
        result["home_lose"] = int(t[3])
        result["home_scored"] = int(t[4])
        result["home_conceded"] = int(t[5])
    
    # FT Trasferta
    ft_away_total = re.search(total_pattern, ft_away_section, re.IGNORECASE)
    if ft_away_total:
        t = ft_away_total.groups()
        result["away_matches"] = int(t[0])
        result["away_win"] = int(t[1])
        result["away_draw"] = int(t[2])
        result["away_lose"] = int(t[3])
        result["away_scored"] = int(t[4])
        result["away_conceded"] = int(t[5])
    
    # === HT DATA ===
    # Riga Total HT: Matches, Win, Draw, Lose, Scored, Conceded
    # HT Casa
    ht_home_total = re.search(total_pattern, ht_home_section, re.IGNORECASE)
    if ht_home_total:
        t = ht_home_total.groups()
        result["home_ht_win"] = int(t[1])
        result["home_ht_draw"] = int(t[2])
        result["home_ht_lose"] = int(t[3])
    
    # HT Trasferta
    ht_away_total = re.search(total_pattern, ht_away_section, re.IGNORECASE)
    if ht_away_total:
        t = ht_away_total.groups()
        result["away_ht_win"] = int(t[1])
        result["away_ht_draw"] = int(t[2])
        result["away_ht_lose"] = int(t[3])
    
    # Riga Home FT (performance in casa)
    # Es: "Home  5  1  2  2  7  7" (matches, win, draw, lose, scored, conceded)
    # NOTA: la prima colonna è Matches, NON Win!
    # NOTA: ci sono DUE righe "Home" nel testo (una per ogni squadra).
    # La PRIMA è quella della squadra di casa che ci interessa!
    home_pattern = r"(?:^|\n)\s*Home\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    home_matches = re.findall(home_pattern, text, re.IGNORECASE | re.MULTILINE)
    if home_matches:
        m = home_matches[0]  # Prima riga "Home" (squadra di casa)
        # Group 0 = Matches (ignorato)
        result["home_home_win"] = int(m[1])
        result["home_home_draw"] = int(m[2])
        result["home_home_lose"] = int(m[3])
        result["home_home_scored"] = int(m[4])
        result["home_home_conceded"] = int(m[5])
    
    # Riga Away FT (performance in trasferta)
    # Es: "Away  6  3  0  3  7  6" (matches, win, draw, lose, scored, conceded)
    # NOTA: ci sono QUATTRO righe "Away" nel testo (FT e HT per entrambe le squadre).
    # L'ordine è: 1) FT casa, 2) HT casa, 3) FT trasferta, 4) HT trasferta
    # La TERZA è quella della squadra di trasferta che ci interessa!
    away_pattern = r"(?:^|\n)\s*Away\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    away_matches = re.findall(away_pattern, text, re.IGNORECASE | re.MULTILINE)
    if away_matches:
        # Prendi la TERZA occorrenza (FT trasferta)
        if len(away_matches) >= 3:
            m = away_matches[2]  # Terza riga "Away" (FT squadra trasferta)
        elif len(away_matches) >= 2:
            m = away_matches[1]  # Fallback: seconda riga
        else:
            m = away_matches[0]  # Fallback: prima riga
        # m[0] = Matches (ignorato)
        result["away_away_win"] = int(m[1])
        result["away_away_draw"] = int(m[2])
        result["away_away_lose"] = int(m[3])
        result["away_away_scored"] = int(m[4])
        result["away_away_conceded"] = int(m[5])
    
    # Riga Last 6 (solo blocco FT — vedi sezioni ft_home_section / ft_away_section)
    # Es. Nowgoal: Last 6 | 6 | 2 | 1 | 3 | 8 | 10 | 7 | 33.3%  → scored=8, conceded=10
    last6_re = re.compile(
        r"Last\s*6\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+(\d+)\s+(\d+))?",
        re.IGNORECASE,
    )

    def _apply_last6_row(section: str, prefix: str) -> None:
        m6 = last6_re.search(section)
        if not m6:
            return
        result[f"{prefix}_last6_win"] = int(m6.group(2))
        result[f"{prefix}_last6_draw"] = int(m6.group(3))
        result[f"{prefix}_last6_lose"] = int(m6.group(4))
        if m6.group(5) and m6.group(6):
            result[f"{prefix}_last6_scored"] = int(m6.group(5))
            result[f"{prefix}_last6_conceded"] = int(m6.group(6))

    if ft_home_section:
        _apply_last6_row(ft_home_section, "home")
    if ft_away_section:
        _apply_last6_row(ft_away_section, "away")

    # Fallback su tutto il testo: ordine Last 6 = FT casa, HT casa, FT trasferta, HT trasferta
    _home_l6_empty = result["home_last6_win"] + result["home_last6_draw"] + result["home_last6_lose"] == 0
    _away_l6_empty = result["away_last6_win"] + result["away_last6_draw"] + result["away_last6_lose"] == 0
    rows_fb = last6_re.findall(text) if (_home_l6_empty or _away_l6_empty) else []
    if _home_l6_empty and len(rows_fb) >= 1:
        l = rows_fb[0]
        result["home_last6_win"] = int(l[1])
        result["home_last6_draw"] = int(l[2])
        result["home_last6_lose"] = int(l[3])
        if l[4] and l[5]:
            result["home_last6_scored"] = int(l[4])
            result["home_last6_conceded"] = int(l[5])
    if _away_l6_empty and rows_fb:
        if len(rows_fb) >= 3:
            l = rows_fb[2]
        elif len(rows_fb) >= 2:
            l = rows_fb[1]
        else:
            l = None
        if l is not None:
            result["away_last6_win"] = int(l[1])
            result["away_last6_draw"] = int(l[2])
            result["away_last6_lose"] = int(l[3])
            if l[4] and l[5]:
                result["away_last6_scored"] = int(l[4])
                result["away_last6_conceded"] = int(l[5])
    
    # === RANK ===
    # Es: "[SPA D2-3]" → rank=3 o "Rank: 3"
    rank_pattern = r"\[[^\]]*-(\d+)\]"
    rank_matches = re.findall(rank_pattern, text)
    if rank_matches:
        if len(rank_matches) >= 1:
            result["home_rank"] = int(rank_matches[0])
        if len(rank_matches) >= 2:
            result["away_rank"] = int(rank_matches[1])
    
    # === PREVIOUS SCORES ===
    # FIX: Cerca nella sezione "Previous Scores Statistics" per evitare di
    # confondere con H2H Win (che appare prima nel testo con lo stesso pattern).
    # Se la sezione non viene trovata, fallback sul testo intero.
    _prev_section = re.search(
        r'Previous\s+Score.*?(?=\*\*Last Updated|HT/FT|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    _prev_text = _prev_section.group(0) if _prev_section else text

    # Es: "Win 8 (80%)" nella sezione Previous
    prev_win_pattern = r"Win\s+(\d+)\s*\((\d+(?:\.\d+)?)\s*%\)"
    prev_matches = re.findall(prev_win_pattern, _prev_text, re.IGNORECASE)
    if prev_matches:
        if len(prev_matches) >= 1:
            result["prev_home_win_pct"] = float(prev_matches[0][1])
        if len(prev_matches) >= 2:
            result["prev_away_win_pct"] = float(prev_matches[1][1])
    
    # Previous Over %
    # FIX: Usa pattern "% Over" invece di "Over %" per evitare garbage
    prev_over_pattern = r"(\d+(?:\.\d+)?)%\s*Over"
    prev_over_matches = re.findall(prev_over_pattern, _prev_text, re.IGNORECASE)
    if prev_over_matches:
        if len(prev_over_matches) >= 1:
            result["prev_home_over_pct"] = float(prev_over_matches[0])
        if len(prev_over_matches) >= 2:
            result["prev_away_over_pct"] = float(prev_over_matches[1])
    
    # Previous avg goals
    # FIX: Il formato reale è "1.3 goals Goal Score/Loss per Game 1.4 goals"
    # Il vecchio pattern cercava "Score" o "per Game" subito dopo "goals",
    # ma il testo ha "Goal Score/Loss per Game" (la parola "Goal" in mezzo).
    prev_goals_pattern = r"(\d+[.,]\d+)\s*goals?\s*Goal\s+Score/Loss\s+per\s+Game\s*(\d+[.,]\d+)\s*goals?"
    prev_goals_matches = re.findall(prev_goals_pattern, _prev_text, re.IGNORECASE)
    if prev_goals_matches:
        # Prima occorrenza = casa
        if len(prev_goals_matches) >= 1:
            result["prev_home_avg_scored"] = float(prev_goals_matches[0][0].replace(",", "."))
            result["prev_home_avg_conceded"] = float(prev_goals_matches[0][1].replace(",", "."))
        # Seconda occorrenza = trasferta
        if len(prev_goals_matches) >= 2:
            result["prev_away_avg_scored"] = float(prev_goals_matches[1][0].replace(",", "."))
            result["prev_away_avg_conceded"] = float(prev_goals_matches[1][1].replace(",", "."))
    
    # === QUOTE INIZIALI ===
    # Es: "1 @2.10  X @3.25  2 @3.40" o "1: 2.10  X: 3.25  2: 3.40"
    odds_pattern = r"(?:1|Home)\s*[@:]\s*(\d+[.,]\d+).*?(?:X|Draw)\s*[@:]\s*(\d+[.,]\d+).*?(?:2|Away)\s*[@:]\s*(\d+[.,]\d+)"
    odds_match = re.search(odds_pattern, text, re.IGNORECASE)
    if odds_match:
        result["mkt_init_1"] = float(odds_match.group(1).replace(",", "."))
        result["mkt_init_x"] = float(odds_match.group(2).replace(",", "."))
        result["mkt_init_2"] = float(odds_match.group(3).replace(",", "."))
    
    # Pattern per tabella markdown Live Odds Analysis (Jina Reader)
    # Formato: | **Bet365** | Initial | AH_home | AH_line | AH_away | 1 | X | 2 | Over | Total | Under |
    # Es: | **Bet365** | Initial | 0.80 | 1.5 | 1.05 | 1.20 | 5.50 | 12.00 | 0.88 | 2/2.5 | 0.98 |
    if result["mkt_init_1"] == 0:
        # Pattern per tabella odds con Initial row
        table_odds_pattern = r'\|\s*\*?\*?Bet365\*?\*?\s*\|\s*Initial\s*\|\s*[\d.]+\s*\|\s*[\d./]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        table_match = re.search(table_odds_pattern, text, re.IGNORECASE)
        if table_match:
            result["mkt_init_1"] = float(table_match.group(1))
            result["mkt_init_x"] = float(table_match.group(2))
            result["mkt_init_2"] = float(table_match.group(3))
    
    # Pattern alternativo: cerca 1X2 in riga con "Initial"
    if result["mkt_init_1"] == 0:
        # Cerca pattern: "Initial" seguito da valori che sembrano quote 1X2
        initial_pattern = r'Initial\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        init_match = re.search(initial_pattern, text, re.IGNORECASE)
        if init_match:
            q1, qx, q2 = float(init_match.group(1)), float(init_match.group(2)), float(init_match.group(3))
            # Verifica che sembrino quote valide (1.0 < quota < 50.0)
            if 1.0 < q1 < 50.0 and 1.0 < qx < 50.0 and 1.0 < q2 < 50.0:
                result["mkt_init_1"] = q1
                result["mkt_init_x"] = qx
                result["mkt_init_2"] = q2
    
    # === INFO PARTITA ===
    id_home, id_away, id_league = _extract_match_identity_from_text(text)
    if id_home:
        result["home_team"] = id_home
    if id_away:
        result["away_team"] = id_away
    if id_league:
        result["league_name"] = id_league

    # Nomi squadre: "Team A vs Team B" o "Team A - Team B"
    # FIX: Usa spazio letterale (non \s) per evitare match su newlines.
    # Aggiunge word boundary alla fine per catturare il nome completo.
    teams_match = re.search(
        r"^([A-Za-z][A-Za-z ]{2,30}?)\s+(?:vs|VS|-|–)\s+([A-Za-z][A-Za-z ]{2,30}?)(?:\s*$|\s*[,\.\|])",
        text, re.MULTILINE
    )
    if teams_match:
        result["home_team"] = _clean_team_name(teams_match.group(1))
        result["away_team"] = _clean_team_name(teams_match.group(2))
    
    # Data partita: usa solo pattern contestualizzati (evita date spurie nel footer/log).
    date_match = re.search(
        r"(?:Match\s*Date|Date|Kick[-\s]?off)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})",
        text,
        re.IGNORECASE,
    )
    if date_match:
        result["match_date"] = date_match.group(1)
    
    # Lega: "League: XXX" o dopo nome squadre
    league_match = re.search(r"(?:League|Lega|Competition)\s*[:\-]?\s*([A-Za-z][A-Za-z\s0-9]{2,30})", text, re.IGNORECASE)
    if league_match:
        result["league_name"] = _clean_league_name(league_match.group(1))
    
    # =====================================================
    # === NUOVI CAMPI: ESTRAZIONE AVANZATA ===
    # =====================================================
    
    # === 1. h_data / a_data: PARTITE RECENTI (da JavaScript Nowgoal) ===
    # Formato: h_data = [[team1_id, team2_id, gol_team1, gol_team2], ...]
    # La squadra di casa è identificata da h2h_home, la trasferta da h2h_away
    
    # Cerca prima gli ID delle squadre
    home_id = None
    away_id = None
    id_match_h = re.search(r'h2h_home\s*=\s*(\d+)', text)
    if id_match_h:
        home_id = int(id_match_h.group(1))
        result["home_id"] = home_id
    id_match_a = re.search(r'h2h_away\s*=\s*(\d+)', text)
    if id_match_a:
        away_id = int(id_match_a.group(1))
        result["away_id"] = away_id
    
    # Estrai h_data (partite recenti squadra casa)
    # Pattern aggiornato per matchare formato Nowgoal
    h_data_match = re.search(r'h_data\s*=\s*(\[(?:\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*,?\s*)+\])', text)
    if h_data_match:
        try:
            h_data_str = h_data_match.group(1)
            h_data_parsed = json.loads(h_data_str)
            home_results = []
            
            for match_data in h_data_parsed[:15]:  # Max 15 partite
                if len(match_data) >= 4:
                    t1, t2, g1, g2 = int(match_data[0]), int(match_data[1]), int(match_data[2]), int(match_data[3])
                    # Determina se la squadra di casa ha giocato in casa o trasferta
                    if home_id and t1 == home_id:
                        # Casa ha giocato in casa
                        home_results.append((g1, g2))  # (gf, gs)
                    elif home_id and t2 == home_id:
                        # Casa ha giocato in trasferta
                        home_results.append((g2, g1))  # (gf, gs)
                    elif not home_id:
                        # Fallback: assume prima squadra è casa
                        if len(home_results) < 10:
                            home_results.append((g1, g2))
            
            result["home_recent_results"] = home_results
            if home_results:
                sc, cs = _compute_streaks_from_results(home_results)
                result["scoring_streak_h"] = sc
                result["clean_sheet_streak_h"] = cs
            
            # Calcola trend forma (prime 5 vs ultime 5)
            if len(home_results) >= 10:
                first5 = home_results[:5]
                last5 = home_results[-5:]
                first5_pts = sum(3 if gf > gs else (1 if gf == gs else 0) for gf, gs in first5)
                last5_pts = sum(3 if gf > gs else (1 if gf == gs else 0) for gf, gs in last5)
                result["home_form_trend"] = (last5_pts - first5_pts) / 15.0  # Normalizzato [-1, 1]
            
            # Calcola xG da partite recenti
            if home_results:
                avg_gf = sum(gf for gf, gs in home_results) / len(home_results)
                avg_gs = sum(gs for gf, gs in home_results) / len(home_results)
                result["home_xg_from_recent"] = round(avg_gf, 2)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            pass
    
    # Estrai a_data (partite recenti squadra trasferta)
    a_data_match = re.search(r'a_data\s*=\s*(\[(?:\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*,?\s*)+\])', text)
    if a_data_match:
        try:
            a_data_str = a_data_match.group(1)
            a_data_parsed = json.loads(a_data_str)
            away_results = []
            
            for match_data in a_data_parsed[:15]:
                if len(match_data) >= 4:
                    t1, t2, g1, g2 = int(match_data[0]), int(match_data[1]), int(match_data[2]), int(match_data[3])
                    if away_id and t1 == away_id:
                        away_results.append((g1, g2))
                    elif away_id and t2 == away_id:
                        away_results.append((g2, g1))
                    elif not away_id:
                        if len(away_results) < 10:
                            away_results.append((g1, g2))
            
            result["away_recent_results"] = away_results
            if away_results:
                sc, cs = _compute_streaks_from_results(away_results)
                result["scoring_streak_a"] = sc
                result["clean_sheet_streak_a"] = cs
            
            if len(away_results) >= 10:
                first5 = away_results[:5]
                last5 = away_results[-5:]
                first5_pts = sum(3 if gf > gs else (1 if gf == gs else 0) for gf, gs in first5)
                last5_pts = sum(3 if gf > gs else (1 if gf == gs else 0) for gf, gs in last5)
                result["away_form_trend"] = (last5_pts - first5_pts) / 15.0
            
            if away_results:
                avg_gf = sum(gf for gf, gs in away_results) / len(away_results)
                avg_gs = sum(gs for gf, gs in away_results) / len(away_results)
                result["away_xg_from_recent"] = round(avg_gf, 2)
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    
    # === 2. Vs_hOdds: QUOTE MULTI-BOOKMAKER (da JavaScript Nowgoal) ===
    # Formato: Vs_hOdds = [[bookmaker_id, timestamp, 'ah_home', 'ah_line', ...], ...]];
    # I valori sono stringhe con apici singoli, JSON vuole doppi apici!
    
    # Pattern più robusto: trova tutto fino al punto e virgola
    vs_odds_match = re.search(r"Vs_hOdds\s*=\s*(\[[\s\S]+?\]\s*\])\s*;", text)
    if vs_odds_match:
        try:
            odds_str = vs_odds_match.group(1)
            # Converti apici singoli in doppi per JSON
            # Attenzione: solo per i valori stringa, non per i numeri
            odds_str = re.sub(r"'([^']*)'", r'"\1"', odds_str)
            odds_data = json.loads(odds_str)
            
            # Trova opening (timestamp=1) e closing (timestamp più alto)
            opening_row = None
            closing_row = None
            max_timestamp = 0
            
            for row in odds_data:
                if len(row) >= 12:
                    ts = int(row[1]) if len(row) > 1 and row[1] else 0
                    if ts == 1:
                        opening_row = row
                    if ts > max_timestamp:
                        max_timestamp = ts
                        closing_row = row
            
            # Estrai dati da opening (timestamp=1)
            if opening_row and len(opening_row) >= 12:
                # AH: index 2=home_odds, 3=line, 4=away_odds
                # NOTA: Le linee Nowgoal hanno formato diverso e segno invertito!
                # - Quarter lines: "0.5/1" → 0.75 (media)
                # - Segno: Nowgoal 0.5 (casa favorita) → Software -0.5
                ah_line_raw = opening_row[3] if opening_row[3] else "0"
                ah_line = convert_nowgoal_line_to_software(ah_line_raw, invert_sign=True)
                ah_home = float(opening_row[2]) if opening_row[2] else 0.0
                ah_away = float(opening_row[4]) if opening_row[4] else 0.0
                # Total: index 8=line, 10=over_odds, 11=under_odds
                # Per Total: quarter lines ma SENZA inversione segno
                total_line_raw = opening_row[8] if opening_row[8] else "0"
                total_line = convert_nowgoal_line_to_software(total_line_raw, invert_sign=False)
                total_over = float(opening_row[10]) if len(opening_row) > 10 and opening_row[10] else 0.0
                total_under = float(opening_row[11]) if len(opening_row) > 11 and opening_row[11] else 0.0
                
                result["ah_line_open"] = ah_line
                result["ah_home_odds_open"] = ah_home
                result["ah_away_odds_open"] = ah_away
                result["total_line_open"] = total_line
                result["total_over_odds_open"] = total_over
                result["total_under_odds_open"] = total_under
            
            # Estrai dati da closing
            if closing_row and len(closing_row) >= 12:
                ah_line_close_raw = closing_row[3] if closing_row[3] else "0"
                ah_line_close = convert_nowgoal_line_to_software(ah_line_close_raw, invert_sign=True)
                total_line_close_raw = closing_row[8] if closing_row[8] else "0"
                total_line_close = convert_nowgoal_line_to_software(total_line_close_raw, invert_sign=False)
                
                result["ah_line_close"] = ah_line_close
                result["total_line_close"] = total_line_close
            
            # Calcola movimento
            if result["ah_line_open"] != 0 and result["ah_line_close"] != 0:
                result["line_movement_ah"] = result["ah_line_close"] - result["ah_line_open"]
            if result["total_line_open"] != 0 and result["total_line_close"] != 0:
                result["line_movement_total"] = result["total_line_close"] - result["total_line_open"]
            
            # Sharp signal: movimento significativo (>0.25) indica informazione
            if abs(result["line_movement_ah"]) >= 0.25 or abs(result["line_movement_total"]) >= 0.25:
                result["odds_sharp_signal"] = max(abs(result["line_movement_ah"]), abs(result["line_movement_total"]))
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    
    # === 3. NOMI SQUADRE (da meta tags o title - più affidabile) ===
    # Pattern 1: Formato Jina Reader markdown "Title: Team A VS Team B Match..."
    jina_title_match = re.search(
        r'^Title:\s*([A-Za-z][A-Za-z\s\.]{1,30}?)\s+(?:VS|vs|Vs)\s+([A-Za-z][A-Za-z\s\.]{1,30}?)(?:\s+(?:Match|Live|Analysis|H2H|Preview))',
        text, re.IGNORECASE | re.MULTILINE
    )
    if jina_title_match:
        home = jina_title_match.group(1).strip()
        away = jina_title_match.group(2).strip()
        # Rimuovi suffissi comuni
        home = re.sub(r'\s*(Match Analysis|H2H Stats|Preview).*$', '', home, flags=re.IGNORECASE).strip()
        away = re.sub(r'\s*(Match Analysis|H2H Stats|Preview).*$', '', away, flags=re.IGNORECASE).strip()
        if home and not result["home_team"]:
            result["home_team"] = _clean_team_name(home)
        if away and not result["away_team"]:
            result["away_team"] = _clean_team_name(away)
    
    # Pattern 2: Formato HTML <title> (se raw HTML disponibile)
    if not result["home_team"] or not result["away_team"]:
        title_match = re.search(
            r'<title>\s*([^<>|]+?)\s+(?:vs|VS|Vs|-|–)\s+([^<>|]+?)(?:\s+(?:Match|Live|Analysis|H2H|Preview))',
            text, re.IGNORECASE
        )
        if title_match:
            home = title_match.group(1).strip()
            away = title_match.group(2).strip()
            home = re.sub(r'\s*(Match Analysis|H2H Stats|Preview).*$', '', home, flags=re.IGNORECASE).strip()
            away = re.sub(r'\s*(Match Analysis|H2H Stats|Preview).*$', '', away, flags=re.IGNORECASE).strip()
            if home and not result["home_team"]:
                result["home_team"] = _clean_team_name(home)
            if away and not result["away_team"]:
                result["away_team"] = _clean_team_name(away)
    
    # Pattern 3: Cerca "# Team A vs Team B" heading (markdown)
    if not result["home_team"] or not result["away_team"]:
        heading_match = re.search(
            r'^#\s*([A-Za-z][A-Za-z\s\.]{1,30}?)\s+(?:vs|VS|Vs)\s+([A-Za-z][A-Za-z\s\.]{1,30}?)\s*$',
            text, re.IGNORECASE | re.MULTILINE
        )
        if heading_match:
            if not result["home_team"]:
                result["home_team"] = _clean_team_name(heading_match.group(1))
            if not result["away_team"]:
                result["away_team"] = _clean_team_name(heading_match.group(2))
    
    # Pattern 4: Cerca nella sezione H2H "Team A Home Same League"
    if not result["home_team"] or not result["away_team"]:
        h2h_home_match = re.search(
            r'Head to Head Statistics.*?\[([A-Za-z][A-Za-z\s]{1,25}?)\].*?Home',
            text, re.IGNORECASE | re.DOTALL
        )
        if h2h_home_match:
            if not result["home_team"]:
                result["home_team"] = _clean_team_name(h2h_home_match.group(1))
    
    # Fallback: cerca in meta description
    if not result["home_team"] or not result["away_team"]:
        meta_match = re.search(
            r'content="([^"]+?)\s+(?:vs|VS|Vs|-|–)\s+([^"]+?)(?:\s+(?:Match|Live|Check|Analysis))',
            text, re.IGNORECASE
        )
        if meta_match:
            if not result["home_team"]:
                result["home_team"] = _clean_team_name(meta_match.group(1))
            if not result["away_team"]:
                result["away_team"] = _clean_team_name(meta_match.group(2))

    # League fallback robusto dalla breadcrumb di Nowgoal.
    if not result["league_name"]:
        breadcrumb = re.search(r"Football>\s*([A-Za-z][A-Za-z0-9\s\-\.\(\)]+?)>\s*$", text, re.MULTILINE)
        if breadcrumb:
            result["league_name"] = _clean_league_name(breadcrumb.group(1))

    # Se la data non e' presente, lascia vuoto invece di fallback ambiguo.
    if result["match_date"] and not re.match(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", result["match_date"]):
        result["match_date"] = ""

    if not result["home_team"] or not result["away_team"]:
        result["extraction_notes"].append("team_names_partial")
    if not result["league_name"]:
        result["extraction_notes"].append("league_missing")
    if result["mkt_init_1"] <= 1.0:
        result["extraction_notes"].append("market_1x2_missing_or_unreadable")
    
    # === 4. PUNTI CLASSIFICA (dalla tabella standings) ===
    # FIX: La tabella standings ha righe FT e HT per OGNI squadra.
    # Dobbiamo prendere SOLO le righe FT Total (non HT Total).
    # Entrambe le squadre sono nella sezione "## Standings ... ## Head to Head".
    # Soluzione: cerca tutte le righe "Total" che seguono "FT Matches" e
    # sono prima del primo "HT Matches" per ciascuna squadra.
    _standings_section = re.search(
        r'## Standings.*?(?:## Head to Head|## Previous|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if _standings_section:
        _st_text = _standings_section.group(0)
        # Estrai i blocchi FT (dopo "FT Matches" fino a "HT Matches" o fine sezione)
        _ft_blocks = re.findall(
            r'FT\s+Matches.*?(?=HT\s+Matches|$)',
            _st_text, re.IGNORECASE | re.DOTALL
        )
        _all_pts = []
        for _block in _ft_blocks:
            _pts = re.findall(
                r"Total\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)",
                _block, re.IGNORECASE
            )
            _all_pts.extend(_pts)
        if len(_all_pts) >= 1:
            result["home_points"] = int(_all_pts[0])
        if len(_all_pts) >= 2:
            result["away_points"] = int(_all_pts[1])
    
    # === 5. MOTIVAZIONE (basata su posizione classifica) ===
    # Calcolata dopo aver estratto rank e matches
    if result["home_rank"] > 0 and result["home_matches"] > 0:
        # Top 3 = lotta titolo (high), Zona salvezza (ultime 3) = high, Metà = normal
        max_teams = 20  # Assumiamo massimo 20 squadre
        if result["home_rank"] <= 3:
            result["home_motivation"] = "high"
        elif result["home_rank"] >= max_teams - 2:
            result["home_motivation"] = "high"  # Lotta salvezza
        elif result["home_rank"] > max_teams // 2 + 3 and result["home_rank"] < max_teams - 3:
            result["home_motivation"] = "low"  # Salvo, senza obiettivi
    
    if result["away_rank"] > 0 and result["away_matches"] > 0:
        max_teams = 20
        if result["away_rank"] <= 3:
            result["away_motivation"] = "high"
        elif result["away_rank"] >= max_teams - 2:
            result["away_motivation"] = "high"
        elif result["away_rank"] > max_teams // 2 + 3 and result["away_rank"] < max_teams - 3:
            result["away_motivation"] = "low"

    # === 6. ASSENZE/INFORTUNI (sezione injuries/suspensions) ===
    abs_h, abs_a, abs_h_players, abs_a_players = _extract_absences_full(text)
    result["home_absences_count"] = abs_h
    result["away_absences_count"] = abs_a
    result["home_absences_players"] = abs_h_players
    result["away_absences_players"] = abs_a_players

    # === 6. TEAM STATISTICS (ultimi 10 match) ===
    # La tabella ha DUE gruppi di colonne: "Recent 3 Matches" e "Recent 10 Matches"
    # DEVO estrarre dall'ULTIMO gruppo (Recent 10 Matches) non dal primo!
    team_stats_section = re.search(
        r'Team Statistics.*?(?=\*\*Last Updated|HT/FT Statistics|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if team_stats_section:
        stats_text = team_stats_section.group(0)

        # Estrai Goals
        goals_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Goal\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if goals_matches:
            result["team_stats_home_goals"] = float(goals_matches[-1][0])
            result["team_stats_away_goals"] = float(goals_matches[-1][1])

        # Estrai Loss (gol subiti)
        loss_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Loss\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if loss_matches:
            result["team_stats_home_conceded"] = float(loss_matches[-1][0])
            result["team_stats_away_conceded"] = float(loss_matches[-1][1])

        # Estrai Opponent Shots
        shots_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Opponent Shots\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if shots_matches:
            result["team_stats_home_shots"] = float(shots_matches[-1][0])
            result["team_stats_away_shots"] = float(shots_matches[-1][1])

        # Estrai Corners
        corners_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Corners\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if corners_matches:
            result["team_stats_home_corners"] = float(corners_matches[-1][0])
            result["team_stats_away_corners"] = float(corners_matches[-1][1])

        # Estrai Yellow Cards
        yellows_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Yellow Cards\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if yellows_matches:
            result["team_stats_home_yellows"] = float(yellows_matches[-1][0])
            result["team_stats_away_yellows"] = float(yellows_matches[-1][1])

        # Estrai Fouls
        fouls_matches = re.findall(r'([\d.]+)\s*[\|]?\s*Fouls\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if fouls_matches:
            result["team_stats_home_fouls"] = float(fouls_matches[-1][0])
            result["team_stats_away_fouls"] = float(fouls_matches[-1][1])

        # Estrai Possession
        poss_matches = re.findall(r'([\d.]+)%?\s*[\|]?\s*Possession\s*[\|]?\s*([\d.]+)%?', stats_text, re.IGNORECASE)
        if poss_matches:
            result["team_stats_home_possession"] = float(poss_matches[-1][0])
            result["team_stats_away_possession"] = float(poss_matches[-1][1])

    result["extraction_section_scores"] = {
        "identity": 1.0 if (result["home_team"] and result["away_team"]) else 0.0,
        "league": 1.0 if bool(result["league_name"]) else 0.0,
        "h2h_core": 1.0 if (result["h2h_home_win_pct"] or result["h2h_draw_pct"] or result["h2h_away_win_pct"]) else 0.0,
        "standings": 1.0 if (result["home_matches"] > 0 and result["away_matches"] > 0) else 0.0,
        "previous_scores": 1.0 if (result["prev_home_avg_scored"] > 0 or result["prev_away_avg_scored"] > 0) else 0.0,
        "market_1x2": 1.0 if (result["mkt_init_1"] > 1.0 and result["mkt_init_x"] > 1.0 and result["mkt_init_2"] > 1.0) else 0.0,
        "team_stats": 1.0 if (result["team_stats_home_goals"] > 0 or result["team_stats_away_goals"] > 0) else 0.0,
        "injuries": 1.0 if (result["home_absences_count"] > 0 or result["away_absences_count"] > 0) else 0.0,
    }

    return result


def _extract_prematch_analysis_from_text(page_text: str) -> PrematchAnalysisExtracted:
    """
    Estrae dati prematch dal testo usando REGEX come metodo PRIMARIO.
    
    Flusso:
    1. Regex estrae tutti i dati possibili (GRATUITO, senza API key)
    2. Se configurato, Gemini arricchisce i dati mancanti (opzionale)
    3. Combina i risultati
    
    Questo approccio garantisce che l'estrazione funzioni sempre,
    anche senza API key Gemini o con restrizioni geografiche.
    """
    import time
    
    # === PASSO 1: Estrazione con REGEX (PRIMARIO) ===
    # NOTA: Usiamo l'intero testo, non troncato, perché i dati JavaScript
    # (h_data, Vs_hOdds) sono nell'HTML grezzo che viene dopo i 20000 caratteri
    regex_data = _extract_all_with_regex(page_text)
    
    # Crea il risultato base dai dati regex
    result = PrematchAnalysisExtracted(
        extraction_success=True,
        error_message="",
        # H2H
        h2h_home_win_pct=regex_data["h2h_home_win_pct"],
        h2h_draw_pct=regex_data["h2h_draw_pct"],
        h2h_away_win_pct=regex_data["h2h_away_win_pct"],
        h2h_over_pct=regex_data["h2h_over_pct"],
        h2h_btts_pct=regex_data["h2h_btts_pct"],
        h2h_matches_count=regex_data["h2h_matches_count"],
        h2h_avg_goals_home=regex_data["h2h_avg_goals_home"],
        h2h_avg_goals_away=regex_data["h2h_avg_goals_away"],
        # Strength
        strength_home=regex_data["strength_home"],
        strength_away=regex_data["strength_away"],
        # Standings casa
        home_rank=regex_data["home_rank"],
        home_matches=regex_data["home_matches"],
        home_win=regex_data["home_win"],
        home_draw=regex_data["home_draw"],
        home_lose=regex_data["home_lose"],
        home_scored=regex_data["home_scored"],
        home_conceded=regex_data["home_conceded"],
        home_home_win=regex_data["home_home_win"],
        home_home_draw=regex_data["home_home_draw"],
        home_home_lose=regex_data["home_home_lose"],
        home_home_scored=float(regex_data["home_home_scored"]),
        home_home_conceded=float(regex_data["home_home_conceded"]),
        home_last6_win=regex_data["home_last6_win"],
        home_last6_draw=regex_data["home_last6_draw"],
        home_last6_lose=regex_data["home_last6_lose"],
        home_last6_scored=regex_data["home_last6_scored"],
        home_last6_conceded=regex_data["home_last6_conceded"],
        # HT standings casa
        home_ht_win=regex_data["home_ht_win"],
        home_ht_draw=regex_data["home_ht_draw"],
        home_ht_lose=regex_data["home_ht_lose"],
        # Standings trasferta
        away_rank=regex_data["away_rank"],
        away_matches=regex_data["away_matches"],
        away_win=regex_data["away_win"],
        away_draw=regex_data["away_draw"],
        away_lose=regex_data["away_lose"],
        away_scored=regex_data["away_scored"],
        away_conceded=regex_data["away_conceded"],
        away_away_win=regex_data["away_away_win"],
        away_away_draw=regex_data["away_away_draw"],
        away_away_lose=regex_data["away_away_lose"],
        away_away_scored=float(regex_data["away_away_scored"]),
        away_away_conceded=float(regex_data["away_away_conceded"]),
        away_last6_win=regex_data["away_last6_win"],
        away_last6_draw=regex_data["away_last6_draw"],
        away_last6_lose=regex_data["away_last6_lose"],
        away_last6_scored=regex_data["away_last6_scored"],
        away_last6_conceded=regex_data["away_last6_conceded"],
        # HT standings trasferta
        away_ht_win=regex_data["away_ht_win"],
        away_ht_draw=regex_data["away_ht_draw"],
        away_ht_lose=regex_data["away_ht_lose"],
        # Previous scores
        home_prev_win_pct=regex_data["prev_home_win_pct"],
        home_prev_avg_scored=regex_data["prev_home_avg_scored"],
        home_prev_avg_conceded=regex_data["prev_home_avg_conceded"],
        home_prev_over_pct=regex_data["prev_home_over_pct"],
        away_prev_win_pct=regex_data["prev_away_win_pct"],
        away_prev_avg_scored=regex_data["prev_away_avg_scored"],
        away_prev_avg_conceded=regex_data["prev_away_avg_conceded"],
        away_prev_over_pct=regex_data["prev_away_over_pct"],
        # Quote iniziali
        mkt_init_1=regex_data["mkt_init_1"],
        mkt_init_x=regex_data["mkt_init_x"],
        mkt_init_2=regex_data["mkt_init_2"],
        # Info partita
        home_team=regex_data["home_team"],
        away_team=regex_data["away_team"],
        league_name=regex_data["league_name"],
        match_date=regex_data["match_date"],
        # === NUOVI CAMPI ===
        # Partite recenti
        home_recent_results=regex_data["home_recent_results"],
        away_recent_results=regex_data["away_recent_results"],
        home_form_trend=regex_data["home_form_trend"],
        away_form_trend=regex_data["away_form_trend"],
        home_xg_from_recent=regex_data["home_xg_from_recent"],
        away_xg_from_recent=regex_data["away_xg_from_recent"],
        scoring_streak_h=regex_data["scoring_streak_h"],
        scoring_streak_a=regex_data["scoring_streak_a"],
        clean_sheet_streak_h=regex_data["clean_sheet_streak_h"],
        clean_sheet_streak_a=regex_data["clean_sheet_streak_a"],
        # Quote multi-bookmaker
        ah_line_open=regex_data["ah_line_open"],
        ah_line_close=regex_data["ah_line_close"],
        ah_home_odds_open=regex_data["ah_home_odds_open"],
        ah_away_odds_open=regex_data["ah_away_odds_open"],
        total_line_open=regex_data["total_line_open"],
        total_line_close=regex_data["total_line_close"],
        total_over_odds_open=regex_data["total_over_odds_open"],
        total_under_odds_open=regex_data["total_under_odds_open"],
        line_movement_ah=regex_data["line_movement_ah"],
        line_movement_total=regex_data["line_movement_total"],
        odds_sharp_signal=regex_data["odds_sharp_signal"],
        # Punti e motivazione
        home_points=regex_data["home_points"],
        away_points=regex_data["away_points"],
        home_motivation=regex_data["home_motivation"],
        away_motivation=regex_data["away_motivation"],
        home_absences_count=regex_data["home_absences_count"],
        away_absences_count=regex_data["away_absences_count"],
        home_absences_players=list(regex_data.get("home_absences_players", []) or []),
        away_absences_players=list(regex_data.get("away_absences_players", []) or []),
        # ID squadre
        home_id=regex_data["home_id"],
        away_id=regex_data["away_id"],
        # Team Statistics (ultimi 10 match)
        team_stats_home_goals=regex_data["team_stats_home_goals"],
        team_stats_home_conceded=regex_data["team_stats_home_conceded"],
        team_stats_home_shots=regex_data["team_stats_home_shots"],
        team_stats_home_corners=regex_data["team_stats_home_corners"],
        team_stats_home_yellows=regex_data["team_stats_home_yellows"],
        team_stats_home_fouls=regex_data["team_stats_home_fouls"],
        team_stats_home_possession=regex_data["team_stats_home_possession"],
        team_stats_away_goals=regex_data["team_stats_away_goals"],
        team_stats_away_conceded=regex_data["team_stats_away_conceded"],
        team_stats_away_shots=regex_data["team_stats_away_shots"],
        team_stats_away_corners=regex_data["team_stats_away_corners"],
        team_stats_away_yellows=regex_data["team_stats_away_yellows"],
        team_stats_away_fouls=regex_data["team_stats_away_fouls"],
        team_stats_away_possession=regex_data["team_stats_away_possession"],
        extraction_notes=list(regex_data.get("extraction_notes", [])),
        extraction_section_scores=dict(regex_data.get("extraction_section_scores", {})),
    )
    
    # Calcola forma_mult
    result.forma_mult_h = _forma_mult_from_standings(
        result.home_win, result.home_draw, result.home_lose,
        result.home_last6_win, result.home_last6_draw, result.home_last6_lose,
        result.home_scored, result.home_conceded, result.home_matches,
    )
    result.forma_mult_a = _forma_mult_from_standings(
        result.away_win, result.away_draw, result.away_lose,
        result.away_last6_win, result.away_last6_draw, result.away_last6_lose,
        result.away_scored, result.away_conceded, result.away_matches,
    )

    _fill_last6_goals_from_recent_results(result)
    
    # Calcola win_rate se non estratto (W / Matches * 100)
    if result.home_win_rate == 0 and result.home_matches > 0 and result.home_win > 0:
        result.home_win_rate = round(result.home_win / result.home_matches * 100, 1)
    if result.away_win_rate == 0 and result.away_matches > 0 and result.away_win > 0:
        result.away_win_rate = round(result.away_win / result.away_matches * 100, 1)
    
    # Se abbiamo dati partite recenti, migliora forma_mult con trend reale
    if result.home_recent_results and len(result.home_recent_results) >= 6:
        # Forma_mult già calcolato da standings, aggiusta con trend partite
        trend_adj = result.home_form_trend * 0.04  # Max ±4% aggiustamento
        result.forma_mult_h = max(0.88, min(1.12, result.forma_mult_h + trend_adj))
    
    if result.away_recent_results and len(result.away_recent_results) >= 6:
        trend_adj = result.away_form_trend * 0.04
        result.forma_mult_a = max(0.88, min(1.12, result.forma_mult_a + trend_adj))
    
    # Calcola fixture_historical_total (media gol H2H totali)
    if result.h2h_avg_goals_home > 0 or result.h2h_avg_goals_away > 0:
        result.fixture_historical_total = result.h2h_avg_goals_home + result.h2h_avg_goals_away

    # Report qualita': quante sezioni chiave sono state davvero estratte.
    section_scores = dict(result.extraction_section_scores)
    section_scores["weather"] = 1.0 if bool(result.weather_condition) else 0.0
    result.extraction_section_scores = section_scores
    if section_scores:
        result.extraction_coverage = sum(section_scores.values()) / len(section_scores)
    
    # === PASSO 2: Se regex ha estratto dati sufficienti, termina qui ===
    has_data = (
        result.h2h_home_win_pct > 0 or result.home_matches > 0 or 
        result.strength_home > 0 or result.home_last6_win + result.home_last6_draw + result.home_last6_lose > 0
    )
    
    if has_data:
        return result
    
    # === PASSO 3: Prova Gemini come FALLBACK (solo se regex non ha trovato nulla) ===
    api_key = _get_gemini_api_key()
    if not api_key:
        # Nessuna API key, ritorna quello che abbiamo (anche se parziale)
        if any([result.h2h_home_win_pct, result.home_matches, result.away_matches]):
            result.error_message = "Dati parziali (solo regex, Gemini non configurato)"
            return result
        return PrematchAnalysisExtracted(
            extraction_success=False,
            error_message="Nessun dato estratto (regex e Gemini non disponibili)",
        )
    
    # Prova Gemini
    # FIX: text_truncated non esisteva — usiamo page_text limitato a 30000 caratteri
    _trunc = page_text[:30000] if page_text else ""
    full_prompt = f"TESTO PAGINA NOWGOAL:\n\n{_trunc}\n\n---\n\n{PREMATCH_ANALYSIS_TEXT_PROMPT}"
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
                        gemini_text = parts_resp[0].get("text", "")
                        if gemini_text:
                            gemini_result = _parse_prematch_analysis_response(gemini_text)
                            # Combina: usa regex dove Gemini ha 0
                            if result.h2h_home_win_pct == 0 and gemini_result.h2h_home_win_pct > 0:
                                result.h2h_home_win_pct = gemini_result.h2h_home_win_pct
                            if result.h2h_draw_pct == 0 and gemini_result.h2h_draw_pct > 0:
                                result.h2h_draw_pct = gemini_result.h2h_draw_pct
                            if result.h2h_away_win_pct == 0 and gemini_result.h2h_away_win_pct > 0:
                                result.h2h_away_win_pct = gemini_result.h2h_away_win_pct
                            if result.home_matches == 0 and gemini_result.home_matches > 0:
                                result.home_matches = gemini_result.home_matches
                                result.home_win = gemini_result.home_win
                                result.home_draw = gemini_result.home_draw
                                result.home_lose = gemini_result.home_lose
                                result.home_scored = gemini_result.home_scored
                                result.home_conceded = gemini_result.home_conceded
                            if result.away_matches == 0 and gemini_result.away_matches > 0:
                                result.away_matches = gemini_result.away_matches
                                result.away_win = gemini_result.away_win
                                result.away_draw = gemini_result.away_draw
                                result.away_lose = gemini_result.away_lose
                                result.away_scored = gemini_result.away_scored
                                result.away_conceded = gemini_result.away_conceded
                            if result.strength_home == 0 and gemini_result.strength_home > 0:
                                result.strength_home = gemini_result.strength_home
                                result.strength_away = gemini_result.strength_away
                            # Info partita da Gemini se regex non ha trovato
                            if not result.home_team and gemini_result.home_team:
                                result.home_team = gemini_result.home_team
                            if not result.away_team and gemini_result.away_team:
                                result.away_team = gemini_result.away_team
                            if not result.league_name and gemini_result.league_name:
                                result.league_name = gemini_result.league_name
                            return result
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
    
    # Ritorna risultato regex anche se Gemini ha fallito
    if has_data:
        result.error_message = f"Dati da regex (Gemini: {last_error})"
        return result
    
    return PrematchAnalysisExtracted(
        extraction_success=False,
        error_message=f"Nessun dato estratto (regex vuoto, Gemini: {last_error})",
    )


def _extract_prematch_single_url_attempt(
    h2h_url: str,
    live_url: str | None,
    domain_for_fallback: str,
    match_id: str | None,
) -> PrematchAnalysisExtracted:
    """Un singolo fetch Jina+HTML + parsing (usato per retry server-side su mirror)."""
    page_text = ""
    raw_html = ""

    def _safe_raw() -> str:
        try:
            return _fetch_raw_html(h2h_url)
        except Exception:
            return ""

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_j = pool.submit(_fetch_jina_reader, h2h_url)
            fut_h = pool.submit(_safe_raw)
            page_text = fut_j.result()
            raw_html = fut_h.result() or ""
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

    combined_text = page_text
    if raw_html:
        cap = min(len(raw_html), _RAW_HTML_APPEND_MAX)
        combined_text = page_text + "\n\n=== RAW HTML ===\n" + raw_html[:cap]

    result = _extract_prematch_analysis_from_text(combined_text)
    u_home, u_away, u_league = _extract_identity_from_url(h2h_url)
    if result.extraction_success:
        if not result.home_team and u_home:
            result.home_team = u_home
        if not result.away_team and u_away:
            result.away_team = u_away
        if not result.league_name and u_league:
            result.league_name = u_league
            result.league_source = "mirror"
        elif result.league_name:
            result.league_source = "nowgoal"

    # Stessi parser della pagina live sul blob H2H+HTML (spesso include Team/HT-FT senza secondo fetch).
    if result.extraction_success:
        embedded = _extract_live_page_data(combined_text)
        _merge_live_snapshot_into_result(result, embedded)
        _refresh_prematch_quality_scores(result)

    if live_url and result.extraction_success and _prematch_needs_live_jina_page(result):
        try:
            live_page_text = _fetch_jina_reader(live_url)
            if live_page_text and len(live_page_text) > 200:
                live_data = _extract_live_page_data(live_page_text)
                _merge_live_snapshot_into_result(result, live_data)
                _refresh_prematch_quality_scores(result)
                if not result.league_name:
                    _, _, live_league = _extract_match_identity_from_text(live_page_text)
                    if live_league:
                        result.league_name = live_league
        except Exception:
            pass

    if result.extraction_success and not result.weather_condition and WEATHER_MODULE_AVAILABLE and _get_weather_for_match:
        try:
            weather = _get_weather_for_match(result.home_team, result.away_team, result.league_name)
            if weather.extraction_success:
                result.weather_condition = weather.condition
                result.weather_temp = weather.temp_celsius
                result.weather_impact = weather.xg_impact
        except Exception:
            pass

    if result.extraction_success and result.weather_impact != 0:
        result.forma_mult_h *= (1.0 + result.weather_impact)
        result.forma_mult_a *= (1.0 + result.weather_impact)

    if result.extraction_success and not result.league_name and match_id:
        mirror_league = _fallback_league_from_nowgoal_mirrors(match_id, domain_for_fallback)
        if mirror_league:
            result.league_name = mirror_league
            result.league_source = "mirror"

    if result.extraction_success and not result.league_name and result.home_team and result.away_team:
        ext_league = _fallback_league_external(result.home_team, result.away_team)
        if ext_league:
            result.league_name = ext_league
            result.league_source = "external"

    return result


def extract_prematch_analysis_from_url(url: str) -> PrematchAnalysisExtracted:
    """
    Estrae analisi prematch da un URL Nowgoal.

    METODO PRIMARIO: Regex (GRATUITO, senza API key, senza limiti)
    FALLBACK OPZIONALE: Gemini (richiede API key)

    AUTOMAZIONE LIVE: stessi parser anche su H2H+HTML; seconda richiesta Jina su `live-*`
    solo se mancano HT/FT, team stats core/dettaglio o quote live (nessun dato in meno).

    Retry server-side: con `match_id`, se copertura bassa o fallimento sul dominio
    richiesto, tenta automaticamente `live5.nowgoal26.com` (pausa breve tra Jina).

    Jina H2H e fetch HTML grezzo partono in parallelo per ridurre la latenza.

    Nota: un fetch headless (es. Playwright) per HTML dipendente da JS resta possibile
    come dipendenza opzionale; non è abilitato di default per restare leggeri.

    Args:
        url: URL Analysis / live / detail Nowgoal.

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

    match_id_match = re.search(r'/match/(?:h2h|live|detail)-(\d+)', url, re.IGNORECASE)
    match_id = match_id_match.group(1) if match_id_match else None

    domain_match = re.search(r'https?://([^/]+)', url)
    original_domain = domain_match.group(1) if domain_match else "live5.nowgoal26.com"

    if match_id:
        h2h_url = f"https://{original_domain}/match/h2h-{match_id}"
        live_url = f"https://{original_domain}/match/live-{match_id}"
    else:
        h2h_url = url
        live_url = None
        if not _is_valid_nowgoal_url(url):
            return PrematchAnalysisExtracted(
                extraction_success=False,
                error_message=(
                    "URL non riconosciuto. Usa un link Nowgoal Analysis "
                    "(es. nowgoal.com/match/h2h-XXXXXX)"
                ),
            )

    candidates: list[tuple[str, str | None, str]] = []
    if match_id:
        candidates.append((h2h_url, live_url, original_domain))
        if original_domain.rstrip("/") != "live5.nowgoal26.com":
            candidates.append((
                f"https://live5.nowgoal26.com/match/h2h-{match_id}",
                f"https://live5.nowgoal26.com/match/live-{match_id}",
                "live5.nowgoal26.com",
            ))
    else:
        candidates.append((h2h_url, live_url, original_domain))

    _COVERAGE_OK = 0.40
    best: PrematchAnalysisExtracted | None = None
    for i, (hu, lu, dom_fb) in enumerate(candidates):
        if i > 0:
            time.sleep(0.35)
        attempt = _extract_prematch_single_url_attempt(hu, lu, dom_fb, match_id)
        if best is None or (
            (attempt.extraction_success and not best.extraction_success)
            or (
                attempt.extraction_coverage > best.extraction_coverage
                and attempt.extraction_success == best.extraction_success
                )
        ):
            best = attempt
        if attempt.extraction_success and attempt.extraction_coverage >= _COVERAGE_OK:
            return attempt

    return best if best is not None else PrematchAnalysisExtracted(
        extraction_success=False,
        error_message="Nessun tentativo di estrazione riuscito",
    )


def _extract_live_page_data(text: str) -> dict:
    """
    Estrae dati aggiuntivi dalla pagina LIVE di Nowgoal.
    
    Estrae:
    - Meteo (condizione, temperatura)
    - HT/FT Statistics
    - Team Statistics (ultimi 10 match)
    - Quote live (informativo)
    
    Returns:
        Dict con i dati estratti.
    """
    result = {
        # Meteo
        "weather_condition": "",
        "weather_temp": 0,
        "weather_impact": 0.0,
        # HT/FT Stats - Casa
        "htft_home_htw_ftw": 0,
        "htft_home_htd_ftw": 0,
        "htft_home_htl_ftw": 0,
        "htft_home_htw_ftd": 0,
        "htft_home_htd_ftd": 0,
        "htft_home_htl_ftd": 0,
        "htft_home_htw_ftl": 0,
        "htft_home_htd_ftl": 0,
        "htft_home_htl_ftl": 0,
        # HT/FT Stats - Trasferta
        "htft_away_htw_ftw": 0,
        "htft_away_htd_ftw": 0,
        "htft_away_htl_ftw": 0,
        "htft_away_htw_ftd": 0,
        "htft_away_htd_ftd": 0,
        "htft_away_htl_ftd": 0,
        "htft_away_htw_ftl": 0,
        "htft_away_htd_ftl": 0,
        "htft_away_htl_ftl": 0,
        # Team Stats
        "team_stats_home_goals": 0.0,
        "team_stats_home_conceded": 0.0,
        "team_stats_home_shots": 0.0,
        "team_stats_home_corners": 0.0,
        "team_stats_home_yellows": 0.0,
        "team_stats_home_fouls": 0.0,
        "team_stats_home_possession": 0.0,
        "team_stats_away_goals": 0.0,
        "team_stats_away_conceded": 0.0,
        "team_stats_away_shots": 0.0,
        "team_stats_away_corners": 0.0,
        "team_stats_away_yellows": 0.0,
        "team_stats_away_fouls": 0.0,
        "team_stats_away_possession": 0.0,
        # Quote live
        "live_ah_line": 0.0,
        "live_ah_home_odds": 0.0,
        "live_ah_away_odds": 0.0,
        "live_total_line": 0.0,
        "live_over_odds": 0.0,
        "live_under_odds": 0.0,
        "home_absences_count": 0,
        "away_absences_count": 0,
        "live_data_notes": [],
    }
    
    # === 1. METEO ===
    # Pattern: "Partly cloudy, 10°C" o "Rain, 15°C" o "Sunny, 25°C"
    # FIX: Gestisce sia °C (U+00B0+C) che ℃ (U+2103) e range "10℃～11℃"
    weather_match = re.search(
        r'(Sunny|Clear|Partly cloudy|Cloudy|Overcast|Rain|Light rain|Heavy rain|Drizzle|Thunderstorm|Snow|Fog|Mist|Windy),?\s*(\d{1,2})[°℃]\s*[～~]?\s*(\d{1,2})?[°℃]?',
        text, re.IGNORECASE
    )
    if weather_match:
        result["weather_condition"] = weather_match.group(1).strip()
        # Se è un range (es. 10～11), prendi la media; altrimenti il singolo valore
        temp_low = int(weather_match.group(2))
        temp_high_str = weather_match.group(3)
        if temp_high_str:
            temp_high = int(temp_high_str)
            result["weather_temp"] = (temp_low + temp_high) // 2
        else:
            result["weather_temp"] = temp_low
        
        # Calcola impatto meteo sui gol
        condition_lower = result["weather_condition"].lower()
        if any(x in condition_lower for x in ["rain", "thunderstorm", "heavy"]):
            result["weather_impact"] = -0.08  # -8% xG per pioggia forte
        elif any(x in condition_lower for x in ["drizzle", "light rain", "mist"]):
            result["weather_impact"] = -0.04  # -4% xG per pioggia leggera
        elif "windy" in condition_lower:
            result["weather_impact"] = -0.03  # -3% xG per vento
        elif result["weather_temp"] >= 30:
            result["weather_impact"] = -0.02  # -2% xG per caldo eccessivo
    
    # === 2. HT/FT STATISTICS ===
    # La tabella HT/FT ha questo formato:
    # | HT-W / FT-W | 4 | 2 | ...
    # Cerca la sezione HT/FT
    htft_section = re.search(
        r'HT/FT Statistics.*?(?=\*\*Last Updated|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if htft_section:
        htft_text = htft_section.group(0)
        
        # Pattern per ogni riga: "| HT-W / FT-W | 4 | 2 |"
        # Le colonne sono: tipo | casa_home | casa_away | trasf_home | trasf_away
        htft_patterns = {
            "htw_ftw": r'HT-W\s*/\s*FT-W\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htd_ftw": r'HT-D\s*/\s*FT-W\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htl_ftw": r'HT-L\s*/\s*FT-W\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htw_ftd": r'HT-W\s*/\s*FT-D\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htd_ftd": r'HT-D\s*/\s*FT-D\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htl_ftd": r'HT-L\s*/\s*FT-D\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htw_ftl": r'HT-W\s*/\s*FT-L\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htd_ftl": r'HT-D\s*/\s*FT-L\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
            "htl_ftl": r'HT-L\s*/\s*FT-L\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',
        }
        
        for key, pattern in htft_patterns.items():
            match = re.search(pattern, htft_text, re.IGNORECASE)
            if match:
                # Gruppo 1 = casa_home, Gruppo 2 = casa_away, Gruppo 3 = trasf_home, Gruppo 4 = trasf_away
                result[f"htft_home_{key}"] = int(match.group(1))  # casa in casa
                result[f"htft_away_{key}"] = int(match.group(4))  # trasferta in trasferta
    
    # === 3. TEAM STATISTICS ===
    # La tabella ha DUE gruppi di colonne: "Recent 3 Matches" e "Recent 10 Matches"
    # Jina Reader converte HTML in markdown con **bold** attorno ai nomi delle stat:
    #   | 1.3 | **Goal** | 1.3 | 1.3 | **Goal** | 1.3 |
    # FIX: Aggiunto \*\*? prima del nome per gestire il markdown bold di Jina Reader.
    # Prendi sempre l'ULTIMO match (Recent 10 Matches).
    
    team_stats_section = re.search(
        r'Team Statistics.*?(?=\*\*Last Updated|HT/FT|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if team_stats_section:
        stats_text = team_stats_section.group(0)
        
        # Estrai valori dalla tabella usando findall per trovare TUTTI i match
        # Supporta sia formato "|" che formato TAB/spazi
        # FIX: \*\*? gestisce il markdown bold di Jina Reader
        
        # Riga Goal
        goals_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Goal\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if goals_matches:
            # Prendi l'ULTIMO match (Recent 10 Matches)
            result["team_stats_home_goals"] = float(goals_matches[-1][0])
            result["team_stats_away_goals"] = float(goals_matches[-1][1])
        
        # Riga Loss (gol subiti)
        loss_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Loss\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if loss_matches:
            result["team_stats_home_conceded"] = float(loss_matches[-1][0])
            result["team_stats_away_conceded"] = float(loss_matches[-1][1])
        
        # Riga Shots (Opponent Shots)
        shots_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Opponent Shots\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if shots_matches:
            result["team_stats_home_shots"] = float(shots_matches[-1][0])
            result["team_stats_away_shots"] = float(shots_matches[-1][1])
        
        # Riga Corners
        corners_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Corners\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if corners_matches:
            result["team_stats_home_corners"] = float(corners_matches[-1][0])
            result["team_stats_away_corners"] = float(corners_matches[-1][1])
        
        # Riga Yellow Cards
        yellows_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Yellow Cards\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if yellows_matches:
            result["team_stats_home_yellows"] = float(yellows_matches[-1][0])
            result["team_stats_away_yellows"] = float(yellows_matches[-1][1])
        
        # Riga Fouls
        fouls_matches = re.findall(r'([\d.]+)\s*[\|]?\s*\*\*?Fouls\*\*?\s*[\|]?\s*([\d.]+)', stats_text, re.IGNORECASE)
        if fouls_matches:
            result["team_stats_home_fouls"] = float(fouls_matches[-1][0])
            result["team_stats_away_fouls"] = float(fouls_matches[-1][1])
        
        # Riga Possession
        poss_matches = re.findall(r'([\d.]+)%?\s*[\|]?\s*\*\*?Possession\*\*?\s*[\|]?\s*([\d.]+)%?', stats_text, re.IGNORECASE)
        if poss_matches:
            result["team_stats_home_possession"] = float(poss_matches[-1][0])
            result["team_stats_away_possession"] = float(poss_matches[-1][1])
    
    # === 4. QUOTE LIVE ===
    # Cerca la tabella Live Odds con righe "Live" (pre-kickoff, score vuoto)
    #
    # Formato reale da Jina Reader (20 colonne totali):
    # | Live |  | AH_init_H | AH_line | AH_init_A | AH_live_H | AH_line | AH_live_A |
    #   | 1X2_i1 | 1X2_iX | 1X2_i2 | 1X2_l1 | 1X2_lX | 1X2_l2 |
    #   | OU_iO | OU_line | OU_iU | OU_lO | OU_line | OU_lU |
    #
    # Le colonne sono ACCOPPIATE (Initial/Live) per ogni tipo di quota.
    # Prendiamo SOLO i valori "Live" (gruppi 2-4 per AH, gruppi 6-8 per O/U).
    
    live_odds_match = re.search(
        r'\|\s*Live\s*\|\s*\|'                                    # | Live | (score vuoto) |
        r'\s*[\d.]+\s*\|\s*(-?[\d./]+)\s*\|\s*[\d.]+\s*\|'       # AH_init_H | AH_line(1) | AH_init_A
        r'\s*([\d.]+)\s*\|\s*(-?[\d./]+)\s*\|\s*([\d.]+)\s*\|'   # AH_live_H(2) | AH_line(3) | AH_live_A(4)
        r'\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'            # 1X2_init_1 | 1X2_init_X | 1X2_init_2
        r'\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'            # 1X2_live_1 | 1X2_live_X | 1X2_live_2
        r'\s*[\d.]+\s*\|\s*(-?[\d./]+)\s*\|\s*[\d.]+\s*\|'       # OU_init_O | OU_line(5) | OU_init_U
        r'\s*([\d.]+)\s*\|\s*(-?[\d./]+)\s*\|\s*([\d.]+)\s*\|',  # OU_live_O(6) | OU_line(7) | OU_live_U(8)
        text, re.IGNORECASE
    )
    if live_odds_match:
        try:
            result["live_ah_home_odds"] = float(live_odds_match.group(2))
            ah_line_raw = live_odds_match.group(3)
            result["live_ah_line"] = convert_nowgoal_line_to_software(ah_line_raw, invert_sign=True)
            result["live_ah_away_odds"] = float(live_odds_match.group(4))
            result["live_over_odds"] = float(live_odds_match.group(6))
            total_line_raw = live_odds_match.group(7)
            result["live_total_line"] = convert_nowgoal_line_to_software(total_line_raw, invert_sign=False)
            result["live_under_odds"] = float(live_odds_match.group(8))
        except (ValueError, TypeError):
            pass
    else:
        result["live_data_notes"].append("live_odds_not_available_no_js")

    # === 5. INJURIES/SUSPENSIONS ===
    abs_h, abs_a = _extract_absence_counts(text)
    result["home_absences_count"] = abs_h
    result["away_absences_count"] = abs_a
    
    return result


_HTFT_LIVE_KEYS: tuple[str, ...] = (
    "htft_home_htw_ftw", "htft_home_htd_ftw", "htft_home_htl_ftw",
    "htft_home_htw_ftd", "htft_home_htd_ftd", "htft_home_htl_ftd",
    "htft_home_htw_ftl", "htft_home_htd_ftl", "htft_home_htl_ftl",
    "htft_away_htw_ftw", "htft_away_htd_ftw", "htft_away_htl_ftw",
    "htft_away_htw_ftd", "htft_away_htd_ftd", "htft_away_htl_ftd",
    "htft_away_htw_ftl", "htft_away_htd_ftl", "htft_away_htl_ftl",
)

_TEAM_STATS_LIVE_KEYS: tuple[str, ...] = (
    "team_stats_home_goals", "team_stats_home_conceded", "team_stats_home_shots",
    "team_stats_home_corners", "team_stats_home_yellows", "team_stats_home_fouls",
    "team_stats_home_possession", "team_stats_away_goals", "team_stats_away_conceded",
    "team_stats_away_shots", "team_stats_away_corners", "team_stats_away_yellows",
    "team_stats_away_fouls", "team_stats_away_possession",
)

_TEAM_STATS_DETAIL_KEYS: tuple[str, ...] = (
    "team_stats_home_shots", "team_stats_away_shots",
    "team_stats_home_corners", "team_stats_away_corners",
    "team_stats_home_yellows", "team_stats_away_yellows",
    "team_stats_home_fouls", "team_stats_away_fouls",
    "team_stats_home_possession", "team_stats_away_possession",
)


def _prematch_any_htft(result: PrematchAnalysisExtracted) -> bool:
    return any(getattr(result, k, 0) for k in _HTFT_LIVE_KEYS)


def _prematch_needs_live_jina_page(result: PrematchAnalysisExtracted) -> bool:
    """True se mancano ancora blocchi tipici della pagina live-* (secondo fetch Jina)."""
    if not _prematch_any_htft(result):
        return True
    if result.team_stats_home_goals == 0 and result.team_stats_away_goals == 0:
        return True
    if result.live_ah_line == 0 and result.live_total_line == 0:
        return True
    # Goal/Loss ok ma nessun dettaglio: spesso parse parziale su H2H — completa da live.
    if (result.team_stats_home_goals > 0 or result.team_stats_away_goals > 0) and all(
        float(getattr(result, k, 0.0) or 0.0) == 0.0 for k in _TEAM_STATS_DETAIL_KEYS
    ):
        return True
    return False


def _refresh_prematch_quality_scores(result: PrematchAnalysisExtracted) -> None:
    """Aggiorna team_stats / weather / injuries in extraction_section_scores dopo merge."""
    d = dict(result.extraction_section_scores or {})
    if result.team_stats_home_goals or result.team_stats_away_goals:
        d["team_stats"] = 1.0
    d["weather"] = 1.0 if result.weather_condition else float(d.get("weather", 0.0))
    if result.home_absences_count > 0 or result.away_absences_count > 0:
        d["injuries"] = 1.0
    result.extraction_section_scores = d
    if d:
        result.extraction_coverage = sum(d.values()) / len(d)


def _merge_live_snapshot_into_result(result: PrematchAnalysisExtracted, live_data: dict) -> None:
    """
    Unisce dati estratti con gli stessi parser della pagina LIVE (testo Jina o blob H2H+HTML).
    Non sovrascrive valori gia' popolati da regex/HTML: stessa o migliore precisione, meno fetch.
    """
    if live_data.get("weather_condition") and not (result.weather_condition or "").strip():
        result.weather_condition = live_data["weather_condition"]
        result.weather_temp = int(live_data.get("weather_temp", 0) or 0)
        result.weather_impact = float(live_data.get("weather_impact", 0.0) or 0.0)

    for key in _HTFT_LIVE_KEYS:
        v = int(live_data.get(key, 0) or 0)
        if v and getattr(result, key, 0) == 0:
            setattr(result, key, v)

    for key in _TEAM_STATS_LIVE_KEYS:
        v = float(live_data.get(key, 0.0) or 0.0)
        if v > 0 and float(getattr(result, key, 0.0) or 0.0) == 0.0:
            setattr(result, key, v)

    lah = float(live_data.get("live_ah_line", 0.0) or 0.0)
    ltl = float(live_data.get("live_total_line", 0.0) or 0.0)
    if lah != 0.0 and result.live_ah_line == 0.0:
        result.live_ah_line = lah
        result.live_ah_home_odds = float(live_data.get("live_ah_home_odds", 0.0) or 0.0)
        result.live_ah_away_odds = float(live_data.get("live_ah_away_odds", 0.0) or 0.0)
    if ltl != 0.0 and result.live_total_line == 0.0:
        result.live_total_line = ltl
        result.live_over_odds = float(live_data.get("live_over_odds", 0.0) or 0.0)
        result.live_under_odds = float(live_data.get("live_under_odds", 0.0) or 0.0)

    if result.home_absences_count <= 0:
        h = int(live_data.get("home_absences_count", 0) or 0)
        if h > 0:
            result.home_absences_count = h
    if result.away_absences_count <= 0:
        a = int(live_data.get("away_absences_count", 0) or 0)
        if a > 0:
            result.away_absences_count = a

    for note in live_data.get("live_data_notes", []):
        if note not in result.extraction_notes:
            result.extraction_notes.append(note)


def _apply_live_data_to_result(result: PrematchAnalysisExtracted, live_data: dict) -> None:
    """Compat: stesso comportamento sicuro del merge (solo campi vuoti)."""
    _merge_live_snapshot_into_result(result, live_data)


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
