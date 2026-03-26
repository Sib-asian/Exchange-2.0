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
from dataclasses import dataclass
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

LIVE_STATS_PROMPT = """Analizza questo screenshot di statistiche live di una partita di calcio (da Nowgoal, FlashScore, SofaScore o simili).

Estrai TUTTE le statistiche visibili e restituiscile in formato JSON esattamente come mostrato:

{
    "minuto": 0,
    "gol_casa": 0,
    "gol_trasf": 0,
    "rossi_casa": 0,
    "rossi_trasf": 0,
    "gialli_casa": 0,
    "gialli_trasf": 0,
    "tiri_porta_casa": 0,
    "tiri_porta_trasf": 0,
    "tiri_fuori_casa": 0,
    "tiri_fuori_trasf": 0,
    "tiri_bloccati_casa": 0,
    "tiri_bloccati_trasf": 0,
    "corner_casa": 0,
    "corner_trasf": 0,
    "possesso_casa": 0.0,
    "possesso_trasf": 0.0,
    "attacchi_casa": 0,
    "attacchi_trasf": 0,
    "attacchi_pericolosi_casa": 0,
    "attacchi_pericolosi_trasf": 0,
    "falli_casa": 0,
    "falli_trasf": 0,
    "confidence": "high/medium/low"
}

REGOLE DI ESTRAZIONE:

1. PUNTEGGIO E MINUTO:
   - Il punteggio è solitamente in formato "X - Y" al centro dello schermo
   - Il minuto può essere mostrato come "45'" o "HT" (halftime=45) o "FT" (fulltime=90)
   - La squadra a SINISTRA è CASA, quella a DESTRA è TRASFERTA

2. STATISTICHE:
   - Ogni statistica ha due valori: uno per casa (sinistra) e uno per trasferta (destra)
   - "Shots on Target" / "Tiri in Porta" = tiri_porta
   - "Shots off Target" / "Tiri Fuori" = tiri_fuori
   - "Blocked Shots" / "Tiri Bloccati" = tiri_bloccati
   - "Corners" / "Corner" / "Calci d'angolo" = corner
   - "Possession" / "Possesso" = possesso (in percentuale, es. 55.0)
   - "Attacks" / "Attacchi" = attacchi
   - "Dangerous Attacks" / "Attacchi Pericolosi" = attacchi_pericolosi
   - "Fouls" / "Falli" = falli
   - "Yellow Cards" / "Gialli" / "Ammonizioni" = gialli
   - "Red Cards" / "Rossi" / "Espulsioni" = rossi

3. NOTA SUI TIRI:
   - Se vedi solo "Total Shots" e "Shots on Target":
     tiri_fuori = total_shots - shots_on_target
   - Se vedi "Shots" senza specificare: usa come tiri_porta

4. CONFIDENCE:
   - "high" = tutti i dati chiaramente visibili
   - "medium" = alcuni dati mancanti o poco chiari
   - "low" = immagine sfocata o molti dati non leggibili

SE UN DATO NON È VISIBILE: imposta a 0 (numeri) o 0.0 (percentuali).

IMPORTANTE: Restituisci SOLO il JSON, nessun altro testo."""


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
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
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

        # Trova inizio JSON
        lines = json_str.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_str = "\n".join(lines[i:])
                break

        if "```" in json_str:
            json_str = json_str.split("```")[0]

        data = json.loads(json_str.strip())

        return LiveStatsExtracted(
            minuto=_safe_int(data.get("minuto")),
            gol_casa=_safe_int(data.get("gol_casa")),
            gol_trasf=_safe_int(data.get("gol_trasf")),
            rossi_casa=_safe_int(data.get("rossi_casa")),
            rossi_trasf=_safe_int(data.get("rossi_trasf")),
            gialli_casa=_safe_int(data.get("gialli_casa")),
            gialli_trasf=_safe_int(data.get("gialli_trasf")),
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
