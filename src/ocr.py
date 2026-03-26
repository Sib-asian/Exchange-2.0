"""
ocr.py — Estrazione automatica dati da screenshot di siti scommesse.

Utilizza l'API OpenAI Vision (GPT-4o-mini) per leggere immagini da:
  - Siti di scommesse (Bet365, Betfair, Pinnacle, ecc.)
  - App mobili
  - Desktop screenshot

Estrae automaticamente:
  - Nomi delle squadre
  - Quote 1X2
  - Quote Over/Under
  - Quote BTTS (GG/NG)

Configurazione:
  - OPENAI_API_KEY: via st.secrets["OPENAI_API_KEY"] o variabile d'ambiente
  - OPENAI_MODEL: modello vision (default: gpt-4o-mini)

Il modulo restituisce i dati in formato strutturato per l'uso nell'UI Streamlit.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


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

# Modello OpenAI Vision di default
_DEFAULT_MODEL = "gpt-4o-mini"


def _get_openai_api_key() -> str | None:
    """Recupera l'API key OpenAI da st.secrets o variabile d'ambiente."""
    # 1. Prova Streamlit secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # 2. Prova variabile d'ambiente
    return os.environ.get("OPENAI_API_KEY")


def _get_openai_model() -> str:
    """Recupera il modello OpenAI da configurare."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
            return st.secrets["OPENAI_MODEL"]
    except Exception:
        pass

    return os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)


def _call_openai_vision(image_b64: str, mime_type: str) -> str:
    """
    Chiama l'API OpenAI Vision con l'immagine in base64.

    Args:
        image_b64: Immagine codificata in base64.
        mime_type: MIME type dell'immagine.

    Returns:
        Risposta testuale dal modello.

    Raises:
        RuntimeError: Se l'API key non è configurata o la chiamata fallisce.
    """
    api_key = _get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non configurata. "
            "Aggiungi la chiave in .streamlit/secrets.toml o come variabile d'ambiente."
        )

    client = OpenAI(api_key=api_key)
    model = _get_openai_model()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    return response.choices[0].message.content or ""


def extract_from_image_file(image_path: str | Path) -> ExtractedData:
    """
    Estrae i dati da un file immagine usando OpenAI Vision.

    Args:
        image_path: Path del file immagine (PNG, JPG, WEBP).

    Returns:
        ExtractedData con tutti i dati estratti.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return ExtractedData(
            extraction_success=False,
            error_message=f"File non trovato: {image_path}",
        )

    try:
        image_bytes = image_path.read_bytes()
        ext = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime_type = mime_map.get(ext, "image/png")

        image_b64 = base64.b64encode(image_bytes).decode()
        response_text = _call_openai_vision(image_b64, mime_type)
        return _parse_vlm_response(response_text)

    except RuntimeError as e:
        return ExtractedData(
            extraction_success=False,
            error_message=str(e),
        )
    except Exception as e:
        return ExtractedData(
            extraction_success=False,
            error_message=f"Errore imprevisto: {e}",
        )


def extract_from_bytes(image_bytes: bytes, extension: str = ".png") -> ExtractedData:
    """
    Estrae i dati da bytes di un'immagine.

    Args:
        image_bytes: Bytes dell'immagine.
        extension: Estensione del file (.png, .jpg, .webp).

    Returns:
        ExtractedData con tutti i dati estratti.
    """
    try:
        ext_to_mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = ext_to_mime.get(extension, "image/png")
        image_b64 = base64.b64encode(image_bytes).decode()
        response_text = _call_openai_vision(image_b64, mime_type)
        return _parse_vlm_response(response_text)

    except RuntimeError as e:
        return ExtractedData(
            extraction_success=False,
            error_message=str(e),
        )
    except Exception as e:
        return ExtractedData(
            extraction_success=False,
            error_message=f"Errore imprevisto: {e}",
        )


def extract_from_base64(
    base64_data: str,
    mime_type: str = "image/png",
) -> ExtractedData:
    """
    Estrae i dati da una stringa base64.

    Args:
        base64_data: Stringa base64 dell'immagine.
        mime_type: MIME type dell'immagine.

    Returns:
        ExtractedData con tutti i dati estratti.
    """
    try:
        # Verifica che il base64 sia valido
        base64.b64decode(base64_data)
        response_text = _call_openai_vision(base64_data, mime_type)
        return _parse_vlm_response(response_text)

    except RuntimeError as e:
        return ExtractedData(
            extraction_success=False,
            error_message=str(e),
        )
    except Exception as e:
        return ExtractedData(
            extraction_success=False,
            error_message=f"Errore decodifica base64: {e}",
        )


def _parse_vlm_response(response: str) -> ExtractedData:
    """
    Parsa la risposta del VLM e estrae i dati strutturati.

    Args:
        response: Risposta testuale del VLM.

    Returns:
        ExtractedData con i dati estratti.
    """
    if not response or not response.strip():
        return ExtractedData(
            extraction_success=False,
            error_message="Risposta vuota dal VLM",
        )

    try:
        json_str = response.strip()

        # Rimuovi eventuali markdown code blocks PRIMA di cercare il JSON
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 2:
                json_str = parts[1]

        json_str = json_str.strip()

        # Cerca l'inizio del JSON ({ o [)
        lines = json_str.split("\n")
        json_start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                json_start_idx = i
                break

        if json_start_idx >= 0:
            json_str = "\n".join(lines[json_start_idx:])

        # Rimuovi eventuali code block di chiusura rimasti
        if "```" in json_str:
            json_str = json_str.split("```")[0]

        json_str = json_str.strip()
        data = json.loads(json_str)

        # Estrai il contenuto dal formato API response
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            if content:
                content = content.strip()
                if content.startswith("{"):
                    data = json.loads(content)

        return ExtractedData(
            squadra_casa=str(data.get("squadra_casa", "")).strip(),
            squadra_trasf=str(data.get("squadra_trasf", "")).strip(),
            quota_1=_safe_float(data.get("quota_1", 0)),
            quota_x=_safe_float(data.get("quota_x", 0)),
            quota_2=_safe_float(data.get("quota_2", 0)),
            linea_ou=_safe_float(data.get("linea_ou", 0)),
            quota_over=_safe_float(data.get("quota_over", 0)),
            quota_under=_safe_float(data.get("quota_under", 0)),
            quota_gg=_safe_float(data.get("quota_gg", 0)),
            quota_ng=_safe_float(data.get("quota_ng", 0)),
            confidence=str(data.get("confidence", "medium")).lower(),
            raw_response=response,
            extraction_success=True,
        )

    except json.JSONDecodeError as e:
        return _fallback_extraction(response, str(e))

    except Exception as e:
        return ExtractedData(
            extraction_success=False,
            error_message=f"Errore parsing risposta: {e}",
            raw_response=response,
        )


def _fallback_extraction(response: str, original_error: str) -> ExtractedData:
    """
    Fallback per estrarre dati quando il JSON parsing fallisce.
    """
    data = ExtractedData(
        extraction_success=False,
        error_message=f"JSON parsing fallito: {original_error}",
        raw_response=response,
    )

    try:
        # Cerca nomi squadre
        match = re.search(
            r"([A-Za-zÀ-ÿ\s]+)\s+(?:vs|-|–|vs\.)\s+([A-Za-zÀ-ÿ\s]+)",
            response,
            re.IGNORECASE,
        )
        if match:
            data.squadra_casa = match.group(1).strip()
            data.squadra_trasf = match.group(2).strip()

        # Cerca quote 1X2
        quote_pattern = r"(\d+[.,]\d{2,3})"
        quotes = re.findall(quote_pattern, response)

        if len(quotes) >= 3:
            data.quota_1 = _safe_float(quotes[0])
            data.quota_x = _safe_float(quotes[1])
            data.quota_2 = _safe_float(quotes[2])

        # Cerca linea Over/Under
        line_pattern = r"(?:over|under|totale?|total)\s*[.:]?\s*(\d+[.,]?\d*)"
        match = re.search(line_pattern, response, re.IGNORECASE)
        if match:
            data.linea_ou = _safe_float(match.group(1))

        # Se abbiamo estratto qualcosa, consideriamo successo parziale
        if data.squadra_casa or data.quota_1 > 0:
            data.extraction_success = True
            data.confidence = "low"
            data.error_message = f"Estrazione parziale (fallback): {original_error}"

    except Exception:
        pass

    return data


def _safe_float(value: Any) -> float:
    """Converte un valore in float in modo sicuro."""
    if value is None:
        return 0.0
    try:
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def validate_extracted_data(data: ExtractedData) -> tuple[bool, list[str]]:
    """
    Valida i dati estratti e restituisce eventuali problemi.

    Returns:
        (is_valid, warnings): True se i dati sono utilizzabili,
                              lista di warning per dati mancanti/sospetti.
    """
    warnings = []

    if not data.squadra_casa:
        warnings.append("Squadra casa non rilevata")
    if not data.squadra_trasf:
        warnings.append("Squadra trasferta non rilevata")

    if data.quota_1 <= 0 or data.quota_x <= 0 or data.quota_2 <= 0:
        warnings.append("Quote 1X2 incomplete o non rilevate")
    else:
        for q, name in [(data.quota_1, "1"), (data.quota_x, "X"), (data.quota_2, "2")]:
            if q < 1.01 or q > 50.0:
                warnings.append(f"Quota {name}={q} sembra fuori range")

    if data.quota_over <= 0 and data.quota_under <= 0:
        warnings.append("Quote Over/Under non rilevate")
    elif data.linea_ou <= 0:
        warnings.append("Linea Over/Under non rilevata")

    if data.quota_gg <= 0 and data.quota_ng <= 0:
        warnings.append("Quote BTTS (GG/NG) non rilevate")

    is_valid = data.quota_1 > 0 and data.quota_x > 0 and data.quota_2 > 0

    return is_valid, warnings
