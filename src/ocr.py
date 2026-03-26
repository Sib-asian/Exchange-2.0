"""
ocr.py — Estrazione automatica dati da screenshot di siti scommesse.

Utilizza il VLM (Vision Language Model) per leggere immagini da:
  - Siti di scommesse (Bet365, Betfair, Pinnacle, ecc.)
  - App mobili
  - Desktop screenshot

Estrae automaticamente:
  - Nomi delle squadre
  - Quote 1X2
  - Quote Over/Under
  - Quote BTTS (GG/NG)

Il modulo restituisce i dati in formato strutturato per l'uso nell'UI Streamlit.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def extract_from_image_file(image_path: str | Path) -> ExtractedData:
    """
    Estrae i dati da un file immagine usando il VLM CLI.

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
        # Chiama il CLI z-ai vision
        # Usa percorso assoluto per trovare il comando
        zai_cmd = os.environ.get("ZAI_CMD", "/usr/local/bin/z-ai")
        result = subprocess.run(
            [
                zai_cmd, "vision",
                "-p", EXTRACTION_PROMPT,
                "-i", str(image_path),
                "-o", "-",  # output to stdout
            ],
            capture_output=True,
            text=True,
            timeout=60,  # 60 secondi timeout
        )

        if result.returncode != 0:
            return ExtractedData(
                extraction_success=False,
                error_message=f"Errore CLI: {result.stderr}",
            )

        # Parsa la risposta
        return _parse_vlm_response(result.stdout)

    except subprocess.TimeoutExpired:
        return ExtractedData(
            extraction_success=False,
            error_message="Timeout: l'analisi dell'immagine ha impiegato troppo tempo",
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
        # Crea un file temporaneo
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            prefix="ocr_match_",
        ) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        try:
            return extract_from_image_file(tmp_path)
        finally:
            # Cleanup file temporaneo
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    except Exception as e:
        return ExtractedData(
            extraction_success=False,
            error_message=f"Errore salvataggio immagine: {e}",
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
        # Determina l'estensione dal MIME type
        mime_to_ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        extension = mime_to_ext.get(mime_type, ".png")

        # Decodifica base64
        image_bytes = base64.b64decode(base64_data)

        return extract_from_bytes(image_bytes, extension)

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
        # Cerca JSON nella risposta (potrebbe essere racchiuso in ```json ... ```)
        json_str = response.strip()

        # Rimuovi eventuali markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]

        json_str = json_str.strip()

        # Parsa JSON
        data = json.loads(json_str)

        # Crea ExtractedData con valori di default per campi mancanti
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
        # Tenta estrazione con regex come fallback
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

    Cerca pattern comuni nella risposta testuale.
    """
    data = ExtractedData(
        extraction_success=False,
        error_message=f"JSON parsing fallito: {original_error}",
        raw_response=response,
    )

    try:
        # Cerca nomi squadre (pattern: "Squadra A vs Squadra B" o simili)
        match = re.search(r"([A-Za-zÀ-ÿ\s]+)\s+(?:vs|-|–|vs\.)\s+([A-Za-zÀ-ÿ\s]+)", response, re.IGNORECASE)
        if match:
            data.squadra_casa = match.group(1).strip()
            data.squadra_trasf = match.group(2).strip()

        # Cerca quote 1X2
        quote_pattern = r"(\d+[.,]\d{2,3})"
        quotes = re.findall(quote_pattern, response)

        if len(quotes) >= 3:
            # Assume le prime tre quote siano 1, X, 2
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
            # Gestisce virgola come separatore decimale
            value = value.replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# Funzione di utilità per validare i dati estratti
def validate_extracted_data(data: ExtractedData) -> tuple[bool, list[str]]:
    """
    Valida i dati estratti e restituisce eventuali problemi.

    Returns:
        (is_valid, warnings): True se i dati sono utilizzabili,
                              lista di warning per dati mancanti/sospetti.
    """
    warnings = []

    # Controlla squadre
    if not data.squadra_casa:
        warnings.append("Squadra casa non rilevata")
    if not data.squadra_trasf:
        warnings.append("Squadra trasferta non rilevata")

    # Controlla quote 1X2
    if data.quota_1 <= 0 or data.quota_x <= 0 or data.quota_2 <= 0:
        warnings.append("Quote 1X2 incomplete o non rilevate")
    else:
        # Verifica che le quote siano in range plausibile
        for q, name in [(data.quota_1, "1"), (data.quota_x, "X"), (data.quota_2, "2")]:
            if q < 1.01 or q > 50.0:
                warnings.append(f"Quota {name}={q} sembra fuori range")

    # Controlla Over/Under
    if data.quota_over <= 0 and data.quota_under <= 0:
        warnings.append("Quote Over/Under non rilevate")
    elif data.linea_ou <= 0:
        warnings.append("Linea Over/Under non rilevata")

    # Controlla BTTS
    if data.quota_gg <= 0 and data.quota_ng <= 0:
        warnings.append("Quote BTTS (GG/NG) non rilevate")

    # Dati utilizzabili se abbiamo almeno le quote 1X2
    is_valid = data.quota_1 > 0 and data.quota_x > 0 and data.quota_2 > 0

    return is_valid, warnings
