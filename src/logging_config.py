"""
logging_config.py — Configurazione logging strutturato per debug e analisi.

Fornisce logging JSON per:
- Debug di calcoli intermedi
- Analisi post-partita
- Audit trail delle operazioni
- Monitoraggio performance
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

# Formattatore JSON personalizzato
class JSONFormatter(logging.Formatter):
    """Formattatore che produce log in formato JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Aggiungi extra fields se presenti
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["data"] = self._serialize(record.extra_data)

        # Aggiungi eccezione se presente
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)

    def _serialize(self, obj: Any) -> Any:
        """Serializza oggetti complessi per JSON."""
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        if isinstance(obj, float):
            return round(obj, 6)  # Evita problemi di precisione
        return obj


def setup_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """
    Configura il logging dell'applicazione.

    Args:
        level: Livello di logging (default: INFO).
        json_output: Se True, usa formato JSON per machine parsing.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Rimuovi handler esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Crea handler per stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        # Formato human-readable
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Ottiene un logger con il nome specificato.

    Args:
        name: Nome del logger (solitamente __name__).

    Returns:
        Logger configurato.
    """
    return logging.getLogger(name)


# Logger pre-configurati per i moduli principali
engine_logger = get_logger("exchange.engine")
calibration_logger = get_logger("exchange.calibration")
market_logger = get_logger("exchange.market")
signal_logger = get_logger("exchange.signal")


class AnalysisLogger:
    """
    Context manager per loggare un'analisi completa.

    Usage:
        with AnalysisLogger(match_id="12345") as log:
            log.data("xG_calculated", {"home": 1.5, "away": 0.8})
            log.data("probabilities", {"p1": 0.55, "px": 0.25, "p2": 0.20})
    """

    def __init__(self, match_id: str = None, minute: int = None):
        self.match_id = match_id
        self.minute = minute
        self.start_time = None
        self.data_points: list[dict] = []
        self.logger = engine_logger

    def __enter__(self) -> "AnalysisLogger":
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"Analysis started: match={self.match_id}, minute={self.minute}",
            extra={"extra_data": {"match_id": self.match_id, "minute": self.minute}},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        if exc_type:
            self.logger.error(
                f"Analysis failed: {exc_val}",
                extra={"extra_data": {
                    "match_id": self.match_id,
                    "duration_ms": duration_ms,
                    "error": str(exc_val),
                }},
            )
        else:
            self.logger.info(
                f"Analysis completed in {duration_ms:.1f}ms",
                extra={"extra_data": {
                    "match_id": self.match_id,
                    "duration_ms": duration_ms,
                    "data_points": len(self.data_points),
                }},
            )

    def data(self, name: str, value: dict) -> None:
        """Logga un punto dati dell'analisi."""
        entry = {"name": name, "value": value, "timestamp": datetime.utcnow().isoformat()}
        self.data_points.append(entry)
        self.logger.debug(
            f"Data: {name}",
            extra={"extra_data": {name: value}},
        )

    def warning(self, message: str, details: dict = None) -> None:
        """Logga un warning durante l'analisi."""
        self.logger.warning(
            message,
            extra={"extra_data": details or {}},
        )
