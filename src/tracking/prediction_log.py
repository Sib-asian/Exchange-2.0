"""
prediction_log.py — Salvataggio persistente delle previsioni.

Ogni analisi viene salvata automaticamente in un file JSON.
I risultati vengono aggiunti dopo la partita con un semplice input.

File: /home/z/my-project/data/predictions.json

Il file sopravvive a:
- Chiusura Streamlit
- Riavvio computer
- Qualsiasi interruzione
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Percorso del file dati (relativo alla directory dell'app)
# Usa la directory corrente o una sottodirectory 'data'
import os
_APP_DIR = Path(__file__).parent.parent.parent  # sale da src/tracking -> src -> root
DATA_DIR = _APP_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


@dataclass
class PredictionRecord:
    """Singola previsione salvata."""

    # Identificativo
    id: str                          # es. "20260401_vissel_kobe_shimizu"
    timestamp: str                   # ISO format
    squadra_casa: str = ""
    squadra_trasf: str = ""
    lega: str = ""

    # Input del modello
    ah_op: float = 0.0
    tot_op: float = 0.0
    xg_h: float = 0.0
    xg_a: float = 0.0

    # Previsioni modello
    p1: float = 0.0
    px: float = 0.0
    p2: float = 0.0
    p_over_25: float = 0.0
    p_under_25: float = 0.0
    p_btts: float = 0.0

    # Quote mercato (per calcolo edge)
    quota_1: float = 0.0
    quota_x: float = 0.0
    quota_2: float = 0.0
    quota_over: float = 0.0
    quota_under: float = 0.0
    quota_btts_si: float = 0.0
    quota_btts_no: float = 0.0

    # Confidenza modello
    model_confidence: float = 0.0

    # Risultato (vuoto fino a fine partita)
    gol_casa: int | None = None
    gol_trasf: int | None = None
    status: str = "PENDING"  # PENDING, COMPLETED
    completed_at: str = ""

    # Campi calcolati automaticamente dal risultato
    risultato_1x2: str = ""          # "1", "X", "2"
    over_25_hit: bool | None = None  # True se Over 2.5 entra
    btts_hit: bool | None = None     # True se entrambe segnano

    def is_completed(self) -> bool:
        return self.status == "COMPLETED" and self.gol_casa is not None

    def compute_derived_fields(self) -> None:
        """Calcola i campi derivati dal risultato."""
        if self.gol_casa is None or self.gol_trasf is None:
            return

        # Risultato 1X2
        if self.gol_casa > self.gol_trasf:
            self.risultato_1x2 = "1"
        elif self.gol_casa < self.gol_trasf:
            self.risultato_1x2 = "2"
        else:
            self.risultato_1x2 = "X"

        # Over 2.5
        total = self.gol_casa + self.gol_trasf
        self.over_25_hit = total > 2.5

        # BTTS
        self.btts_hit = self.gol_casa > 0 and self.gol_trasf > 0

        self.status = "COMPLETED"
        self.completed_at = datetime.now().isoformat()


class PredictionLog:
    """
    Gestore del log delle previsioni.

    Salvataggio persistente su file JSON.
    """

    def __init__(self) -> None:
        self._ensure_file_exists()
        self._records: list[PredictionRecord] = self._load()

    def _ensure_file_exists(self) -> None:
        """Crea il file se non esiste."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not PREDICTIONS_FILE.exists():
            self._save_to_file([])

    def _load(self) -> list[PredictionRecord]:
        """Carica le previsioni dal file."""
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [PredictionRecord(**r) for r in data.get("predictions", [])]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_to_file(self, records: list[PredictionRecord]) -> None:
        """Salva le previsioni su file."""
        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "predictions": [asdict(r) for r in records],
        }
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, record: PredictionRecord) -> None:
        """Aggiunge una nuova previsione."""
        self._records.append(record)
        self._save_to_file(self._records)

    def get_pending(self) -> list[PredictionRecord]:
        """Restituisce le partite non ancora chiuse."""
        return [r for r in self._records if r.status == "PENDING"]

    def get_completed(self) -> list[PredictionRecord]:
        """Restituisce le partite chiuse."""
        return [r for r in self._records if r.status == "COMPLETED"]

    def get_all(self) -> list[PredictionRecord]:
        """Restituisce tutte le previsioni."""
        return self._records.copy()

    def complete(self, record_id: str, gol_casa: int, gol_trasf: int) -> bool:
        """
        Chiude una previsione con il risultato.

        Args:
            record_id: ID della previsione
            gol_casa: Gol finali casa
            gol_trasf: Gol finali trasferta

        Returns:
            True se trovato e aggiornato, False altrimenti
        """
        for record in self._records:
            if record.id == record_id:
                record.gol_casa = gol_casa
                record.gol_trasf = gol_trasf
                record.compute_derived_fields()
                self._save_to_file(self._records)
                return True
        return False

    def delete(self, record_id: str) -> bool:
        """Elimina una previsione."""
        for i, r in enumerate(self._records):
            if r.id == record_id:
                self._records.pop(i)
                self._save_to_file(self._records)
                return True
        return False

    def count(self) -> dict[str, int]:
        """Conta le previsioni."""
        return {
            "total": len(self._records),
            "pending": len(self.get_pending()),
            "completed": len(self.get_completed()),
        }

    def clear_all(self) -> None:
        """Elimina tutte le previsioni (usare con cautela)."""
        self._records = []
        self._save_to_file([])


# Istanza globale singleton
_log_instance: PredictionLog | None = None


def get_prediction_log() -> PredictionLog:
    """Restituisce l'istanza globale del log."""
    global _log_instance
    if _log_instance is None:
        _log_instance = PredictionLog()
    return _log_instance


def create_record_from_analysis(
    squadra_casa: str,
    squadra_trasf: str,
    lega: str,
    input_data: dict[str, Any],
    predictions: dict[str, float],
    market_quotes: dict[str, float] | None = None,
) -> PredictionRecord:
    """
    Crea un record da salvare dai dati dell'analisi.

    Args:
        squadra_casa: Nome squadra casa
        squadra_trasf: Nome squadra trasferta
        lega: Nome lega
        input_data: Dict con ah_op, tot_op, xg_h, xg_a
        predictions: Dict con p1, px, p2, p_over, p_under, p_btts, model_confidence
        market_quotes: Dict con quote mercato (opzionale)

    Returns:
        PredictionRecord pronto da salvare
    """
    now = datetime.now()
    # Crea ID univoco
    safe_home = squadra_casa.lower().replace(" ", "_")[:20]
    safe_away = squadra_trasf.lower().replace(" ", "_")[:20]
    record_id = f"{now.strftime('%Y%m%d')}_{safe_home}_{safe_away}"

    quotes = market_quotes or {}

    return PredictionRecord(
        id=record_id,
        timestamp=now.isoformat(),
        squadra_casa=squadra_casa,
        squadra_trasf=squadra_trasf,
        lega=lega,
        ah_op=input_data.get("ah_op", 0.0),
        tot_op=input_data.get("tot_op", 0.0),
        xg_h=input_data.get("xg_h", 0.0),
        xg_a=input_data.get("xg_a", 0.0),
        p1=predictions.get("p1", 0.0),
        px=predictions.get("px", 0.0),
        p2=predictions.get("p2", 0.0),
        p_over_25=predictions.get("p_over", 0.0),
        p_under_25=predictions.get("p_under", 0.0),
        p_btts=predictions.get("p_btts", 0.0),
        model_confidence=predictions.get("model_confidence", 0.0),
        quota_1=quotes.get("quota_1", 0.0),
        quota_x=quotes.get("quota_x", 0.0),
        quota_2=quotes.get("quota_2", 0.0),
        quota_over=quotes.get("quota_over", 0.0),
        quota_under=quotes.get("quota_under", 0.0),
        quota_btts_si=quotes.get("quota_btts_si", 0.0),
        quota_btts_no=quotes.get("quota_btts_no", 0.0),
    )
