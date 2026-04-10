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
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

# Percorso del file dati (relativo alla directory dell'app)
_APP_DIR = Path(__file__).parent.parent.parent  # sale da src/tracking -> src -> root
DATA_DIR = _APP_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


def tot_op_band(tot_op: float) -> str:
    """Fascia della linea total apertura per segmentazione statistiche."""
    if tot_op <= 0:
        return "unknown"
    if tot_op < 2.25:
        return "<2.25"
    if tot_op <= 2.75:
        return "2.25-2.75"
    return ">2.75"


def record_from_dict(data: dict[str, Any]) -> PredictionRecord:
    """Costruisce un record ignorando chiavi sconosciute (retrocompatibilità JSON)."""
    allowed = {f.name for f in fields(PredictionRecord)}
    return PredictionRecord(**{k: v for k, v in data.items() if k in allowed})


def assess_quote_quality(
    quotes: dict[str, float] | None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Classifica la qualità quote per uso tracking:
    - trusted: tripla 1X2 valida e overround plausibile
    - untrusted: quote assenti/parziali/incoerenti
    """
    q = quotes or {}
    q1 = float(q.get("quota_1", 0.0) or 0.0)
    qx = float(q.get("quota_x", 0.0) or 0.0)
    q2 = float(q.get("quota_2", 0.0) or 0.0)
    source = str((metadata or {}).get("quote_source", "")).strip() or "unknown"
    if not (q1 > 1.0 and qx > 1.0 and q2 > 1.0):
        return "untrusted", f"missing_or_partial_1x2:{source}"
    overround = 1.0 / q1 + 1.0 / qx + 1.0 / q2
    if not (1.0 <= overround <= 1.30):
        return "untrusted", f"bad_overround_1x2:{overround:.3f}:{source}"
    if not all(math.isfinite(v) for v in (q1, qx, q2)):
        return "untrusted", f"non_finite_quote:{source}"
    return "trusted", f"ok:{source}"


@dataclass
class PredictionRecord:
    """Singola previsione salvata."""

    # Identificativo
    id: str                          # es. "20260401_vissel_kobe_shimizu"
    timestamp: str                   # ISO format
    squadra_casa: str = ""
    squadra_trasf: str = ""
    lega: str = ""
    minuto: int = 0
    is_prematch: bool = False

    # Input del modello
    ah_op: float = 0.0
    tot_op: float = 0.0
    ou_line: float = 2.5  # linea O/U usata per p_over/p_under e per esito Over nel tracker
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

    # Osservabilità / qualità estrazione e motore
    extraction_coverage: float = 0.0
    league_source: str = ""
    model_agreement: float = 0.0
    quality_score: float = 0.0
    signals_blocked: bool = False
    signals_block_reason: str = ""
    tot_band: str = ""
    software_version: str = ""
    # Revisione motore/pipeline (config ENGINE.MODEL_REVISION) per audit e champion/challenger.
    model_revision: str = ""
    consensus_w_bp: float = 0.0
    consensus_w_cop: float = 0.0
    consensus_w_mk: float = 0.0
    # Probabilità 1X2 grezze per modello (pre-consensus calibrato) — apprendimento pesi
    p1_bp: float = 0.0
    px_bp: float = 0.0
    p2_bp: float = 0.0
    p1_cop: float = 0.0
    px_cop: float = 0.0
    p2_cop: float = 0.0
    p1_mk: float = 0.0
    px_mk: float = 0.0
    p2_mk: float = 0.0
    # Optional closing market snapshots (for CLV proxy)
    quota_1_close: float = 0.0
    quota_x_close: float = 0.0
    quota_2_close: float = 0.0
    quota_over_close: float = 0.0
    quota_under_close: float = 0.0
    quota_btts_si_close: float = 0.0
    quota_btts_no_close: float = 0.0
    quote_quality: str = "unknown"  # trusted | untrusted | unknown
    quote_quality_reason: str = ""

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

        # Over/Under sulla linea salvata (tipicamente 1.5 o 2.5)
        total = self.gol_casa + self.gol_trasf
        self.over_25_hit = total > float(self.ou_line)

        # BTTS
        self.btts_hit = self.gol_casa > 0 and self.gol_trasf > 0

        self.status = "COMPLETED"
        self.completed_at = datetime.now().isoformat()


class PredictionLog:
    """
    Gestore del log delle previsioni.

    Salvataggio: file JSON locale, oppure Supabase se configurato
    (SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in env o Streamlit Secrets).
    """

    def __init__(self) -> None:
        from src.tracking import supabase_store as _sb

        self._use_supabase = _sb.is_enabled()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Carica prima da Supabase/file: non creare un file vuoto prima del load
        # (eviterebbe di sovrascrivere dati remoti al redeploy senza file locale).
        self._records: list[PredictionRecord] = self._load()

    def _load(self) -> list[PredictionRecord]:
        """Carica da Supabase se attivo, altrimenti (o se errore rete) da file."""
        from src.tracking import supabase_store as _sb

        if self._use_supabase:
            remote = _sb.fetch_payload()
            if remote is not None:
                try:
                    return [record_from_dict(r) for r in remote.get("predictions", [])]
                except (TypeError, ValueError):
                    pass
        try:
            with open(PREDICTIONS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return [record_from_dict(r) for r in data.get("predictions", [])]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_to_file(self, records: list[PredictionRecord]) -> None:
        """Salva il blob completo (file + opzionalmente Supabase)."""
        from src.tracking import supabase_store as _sb

        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "predictions": [asdict(r) for r in records],
        }
        if self._use_supabase:
            _sb.save_payload(data)
        try:
            with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

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

    def backfill_quote_quality(self, *, overwrite: bool = False) -> dict[str, int]:
        """
        Retro-tagga quote_quality sui record esistenti.
        overwrite=False: aggiorna solo record unknown/vuoti.
        """
        updated = 0
        trusted = 0
        untrusted = 0
        for r in self._records:
            current = str(getattr(r, "quote_quality", "") or "").strip().lower()
            if (not overwrite) and current in {"trusted", "untrusted"}:
                if current == "trusted":
                    trusted += 1
                else:
                    untrusted += 1
                continue
            q_quality, q_reason = assess_quote_quality(
                {
                    "quota_1": float(getattr(r, "quota_1", 0.0) or 0.0),
                    "quota_x": float(getattr(r, "quota_x", 0.0) or 0.0),
                    "quota_2": float(getattr(r, "quota_2", 0.0) or 0.0),
                },
                {"quote_source": "retro-tag"},
            )
            r.quote_quality = q_quality
            r.quote_quality_reason = q_reason
            updated += 1
            if q_quality == "trusted":
                trusted += 1
            else:
                untrusted += 1
        if updated > 0:
            self._save_to_file(self._records)
        return {
            "updated": updated,
            "trusted": trusted,
            "untrusted": untrusted,
            "total": len(self._records),
        }


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
    metadata: dict[str, Any] | None = None,
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
    # ID univoco: data + orario con microsecondi (evita collisioni stesso giorno / stessa coppia).
    safe_home = squadra_casa.lower().replace(" ", "_")[:20]
    safe_away = squadra_trasf.lower().replace(" ", "_")[:20]
    record_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{safe_home}_{safe_away}"

    quotes = market_quotes or {}
    meta = metadata or {}
    tot_op_val = float(input_data.get("tot_op", 0.0))

    _qq, _qq_reason = assess_quote_quality(quotes, meta)
    return PredictionRecord(
        id=record_id,
        timestamp=now.isoformat(),
        squadra_casa=squadra_casa,
        squadra_trasf=squadra_trasf,
        lega=lega,
        ah_op=input_data.get("ah_op", 0.0),
        tot_op=tot_op_val,
        ou_line=float(input_data.get("linea_ou", 2.5)),
        xg_h=input_data.get("xg_h", 0.0),
        xg_a=input_data.get("xg_a", 0.0),
        minuto=int(input_data.get("minuto", 0)),
        is_prematch=bool(input_data.get("minuto", 0) == 0),
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
        extraction_coverage=float(meta.get("extraction_coverage", 0.0)),
        league_source=str(meta.get("league_source", "")),
        model_agreement=float(meta.get("model_agreement", 0.0)),
        quality_score=float(meta.get("quality_score", 0.0)),
        signals_blocked=bool(meta.get("signals_blocked", False)),
        signals_block_reason=str(meta.get("signals_block_reason", "")),
        tot_band=str(meta.get("tot_band", tot_op_band(tot_op_val))),
        software_version=str(meta.get("software_version", "")),
        model_revision=str(meta.get("model_revision", "")),
        consensus_w_bp=float(meta.get("consensus_w_bp", 0.0)),
        consensus_w_cop=float(meta.get("consensus_w_cop", 0.0)),
        consensus_w_mk=float(meta.get("consensus_w_mk", 0.0)),
        p1_bp=float(meta.get("p1_bp", 0.0)),
        px_bp=float(meta.get("px_bp", 0.0)),
        p2_bp=float(meta.get("p2_bp", 0.0)),
        p1_cop=float(meta.get("p1_cop", 0.0)),
        px_cop=float(meta.get("px_cop", 0.0)),
        p2_cop=float(meta.get("p2_cop", 0.0)),
        p1_mk=float(meta.get("p1_mk", 0.0)),
        px_mk=float(meta.get("px_mk", 0.0)),
        p2_mk=float(meta.get("p2_mk", 0.0)),
        quota_1_close=float(quotes.get("quota_1_close", 0.0)),
        quota_x_close=float(quotes.get("quota_x_close", 0.0)),
        quota_2_close=float(quotes.get("quota_2_close", 0.0)),
        quota_over_close=float(quotes.get("quota_over_close", 0.0)),
        quota_under_close=float(quotes.get("quota_under_close", 0.0)),
        quota_btts_si_close=float(quotes.get("quota_btts_si_close", 0.0)),
        quota_btts_no_close=float(quotes.get("quota_btts_no_close", 0.0)),
        quote_quality=str(meta.get("quote_quality", _qq)),
        quote_quality_reason=str(meta.get("quote_quality_reason", _qq_reason)),
    )
