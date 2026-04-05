"""
session_storage.py — Persistenza partite salvate su file JSON.

Ogni partita salvata contiene il setup prematch (linee, bankroll, ricerca)
e l'ultimo stato live. Max 8 partite — le più vecchie vengono rimpiazzate.

Il file è in data/partite_salvate.json (locale, solo per l'utente corrente).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_STORAGE_PATH = Path("data/partite_salvate.json")
_MAX_PARTITE = 8


@dataclass
class PartitaSalvata:
    """Snapshot di una partita (setup prematch + stato live al momento del salvataggio)."""

    id: str                         # ID univoco basato su timestamp
    nome: str                       # "Casa vs Trasf" (label nel dropdown)
    saved_at: str                   # Timestamp leggibile, es. "28/03 14:32"

    # Stato dei widget da ripristinare (chiavi session_state → valori)
    widget_state: dict[str, Any] = field(default_factory=dict)

    # Risultato della Ricerca Pre-Partita serializzato (None se non disponibile)
    ricerca: dict[str, Any] | None = None

    # Dati estratti dallo screen Analysis di Nowgoal (pre-partita)
    # Contiene: fixture_historical_total, forma_mult_h/a, H2H %, standings
    prematch_analysis: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# I/O su file JSON
# ---------------------------------------------------------------------------

def load_partite() -> list[PartitaSalvata]:
    """Carica tutte le partite salvate dal file JSON. Restituisce lista vuota se il file non esiste."""
    if not _STORAGE_PATH.exists():
        return []
    try:
        raw = json.loads(_STORAGE_PATH.read_text(encoding="utf-8"))
        return [PartitaSalvata(**p) for p in raw]
    except Exception:
        # File corrotto o versione incompatibile: ignora silenziosamente
        return []


def save_partita(partita: PartitaSalvata) -> None:
    """
    Salva o aggiorna una partita nel file JSON.

    Se esiste già una partita con lo stesso ID, la aggiorna (overwrite).
    Se il numero totale supera _MAX_PARTITE, rimuove la più vecchia (in testa alla lista).
    """
    _STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    partite = load_partite()

    # Aggiorna se già esiste, altrimenti aggiungi in coda
    for i, p in enumerate(partite):
        if p.id == partita.id:
            partite[i] = partita
            _write(partite)
            return

    # Nuova partita: rimuovi la più vecchia se siamo al limite
    if len(partite) >= _MAX_PARTITE:
        partite.pop(0)

    partite.append(partita)
    _write(partite)


def delete_partita(pid: str) -> None:
    """Rimuove la partita con l'ID specificato."""
    partite = [p for p in load_partite() if p.id != pid]
    _write(partite)


def _write(partite: list[PartitaSalvata]) -> None:
    _STORAGE_PATH.write_text(
        json.dumps([asdict(p) for p in partite], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Helpers per app.py
# ---------------------------------------------------------------------------

def build_partita_id() -> str:
    """Genera un ID univoco basato sul timestamp corrente."""
    return str(int(time.time() * 1000))


def build_saved_at_label() -> str:
    """Etichetta leggibile per il timestamp corrente, es. '28/03 14:32'."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d/%m %H:%M")


def collect_widget_state(session_state: Any) -> dict[str, Any]:
    """
    Estrae i valori rilevanti dal session_state di Streamlit da salvare.

    Salva solo i valori primitivi (float, int, bool) che corrispondono
    ai widget keyed dell'app. Ignora oggetti non serializzabili.
    """
    keys = [
        # Linee asiatiche
        "lines_ah_op",
        "lines_tot_op",
        "ah_cur_raw_input",
        "tot_cur_raw_input",
        "prematch_ou_line_select",
        # Stato live
        "live_minuto",
        "live_gol_casa",
        "live_gol_trasf",
        "live_rossi_casa",
        "live_rossi_trasf",
        "live_gialli_casa",
        "live_gialli_trasf",
        "live_sot_h",
        "live_soff_h",
        "live_sot_a",
        "live_soff_a",
        "live_blk_h",
        "live_blk_a",
        "live_corner_h",
        "live_corner_a",
        "live_poss_h",
        "live_poss_a",
        "live_att_per_h",
        "live_att_per_a",
        "live_att_h",
        "live_att_a",
        "live_falli_casa",
        "live_falli_trasf",
        # Bankroll
        "bankroll_value",
        "comm_pct_value",
    ]
    state: dict[str, Any] = {}
    for k in keys:
        v = session_state.get(k)
        if v is not None and isinstance(v, (int, float, bool, str)):
            state[k] = v
    return state


def restore_widget_state(session_state: Any, widget_state: dict[str, Any]) -> None:
    """
    Scrive i valori salvati nel session_state prima del render dei widget.

    Streamlit usa session_state come fonte di verità: impostare i valori
    PRIMA del render forza i widget a mostrare i valori ripristinati.
    """
    for k, v in widget_state.items():
        session_state[k] = v


def collect_prematch_analysis(session_state: Any) -> dict[str, Any] | None:
    """
    Serializza i dati estratti dallo screen Analysis (PrematchAnalysisExtracted)
    per il salvataggio su JSON.

    Salva solo i campi numerici derivati — non l'immagine né la raw_response.
    """
    pa = session_state.get("prematch_analysis")
    if pa is None or not getattr(pa, "extraction_success", False):
        return None
    pa_dict = asdict(pa)
    # Evita payload inutilmente grande nei salvataggi locali.
    pa_dict.pop("raw_response", None)
    pa_dict.pop("error_message", None)
    return pa_dict


def restore_prematch_analysis(session_state: Any, data: dict[str, Any] | None) -> None:
    """
    Ripristina i dati estratti dallo screen Analysis nel session_state.

    Ricostruisce un oggetto PrematchAnalysisExtracted con extraction_success=True
    dai dati serializzati, così l'UI può mostrare il riepilogo e il motore
    può usare i parametri senza richiedere un nuovo screen.
    """
    if not data:
        return
    try:
        from src.ocr import PrematchAnalysisExtracted
        payload = dict(data)
        payload["extraction_success"] = True
        pa = PrematchAnalysisExtracted(**payload)
        session_state["prematch_analysis"] = pa
    except Exception:
        pass  # Dati corrotti o versione incompatibile: ignora
