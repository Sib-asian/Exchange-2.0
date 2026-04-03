"""
inputs.py — Widget di input Streamlit per Radar Pro Live.

Centralizza tutta la logica di input dell'interfaccia utente:
  - Upload screenshot con estrazione automatica dati (VLM)
  - Stato partita (minuto, gol, cartellini)
  - Linee asiatiche (apertura + corrente)
  - Selezione linea Over/Under
  - Bankroll e commissione
  - Tiri live
  - Quote exchange (opzionali)

Restituisce oggetti tipizzati (MatchState, ExchangeQuotes) pronti per il motore.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import BAYES, INPUT_VALIDATION, UI
from src.engine import ExchangeQuotes, MatchState
from src.ocr import (
    ExtractedData,
    LiveStatsExtracted,
    PrematchAnalysisExtracted,
    extract_from_bytes,
    extract_live_stats_from_bytes,
    extract_prematch_analysis_from_bytes,
    extract_prematch_analysis_from_url,
    validate_extracted_data,
)


# ---------------------------------------------------------------------------
# Upload Screenshot con estrazione automatica
# ---------------------------------------------------------------------------

def render_image_upload() -> ExtractedData | None:
    """
    Render del widget per caricare screenshot da siti di scommesse.

    L'immagine viene analizzata automaticamente dal VLM per estrarre:
      - Nomi delle squadre
      - Quote 1X2
      - Quote Over/Under
      - Quote BTTS (GG/NG)

    Returns:
        ExtractedData se l'immagine è stata caricata ed elaborata,
        None altrimenti.
    """
    st.markdown("### 📷 Lettura Automatica da Screenshot")

    # File uploader
    uploaded_file = st.file_uploader(
        "Carica uno screenshot da sito di scommesse o app",
        type=["png", "jpg", "jpeg", "webp"],
        help=(
            "Accetta screenshot da: Bet365, Betfair, Pinnacle, William Hill, "
            "e altre app/siti di scommesse. L'AI estrarrà automaticamente "
            "quote e nomi delle squadre."
        ),
        key="screenshot_uploader",
    )

    if uploaded_file is None:
        # Mostra istruzioni
        st.info(
            "📋 **Come funziona:**\n\n"
            "1. Fai uno screenshot delle quote dal tuo sito/app di scommesse\n"
            "2. Carica l'immagine qui sopra\n"
            "3. L'AI estrarrà automaticamente: squadre, quote 1X2, Over/Under, BTTS\n"
            "4. Dovrai inserire manualmente le **linee asiatiche** (AH e Total)\n\n"
            "⚡ L'analisi è automatica al caricamento!"
        )
        return None

    # Verifica se abbiamo già elaborato questa immagine
    file_id = f"ocr_{uploaded_file.name}_{uploaded_file.size}"

    if "last_ocr_file_id" in st.session_state and st.session_state.last_ocr_file_id == file_id:
        # Restituisci i dati già elaborati
        return st.session_state.get("extracted_data")

    # Mostra spinner durante l'elaborazione
    with st.spinner("🔍 Analisi immagine in corso..."):
        try:
            # Leggi i bytes del file
            image_bytes = uploaded_file.read()

            # Determina il tipo MIME
            mime_type = uploaded_file.type or "image/png"

            # Estrai i dati
            extracted = extract_from_bytes(image_bytes, extension=_get_extension(mime_type))

            # Salva in session state
            st.session_state.last_ocr_file_id = file_id
            st.session_state.extracted_data = extracted

            return extracted

        except Exception as e:
            st.error(f"❌ Errore durante l'analisi dell'immagine: {e}")
            return ExtractedData(
                extraction_success=False,
                error_message=str(e),
            )


def render_extracted_data_panel(data: ExtractedData) -> dict:
    """
    Mostra i dati estratti in un pannello con possibilità di modifica.

    Args:
        data: Dati estratti dall'immagine.

    Returns:
        Dict con i dati (eventualmente modificati dall'utente).
    """
    if not data.extraction_success:
        st.warning(f"⚠️ Estrazione parziale o fallita: {data.error_message}")
        st.caption("Puoi inserire i dati manualmente nei campi sottostanti.")
        return {}

    # Mostra confidence
    confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(data.confidence, "⚪")
    st.markdown(f"**Affidabilità estrazione:** {confidence_icon} {data.confidence.upper()}")

    # Valida i dati
    _is_valid, validation_warnings = validate_extracted_data(data)
    if validation_warnings:
        for w in validation_warnings:
            st.caption(f"⚠️ {w}")

    st.divider()

    # Form con dati estratti (modificabili)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏟️ Partita**")
        squadra_casa = st.text_input(
            "Squadra Casa",
            value=data.squadra_casa,
            key="ocr_squadra_casa",
        )
        squadra_trasf = st.text_input(
            "Squadra Trasferta",
            value=data.squadra_trasf,
            key="ocr_squadra_trasf",
        )

    with col2:
        st.markdown("**📊 Quote 1X2**")
        q1, qx, q2 = st.columns(3)
        with q1:
            quota_1 = st.number_input(
                "Quota 1",
                value=data.quota_1,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_1",
            )
        with qx:
            quota_x = st.number_input(
                "Quota X",
                value=data.quota_x,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_x",
            )
        with q2:
            quota_2 = st.number_input(
                "Quota 2",
                value=data.quota_2,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_2",
            )

    st.divider()

    # Quote Over/Under e BTTS
    col_ou, col_btts = st.columns(2)

    with col_ou:
        st.markdown("**📈 Over/Under**")
        ou_linee = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        # Trova la linea più vicina a quella estratta
        if data.linea_ou > 0:
            default_linea_idx = min(
                range(len(ou_linee)),
                key=lambda i: abs(ou_linee[i] - data.linea_ou),
            )
        else:
            default_linea_idx = ou_linee.index(2.5)  # Default a 2.5

        linea_ou = st.selectbox(
            "Linea O/U",
            ou_linee,
            index=default_linea_idx,
            key="ocr_linea_ou",
        )

        qo, qu = st.columns(2)
        with qo:
            quota_over = st.number_input(
                f"Quota Over {linea_ou}",
                value=data.quota_over,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_over",
            )
        with qu:
            quota_under = st.number_input(
                f"Quota Under {linea_ou}",
                value=data.quota_under,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_under",
            )

    with col_btts:
        st.markdown("**⚽ BTTS (Goal/No Goal)**")
        qgg, qng = st.columns(2)
        with qgg:
            quota_gg = st.number_input(
                "Quota GG (Sì)",
                value=data.quota_gg,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_gg",
                help="Entrambe le squadre segnano",
            )
        with qng:
            quota_ng = st.number_input(
                "Quota NG (No)",
                value=data.quota_ng,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="ocr_quota_ng",
                help="Almeno una squadra non segna",
            )

    # Calcola probabilità implicite per mostrare allineamento
    if quota_1 > 1 and quota_x > 1 and quota_2 > 1:
        st.divider()
        st.markdown("**📉 Probabilità Implicite di Mercato**")

        # Margin del bookmaker
        overround = (1/quota_1 + 1/quota_x + 1/quota_2 - 1) * 100

        # Probabilità implicite normalizzate
        total_implied = 1/quota_1 + 1/quota_x + 1/quota_2
        p1_imp = (1/quota_1) / total_implied * 100
        px_imp = (1/quota_x) / total_implied * 100
        p2_imp = (1/quota_2) / total_implied * 100

        c1, cx, c2, cm = st.columns(4)
        with c1:
            st.metric("1 (Casa)", f"{p1_imp:.1f}%")
        with cx:
            st.metric("X (Pareggio)", f"{px_imp:.1f}%")
        with c2:
            st.metric("2 (Trasferta)", f"{p2_imp:.1f}%")
        with cm:
            st.metric("Margin", f"+{overround:.1f}%", delta_color="inverse")

    return {
        "squadra_casa": squadra_casa,
        "squadra_trasf": squadra_trasf,
        "quota_1": quota_1,
        "quota_x": quota_x,
        "quota_2": quota_2,
        "linea_ou": linea_ou,
        "quota_over": quota_over,
        "quota_under": quota_under,
        "quota_gg": quota_gg,
        "quota_ng": quota_ng,
    }


def _get_extension(mime_type: str) -> str:
    """Converte MIME type in estensione file."""
    mime_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    return mime_map.get(mime_type, ".png")


# ---------------------------------------------------------------------------
# Input Originali
# ---------------------------------------------------------------------------

_LIVE_WIDGET_KEYS = [
    "live_minuto", "live_gol_casa", "live_gol_trasf",
    "live_rossi_casa", "live_rossi_trasf",
    "live_gialli_casa", "live_gialli_trasf",
    "live_sot_h", "live_soff_h", "live_sot_a", "live_soff_a",
    "live_blk_h", "live_blk_a",
    "live_corner_h", "live_corner_a",
    "live_poss_h", "live_poss_a",
    "live_att_per_h", "live_att_per_a",
    "live_att_h", "live_att_a",
    "live_falli_casa", "live_falli_trasf",
]


def _push_live_data_to_session(data: LiveStatsExtracted) -> None:
    """Scrive i valori estratti direttamente nel session state dei widget.

    Streamlit usa il session state come fonte di verità per i widget.
    Scrivendo i valori PRIMA che i widget vengano renderizzati, forziamo
    Streamlit a mostrare i valori aggiornati.

    NON sovrascrive minuto, gol e rossi se lo screenshot non li contiene
    (valore 0) — l'utente li inserisce manualmente prima di caricare lo screen.
    """
    # Minuto e punteggio: scrivi se estratti dalla pagina detail di Nowgoal.
    # IMPORTANTE: non scrivere direttamente ai widget-bound keys qui —
    # usa _pending_live_data che viene applicato PRIMA del render dei widget.
    # Il minuto 0 è il default — non sovrascrivere se non rilevato.
    pending: dict = {}
    if data.minuto > 0:
        pending["live_minuto"] = data.minuto
    pending["live_gol_casa"]  = data.gol_casa
    pending["live_gol_trasf"] = data.gol_trasf

    # Cartellini: sempre (derivati dagli eventi o conteggio diretto)
    pending["live_rossi_casa"]   = data.rossi_casa
    pending["live_rossi_trasf"]  = data.rossi_trasf
    pending["live_gialli_casa"]  = data.gialli_casa
    pending["live_gialli_trasf"] = data.gialli_trasf

    # Statistiche live (scritte direttamente — non hanno widget in render_live_semplice)
    st.session_state["live_sot_h"] = data.tiri_porta_casa
    st.session_state["live_soff_h"] = data.tiri_fuori_casa
    st.session_state["live_sot_a"] = data.tiri_porta_trasf
    st.session_state["live_soff_a"] = data.tiri_fuori_trasf
    st.session_state["live_blk_h"] = data.tiri_bloccati_casa
    st.session_state["live_blk_a"] = data.tiri_bloccati_trasf
    st.session_state["live_corner_h"] = data.corner_casa
    st.session_state["live_corner_a"] = data.corner_trasf
    st.session_state["live_poss_h"] = data.possesso_casa
    st.session_state["live_poss_a"] = data.possesso_trasf
    st.session_state["live_att_per_h"] = data.attacchi_pericolosi_casa
    st.session_state["live_att_per_a"] = data.attacchi_pericolosi_trasf
    st.session_state["live_att_h"] = data.attacchi_casa
    st.session_state["live_att_a"] = data.attacchi_trasf
    st.session_state["live_falli_casa"] = data.falli_casa
    st.session_state["live_falli_trasf"] = data.falli_trasf

    # Salva i valori widget-bound in una chiave pending: verranno applicati
    # PRIMA del render dei widget al prossimo ciclo (dopo st.rerun())
    st.session_state["_pending_live_data"] = pending


def render_live_screenshot_upload() -> LiveStatsExtracted | None:
    """
    Render del widget per caricare screenshot di statistiche live (Nowgoal, ecc.).

    Returns:
        LiveStatsExtracted se elaborato, None altrimenti.
    """
    uploaded_file = st.file_uploader(
        "Carica screenshot statistiche live (Nowgoal, FlashScore, SofaScore...)",
        type=["png", "jpg", "jpeg", "webp"],
        help=(
            "Carica uno screenshot dalla pagina delle statistiche live. "
            "L'AI leggerà automaticamente: punteggio, minuto, tiri, corner, "
            "possesso, attacchi, cartellini e altro."
        ),
        key="live_stats_uploader",
    )

    if uploaded_file is None:
        return None

    file_id = f"live_{uploaded_file.name}_{uploaded_file.size}"
    cached = st.session_state.get("live_stats_data")
    if (
        "last_live_file_id" in st.session_state
        and st.session_state.last_live_file_id == file_id
        and cached is not None
        and cached.extraction_success
    ):
        return cached

    with st.spinner("📊 Lettura statistiche live in corso..."):
        try:
            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.type or "image/png"
            extracted = extract_live_stats_from_bytes(
                image_bytes, extension=_get_extension(mime_type),
            )
            st.session_state.last_live_file_id = file_id
            st.session_state.live_stats_data = extracted
            # Scrivi i valori nei widget e forza re-render
            if extracted.extraction_success:
                _push_live_data_to_session(extracted)
                st.rerun()
            return extracted
        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {e}")
            return LiveStatsExtracted(
                extraction_success=False, error_message=str(e),
            )


def _render_live_stats_panel(data: LiveStatsExtracted) -> dict:
    """Mostra le statistiche live estratte con possibilità di modifica."""
    if not data.extraction_success:
        st.warning(f"⚠️ Estrazione fallita: {data.error_message}")
        st.caption("Inserisci i dati manualmente nei campi sottostanti.")

    confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
        data.confidence, "⚪",
    )
    if data.extraction_success:
        st.markdown(
            f"**Affidabilità lettura:** {confidence_icon} {data.confidence.upper()}",
        )
        # Debug: mostra risposta raw per diagnostica
        if data.raw_response:
            with st.expander("🔍 Debug: risposta AI raw", expanded=False):
                st.code(data.raw_response, language="json")
    st.divider()

    # Riga 1: Minuto e Punteggio (sempre manuali, lo screenshot non li contiene)
    col_min, col_g1, col_g2 = st.columns([1, 1, 1])
    with col_min:
        minuto = st.slider("Minuto", 0, 90, 0, 1, key="live_minuto")
    with col_g1:
        gol_casa = st.number_input(
            "Gol CASA", min_value=0, max_value=20,
            key="live_gol_casa",
        )
    with col_g2:
        gol_trasf = st.number_input(
            "Gol TRASF.", min_value=0, max_value=20,
            key="live_gol_trasf",
        )

    # Riga 2: Cartellini (manuali — rossi sempre manuali, gialli da OCR o manuali)
    col_r1, col_r2, col_y1, col_y2 = st.columns(4)
    with col_r1:
        rossi_casa = st.number_input(
            "🟥 Rossi CASA", min_value=0, max_value=4,
            key="live_rossi_casa",
        )
    with col_r2:
        rossi_trasf = st.number_input(
            "🟥 Rossi TRASF.", min_value=0, max_value=4,
            key="live_rossi_trasf",
        )
    with col_y1:
        gialli_casa = st.number_input(
            "🟨 Gialli CASA", min_value=0, max_value=20,
            key="live_gialli_casa",
        )
    with col_y2:
        gialli_trasf = st.number_input(
            "🟨 Gialli TRASF.", min_value=0, max_value=20,
            key="live_gialli_trasf",
        )

    st.divider()
    st.markdown("**📊 Statistiche Live**")

    # Riga 3: Tiri
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        sot_h = st.number_input(
            "Tiri porta 🏠", min_value=0,
            key="live_sot_h",
        )
    with col_t2:
        soff_h = st.number_input(
            "Tiri fuori 🏠", min_value=0,
            key="live_soff_h",
        )
    with col_t3:
        sot_a = st.number_input(
            "Tiri porta ✈️", min_value=0,
            key="live_sot_a",
        )
    with col_t4:
        soff_a = st.number_input(
            "Tiri fuori ✈️", min_value=0,
            key="live_soff_a",
        )

    # Riga 3b: Tiri bloccati
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        blk_h = st.number_input(
            "Bloccati 🏠", min_value=0,
            key="live_blk_h",
        )
    with col_b2:
        blk_a = st.number_input(
            "Bloccati ✈️", min_value=0,
            key="live_blk_a",
        )

    # Riga 4: Corner e Possesso
    col_c1, col_c2, col_p1, col_p2 = st.columns(4)
    with col_c1:
        corner_h = st.number_input(
            "Corner 🏠", min_value=0,
            key="live_corner_h",
        )
    with col_c2:
        corner_a = st.number_input(
            "Corner ✈️", min_value=0,
            key="live_corner_a",
        )
    with col_p1:
        poss_h = st.number_input(
            "Possesso% 🏠",
            min_value=0.0, max_value=100.0, step=1.0,
            key="live_poss_h",
        )
    with col_p2:
        poss_a = st.number_input(
            "Possesso% ✈️",
            min_value=0.0, max_value=100.0, step=1.0,
            key="live_poss_a",
        )

    # Riga 5: Attacchi (pericolosi + totali)
    col_a1, col_a2, col_at1, col_at2 = st.columns(4)
    with col_a1:
        att_per_h = st.number_input(
            "Att. Peric. 🏠",
            min_value=0, key="live_att_per_h",
        )
    with col_a2:
        att_per_a = st.number_input(
            "Att. Peric. ✈️",
            min_value=0, key="live_att_per_a",
        )
    with col_at1:
        att_h = st.number_input(
            "Att. Totali 🏠",
            min_value=0, key="live_att_h",
        )
    with col_at2:
        att_a = st.number_input(
            "Att. Totali ✈️",
            min_value=0, key="live_att_a",
        )

    # Riga 6: Falli
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        falli_casa = st.number_input(
            "Falli CASA", min_value=0,
            key="live_falli_casa",
        )
    with col_f2:
        falli_trasf = st.number_input(
            "Falli TRASF.", min_value=0,
            key="live_falli_trasf",
        )

    return {
        "minuto": minuto,
        "gol_casa": gol_casa,
        "gol_trasf": gol_trasf,
        "rossi_casa": rossi_casa,
        "rossi_trasf": rossi_trasf,
        "gialli_casa": gialli_casa,
        "gialli_trasf": gialli_trasf,
        "sot_h": sot_h,
        "soff_h": soff_h,
        "sot_a": sot_a,
        "soff_a": soff_a,
        "blk_h": blk_h,
        "blk_a": blk_a,
        "corner_h": corner_h,
        "corner_a": corner_a,
        "possesso_h": poss_h,
        "possesso_a": poss_a,
        "att_pericolosi_h": att_per_h,
        "att_pericolosi_a": att_per_a,
        "att_h": att_h,
        "att_a": att_a,
        "falli_casa": falli_casa,
        "falli_trasf": falli_trasf,
    }


def render_match_state_live() -> dict:
    """
    Render del blocco "Stato Partita Live" con screenshot o input manuale.

    Returns:
        Dict con tutti i valori live (minuto, gol, rossi, tiri, corner, ecc.).
    """
    st.header("3. Stato Partita Live")

    live_data = render_live_screenshot_upload()

    if live_data is not None:
        return _render_live_stats_panel(live_data)

    # Nessuno screenshot: mostra info e campi manuali vuoti
    st.info(
        "📋 **Come funziona:**\n\n"
        "1. Inserisci **minuto** e **gol** (li vedi dal live)\n"
        "2. Vai sulla tab **statistiche** di Nowgoal/FlashScore/SofaScore\n"
        "3. Fai uno screenshot e caricalo qui sopra\n"
        "4. L'AI leggerà: tiri, corner, possesso, attacchi, cartellini\n"
        "5. Minuto e gol che hai già inserito NON vengono sovrascritti\n\n"
        "Puoi anche inserire tutto manualmente qui sotto."
    )

    # Fallback manuale compatto
    empty_data = LiveStatsExtracted()
    return _render_live_stats_panel(empty_data)


def render_asian_lines(gol_casa: int = 0, gol_trasf: int = 0, minuto: int = 0, tot_op: float = 2.5) -> dict:
    """
    Render del blocco "Linee Asiatiche" con selettore modalità (full-game o rimanenti).

    MODALITÀ FULL GAME (Betfair Exchange, Pinnacle, la maggior parte degli exchange):
      Le linee live quotano il risultato COMPLETO a 90 minuti.
      Esempio: score 1-0 al 60' → AH live "Home -1.0" significa che la casa
      deve vincere di 1+ goal sull'intera partita (non solo nei 30' rimanenti).

    MODALITÀ GOL RIMANENTI (Asian market puri, alcuni operatori):
      Le linee live quotano solo i GOL RIMANENTI da ora al 90'.
      Esempio: score 1-0 al 60' → AH live "-0.0" potrebbe significare
      che entrambe le squadre hanno 50/50 di segnare di più nei 30' rimanenti.

    Il modello lavora internamente in "gol rimanenti". In modalità full-game
    la conversione avviene automaticamente:
      ah_rimanenti  = ah_full  + (gol_casa − gol_trasf)
      tot_rimanenti = tot_full − (gol_casa + gol_trasf)

    Returns:
        Dict con ah_op, tot_op, ah_cur, tot_cur (sempre in "gol rimanenti").
        Include anche "fullgame_mode" (bool) e i valori raw inseriti dall'utente.
        Include "validation_errors" con eventuali errori di validazione.
    """
    st.header("1. Linee Asiatiche (Spread / Total)")

    # Validazione input: variabile per raccogliere errori
    validation_errors: list[str] = []

    # Selettore modalità — impatta direttamente la semantica delle linee correnti
    fullgame_mode = st.radio(
        "Modalità linee live:",
        options=["Full Game (Betfair, Pinnacle, exchange standard)", "Gol Rimanenti (mercati asiatici puri)"],
        index=0,
        horizontal=True,
        help=(
            "**Full Game** (raccomandato per Betfair/Pinnacle): le linee live quotano "
            "il risultato finale a 90'. Esempio: Over 2.5 = 2.5 gol totali intera partita.\n\n"
            "**Gol Rimanenti**: le linee live quotano solo i gol da ora al fischio finale. "
            "Più raro, usato da alcuni operatori asiatici."
        ),
    ) == "Full Game (Betfair, Pinnacle, exchange standard)"

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Apertura — full 90'**")
        ah_op  = st.number_input("AH Apertura",     value=-0.25, step=0.25,
                                  key="lines_ah_op",
                                  help="Handicap asiatico alla quota di apertura (intera partita).")
        tot_op = st.number_input("Totale Apertura", value=2.50,  step=0.25,
                                  key="lines_tot_op",
                                  help="Over/Under alla quota di apertura (gol totali intera partita).")

    with col_a2:
        if fullgame_mode:
            st.markdown("**Corrente — full game live** *(opzionale, auto-converte in rimanenti)*")
            # Pre-inizializza solo al primo render; i render successivi usano session_state
            # così modificare l'apertura non resetta mai il corrente inserito dall'utente.
            if "ah_cur_raw_input" not in st.session_state:
                st.session_state["ah_cur_raw_input"] = ah_op
            if "tot_cur_raw_input" not in st.session_state:
                st.session_state["tot_cur_raw_input"] = tot_op
            ah_cur_raw  = st.number_input(
                "AH Corrente (full game)",
                step=0.25,
                key="ah_cur_raw_input",
                help=(
                    "AH live come quotato sull'exchange (full 90'). "
                    "Se non hai dati live, lascia uguale all'AH Apertura: il modello scala "
                    "automaticamente al tempo rimanente. "
                    "Aggiorna solo se il mercato si è mosso significativamente."
                ),
            )
            tot_cur_raw = st.number_input(
                "Totale Corrente (full game)",
                step=0.25,
                key="tot_cur_raw_input",
                help=(
                    "Linea Over/Under live come quotata sull'exchange (gol totali da inizio partita). "
                    "Se non hai dati live, lascia uguale al Totale Apertura: il modello scala "
                    "automaticamente al tempo rimanente. "
                    "Aggiorna solo se il mercato si è mosso significativamente."
                ),
            )
            gol_diff = gol_casa - gol_trasf
            gol_tot  = gol_casa + gol_trasf
            ah_cur   = ah_cur_raw  + gol_diff
            tot_cur  = max(BAYES.TOT_BAYES_MIN, tot_cur_raw - gol_tot)  # Fix #2.8: Usa TOT_BAYES_MIN dal config

            if gol_tot > 0:
                st.caption(
                    f"📐 Conversione automatica — AH rimanenti: **{ah_cur:+.2f}** "
                    f"(= {ah_cur_raw:+.2f} + {gol_diff:+d} gol di vantaggio)  "
                    f"· Total rimanenti: **{tot_cur:.2f}** (= {tot_cur_raw:.2f} − {gol_tot} gol segnati)"
                )
            if tot_cur_raw < gol_tot:
                st.error(
                    f"⛔ Impossibile: il Totale Corrente ({tot_cur_raw}) è inferiore ai gol già segnati "
                    f"({gol_tot}). Inserisci la linea full game corretta (es. >{gol_tot + 0.5:.1f})."
                )
        else:
            st.markdown("**Corrente — gol rimanenti** *(opzionale)*")
            if "ah_cur_raw_input" not in st.session_state:
                st.session_state["ah_cur_raw_input"] = ah_op
            if "tot_cur_raw_input" not in st.session_state:
                st.session_state["tot_cur_raw_input"] = tot_op
            ah_cur_raw = st.number_input(
                "AH Corrente (rimanenti)",
                step=0.25,
                key="ah_cur_raw_input",
                help=(
                    "Handicap asiatico live riferito ai soli gol rimanenti da ora al 90'. "
                    "Se non hai dati live, lascia uguale all'apertura."
                ),
            )
            tot_cur_raw = st.number_input(
                "Totale Corrente (rimanenti)",
                step=0.25,
                key="tot_cur_raw_input",
                help=(
                    "Gol attesi rimanenti da ora al 90' secondo il mercato. "
                    "Se non hai dati live, lascia uguale all'apertura."
                ),
            )
            ah_cur  = ah_cur_raw
            tot_cur = tot_cur_raw

    # ===================== VALIDAZIONE INPUT CRITICA =====================
    # Fix #3.1, #3.2: Usa parametri dal config invece di hardcoded
    # Verifica che tot_cur sia plausibile per il minuto corrente.
    # Questo previene risultati sbagliati quando l'utente cambia minuto/punteggio
    # ma dimentica di aggiornare le linee live.

    # FIX: Lista errori BLOCCANTI (non solo warning)
    blocking_errors: list[str] = []

    if minuto > 0:
        mins_rem = max(1, 90 - minuto)
        tot_cap = max(BAYES.TOT_BAYES_MIN, mins_rem / 90.0 * BAYES.TOT_TEMPORAL_MAX)

        if tot_cur > tot_cap * INPUT_VALIDATION.TOT_VALIDATION_MULTIPLIER:
            # ERRORE BLOCCANTE: tot_cur troppo alto per il minuto
            blocking_errors.append(
                f"LINEE NON AGGIORNATE: Totale rimanenti {tot_cur:.2f} è impossibile al minuto {minuto}' "
                f"(massimo realistico: {tot_cap:.2f}). Aggiorna le linee live dall'exchange!"
            )
            # Mostra suggerimento
            st.error(
                f"⛔ **BLOCCANTE**: Linee non aggiornate!\n\n"
                f"Hai inserito **{tot_cur:.2f} gol rimanenti** al minuto **{minuto}'**, "
                f"ma il massimo realistico è **~{tot_cap:.2f} gol**.\n\n"
                f"**Devi aggiornare le linee live dall'exchange!**\n\n"
                f"Suggerimento: al {minuto}' rimangono ~{mins_rem} minuti, "
                f"cerca linee Total intorno a **{tot_cap:.1f}** gol rimanenti."
            )
            validation_errors.append(f"tot_cur={tot_cur:.2f} > tot_cap={tot_cap:.2f} al minuto {minuto}")

        elif tot_cur > tot_cap * INPUT_VALIDATION.TOT_VALIDATION_WARNING:
            # WARNING: tot_cur sospettosamente alto
            st.warning(
                f"⚠️ **Attenzione**: Total rimanenti ({tot_cur:.2f}) sembra alto per il minuto {minuto}'. "
                f"Massimo realistico: {tot_cap:.2f} gol. Verifica le linee live."
            )

    # Fix #3.2: Validazione AH con buffer dal config
    if abs(ah_cur) > tot_cur + INPUT_VALIDATION.AH_VALIDATION_BUFFER:
        blocking_errors.append(
            f"AH impossibile: |{ah_cur:.2f}| > Total {tot_cur:.2f}"
        )
        st.error(
            f"⛔ **AH IMPOSSIBILE!**\n\n"
            f"Hai inserito AH = **{ah_cur:+.2f}** con Total = **{tot_cur:.2f}**.\n\n"
            f"L'handicap non può superare il totale dei gol attesi. "
            f"Verifica i valori inseriti."
        )
        validation_errors.append(f"|ah_cur|={abs(ah_cur):.2f} > tot_cur={tot_cur:.2f}")

    # FIX: Validazione AH vs punteggio attuale
    # Se il punteggio è cambiato ma AH non si è mosso, probabile errore utente
    if (minuto >= INPUT_VALIDATION.AH_SCORE_MOVE_MINUTE
            and gol_casa + gol_trasf > 0 and fullgame_mode):
        # In modalità full-game, se AH è uguale all'apertura ma ci sono gol,
        # l'utente probabilmente non ha aggiornato
        ah_expected_move = abs(ah_cur_raw - ah_op)
        if ah_expected_move < INPUT_VALIDATION.AH_SCORE_MOVE_THRESHOLD:
            st.warning(
                f"⚠️ **Possibile AH non aggiornato**: Ci sono gol ({gol_casa}-{gol_trasf}) ma "
                f"AH corrente ({ah_cur_raw:+.2f}) è ancora uguale all'apertura ({ah_op:+.2f}). "
                f"Verifica le linee live."
            )

    return {
        "ah_op":        ah_op,
        "tot_op":       tot_op,
        "ah_cur":       ah_cur,
        "tot_cur":      tot_cur,
        "fullgame_mode": fullgame_mode,
        "tot_cur_raw":  tot_cur_raw,   # valore grezzo pre-conversione (per OU default)
        "validation_errors": validation_errors,  # errori di validazione (vuoto = OK)
        "blocking_errors": blocking_errors,      # FIX: errori bloccanti
    }


def render_ou_selector(
    tot_cur_raw: float,
    gol_attuali: int,
    fullgame_mode: bool,
) -> float:
    """
    Render del selettore linea Over/Under.

    La linea analizzata è sempre una linea FULL GAME (gol totali partita).
    Il default viene calcolato come:
      - Full game mode:   closest a tot_cur_raw          (già full game)
      - Remaining mode:   closest a gol_attuali + tot_cur  (remaining → full game)

    Args:
        tot_cur_raw: Valore inserito dall'utente (full game o rimanenti a seconda della modalità).
        gol_attuali: Gol totali già segnati.
        fullgame_mode: True = modalità full game, False = modalità gol rimanenti.

    Returns:
        Linea Over/Under selezionata (sempre full game).
    """
    linee = list(UI.LINEE_OU)

    target = tot_cur_raw if fullgame_mode else gol_attuali + tot_cur_raw

    idx_default = min(range(len(linee)), key=lambda i: abs(linee[i] - target))
    return st.selectbox(
        "Linea U/O da analizzare (full game):",
        linee,
        index=idx_default,
        help=(
            "Linea full game (include i gol già segnati). "
            "Selezionata automaticamente come la più vicina al totale atteso. "
            "Le linee X.75/X.25 sono Asian quarter lines (mezzo stake su ogni semi-linea)."
        ),
    )


def render_bankroll() -> tuple[float, float, float]:
    """
    Render dei widget bankroll e commissione.

    Returns:
        (bankroll, comm_pct, comm_rate)
    """
    col_bk, col_cm = st.columns(2)
    with col_bk:
        bankroll = st.number_input(
            "Bankroll (€)", value=UI.BANKROLL_DEFAULT, step=UI.BANKROLL_STEP, min_value=1.0,
            key="bankroll_value",
        )
    with col_cm:
        comm_pct = st.number_input(
            "Commissione exchange (%)",
            value=UI.COMM_DEFAULT,
            min_value=0.0,
            max_value=UI.COMM_MAX,
            step=UI.COMM_STEP,
            key="comm_pct_value",
            help="Betfair Standard: 2-5%. Usata per calcolare l'edge netto.",
        )
    return bankroll, comm_pct, comm_pct / 100.0


def render_shots(minuto: int) -> tuple[int, int, int, int]:
    """
    Render dei widget per i tiri live con validazione.

    DEPRECATO: Mantenuto per compatibilità. Usare render_match_state_live().

    Args:
        minuto: Minuto attuale (per validazione).

    Returns:
        (sot_h, soff_h, sot_a, soff_a)
    """
    return 0, 0, 0, 0


def render_exchange_quotes(linea_ou: float) -> ExchangeQuotes:
    """
    Render del pannello quote exchange (opzionale, in expander).

    Args:
        linea_ou: Linea Over/Under selezionata (per le label).

    Returns:
        ExchangeQuotes con le quote inserite.
    """
    with st.expander("⚙️ Analisi avanzata con quote exchange (opzionale)"):
        st.caption(
            "Inserisci le quote che vedi sull'exchange per ottenere edge preciso, "
            "stake Kelly in € e EV atteso. Lascia a 0 i mercati che non ti interessano."
        )
        eq1, eqx, eq2 = st.columns(3)
        q_1 = eq1.number_input("Quota 1 (Casa)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_x = eqx.number_input("Quota X (Pareggio)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_2 = eq2.number_input("Quota 2 (Trasf.)", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        equ, eqo, eqb = st.columns(3)
        q_under = equ.number_input(f"Quota Under {linea_ou}", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_over = eqo.number_input(f"Quota Over {linea_ou}", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        q_btts_si = eqb.number_input("Quota BTTS Sì", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        eqbn_col, _, _ = st.columns(3)
        q_btts_no = eqbn_col.number_input("Quota BTTS No", min_value=0.0, value=0.0, step=0.05, format="%.2f")

    return ExchangeQuotes(
        q_1=q_1, q_x=q_x, q_2=q_2,
        q_over=q_over, q_under=q_under,
        q_btts_si=q_btts_si, q_btts_no=q_btts_no,
    )


def build_match_state(
    match: dict,
    lines: dict,
    linea_ou: float,
    bankroll: float,
    comm_rate: float,
    shots: tuple[int, int, int, int] | None = None,
    ocr_imp_total: float = 0.0,
    ocr_quota_1: float = 0.0,
    ocr_quota_x: float = 0.0,
    ocr_quota_2: float = 0.0,
    ocr_quota_over: float = 0.0,
    ocr_quota_under: float = 0.0,
    ocr_quota_gg: float = 0.0,
    ocr_quota_ng: float = 0.0,
    fixture_historical_total: float = 0.0,
    movement_quality: float = 1.0,
    ocr_confidence_scale: float = 1.0,
    absence_mult_h: float = 1.0,
    absence_mult_a: float = 1.0,
    forma_mult_h: float = 1.0,
    forma_mult_a: float = 1.0,
    mkt_init_1: float = 0.0,
    mkt_init_x: float = 0.0,
    mkt_init_2: float = 0.0,
    h2h_home_win_pct: float = 0.0,
    h2h_draw_pct: float = 0.0,
    h2h_away_win_pct: float = 0.0,
    h2h_over_pct: float = 0.0,
    strength_home: int = 0,
    strength_away: int = 0,
    prev_avg_scored_h: float = 0.0,
    prev_avg_conceded_h: float = 0.0,
    prev_avg_scored_a: float = 0.0,
    prev_avg_conceded_a: float = 0.0,
    # === FORM ANALYSIS (da standings Nowgoal) ===
    standings_rank_h: int = 0,
    standings_rank_a: int = 0,
    standings_points_h: int = 0,
    standings_points_a: int = 0,
    standings_played_h: int = 0,
    standings_played_a: int = 0,
    standings_total_teams: int = 20,
    last6_points_h: float = 0.0,
    last6_points_a: float = 0.0,
    home_ppg_h: float = 0.0,
    away_ppg_a: float = 0.0,
    home_gf_h: float = 0.0,
    home_ga_h: float = 0.0,
    away_gf_a: float = 0.0,
    away_ga_a: float = 0.0,
    weather_xg_impact: float = 0.0,
    h2h_btts_pct: float = 0.0,
    scoring_streak_h: int = 0,
    scoring_streak_a: int = 0,
    clean_sheet_streak_h: int = 0,
    clean_sheet_streak_a: int = 0,
    h2h_matches_count: int = 0,
    late_goals_pct_h: float = 0.0,
    late_goals_pct_a: float = 0.0,
) -> MatchState:
    """
    Costruisce il MatchState validato dai valori dei widget.

    Args:
        match: Dict con minuto, gol, rossi e (opzionale) tiri, corner, possesso, attacchi.
        lines: Dict con linee asiatiche.
        linea_ou: Linea Over/Under.
        bankroll: Capitale.
        comm_rate: Commissione.
        shots: (sot_h, soff_h, sot_a, soff_a) — legacy, se None usa match dict.

    Returns:
        MatchState validato.
    """
    if shots is not None:
        sot_h, soff_h, sot_a, soff_a = shots
    else:
        sot_h = match.get("sot_h", 0)
        soff_h = match.get("soff_h", 0)
        sot_a = match.get("sot_a", 0)
        soff_a = match.get("soff_a", 0)

    return MatchState(
        minuto=match["minuto"],
        gol_casa=match["gol_casa"],
        gol_trasf=match["gol_trasf"],
        rossi_casa=match["rossi_casa"],
        rossi_trasf=match["rossi_trasf"],
        ah_op=lines["ah_op"],
        tot_op=lines["tot_op"],
        ah_cur=lines["ah_cur"],
        tot_cur=lines["tot_cur"],
        linea_ou=linea_ou,
        sot_h=sot_h, soff_h=soff_h,
        sot_a=sot_a, soff_a=soff_a,
        corner_h=match.get("corner_h", 0),
        corner_a=match.get("corner_a", 0),
        possesso_h=match.get("possesso_h", 0.0),
        possesso_a=match.get("possesso_a", 0.0),
        att_pericolosi_h=match.get("att_pericolosi_h", 0),
        att_pericolosi_a=match.get("att_pericolosi_a", 0),
        gialli_casa=match.get("gialli_casa", 0),
        gialli_trasf=match.get("gialli_trasf", 0),
        blk_h=match.get("blk_h", 0),
        blk_a=match.get("blk_a", 0),
        att_h=match.get("att_h", 0),
        att_a=match.get("att_a", 0),
        falli_casa=match.get("falli_casa", 0),
        falli_trasf=match.get("falli_trasf", 0),
        ocr_imp_total=ocr_imp_total,
        ocr_quota_1=ocr_quota_1,
        ocr_quota_x=ocr_quota_x,
        ocr_quota_2=ocr_quota_2,
        ocr_quota_over=ocr_quota_over,
        ocr_quota_under=ocr_quota_under,
        ocr_quota_gg=ocr_quota_gg,
        ocr_quota_ng=ocr_quota_ng,
        fixture_historical_total=fixture_historical_total,
        movement_quality=movement_quality,
        ocr_confidence_scale=ocr_confidence_scale,
        absence_mult_h=absence_mult_h,
        absence_mult_a=absence_mult_a,
        forma_mult_h=forma_mult_h,
        forma_mult_a=forma_mult_a,
        mkt_init_1=mkt_init_1,
        mkt_init_x=mkt_init_x,
        mkt_init_2=mkt_init_2,
        h2h_home_win_pct=h2h_home_win_pct,
        h2h_draw_pct=h2h_draw_pct,
        h2h_away_win_pct=h2h_away_win_pct,
        h2h_over_pct=h2h_over_pct,
        strength_home=strength_home,
        strength_away=strength_away,
        prev_avg_scored_h=prev_avg_scored_h,
        prev_avg_conceded_h=prev_avg_conceded_h,
        prev_avg_scored_a=prev_avg_scored_a,
        prev_avg_conceded_a=prev_avg_conceded_a,
        # Form Analysis
        standings_rank_h=standings_rank_h,
        standings_rank_a=standings_rank_a,
        standings_points_h=standings_points_h,
        standings_points_a=standings_points_a,
        standings_played_h=standings_played_h,
        standings_played_a=standings_played_a,
        standings_total_teams=standings_total_teams,
        last6_points_h=last6_points_h,
        last6_points_a=last6_points_a,
        home_ppg_h=home_ppg_h,
        away_ppg_a=away_ppg_a,
        home_gf_h=home_gf_h,
        home_ga_h=home_ga_h,
        away_gf_a=away_gf_a,
        away_ga_a=away_ga_a,
        weather_xg_impact=weather_xg_impact,
        h2h_btts_pct=h2h_btts_pct,
        scoring_streak_h=scoring_streak_h,
        scoring_streak_a=scoring_streak_a,
        clean_sheet_streak_h=clean_sheet_streak_h,
        clean_sheet_streak_a=clean_sheet_streak_a,
        h2h_matches_count=h2h_matches_count,
        late_goals_pct_h=late_goals_pct_h,
        late_goals_pct_a=late_goals_pct_a,
        bankroll=bankroll,
        comm_rate=comm_rate,
    )


# ---------------------------------------------------------------------------
# Nuovi renderer semplificati
# ---------------------------------------------------------------------------

def render_prematch_analysis_screen() -> PrematchAnalysisExtracted | None:
    """
    Sezione Analysis di Nowgoal (pre-partita).

    Offre due modalità di input:
    - Tab "URL": incolla il link Nowgoal → estrazione automatica via Jina Reader
    - Tab "Screenshot": carica 1-2 immagini → estrazione via Gemini Vision

    Restituisce PrematchAnalysisExtracted se l'estrazione ha avuto successo,
    None se l'utente non ha inserito nulla.
    I dati estratti vengono salvati in session_state["prematch_analysis"].
    """
    st.caption("📊 **Analysis** di Nowgoal · Inserisci prima del fischio (opzionale)")

    cached: PrematchAnalysisExtracted | None = st.session_state.get("prematch_analysis")

    # Se c'è già un'estrazione valida, mostra subito il riepilogo senza ridisegnare i tab
    if cached and cached.extraction_success:
        # Mostra se la cache ha il meteo
        if cached.weather_condition:
            st.success(f"📍 Dati da cache — Meteo: {cached.weather_condition}, {cached.weather_temp}°C")
        else:
            st.info("📍 Dati da cache (senza meteo) — Clicca 'Rimuovi analisi' per estrarre di nuovo con OpenWeather")
        _model = st.session_state.get("prematch_last_model_1x2")
        _render_prematch_analysis_summary(cached, model_probs=_model if isinstance(_model, dict) else None)
        return cached

    tab_url, tab_screen = st.tabs(["🔗 URL Nowgoal", "📷 Screenshot"])

    # ── TAB 1: URL ──────────────────────────────────────────────────────────
    with tab_url:
        st.caption(
            "Apri Nowgoal → vai sulla partita → tab **H2H** → copia l'URL dalla barra del browser"
        )
        url_input = st.text_input(
            "URL pagina H2H",
            placeholder="https://www.nowgoal.com/match/h2h-XXXXXX",
            key="prematch_analysis_url_input",
            label_visibility="collapsed",
        )
        if st.button("Estrai da URL", key="_extract_url_btn", type="primary"):
            if url_input.strip():
                with st.spinner("Lettura pagina e analisi..."):
                    # Un solo giro: `extract_prematch_analysis_from_url` già prova mirror live5
                    # e merge da H2H+HTML; un secondo estratto raddoppiava tempi senza guadagni certi.
                    result = extract_prematch_analysis_from_url(url_input.strip())

                st.session_state["_prematch_analysis_url"] = url_input.strip()
                # Pulisce cache file per evitare conflitti
                st.session_state.pop("_prematch_analysis_file_id", None)
                if result.extraction_success:
                    st.session_state["prematch_analysis"] = result
                    cached = result
                    st.rerun()
                else:
                    st.error(f"Estrazione fallita: {result.error_message}")
            else:
                st.warning("Inserisci l'URL Nowgoal prima di procedere.")

    # ── TAB 2: SCREENSHOT ───────────────────────────────────────────────────
    with tab_screen:
        uploaded = st.file_uploader(
            "Screen Analisi pre-partita",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="prematch_analysis_upload",
            label_visibility="collapsed",
            help="Carica 1 o 2 screenshot della tab 'Analysis' di Nowgoal. "
                 "Gemini estrae H2H, classifica e forma delle squadre.",
        )

        if uploaded:
            file_id = "_".join(f"{f.name}_{f.size}" for f in uploaded)
            last_id = st.session_state.get("_prematch_analysis_file_id", "")

            if file_id != last_id:
                images = [(f.read(), "." + f.name.rsplit(".", 1)[-1].lower()) for f in uploaded]
                with st.spinner("Gemini analizza lo screen..."):
                    result = extract_prematch_analysis_from_bytes(images)

                st.session_state["_prematch_analysis_file_id"] = file_id
                # Pulisce cache URL per evitare conflitti
                st.session_state.pop("_prematch_analysis_url", None)
                if result.extraction_success:
                    st.session_state["prematch_analysis"] = result
                    cached = result
                    st.rerun()
                else:
                    st.error(f"Estrazione fallita: {result.error_message}")

    return None


def _implied_probs_1x2(o1: float, ox: float, o2: float) -> tuple[float, float, float] | None:
    """Probabilità implicite normalizzate (rimuove overround)."""
    if o1 <= 1.0 or ox <= 1.0 or o2 <= 1.0:
        return None
    i1, ix, i2 = 1.0 / o1, 1.0 / ox, 1.0 / o2
    s = i1 + ix + i2
    if s <= 0:
        return None
    return i1 / s, ix / s, i2 / s


def _render_prematch_market_synthesis(data: PrematchAnalysisExtracted) -> None:
    """Aperture vs attuali: 1X2, Asian, Total, eventuale riga Live — in parole povere."""
    st.markdown("##### Mercato (da estrazione URL/Live)")
    rows_mkt: list[str] = []

    imp = _implied_probs_1x2(data.mkt_init_1, data.mkt_init_x, data.mkt_init_2)
    if imp:
        p1, px, p2 = imp
        rows_mkt.append(
            f"**1X2 implicito** (da quote *iniziali* Nowgoal): "
            f"Casa **{p1 * 100:.1f}%** · X **{px * 100:.1f}%** · Trasf. **{p2 * 100:.1f}%** "
            f"— quote {data.mkt_init_1:.2f} / {data.mkt_init_x:.2f} / {data.mkt_init_2:.2f}"
        )
    elif data.mkt_init_1 > 0:
        rows_mkt.append(
            f"**1X2** (quote iniziali): {data.mkt_init_1:.2f} / {data.mkt_init_x:.2f} / {data.mkt_init_2:.2f}"
        )

    ah_o = float(data.ah_line_open or 0.0)
    ah_c = float(data.ah_line_close or 0.0)
    if ah_o != 0.0 or ah_c != 0.0:
        d_ah = float(data.line_movement_ah or 0.0)
        mov = f" · movimento linea **{d_ah:+.2f}**" if (ah_o != 0 and ah_c != 0) else ""
        extra = ""
        if data.ah_home_odds_open > 0 or data.ah_away_odds_open > 0:
            extra = (
                f" · quote AH apertura casa/tr. **{data.ah_home_odds_open:.2f}** / "
                f"**{data.ah_away_odds_open:.2f}**"
            )
        rows_mkt.append(
            f"**Asian (consensus tabella)**: apertura **{ah_o:+.2f}** → attuale **{ah_c:+.2f}**{mov}{extra}"
        )

    to_o = float(data.total_line_open or 0.0)
    to_c = float(data.total_line_close or 0.0)
    if to_o != 0.0 or to_c != 0.0:
        d_tot = float(data.line_movement_total or 0.0)
        mov = f" · movimento **{d_tot:+.2f}**" if (to_o != 0 and to_c != 0) else ""
        ou = ""
        if data.total_over_odds_open > 0 or data.total_under_odds_open > 0:
            ou = f" · O/U apertura **{data.total_over_odds_open:.2f}** / **{data.total_under_odds_open:.2f}**"
        rows_mkt.append(
            f"**Over/Under (consensus)**: apertura **{to_o:.2f}** → attuale **{to_c:.2f}**{mov}{ou}"
        )

    liv_ah = float(data.live_ah_line or 0.0)
    liv_tot = float(data.live_total_line or 0.0)
    if liv_ah != 0.0 or liv_tot != 0.0:
        la = (
            f"AH **{liv_ah:+.2f}** @ {data.live_ah_home_odds:.2f} / {data.live_ah_away_odds:.2f}"
            if liv_ah != 0.0
            else ""
        )
        lt = (
            f" · Total **{liv_tot:.2f}** @ O {data.live_over_odds:.2f} / U {data.live_under_odds:.2f}"
            if liv_tot != 0.0
            else ""
        )
        rows_mkt.append(f"**Da pagina Live (Jina)** — {la}{lt}")

    sig = float(getattr(data, "odds_sharp_signal", 0.0) or 0.0)
    if sig > 0:
        rows_mkt.append(f"Segnale movimento quote (interno): **{sig:.3f}**")

    if rows_mkt:
        for line in rows_mkt:
            st.markdown(line)
    else:
        st.caption(
            "Nessuna linea AH/Total o quote 1X2 estratta da questa pagina. "
            "Compila le linee nel pannello principale o riprova l'URL."
        )


def _render_model_vs_market_1x2(
    data: PrematchAnalysisExtracted,
    model_probs: dict[str, Any] | None,
) -> None:
    """Confronto ultimo ANALIZZA prematch vs implicito del mercato."""
    if not model_probs:
        st.caption(
            "Dopo **ANALIZZA** (minuto 0) qui comparirà il confronto **modello ↔ mercato** sul 1X2."
        )
        return
    try:
        mp1 = float(model_probs.get("p1", 0.0))
        mpx = float(model_probs.get("px", 0.0))
        mp2 = float(model_probs.get("p2", 0.0))
    except (TypeError, ValueError):
        return
    if mp1 <= 0 and mpx <= 0 and mp2 <= 0:
        return

    imp = _implied_probs_1x2(data.mkt_init_1, data.mkt_init_x, data.mkt_init_2)
    if not imp:
        st.caption("Modello: Casa / X / Trasf. calcolati; mercato 1X2 non disponibile per il confronto.")
        return

    m1, mx, m2 = imp
    st.markdown("##### Modello vs mercato (1X2)")
    h = data.home_team or "Casa"
    a = data.away_team or "Trasferta"
    d1 = (mp1 - m1) * 100.0
    dx = (mpx - mx) * 100.0
    d2 = (mp2 - m2) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric(h, f"{mp1 * 100:.1f}%", f"{d1:+.1f} pp vs mercato", help="Differenza in punti percentuali vs implicito delle quote iniziali")
    c2.metric("X", f"{mpx * 100:.1f}%", f"{dx:+.1f} pp vs mercato")
    c3.metric(a, f"{mp2 * 100:.1f}%", f"{d2:+.1f} pp vs mercato")

    st.caption(
        "Mercato = probabilità implicite dalle **quote 1X2 iniziali** Nowgoal (normalizzate). "
        "Modello = ultimo **ANALIZZA** a minuto 0."
    )


def _render_prematch_analysis_summary(
    data: PrematchAnalysisExtracted,
    model_probs: dict[str, Any] | None = None,
) -> None:
    """Mostra un riepilogo compatto dei dati estratti dallo screen Analysis."""
    with st.expander("✅ Dati Analisi estratti", expanded=True):
        # Nomi squadre
        if data.home_team or data.away_team:
            st.markdown(f"**{data.home_team}** vs **{data.away_team}**")
            if data.league_name:
                st.caption(f"📍 {data.league_name}")
                src = getattr(data, "league_source", "unknown")
                if src and src != "unknown":
                    st.caption(f"Sorgente lega: `{src}`")

        _render_prematch_market_synthesis(data)
        _render_model_vs_market_1x2(data, model_probs)

        # H2H data — mostra messaggio chiaro se non disponibile
        h2h_available = data.h2h_home_win_pct > 0 or data.h2h_draw_pct > 0 or data.h2h_away_win_pct > 0
        if h2h_available:
            c1, c2, c3 = st.columns(3)
            c1.metric("H2H Casa", f"{data.h2h_home_win_pct:.0f}%")
            c2.metric("H2H X", f"{data.h2h_draw_pct:.0f}%")
            c3.metric("H2H Trasf.", f"{data.h2h_away_win_pct:.0f}%")

            if data.h2h_avg_goals_home > 0 or data.h2h_avg_goals_away > 0:
                tot = data.h2h_avg_goals_home + data.h2h_avg_goals_away
                st.caption(f"Media gol H2H: {data.h2h_avg_goals_home:.1f} + {data.h2h_avg_goals_away:.1f} = **{tot:.1f}** per partita")
        else:
            st.info("📊 **Nessun dato H2H disponibile** — Le squadre non si sono mai incontrate o dati non trovati.")

        if data.home_rank > 0 or data.away_rank > 0:
            r1, r2 = st.columns(2)
            r1.caption(
                f"**Casa** — {data.home_rank}° · {data.home_win_rate:.1f}% win rate · "
                f"Last 6: {data.home_last6_win}W {data.home_last6_draw}D {data.home_last6_lose}L"
            )
            r2.caption(
                f"**Trasf.** — {data.away_rank}° · {data.away_win_rate:.1f}% win rate · "
                f"Last 6: {data.away_last6_win}W {data.away_last6_draw}D {data.away_last6_lose}L"
            )

        fm1, fm2 = st.columns(2)
        fm1.metric("Forma mult. Casa", f"{data.forma_mult_h:.3f}")
        fm2.metric("Forma mult. Trasf.", f"{data.forma_mult_a:.3f}")

        nh = len(data.home_absences_players or [])
        na = len(data.away_absences_players or [])
        if nh or na or data.home_absences_count or data.away_absences_count:
            st.caption(
                f"Infortuni/assenze estratti — **Casa**: {nh or data.home_absences_count} · "
                f"**Trasf.**: {na or data.away_absences_count}"
            )
            if nh or na:
                with st.expander("Dettaglio assenze (per il moltiplicatore xG)", expanded=False):
                    if data.home_absences_players:
                        st.markdown("**Casa**")
                        for line in data.home_absences_players:
                            st.markdown(f"- `{line}`")
                    if data.away_absences_players:
                        st.markdown("**Trasferta**")
                        for line in data.away_absences_players:
                            st.markdown(f"- `{line}`")

        if data.extraction_coverage > 0:
            st.caption(f"Qualita' estrazione URL: **{data.extraction_coverage * 100:.0f}%**")
            if data.extraction_coverage < 0.55:
                st.warning("Qualita' bassa: il motore usera' solo una parte dei dati prematch estratti.")
        if data.extraction_notes:
            st.caption("Note parser: " + ", ".join(data.extraction_notes))
        if data.extraction_section_scores:
            labels = {
                "identity": "Identita'",
                "league": "Lega",
                "h2h_core": "H2H",
                "standings": "Classifica",
                "previous_scores": "Previous",
                "market_1x2": "Quote 1X2",
                "team_stats": "Team Stats",
                "weather": "Meteo",
                "injuries": "Infortuni",
            }
            parts = []
            for key, score in data.extraction_section_scores.items():
                icon = "🟢" if score >= 1.0 else "🟡"
                parts.append(f"{icon} {labels.get(key, key)}")
            st.caption("Sezioni estratte: " + " · ".join(parts))
        with st.expander("Mappa campi usati dal motore", expanded=False):
            st.caption(
                "Usati nel calcolo: H2H 1X2/Over, standings, previous scores, team stats goal/loss, "
                "strength, quote iniziali 1X2, meteo/quality."
            )
            st.caption(
                "Solo informativi: HT/FT completo, quote live, parte delle metriche di tabella non "
                "ancora collegate direttamente al modello."
            )

        # HT H2H
        if data.h2h_ht_home_win_pct > 0 or data.h2h_ht_draw_pct > 0:
            st.caption(
                f"H2H all'intervallo: Casa {data.h2h_ht_home_win_pct:.0f}% · "
                f"X {data.h2h_ht_draw_pct:.0f}% · Trasf. {data.h2h_ht_away_win_pct:.0f}%"
            )

        # Goal timing
        if data.home_goals_1h > 0 or data.away_goals_1h > 0:
            st.caption(
                f"Gol per partita — Casa: {data.home_goals_1h/max(1,data.home_matches):.2f} (1T) "
                f"+ {data.home_goals_2h/max(1,data.home_matches):.2f} (2T) · "
                f"Trasf.: {data.away_goals_1h/max(1,data.away_matches):.2f} (1T) "
                f"+ {data.away_goals_2h/max(1,data.away_matches):.2f} (2T)"
            )

        # === METEO (estratto dalla pagina LIVE) ===
        if data.weather_condition:
            weather_icon = "☀️" if "sun" in data.weather_condition.lower() or "clear" in data.weather_condition.lower() else \
                           "🌧️" if "rain" in data.weather_condition.lower() else \
                           "⛈️" if "thunder" in data.weather_condition.lower() else \
                           "🌫️" if "fog" in data.weather_condition.lower() or "mist" in data.weather_condition.lower() else \
                           "💨" if "wind" in data.weather_condition.lower() else \
                           "⛅" if "cloud" in data.weather_condition.lower() else "🌡️"
            
            impact_str = ""
            if data.weather_impact != 0:
                impact_str = f" → **xG {'↓' if data.weather_impact < 0 else '↑'} {abs(data.weather_impact)*100:.0f}%**"
            
            st.info(f"{weather_icon} **Meteo:** {data.weather_condition}, {data.weather_temp}°C{impact_str}")

        # === HT/FT STATISTICS (estratto dalla pagina LIVE) ===
        has_htft = any([
            data.htft_home_htw_ftw, data.htft_home_htd_ftw, data.htft_home_htl_ftw,
            data.htft_away_htw_ftw, data.htft_away_htd_ftw, data.htft_away_htl_ftw,
        ])
        if has_htft:
            st.markdown("**📊 HT/FT Statistics** (da pagina LIVE)")
            
            # Tabella HT/FT per casa
            htft_home = [
                ["HT-W → FT-W", data.htft_home_htw_ftw, data.htft_home_htd_ftw, data.htft_home_htl_ftw],
                ["HT-D → FT-W", 0, 0, 0],  # Placeholder
                ["HT-L → FT-W", data.htft_home_htl_ftw, 0, 0],
            ]
            
            # Calcola righe HT-D e HT-L per casa
            htft_home[1] = ["HT-D → FT-W", data.htft_home_htw_ftd, data.htft_home_htd_ftd, data.htft_home_htl_ftd]
            
            # Calcola totale casa per percentuali
            total_home_htw = data.htft_home_htw_ftw + data.htft_home_htd_ftw + data.htft_home_htl_ftw
            total_home_htd = data.htft_home_htw_ftd + data.htft_home_htd_ftd + data.htft_home_htl_ftd
            total_home_htl = data.htft_home_htw_ftl + data.htft_home_htd_ftl + data.htft_home_htl_ftl
            
            col_ht, col_ft = st.columns(2)
            with col_ht:
                st.caption("**Casa - Risultato FT dopo HT:**")
                st.caption(f"HT in vantaggio → FT-W: **{data.htft_home_htw_ftw}** | FT-D: {data.htft_home_htw_ftd} | FT-L: {data.htft_home_htw_ftl}")
                st.caption(f"HT pareggio → FT-W: {data.htft_home_htd_ftw} | FT-D: **{data.htft_home_htd_ftd}** | FT-L: {data.htft_home_htd_ftl}")
                st.caption(f"HT in svantaggio → FT-W: {data.htft_home_htl_ftw} | FT-D: {data.htft_home_htl_ftd} | FT-L: **{data.htft_home_htl_ftl}**")
            
            with col_ft:
                st.caption("**Trasferta - Risultato FT dopo HT:**")
                st.caption(f"HT in vantaggio → FT-W: **{data.htft_away_htw_ftw}** | FT-D: {data.htft_away_htw_ftd} | FT-L: {data.htft_away_htw_ftl}")
                st.caption(f"HT pareggio → FT-W: {data.htft_away_htd_ftw} | FT-D: **{data.htft_away_htd_ftd}** | FT-L: {data.htft_away_htd_ftl}")
                st.caption(f"HT in svantaggio → FT-W: {data.htft_away_htl_ftw} | FT-D: {data.htft_away_htl_ftd} | FT-L: **{data.htft_away_htl_ftl}**")

        # === TEAM STATISTICS (ultimi 10 match, da pagina LIVE) ===
        has_team_stats = data.team_stats_home_goals > 0 or data.team_stats_away_goals > 0
        if has_team_stats:
            st.markdown("**📈 Team Statistics** (ultimi 10 match)")
            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                st.caption(f"**Casa:** {data.team_stats_home_goals:.1f} gol/partita, {data.team_stats_home_conceded:.1f} subiti")
                if data.team_stats_home_possession > 0:
                    st.caption(f"Possesso: {data.team_stats_home_possession:.1f}% | Corner: {data.team_stats_home_corners:.1f}")
            with ts_col2:
                st.caption(f"**Trasferta:** {data.team_stats_away_goals:.1f} gol/partita, {data.team_stats_away_conceded:.1f} subiti")
                if data.team_stats_away_possession > 0:
                    st.caption(f"Possesso: {data.team_stats_away_possession:.1f}% | Corner: {data.team_stats_away_corners:.1f}")

        if st.button("🗑 Rimuovi analisi", key="_remove_prematch_analysis"):
            st.session_state.pop("prematch_analysis", None)
            st.session_state.pop("_prematch_analysis_file_id", None)
            st.rerun()


def render_linee_semplici(gol_casa: int = 0, gol_trasf: int = 0) -> dict:
    """
    Render semplificato delle linee: 4 campi in 2 colonne.
    Sempre modalità Full Game — nessun radio button.

    Returns:
        Dict compatibile con render_asian_lines (ah_op, tot_op, ah_cur, tot_cur, ...).
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spread (AH)**")
        ah_op = st.number_input(
            "Apertura", value=-0.25, step=0.25, key="lines_ah_op",
            help="Handicap asiatico all'apertura del mercato.",
        )
        if "ah_cur_raw_input" not in st.session_state:
            st.session_state["ah_cur_raw_input"] = ah_op
        ah_cur_raw = st.number_input(
            "Chiusura / Live", step=0.25, key="ah_cur_raw_input",
            help="AH corrente sull'exchange (full game). Uguale all'apertura se non aggiornato.",
        )

    with col2:
        st.markdown("**Total (O/U)**")
        tot_op = st.number_input(
            "Apertura", value=2.50, step=0.25, key="lines_tot_op",
            help="Linea Over/Under all'apertura (gol totali intera partita).",
        )
        if "tot_cur_raw_input" not in st.session_state:
            st.session_state["tot_cur_raw_input"] = tot_op
        tot_cur_raw = st.number_input(
            "Chiusura / Live", step=0.25, key="tot_cur_raw_input",
            help="Total corrente sull'exchange (full game). Aggiorna se il mercato si è mosso.",
        )

    # Conversione Full Game → gol rimanenti (automatica)
    gol_diff = gol_casa - gol_trasf
    gol_tot  = gol_casa + gol_trasf
    ah_cur   = ah_cur_raw + gol_diff
    tot_cur  = max(BAYES.TOT_BAYES_MIN, tot_cur_raw - gol_tot)
    validation_errors: list[str] = []
    blocking_errors: list[str] = []

    if gol_tot > 0:
        st.caption(
            f"Rimanenti calcolati automaticamente — "
            f"AH: **{ah_cur:+.2f}** · Total: **{tot_cur:.2f}**"
        )

    if tot_cur_raw < gol_tot:
        blocking_errors.append("Total corrente inferiore ai gol gia' segnati.")
    if abs(ah_cur) > tot_cur + 0.25:
        blocking_errors.append("AH rimanente incoerente rispetto al Total rimanente.")
    if abs(ah_cur_raw - ah_op) < 0.01 and abs(tot_cur_raw - tot_op) < 0.01 and gol_tot > 0:
        validation_errors.append("Linee correnti uguali all'apertura con gol gia' segnati.")

    for msg in blocking_errors:
        st.error(f"⛔ {msg}")
    for msg in validation_errors:
        st.warning(f"⚠️ {msg}")

    return {
        "ah_op": ah_op,
        "tot_op": tot_op,
        "ah_cur": ah_cur,
        "tot_cur": tot_cur,
        "tot_cur_raw": tot_cur_raw,
        "fullgame_mode": True,
        "validation_errors": validation_errors,
        "blocking_errors": blocking_errors,
    }


def render_live_semplice() -> dict:
    """
    Render compatto per i dati live.

    Minuto + gol: sempre manuali.
    Tiri/corner/possesso/attacchi: letti dallo screenshot (Gemini).
    Nessun st.rerun() → l'expander rimane aperto dopo il caricamento.

    Returns:
        Dict con tutti i campi live (compatibile con build_match_state).
    """
    # ── Applica valori pending PRIMA di rendere i widget ─────────────────────
    # _pending_live_data viene scritto da _push_live_data_to_session() e
    # contiene i valori estratti dallo screenshot (minuto, gol, cartellini).
    # Va applicato qui, PRIMA che i widget vengano istanziati, altrimenti
    # Streamlit lancia "cannot be modified after widget instantiated".
    _pending = st.session_state.pop("_pending_live_data", None)
    if _pending:
        for _k, _v in _pending.items():
            st.session_state[_k] = _v

    # ── Minuto + Punteggio ────────────────────────────────────────────────────
    col_m, col_h, col_a = st.columns(3)
    with col_m:
        minuto = st.slider("Minuto", 0, 90, key="live_minuto")
    with col_h:
        gol_casa = st.number_input("Gol Casa", min_value=0, max_value=20, key="live_gol_casa")
    with col_a:
        gol_trasf = st.number_input("Gol Trasf.", min_value=0, max_value=20, key="live_gol_trasf")

    # ── Cartellini rossi (impatto diretto sul modello) ────────────────────────
    cr1, cr2 = st.columns(2)
    with cr1:
        st.number_input("🟥 Rossi Casa", min_value=0, max_value=4, key="live_rossi_casa")
    with cr2:
        st.number_input("🟥 Rossi Trasf.", min_value=0, max_value=4, key="live_rossi_trasf")

    st.divider()

    # ── Screenshot live (Gemini estrae tiri/corner/possesso/attacchi) ─────────
    st.markdown("**📷 Screenshot statistiche live**")
    uploaded = st.file_uploader(
        "Carica screenshot (Nowgoal, FlashScore, SofaScore...)",
        type=["png", "jpg", "jpeg", "webp"],
        key="live_stats_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_id = f"live_{uploaded.name}_{uploaded.size}"

        # Processa solo se è un file nuovo
        if st.session_state.get("last_live_file_id") != file_id:
            with st.spinner("Gemini legge le statistiche..."):
                try:
                    from src.ocr import extract_live_stats_from_bytes
                    img_bytes = uploaded.read()
                    extracted = extract_live_stats_from_bytes(
                        img_bytes, extension=_get_extension(uploaded.type or "image/png"),
                    )
                    st.session_state["last_live_file_id"] = file_id
                    st.session_state["live_stats_data"] = extracted
                    if extracted.extraction_success:
                        _push_live_data_to_session(extracted)
                        # Segnala che l'expander deve restare aperto dopo il rerun
                        st.session_state["_live_expander_open"] = True
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Errore lettura screenshot: {e}")

        # Mostra riepilogo di quanto estratto
        cached = st.session_state.get("live_stats_data")
        if cached and cached.extraction_success:
            _min = st.session_state.get("live_minuto", 0)
            _gh  = st.session_state.get("live_gol_casa", 0)
            _ga  = st.session_state.get("live_gol_trasf", 0)
            _sh  = st.session_state.get("live_sot_h", 0)
            _sa  = st.session_state.get("live_sot_a", 0)
            _ch  = st.session_state.get("live_corner_h", 0)
            _ca  = st.session_state.get("live_corner_a", 0)
            _ph  = st.session_state.get("live_poss_h", 0.0)
            _pa  = st.session_state.get("live_poss_a", 0.0)
            _ah  = st.session_state.get("live_att_per_h", 0)
            _aa  = st.session_state.get("live_att_per_a", 0)
            _rh  = st.session_state.get("live_rossi_casa", 0)
            _ra  = st.session_state.get("live_rossi_trasf", 0)
            _yh  = st.session_state.get("live_gialli_casa", 0)
            _ya  = st.session_state.get("live_gialli_trasf", 0)

            st.success(
                f"✅ **{_min}'  {_gh}–{_ga}**  ·  "
                f"Tiri porta: {_sh}–{_sa}  ·  "
                f"Corner: {_ch}–{_ca}  ·  "
                f"Poss: {_ph:.0f}%–{_pa:.0f}%  ·  "
                f"Att.per: {_ah}–{_aa}"
                + (f"  ·  🟥 {_rh}–{_ra}" if _rh or _ra else "")
                + (f"  ·  🟨 {_yh}–{_ya}" if _yh or _ya else "")
            )

            # Cronologia eventi
            _ev = getattr(cached, "eventi", [])
            if _ev:
                st.markdown("**📋 Cronologia eventi**")
                _ico = {"goal": "⚽", "yellow": "🟨", "red": "🟥", "sub": "🔄"}
                for e in sorted(_ev, key=lambda x: x.get("min", 0)):
                    _side = "Casa" if e["sq"] == "h" else "Trasf."
                    _icon = _ico.get(e["t"], "•")
                    _pl   = e.get("pl", "")
                    _mn   = e.get("min", 0)
                    st.caption(f"{_icon} {_mn}' — {_pl} ({_side})")

            # Tabella completa valori estratti
            with st.expander("Vedi tutti i valori estratti", expanded=False):
                _t1, _t2 = st.columns(2)
                with _t1:
                    st.markdown("**Casa**")
                    for label, key in [
                        ("Tiri in porta", "live_sot_h"), ("Tiri fuori", "live_soff_h"),
                        ("Tiri bloccati", "live_blk_h"), ("Corner", "live_corner_h"),
                        ("Possesso", "live_poss_h"), ("Att. peric.", "live_att_per_h"),
                        ("Att. totali", "live_att_h"), ("Gialli", "live_gialli_casa"),
                        ("Rossi", "live_rossi_casa"), ("Falli", "live_falli_casa"),
                    ]:
                        v = st.session_state.get(key, 0)
                        st.write(f"{label}: {v:.0f}" if isinstance(v, float) else f"{label}: {v}")
                with _t2:
                    st.markdown("**Trasferta**")
                    for label, key in [
                        ("Tiri in porta", "live_sot_a"), ("Tiri fuori", "live_soff_a"),
                        ("Tiri bloccati", "live_blk_a"), ("Corner", "live_corner_a"),
                        ("Possesso", "live_poss_a"), ("Att. peric.", "live_att_per_a"),
                        ("Att. totali", "live_att_a"), ("Gialli", "live_gialli_trasf"),
                        ("Rossi", "live_rossi_trasf"), ("Falli", "live_falli_trasf"),
                    ]:
                        v = st.session_state.get(key, 0)
                        st.write(f"{label}: {v:.0f}" if isinstance(v, float) else f"{label}: {v}")
        elif cached:
            st.warning(f"⚠️ Lettura parziale: {cached.error_message}")

    # ── Leggi tutti i valori dal session_state ────────────────────────────────
    def _ss(k: str, default=0):
        return st.session_state.get(k, default)

    return {
        "minuto":           minuto,
        "gol_casa":         gol_casa,
        "gol_trasf":        gol_trasf,
        "rossi_casa":       _ss("live_rossi_casa"),
        "rossi_trasf":      _ss("live_rossi_trasf"),
        "gialli_casa":      _ss("live_gialli_casa"),
        "gialli_trasf":     _ss("live_gialli_trasf"),
        "sot_h":            _ss("live_sot_h"),
        "soff_h":           _ss("live_soff_h"),
        "sot_a":            _ss("live_sot_a"),
        "soff_a":           _ss("live_soff_a"),
        "blk_h":            _ss("live_blk_h"),
        "blk_a":            _ss("live_blk_a"),
        "corner_h":         _ss("live_corner_h"),
        "corner_a":         _ss("live_corner_a"),
        "possesso_h":       _ss("live_poss_h", 0.0),
        "possesso_a":       _ss("live_poss_a", 0.0),
        "att_pericolosi_h": _ss("live_att_per_h"),
        "att_pericolosi_a": _ss("live_att_per_a"),
        "att_h":            _ss("live_att_h"),
        "att_a":            _ss("live_att_a"),
        "falli_casa":       _ss("live_falli_casa"),
        "falli_trasf":      _ss("live_falli_trasf"),
    }
