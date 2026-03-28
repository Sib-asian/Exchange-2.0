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

import streamlit as st

from src.config import BAYES, INPUT_VALIDATION, UI
from src.engine import ExchangeQuotes, MatchState
from src.ocr import (
    ExtractedData,
    LiveStatsExtracted,
    extract_from_bytes,
    extract_live_stats_from_bytes,
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
    # Minuto e gol: NON sovrascrivere mai dallo screenshot.
    # La tab statistiche di Nowgoal mostra "FT"/"HT" che non corrisponde
    # al minuto reale della partita live. L'utente li imposta manualmente.
    # Statistiche: sovrascivi sempre (sono il motivo principale dello screenshot)
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
    st.session_state["live_gialli_casa"] = data.gialli_casa
    st.session_state["live_gialli_trasf"] = data.gialli_trasf
    st.session_state["live_falli_casa"] = data.falli_casa
    st.session_state["live_falli_trasf"] = data.falli_trasf


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
                                  help="Handicap asiatico alla quota di apertura (intera partita).")
        tot_op = st.number_input("Totale Apertura", value=2.50,  step=0.25,
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
        )
    with col_cm:
        comm_pct = st.number_input(
            "Commissione exchange (%)",
            value=UI.COMM_DEFAULT,
            min_value=0.0,
            max_value=UI.COMM_MAX,
            step=UI.COMM_STEP,
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
        bankroll=bankroll,
        comm_rate=comm_rate,
    )
