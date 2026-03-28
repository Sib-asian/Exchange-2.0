"""
app.py — Radar Pro Live v2.0
Layout semplificato: Linee → Screenshot prematch → ANALIZZA
                     ▶ Dati Live (expander) → ANALIZZA
"""

import streamlit as st

from src.config import INPUT_VALIDATION, SIGNALS, UI
from src.session_storage import (
    PartitaSalvata,
    build_saved_at_label,
    collect_widget_state,
    delete_partita,
    load_partite,
    restore_widget_state,
    save_partita,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=UI.PAGE_TITLE,
    page_icon=UI.PAGE_ICON,
    layout=UI.LAYOUT,
)
st.title(f"{UI.PAGE_ICON} {UI.PAGE_TITLE}")
st.caption(f"v{UI.VERSION} · Bivariate Poisson + Dixon-Coles + Kelly frazionato")

# ── Match Memory ──────────────────────────────────────────────────────────────
_saved_matches = load_partite()
if _saved_matches:
    _match_options = {f"{p.nome}  [{p.saved_at}]": p for p in _saved_matches}
    _col_sel, _col_load, _col_del = st.columns([5, 1, 1])
    with _col_sel:
        _selected_label = st.selectbox(
            "Partite salvate",
            options=list(_match_options.keys()),
            index=None,
            placeholder="Seleziona una partita salvata...",
            key="match_memory_selector",
            label_visibility="collapsed",
        )
    with _col_load:
        if _selected_label and st.button("Carica", key="btn_load_match", use_container_width=True):
            _p = _match_options[_selected_label]
            restore_widget_state(st.session_state, _p.widget_state)
            if _p.ricerca is not None:
                from dataclasses import fields as _dc_fields
                from src.research import RicercaPartita as _RicercaPartita
                try:
                    _r_obj = _RicercaPartita(**{
                        k: v for k, v in _p.ricerca.items()
                        if k in {f.name for f in _dc_fields(_RicercaPartita)}
                    })
                    st.session_state["ricerca_risultato"] = _r_obj
                except Exception:
                    pass
            st.session_state["active_match_id"] = _p.id
            st.rerun()
    with _col_del:
        if _selected_label and st.button("✕", key="btn_del_match", use_container_width=True):
            _p = _match_options[_selected_label]
            delete_partita(_p.id)
            if st.session_state.get("active_match_id") == _p.id:
                st.session_state.pop("active_match_id", None)
            st.rerun()

    _active_id = st.session_state.get("active_match_id")
    _active_p = next((p for p in _saved_matches if p.id == _active_id), None) if _active_id else None
    if _active_p:
        st.caption(f"Attiva: **{_active_p.nome}** · salvata {_active_p.saved_at}")

st.divider()

# ── Stato live corrente (usato per la conversione Full Game nelle linee) ───────
_live_min = st.session_state.get("live_minuto", 0)
_live_gh  = st.session_state.get("live_gol_casa", 0)
_live_ga  = st.session_state.get("live_gol_trasf", 0)

# ── Linee ─────────────────────────────────────────────────────────────────────
from src.ui.inputs import (
    build_match_state,
    render_bankroll,
    render_extracted_data_panel,
    render_image_upload,
    render_linee_semplici,
    render_live_semplice,
)

# Calcola linea OU automaticamente (closest a tot_cur_raw)
def _auto_linea_ou(tot_raw: float, gol_tot: int = 0) -> float:
    _linee = list(UI.LINEE_OU)
    return min(_linee, key=lambda x: abs(x - tot_raw))

lines   = render_linee_semplici(gol_casa=_live_gh, gol_trasf=_live_ga)
linea_ou = _auto_linea_ou(lines["tot_cur_raw"])

st.divider()

# ── Screenshot Quote Prematch ─────────────────────────────────────────────────
extracted_data = None
with st.expander("📷 Screenshot quote prematch", expanded=False):
    extracted_data = render_image_upload()
    if extracted_data is not None:
        render_extracted_data_panel(extracted_data)

# ── Bankroll / Impostazioni ───────────────────────────────────────────────────
# I widget sono sempre renderizzati (anche se nascosti nell'expander) così
# i valori sono sempre disponibili nel session_state dopo il primo render.
with st.expander("⚙️ Bankroll e commissione", expanded=False):
    render_bankroll()

bankroll  = float(st.session_state.get("bankroll_value",  UI.BANKROLL_DEFAULT))
comm_pct  = float(st.session_state.get("comm_pct_value",  UI.COMM_DEFAULT))
comm_rate = comm_pct / 100.0

# ── Badge Ricerca Pre-Partita ─────────────────────────────────────────────────
_ricerca     = st.session_state.get("ricerca_risultato")
_ricerca_ok  = _ricerca is not None and _ricerca.success

if _ricerca_ok:
    _aff_icon = {"alta": "🟢", "media": "🟡", "bassa": "🔴"}.get(_ricerca.affidabilita, "⚪")
    _parts = [f"{_ricerca.squadra_casa} vs {_ricerca.squadra_trasf} {_aff_icon}"]
    if _ricerca.h2h_media_gol > 0:
        _parts.append(f"H2H {_ricerca.h2h_media_gol:.1f} gol/p")
    _abs_tot = len(_ricerca.assenze_casa) + len(_ricerca.assenze_trasf)
    if _abs_tot:
        _parts.append(f"{_abs_tot} assenz{'a' if _abs_tot == 1 else 'e'}")
    _bc, _xc = st.columns([6, 1])
    with _bc:
        st.success("🔍 " + " · ".join(_parts))
    with _xc:
        if st.button("✕", key="remove_ricerca"):
            del st.session_state["ricerca_risultato"]
            st.rerun()
else:
    st.page_link("pages/2_🔍_Ricerca_Partita.py", label="🔍 Aggiungi contesto (assenze, H2H, forma)", icon="🔍")

# ── ANALIZZA Prematch ─────────────────────────────────────────────────────────
_btn_prematch = st.button("ANALIZZA", use_container_width=True, type="primary", key="btn_analizza_prematch")

st.divider()

# ── Dati Live ─────────────────────────────────────────────────────────────────
with st.expander("📺 Dati Live (aggiorna dopo il fischio)", expanded=False):
    _match_live = render_live_semplice()
    _btn_live   = st.button("ANALIZZA", use_container_width=True, type="primary", key="btn_analizza_live")

# ── Pipeline di analisi (si attiva con uno dei due bottoni) ───────────────────
if _btn_prematch or _btn_live:

    # Dati live: usa quelli del expander se live, altrimenti prematch (minuto=0)
    _match = _match_live if _btn_live else {
        "minuto": 0, "gol_casa": 0, "gol_trasf": 0,
        "rossi_casa": 0, "rossi_trasf": 0,
        "gialli_casa": 0, "gialli_trasf": 0,
        "sot_h": 0, "soff_h": 0, "sot_a": 0, "soff_a": 0,
        "blk_h": 0, "blk_a": 0,
        "corner_h": 0, "corner_a": 0,
        "possesso_h": 0.0, "possesso_a": 0.0,
        "att_pericolosi_h": 0, "att_pericolosi_a": 0,
        "att_h": 0, "att_a": 0,
        "falli_casa": 0, "falli_trasf": 0,
    }

    # Se si preme ANALIZZA dal prematch ma c'è già uno stato live, usalo
    if _btn_prematch and _live_min > 0:
        _match = _match_live

    # Ricalcola linee con il minuto/gol correnti
    _gol_tot_match = _match["gol_casa"] + _match["gol_trasf"]
    _linea_ou_eff  = _auto_linea_ou(lines["tot_cur_raw"], _gol_tot_match)

    # OCR data
    _ocr_total  = 0.0
    _ocr_q1     = 0.0
    _ocr_qx     = 0.0
    _ocr_q2     = 0.0
    _ocr_qover  = 0.0
    _ocr_qund   = 0.0
    _ocr_qgg    = 0.0
    _ocr_qng    = 0.0
    if extracted_data is not None and extracted_data.extraction_success:
        if extracted_data.linea_ou > 0.5:
            _ocr_total = extracted_data.linea_ou
        _ocr_q1    = extracted_data.quota_1
        _ocr_qx    = extracted_data.quota_x
        _ocr_q2    = extracted_data.quota_2
        _ocr_qover = extracted_data.quota_over
        _ocr_qund  = extracted_data.quota_under
        _ocr_qgg   = extracted_data.quota_gg
        _ocr_qng   = extracted_data.quota_ng

    # Moltiplicatori AI da ricerca
    _fixture_prior      = 0.0
    _movement_quality   = 1.0
    _ocr_confidence_scale = 1.0
    _ricerca_adj_tot    = 0.0
    _ricerca_adj_ah     = 0.0
    _absence_mult_h     = 1.0
    _absence_mult_a     = 1.0
    _forma_mult_h       = 1.0
    _forma_mult_a       = 1.0

    if _ricerca_ok:
        from src.config import AI_ADJ
        from src.models.ai_adjustments import calcola_assenze_mult, calcola_forma_mult

        _aff_scale = {
            "alta":  AI_ADJ.AFFIDABILITA_ALTA,
            "media": AI_ADJ.AFFIDABILITA_MEDIA,
            "bassa": AI_ADJ.AFFIDABILITA_BASSA,
        }.get(_ricerca.affidabilita, AI_ADJ.AFFIDABILITA_MEDIA)

        if _ricerca.h2h_media_gol > 0:
            _fixture_prior = _ricerca.h2h_media_gol
        _ricerca_adj_tot = _ricerca.adj_tot * _aff_scale
        _ricerca_adj_ah  = _ricerca.adj_ah  * _aff_scale

        _own_h = calcola_assenze_mult(_ricerca.assenze_casa,  per_avversario=False)
        _gk_h  = calcola_assenze_mult(_ricerca.assenze_trasf, per_avversario=True)
        _own_a = calcola_assenze_mult(_ricerca.assenze_trasf, per_avversario=False)
        _gk_a  = calcola_assenze_mult(_ricerca.assenze_casa,  per_avversario=True)
        _absence_mult_h = (1.0 + (_own_h - 1.0) * _aff_scale) * (1.0 + (_gk_h - 1.0) * _aff_scale)
        _absence_mult_a = (1.0 + (_own_a - 1.0) * _aff_scale) * (1.0 + (_gk_a - 1.0) * _aff_scale)

        if _ricerca.forma_casa:
            _forma_mult_h = 1.0 + (calcola_forma_mult(_ricerca.forma_casa) - 1.0) * _aff_scale
        if _ricerca.forma_trasf:
            _forma_mult_a = 1.0 + (calcola_forma_mult(_ricerca.forma_trasf) - 1.0) * _aff_scale

    # Gemini enrichment (solo se ci sono squadre riconosciute)
    _has_ocr_teams = (
        extracted_data is not None and extracted_data.extraction_success
        and bool(extracted_data.squadra_casa) and bool(extracted_data.squadra_trasf)
    )
    _has_ricerca_teams = _ricerca_ok and bool(_ricerca.squadra_casa)
    _is_prematch = _match["minuto"] == 0

    if _has_ocr_teams or _has_ricerca_teams:
        from concurrent.futures import ThreadPoolExecutor
        from src.research import _get_gemini_api_key as _check_key
        from src.research import cerca_prior_storico, interpreta_movimento_linee, valida_quote_ocr

        _sc  = extracted_data.squadra_casa  if _has_ocr_teams else _ricerca.squadra_casa
        _st  = extracted_data.squadra_trasf if _has_ocr_teams else _ricerca.squadra_trasf
        _gol_diff_live  = _match["gol_casa"] - _match["gol_trasf"]
        _ah_cur_full    = lines["ah_cur"] - _gol_diff_live
        _delta_ah_pure  = _ah_cur_full - lines["ah_op"]
        _delta_tot_pure = lines["tot_cur_raw"] - lines["tot_op"]

        _gem_key = (_sc, _st, round(_delta_ah_pure, 2), round(_delta_tot_pure, 2),
                    st.session_state.get("last_ocr_file_id", ""))
        _cached = st.session_state.get("gemini_enrichment")

        if _cached is None or _cached.get("_key") != _gem_key:
            if _check_key():
                with st.spinner("Gemini analizza..."):
                    try:
                        with ThreadPoolExecutor(max_workers=3) as _ex:
                            _f_ocr = _ex.submit(valida_quote_ocr, _sc, _st,
                                                 _ocr_q1, _ocr_qx, _ocr_q2,
                                                 _ocr_qover, _ocr_qund, _linea_ou_eff,
                                                 ) if (_is_prematch and _ocr_q1 > 1.0) else None
                            _f_prior = _ex.submit(cerca_prior_storico, _sc, _st
                                                  ) if (_is_prematch and _fixture_prior <= 0) else None
                            _f_mov = _ex.submit(interpreta_movimento_linee, _sc, _st,
                                                _delta_ah_pure, _delta_tot_pure,
                                                ) if abs(_delta_ah_pure) >= 0.15 or abs(_delta_tot_pure) >= 0.15 else None

                        _ocr_val = _f_ocr.result()   if _f_ocr   else {"confidence_scale": 1.0, "flags": []}
                        _fix_val = _f_prior.result() if _f_prior else _fixture_prior
                        _mov_val = _f_mov.result()   if _f_mov   else 1.0
                        st.session_state["gemini_enrichment"] = {
                            "_key": _gem_key,
                            "ocr_confidence_scale": _ocr_val["confidence_scale"],
                            "ocr_flags": _ocr_val["flags"],
                            "fixture_prior": _fix_val,
                            "movement_quality": _mov_val,
                        }
                    except Exception:
                        st.session_state["gemini_enrichment"] = {
                            "_key": _gem_key,
                            "ocr_confidence_scale": 1.0, "ocr_flags": [],
                            "fixture_prior": 0.0, "movement_quality": 1.0,
                        }

        _g = st.session_state.get("gemini_enrichment", {})
        _ocr_confidence_scale = _g.get("ocr_confidence_scale", 1.0)
        _fixture_prior        = _g.get("fixture_prior", 0.0)
        _movement_quality     = _g.get("movement_quality", 1.0)

        _ocr_flags = _g.get("ocr_flags", [])
        if _ocr_flags:
            st.warning("⚠️ Gemini: " + " | ".join(_ocr_flags[:3]) +
                       f" (confidenza ridotta: {_ocr_confidence_scale:.0%})")

    # Applica aggiustamenti ricerca alle linee
    _adj_lines = dict(lines)
    if _ricerca_adj_tot != 0 or _ricerca_adj_ah != 0:
        _adj_lines["tot_op"] = lines["tot_op"] + _ricerca_adj_tot
        _adj_lines["ah_op"]  = lines["ah_op"]  + _ricerca_adj_ah

    # Build MatchState
    try:
        state = build_match_state(
            _match, _adj_lines, _linea_ou_eff, bankroll, comm_rate,
            ocr_imp_total=_ocr_total,
            ocr_quota_1=_ocr_q1, ocr_quota_x=_ocr_qx, ocr_quota_2=_ocr_q2,
            ocr_quota_over=_ocr_qover, ocr_quota_under=_ocr_qund,
            ocr_quota_gg=_ocr_qgg, ocr_quota_ng=_ocr_qng,
            fixture_historical_total=_fixture_prior,
            movement_quality=_movement_quality,
            ocr_confidence_scale=_ocr_confidence_scale,
            absence_mult_h=_absence_mult_h, absence_mult_a=_absence_mult_a,
            forma_mult_h=_forma_mult_h, forma_mult_a=_forma_mult_a,
        )
    except (AssertionError, ValueError) as e:
        st.error(f"❌ Input non valido: {e}")
        st.stop()

    with st.spinner("Calcolo..."):
        from src.engine import analizza
        risultati = analizza(state)

    # Auto-save
    _nome_match = ""
    if _ricerca_ok:
        _nome_match = f"{_ricerca.squadra_casa} vs {_ricerca.squadra_trasf}"
    elif _has_ocr_teams:
        _nome_match = f"{extracted_data.squadra_casa} vs {extracted_data.squadra_trasf}"

    if _nome_match:
        from dataclasses import asdict as _asdict
        _active_id = st.session_state.get("active_match_id")
        _pid = _active_id if _active_id else str(abs(hash(_nome_match)) % (10 ** 9))
        save_partita(PartitaSalvata(
            id=_pid, nome=_nome_match, saved_at=build_saved_at_label(),
            widget_state=collect_widget_state(st.session_state),
            ricerca=_asdict(_ricerca) if _ricerca_ok else None,
        ))
        st.session_state["active_match_id"] = _pid

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    from src.signals import calcola_soglie, genera_segnali_rapidi
    from src.ui.outputs import (
        render_anomalies,
        render_asian_handicap,
        render_avvisi_affidabilita,
        render_avvisi_incoerenza,
        render_clean_sheet,
        render_coerenza_mercati,
        render_confidence_bands,
        render_correct_score,
        render_debug,
        render_lines_need_update,
        render_mercati_chiusi,
        render_model_confidence,
        render_momentum,
        render_ocr_market_divergence,
        render_pronostici_rapidi,
        render_red_card_impact,
        render_riepilogo_modello,
        render_segnali_rapidi,
        render_session_stats,
    )

    _shots    = (state.sot_h, state.soff_h, state.sot_a, state.soff_a)
    _n_shots  = sum(_shots)
    _gol_att  = state.gol_casa + state.gol_trasf
    _minuto   = state.minuto

    render_lines_need_update(risultati)

    # ── Pronostici rapidi (solo percentuali) ──────────────────────────────────
    st.divider()
    render_pronostici_rapidi(risultati, _linea_ou_eff, _minuto, state.gol_casa, state.gol_trasf)

    # ── Segnali rapidi ────────────────────────────────────────────────────────
    from src.models.kelly import calcola_kelly_fraction
    _kelly_frac = calcola_kelly_fraction(_minuto, _n_shots, risultati.model_confidence)

    _settled = render_mercati_chiusi(
        _gol_att, _linea_ou_eff, state.gol_casa, state.gol_trasf,
        _minuto, risultati.p_btts,
    )
    segnali_rapidi = genera_segnali_rapidi(
        risultati.p1, risultati.px, risultati.p2,
        risultati.p_over, risultati.p_under, risultati.p_btts,
        _minuto, _linea_ou_eff, _gol_att,
        model_confidence=risultati.model_confidence,
        model_agreement=risultati.model_agreement,
    )
    if _settled.get("ou_vinto"):
        segnali_rapidi = [s for s in segnali_rapidi if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
    if _settled.get("btts_si_settled") or _settled.get("btts_no_settled"):
        segnali_rapidi = [s for s in segnali_rapidi if "BTTS" not in s.mercato.upper()]

    render_segnali_rapidi(segnali_rapidi)
    render_avvisi_affidabilita(risultati.flat_lines, _n_shots, _minuto)

    # ── Stop se fine partita ──────────────────────────────────────────────────
    if _minuto >= SIGNALS.GAME_END_THRESHOLD:
        st.error("Fine partita — spread enormi, non entrare.")
        st.stop()
    if _minuto > SIGNALS.RECOVERY_TIME_WARNING:
        if _minuto >= SIGNALS.RECOVERY_TIME_HARD_CAP:
            st.error(f"🚫 Minuto {_minuto}' supera il limite ({SIGNALS.RECOVERY_TIME_HARD_CAP}').")
            st.stop()
        else:
            st.warning(f"⚠️ Recovery time ({_minuto}'): previsioni meno affidabili.")

    # ── Analisi avanzata (expander) ───────────────────────────────────────────
    st.divider()
    with st.expander("📊 Analisi avanzata", expanded=False):
        render_riepilogo_modello(risultati, _linea_ou_eff, _minuto)
        st.divider()
        render_confidence_bands(risultati, _linea_ou_eff)
        render_coerenza_mercati(risultati, _linea_ou_eff)
        render_correct_score(risultati)
        render_asian_handicap(risultati.full_matrix)
        render_clean_sheet(risultati.full_matrix, state.gol_casa, state.gol_trasf)
        render_red_card_impact(state.rossi_casa, state.rossi_trasf, _minuto)
        st.divider()
        render_momentum(risultati.momentum, risultati.delta_ah, risultati.delta_tot)
        if extracted_data is not None and extracted_data.extraction_success:
            render_ocr_market_divergence(
                risultati,
                ocr_quota_1=extracted_data.quota_1,
                ocr_quota_x=extracted_data.quota_x,
                ocr_quota_2=extracted_data.quota_2,
                ocr_quota_over=extracted_data.quota_over,
                ocr_quota_under=extracted_data.quota_under,
            )
        render_avvisi_incoerenza(
            risultati.p_btts, risultati.p_under, risultati.p_over,
            _linea_ou_eff, _gol_att,
            btts_settled=_settled.get("btts_si_settled", False),
        )

    # ── Segnali con quote exchange (expander, opzionale) ──────────────────────
    with st.expander("💰 Segnali con quote exchange", expanded=False):
        from src.signals import genera_segnali_avanzati
        from src.ui.inputs import render_exchange_quotes
        from src.ui.outputs import (
            render_allineamento_mercato,
            render_risk_metrics,
            render_value_bets,
        )

        quotes = render_exchange_quotes(_linea_ou_eff)

        _momentum_factor = max(
            SIGNALS.MOMENTUM_STAKE_FLOOR,
            1.0 - SIGNALS.MOMENTUM_STAKE_REDUCTION_RATE * max(
                0.0, risultati.momentum - SIGNALS.MOMENTUM_STAKE_THRESHOLD,
            ),
        )
        segnali_avanzati = genera_segnali_avanzati(
            risultati.p1, risultati.px, risultati.p2,
            risultati.p_over, risultati.p_under, risultati.p_btts,
            quotes, _minuto, _linea_ou_eff, _gol_att,
            bankroll, comm_rate, _n_shots,
            risultati.momentum,
            model_confidence=risultati.model_confidence,
            model_agreement=risultati.model_agreement,
        )
        if _settled.get("ou_vinto"):
            segnali_avanzati = [s for s in segnali_avanzati if "OVER" not in s.mercato.upper() and "UNDER" not in s.mercato.upper()]
        if _settled.get("btts_si_settled") or _settled.get("btts_no_settled"):
            segnali_avanzati = [s for s in segnali_avanzati if "BTTS" not in s.mercato.upper()]

        from src.ui.outputs import render_segnali_avanzati
        render_segnali_avanzati(segnali_avanzati, quotes.any_active)
        render_allineamento_mercato(risultati, quotes)

        if quotes.any_active:
            render_value_bets(risultati, quotes)
            from src.models.consensus import compute_model_market_divergence
            model_probs_map = {
                "p1": risultati.p1, "px": risultati.px, "p2": risultati.p2,
                "p_over": risultati.p_over, "p_under": risultati.p_under,
                "p_btts": risultati.p_btts,
            }
            market_probs_map: dict[str, float] = {}
            for _k, _q in [("p1", quotes.q_1), ("px", quotes.q_x), ("p2", quotes.q_2),
                            ("p_over", quotes.q_over), ("p_under", quotes.q_under),
                            ("p_btts", quotes.q_btts_si)]:
                if _q > 1.0:
                    market_probs_map[_k] = 1.0 / _q
            risultati.market_divergence = compute_model_market_divergence(model_probs_map, market_probs_map)

        if segnali_avanzati:
            render_risk_metrics(segnali_avanzati, bankroll)

    # ── Debug ─────────────────────────────────────────────────────────────────
    with st.expander("🔧 Debug", expanded=False):
        _anomalies = render_anomalies(state)
        _soglie = calcola_soglie(_minuto, _linea_ou_eff, _gol_att)
        render_debug(
            risultati, _linea_ou_eff, _minuto,
            _soglie, comm_pct, _momentum_factor if "quotes" in dir() else 1.0, _n_shots,
        )
        render_session_stats()
