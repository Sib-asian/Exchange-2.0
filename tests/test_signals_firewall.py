from src.signals import genera_segnali_avanzati, genera_segnali_rapidi
from src.engine import ExchangeQuotes


def test_genera_segnali_rapidi_honors_firewall_block() -> None:
    out = genera_segnali_rapidi(
        0.72, 0.16, 0.12,
        0.60, 0.40, 0.55,
        minuto=0,
        linea_ou=2.5,
        gol_attuali=0,
        model_confidence=0.90,
        model_agreement=0.90,
        signals_blocked=True,
    )
    assert out == []


def test_genera_segnali_avanzati_honors_firewall_block() -> None:
    quotes = ExchangeQuotes(
        q_1=2.20,
        q_x=3.30,
        q_2=3.60,
        q_over=2.00,
        q_under=1.95,
        q_btts_si=1.95,
        q_btts_no=1.95,
    )
    out = genera_segnali_avanzati(
        0.50, 0.27, 0.23,
        0.56, 0.44, 0.52,
        quotes,
        minuto=0,
        linea_ou=2.5,
        gol_attuali=0,
        bankroll=1000.0,
        comm_rate=0.025,
        n_shots_tot=0,
        momentum=0.0,
        model_confidence=0.90,
        model_agreement=0.90,
        signals_blocked=True,
    )
    assert out == []
