"""
Simulazioni per verificare:
1. Segnali duplicati/identici
2. Gestione live (cambio minuto/punteggio)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.engine import MatchState, ExchangeQuotes, analizza
from src.signals import (
    genera_segnali_rapidi,
    genera_segnali_avanzati,
    calcola_soglie,
)

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def simula_partita():
    """Simula una partita cambiando minuto e punteggio."""
    print_separator("SIMULAZIONE PARTITA LIVE")
    
    # Configurazione base
    base_state = {
        "gol_casa": 0,
        "gol_trasf": 0,
        "rossi_casa": 0,
        "rossi_trasf": 0,
        "ah_op": -0.5,      # Casa favorita
        "tot_op": 2.5,
        "ah_cur": -0.5,
        "tot_cur": 2.5,
        "linea_ou": 2.5,
        "sot_h": 0,
        "soff_h": 0,
        "sot_a": 0,
        "soff_a": 0,
    }
    
    minuti = [0, 15, 30, 45, 60, 75, 85]
    
    for minuto in minuti:
        state = MatchState(minuto=minuto, **base_state)
        try:
            risultati = analizza(state)
            
            print(f"\n--- Minuto {minuto}' | Punteggio: {state.gol_casa}-{state.gol_trasf} ---")
            print(f"xG Casa: {risultati.xg_h_final:.3f} | xG Trasf: {risultati.xg_a_final:.3f}")
            print(f"P(1): {risultati.p1:.1%} | P(X): {risultati.px:.1%} | P(2): {risultati.p2:.1%}")
            print(f"P(Over 2.5): {risultati.p_over:.1%} | P(Under 2.5): {risultati.p_under:.1%}")
            print(f"P(BTTS): {risultati.p_btts:.1%}")
            print(f"Momentum: {risultati.momentum:.2f} | Confidence: {risultati.model_confidence:.2f}")
            
            # Genera segnali rapidi
            segnali = genera_segnali_rapidi(
                prob_1=risultati.p1,
                prob_x=risultati.px,
                prob_2=risultati.p2,
                prob_over=risultati.p_over,
                prob_under=risultati.p_under,
                prob_btts=risultati.p_btts,
                minuto=minuto,
                linea_ou=2.5,
                gol_attuali=0,
                model_confidence=risultati.model_confidence,
            )
            
            if segnali:
                print(f"Segnali ({len(segnali)}):")
                for s in segnali:
                    print(f"  - {s.tipo}: {s.mercato} @ {s.quota_fair:.2f} (prob {s.prob_mod:.1%})")
            else:
                print("Nessun segnale")
                
        except Exception as e:
            print(f"ERRORE al minuto {minuto}: {e}")

def simula_cambio_punteggio():
    """Simula cambi di punteggio durante la partita."""
    print_separator("SIMULAZIONE CAMBIO PUNTEGGIO")
    
    # Partita 0-0 fino al 60', poi gol casa
    scenari = [
        {"minuto": 0, "gol_casa": 0, "gol_trasf": 0, "desc": "Inizio partita"},
        {"minuto": 30, "gol_casa": 0, "gol_trasf": 0, "desc": "30' ancora 0-0"},
        {"minuto": 45, "gol_casa": 0, "gol_trasf": 0, "desc": "Fine primo tempo 0-0"},
        {"minuto": 60, "gol_casa": 1, "gol_trasf": 0, "desc": "60' GOL CASA 1-0"},
        {"minuto": 75, "gol_casa": 1, "gol_trasf": 1, "desc": "75' PAREGGIO 1-1"},
        {"minuto": 85, "gol_casa": 2, "gol_trasf": 1, "desc": "85' CASA RADDOPPIA 2-1"},
    ]
    
    base = {
        "rossi_casa": 0, "rossi_trasf": 0,
        "ah_op": -0.5, "tot_op": 2.5,
        "ah_cur": -0.5,  # Simuliamo linee statiche (probabile caso reale)
        "tot_cur": 2.5,
        "linea_ou": 2.5,
        "sot_h": 0, "soff_h": 0, "sot_a": 0, "soff_a": 0,
    }
    
    for s in scenari:
        minuto = s["minuto"]
        gol_casa = s["gol_casa"]
        gol_trasf = s["gol_trasf"]
        
        # Aggiusta linee correnti per riflettere i gol
        # ah_cur dovrebbe riflettere il vantaggio
        ah_cur = base["ah_op"] - (gol_casa - gol_trasf)  # -0.5 - diff
        tot_cur = max(0.5, base["tot_op"] - (gol_casa + gol_trasf) * 0.3)  # Riduce man mano
        
        state = MatchState(
            minuto=minuto,
            gol_casa=gol_casa,
            gol_trasf=gol_trasf,
            ah_cur=ah_cur,
            tot_cur=tot_cur,
            **{k: v for k, v in base.items() if k not in ["ah_cur", "tot_cur"]},
        )
        
        try:
            risultati = analizza(state)
            
            print(f"\n--- {s['desc']} ---")
            print(f"Linee: AH cur={ah_cur:.2f}, Tot cur={tot_cur:.2f}")
            print(f"xG rimanenti: Casa {risultati.xg_h_final:.3f} | Trasf {risultati.xg_a_final:.3f}")
            print(f"P(1): {risultati.p1:.1%} | P(X): {risultati.px:.1%} | P(2): {risultati.p2:.1%}")
            print(f"P(Over 2.5): {risultati.p_over:.1%}")
            
            # Top correct scores
            print(f"Top correct scores:")
            for (h, a), prob in risultati.top_cs[:3]:
                print(f"  {h}-{a}: {prob:.2%}")
                
        except Exception as e:
            print(f"ERRORE: {e}")

def simula_segnali_duplicati():
    """Verifica se i segnali sono spesso identici."""
    print_separator("VERIFICA SEGNALI DUPLICATI")
    
    # Test con diverse configurazioni
    test_cases = [
        {"prob_1": 0.70, "prob_x": 0.20, "prob_2": 0.10, "desc": "Casa forte favorita"},
        {"prob_1": 0.50, "prob_x": 0.30, "prob_2": 0.20, "desc": "Casa moderatamente favorita"},
        {"prob_1": 0.35, "prob_x": 0.30, "prob_2": 0.35, "desc": "Partita equilibrata"},
        {"prob_1": 0.20, "prob_x": 0.30, "prob_2": 0.50, "desc": "Trasferta favorita"},
    ]
    
    for tc in test_cases:
        print(f"\n--- {tc['desc']} ---")
        
        segnali = genera_segnali_rapidi(
            prob_1=tc["prob_1"],
            prob_x=tc["prob_x"],
            prob_2=tc["prob_2"],
            prob_over=0.55,
            prob_under=0.45,
            prob_btts=0.50,
            minuto=45,
            linea_ou=2.5,
            gol_attuali=0,
            model_confidence=0.80,
        )
        
        if segnali:
            mercati = [s.mercato for s in segnali]
            print(f"Segnali: {mercati}")
            
            # Verifica duplicati
            if len(mercati) != len(set(mercati)):
                print("⚠️ ATTENZIONE: Mercati duplicati trovati!")
            
            for s in segnali:
                print(f"  {s.tipo}: {s.mercato} - prob={s.prob_mod:.1%}, fair=@{s.quota_fair:.2f}")
        else:
            print("Nessun segnale generato")

def simula_soglie():
    """Verifica come cambiano le soglie."""
    print_separator("VERIFICA SOGLIE DINAMICHE")
    
    for minuto in [0, 30, 45, 60, 75, 85]:
        soglie = calcola_soglie(minuto, 2.5, 0, model_agreement=0.8)
        print(f"Minuto {minuto:2d}': 1x2={soglie['1x2']:.1%}, OU_over={soglie['ou_over']:.1%}, BTTS_no={soglie['btts_no']:.1%}")

def simula_input_utente():
    """Simula esattamente cosa farebbe un utente."""
    print_separator("SIMULAZIONE INPUT UTENTE REALE")
    
    print("""
SCENARIO: L'utente sta seguendo una partita live.
- Inizio: 0-0, AH -0.5, Tot 2.5
- Al 30': ancora 0-0, le linee non si sono mosse
- Al 60': gol casa 1-0, linee si aggiornano
- Al 80': pareggio 1-1
""")
    
    # Caso 1: Inizio partita
    print("\n--- INIZIO PARTITA (0') ---")
    state = MatchState(
        minuto=0, gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.5, tot_op=2.5,
        ah_cur=-0.5, tot_cur=2.5,
        linea_ou=2.5,
    )
    r = analizza(state)
    print(f"xG Casa: {r.xg_h_final:.3f} (da linee)")
    print(f"P(1)={r.p1:.1%} P(X)={r.px:.1%} P(2)={r.p2:.1%}")
    
    # Caso 2: 30' senza gol, linee non mosse (UTENTE NON AGGIORNA)
    print("\n--- 30' SENZA GOL, LINEE NON AGGIORNATE (errore comune) ---")
    state = MatchState(
        minuto=30, gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.5, tot_op=2.5,
        ah_cur=-0.5, tot_cur=2.5,  # ANCORA 2.5 come se fosse full-game!
        linea_ou=2.5,
    )
    r = analizza(state)
    print(f"⚠️ tot_cur=2.5 è interpretato come 'gol rimanenti'")
    print(f"xG Casa: {r.xg_h_final:.3f} | xG Trasf: {r.xg_a_final:.3f}")
    print(f"P(Over): {r.p_over:.1%}")
    
    # Caso 3: 30' con linee corrette
    print("\n--- 30' SENZA GOL, LINEE CORRETTE ---")
    state = MatchState(
        minuto=30, gol_casa=0, gol_trasf=0,
        rossi_casa=0, rossi_trasf=0,
        ah_op=-0.5, tot_op=2.5,
        ah_cur=-0.5, tot_cur=1.8,  # CORRETTO: gol rimanenti ~1.8
        linea_ou=2.5,
    )
    r = analizza(state)
    print(f"xG Casa: {r.xg_h_final:.3f} | xG Trasf: {r.xg_a_final:.3f}")
    print(f"P(Over): {r.p_over:.1%}")

if __name__ == "__main__":
    print("🔍 AVVIO SIMULAZIONI EXCHANGE-2.0\n")
    
    simula_soglie()
    simula_segnali_duplicati()
    simula_partita()
    simula_cambio_punteggio()
    simula_input_utente()
    
    print("\n✅ SIMULAZIONI COMPLETATE")
