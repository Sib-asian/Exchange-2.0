# Radar Pro Live — Analisi Precisione Pronostici Finali

## 🎯 PRIORITÀ: PRECISIONE DELLE PREDIZIONI FINALI

Analisi approfondita della pipeline `engine.py` (v2.0-phase3-seven-upgrades, ~1460 righe).

---

## ❌ CRITICO: 7 Problemi Identificati che Degrado la Precisione

### 1. 🔴 H2H Data: 6 Punti di Iniezione (Double/Multi-Counting)

**Problema**: I dati H2H entrano nella pipeline in **6 punti diversi**, creando un effetto
cumulativo eccessivo. Ogni punto è individualmente conservativo (α=0.05-0.15), ma la somma
sfugge al controllo.

| # | Funzione | Segnale H2H | Effetto |
|---|----------|-------------|--------|
| 1 | `_apply_h2h_blend` | gol medi H2H per squadra | Blend xG asimmetrico |
| 2 | `_apply_prematch_priors` | H2H AH home cover % | Tilt asimmetrico xG |
| 3 | `_apply_post_consensus_market_blend` | H2H 1X2 win % | Blend probabilità post-consensus |
| 4 | `calibra_btts` (9a) | H2H BTTS % | Calibrazione BTTS |
| 5 | H2H Over % (9d) | H2H Over % | Blend O/U |
| 6 | HT/FT model (9c-bis) | H2H HT % | Calibrazione 1X2 |

**Impatto**: Se H2H indica "casa forte", il sistema applica 6 piccoli boost che si sommano,
sovrastimando sistematicamente la forza relativa della squadra di casa.

**Fix**: Unificare H2H in un singolo punto di iniezione EARLY nella pipeline (step 1b),
e rimuovere/ridurre drasticamente gli altri 5 punti.

---

### 2. 🔴 Circular Dependency: Post-Consensus Market Blend

**Problema**: La pipeline estrae xG DALLE linee di mercato (step 1), calcola probabilità
dal consensus (step 8-9), poi BLEND LE PROBABILITÀ INDIETRO con le quote mercato (step 9c).

```
Linee mercato → xG → modelli → probabilità → BLENDED with market → probabilità finali
     ↑___________________SAME DATA SOURCE___________________|
```

**Impatto**: Il segnale indipendente del modello viene diluito. Le probabilità finali sono
troppo ancorate alle quote bookmaker, riducendo il value edge.

**Fix**: Rimuovere o ridurre drasticamente `_apply_post_consensus_market_blend`. Il mercato
dovrebbe entrare SOLO come input (linee AH/Total), non come output constraint.

---

### 3. 🟡 Live Recalibration Simmetrica (Errata per Situazioni Asimmetriche)

**Problema**: `compute_live_recalibration_factor` restituisce UN SOLO fattore moltiplicativo
applicato IDENTICAMENTE a entrambe le squadre. Se la casa sovraperforma e la trasferta
sottoperforma, il fattore riduce/aumenta entrambe erroneamente.

**Impatto**: In partite sbilanciate, il recal live sposta la previsione nella direzione
sbagliata per una delle due squadre.

**Fix**: Calcolare fattori SEPARATI per casa e trasferta, basati sulla deviazione del
rispettivo xG rispetto ai gol osservati per squadra.

---

### 4. 🟡 O/U 2.5 Canonico: 6 Step di Post-Processing

**Problema**: `p_over_25_ref` viene modificato in 6 passaggi sequenziali:

1. H2H Over % blend (9d)
2. Previous Scores Over% blend (9e)
3. Bridge da linea selezionata (9f)
4. Riconciliazione con distribuzione gol (10)
5. Coerenza O/U con distribuzione (10c)
6. Coerenza O/U + BTTS (10d)

**Impatto**: Ogni step è piccolo (α=0.03-0.08) ma la composizione produce una deviazione
totale di ~15-25% dal valore originale del consensus. Questo aumenta l'errore sistematico
e rende difficile tracciare la fonte di bias.

**Fix**: Consolidare i 6 step in max 2-3 step ben definiti, con alpha totali calcolati
come composizione logica (non sequenza empirica).

---

### 5. 🟡 xG Pipeline: 9 Step Sequenziali Prematch

**Problema**: La pipeline xG prematch ha 9 step di blend/adjust:

1. Bayesian xG from lines
2. H2H goals blend
3. Absences + form + weather
4. Previous scores blend
5. Strength model blend
6. URL signals (recent xG, form trend, AH cover)
7. Form analysis (standings, last6, home/away, timing)
8. Lambda-total coherence
9. Tempo mixture

**Impatto**: Effetto cascata. Ogni blend α_i = 0.05-0.15, ma il prodotto
Π(1 - α_i) ≈ 0.60-0.75, cioè il segnale di mercato originale è ridotto del 25-40%.
Questo significa che il modello sta aggiungendo ~25-40% di segnale "inventato" che
degrada la precisione rispetto alle linee di mercato pure.

**Fix**: Ridurre il numero di step a 4-5, aumentando la qualità di ciascuno e
calcolando un "budget di deviazione" massimo dal mercato (es. ±20% tot).

---

### 6. 🟢 Draw Shrinkage Potenzialmente Aggressivo in Partite Aperte

**Problema**: Per partite con tot > 3.0, il draw shrinkage totale può raggiungere
~5-6% (BASE=0.030 × tot/2.5 + agreement bonus). La letteratura (Dixon-Coles 1997)
suggerisce che il bias della Poisson sul draw è ~2-3%.

**Impatto**: Partite aperte (tot alto) vedono P(X) ridotta troppo → più P(1)+P(2)
→ eccesso di segnali BACK su favoriti.

**Fix**: Cap più aggressivo sul draw shrinkage totale (max 4% invece di ~6%).

---

### 7. 🟢 BTTS One-Goal Boost Potenzialmente Errato

**Problema**: `BTTS_ONE_GOAL_BOOST = 0.08` (+8% BTTS se una squadra ha segnato).
Ma questo boost viene applicato ANCHE in live quando il punteggio è 1-0 al minuto 5.
A quel punto, il boost di +8% è poco informativo perché è statisticamente normale
che una squadra abbia segnato.

**Fix**: Modulare il boost per minuto (più forte a fine partita, più debole all'inizio).

---

## 📊 PRIORITÀ IMPLEMENTAZIONE

| Priorità | Fix | Impatto Stimato | Complessità |
|----------|-----|-----------------|-------------|
| 1 | Fix circular dependency (rimuovi post-consensus market blend) | ★★★★★ | Bassa |
| 2 | Unifica H2H in singolo punto di iniezione | ★★★★ | Media |
| 3 | Live recalibration asimmetrica | ★★★★ | Media |
| 4 | xG pipeline budget di deviazione | ★★★ | Alta |
| 5 | O/U 2.5 consolidate post-processing | ★★☆ | Media |
| 6 | Draw shrinkage cap | ★★☆ | Bassa |
| 7 | BTTS boost time-modulated | ★☆☆ | Bassa |
