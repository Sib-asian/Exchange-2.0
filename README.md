# Exchange 2.0 — Radar Pro Live

> Motore probabilistico per previsioni calcistiche con analisi multi-modello e calibrazione Bayesiana.

## Panoramica

Exchange 2.0 "Radar Pro Live" è un engine di analisi probabilistica per il calcio, progettato per generare previsioni accurate su mercati 1X2, Over/Under, BTTS, Asian Handicap e Correct Score. Utilizza 3 modelli matematici indipendenti con consensus pesato e una pipeline di calibrazione sofisticata.

## Caratteristiche Principali

- **3 Modelli Matematici** — Poisson bivariata + Dixon-Coles, Copula Frank + CMP, Markov chain
- **Consensus Multi-Modello** — Media pesata adattiva con credible intervals
- **Kelly Criterion Frazionario** — Gestione del bankroll con riduzioni conservative
- **Estrazione OCR da NowGoal** — URL parsing + Gemini Vision per dati prematch
- **Calibrazione Storica per Lega** — Platt scaling + parameter learning da log
- **Quality Firewall** — Blocco segnali quando il modello non è affidabile
- **Analisi Meteo** — Integrazione OpenWeather con impatto xG
- **AI Research** — Gemini + Google Search per assenze, formazione, notizie

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Linguaggio | Python ≥ 3.11 |
| UI | Streamlit ≥ 1.35.0 |
| AI/Vision | Google Gemini API, OpenAI SDK |
| Meteo | OpenWeather API |
| Test | pytest + pytest-cov |
| CI/CD | GitHub Actions (Python 3.11/3.12) |

## Installazione

### Prerequisiti
- Python 3.11 o superiore
- API key Google Gemini (opzionale, per OCR avanzato)
- API key OpenAI (opzionale, come fallback OCR)

### Setup

```bash
# Clona il repository
git clone https://github.com/Sib-asian/Exchange-2.0.git
cd Exchange-2.0

# Crea ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

# Configura API key (opzionale)
# Crea .streamlit/secrets.toml:
# GEMINI_API_KEY = "your-key-here"
# OPENAI_API_KEY = "your-key-here"
# OPENWEATHER_API_KEY = "your-key-here"

# Avvia l'applicazione
streamlit run app.py
```

## Struttura Progetto

```
Exchange-2.0/
├── app.py                      # App Streamlit principale
├── src/
│   ├── engine.py               # Motore di analisi (orchestratore)
│   ├── types.py                # Dataclass centralizzate
│   ├── pipeline.py             # Pipeline analisi + calibrazione
│   ├── config.py               # Configurazione centralizzata
│   ├── ocr.py                  # Estrazione dati OCR/URL
│   ├── prematch_app_bridge.py  # Bridge OCR → MatchState
│   ├── signals.py              # Generazione segnali betting
│   ├── weather.py              # Integrazione meteo
│   ├── research.py             # Ricerca AI pre-partita
│   ├── models/                 # Modelli matematici
│   │   ├── poisson.py          # Poisson bivariata + Dixon-Coles
│   │   ├── copula.py           # Copula Frank + CMP
│   │   ├── markov.py           # Catena di Markov
│   │   ├── calibration.py      # Calibrazione xG + devigging
│   │   ├── consensus.py        # Consenso multi-modello
│   │   ├── kelly.py            # Kelly criterion frazionato
│   │   └── ...                 # 25+ modelli aggiuntivi
│   ├── markets/                # Calcolo mercati
│   ├── tracking/               # Tracking previsioni
│   ├── ui/                     # Interfaccia Streamlit
│   └── utils/                  # Utilità (rate limiter, etc.)
├── data/
│   ├── predictions.json        # Log previsioni
│   └── team_city_map.json      # Mapping squadre→città
├── tests/                      # Suite test (~7000 righe)
└── pyproject.toml              # Build system
```

## Pipeline di Analisi

```
Input (linee manuali / URL Nowgoal / screenshot)
    ↓
OCR + Gemini Vision → PrematchAnalysisExtracted
    ↓
build_match_state() → MatchState
    ↓
analizza() → 3 modelli paralleli (Poisson, Copula, Markov)
    ↓
Consensus pesato → ProbabilitaModello
    ↓
Calibrazione storica lega + Platt scaling
    ↓
Segnali betting + Kelly criterion
    ↓
Output: previsioni, stake, quality score
```

## Modelli Matematici

### 1. Poisson Bivariata + Dixon-Coles
Modello standard per il calcio con correzione di Dixon-Coles per bassi punteggi (0-0, 1-0, 0-1, 1-1). Parametrizzato dalle linee AH/Total tramite calibrazione Bayesiana.

### 2. Copula Frank + Conway-Maxwell-Poisson
Cattura la dipendenza tra gol casa/trasferta tramite copula di Frank. CMP gestisce l'overdispersione (sotto-Poisson per leghe equilibrate, sopra-Poisson per leghe offensive).

### 3. Markov Chain (Score-State)
Modello di transizione tra stati di punteggio. Cattura effetti di momentum e dipendenza temporale che i modelli Poisson non possono modellare.

## Devigging

Il sistema utilizza due metodi di rimozione del vig (overround):
- **Shin's Power Method** (primario) — ridistribuisce il vig non uniformemente
- **Normalizzazione Proporzionale** (fallback)

## Testing

```bash
# Esegui tutti i test
pytest

# Con coverage
pytest --cov=src --cov-report=term-missing

# Solo test engine
pytest tests/test_engine.py -v

# Solo test OCR
pytest tests/test_ocr.py -v
```

## Configurazione

Tutti i parametri sono centralizzati in `src/config.py` tramite dataclass frozen. I principali gruppi di configurazione:
- `BAYES` — Parametri calibrazione Bayesiana
- `CONSENSUS` — Pesi e soglie consenso
- `PRECISION` — Quality firewall e soglie operative
- `DECAY` — Time decay e effetti live
- `ENGINE` — Configurazione motore generale

## Licenza

Vedere il file LICENSE per i dettagli.
