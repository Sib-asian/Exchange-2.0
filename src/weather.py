"""
weather.py — Integrazione OpenWeather API per meteo partite.

Usa OpenWeatherMap API per ottenere condizioni meteo in tempo reale
per la città dove si gioca la partita.

L'impatto del meteo sulle probabilità di gol:
- Pioggia: campo scivoloso, palla pesante → -5% xG
- Pioggia forte/tempesta: condizioni difficili → -10% xG
- Vento forte (>10 m/s): traiettorie imprevedibili → -3% xG
- Neve: campo pesante → -7% xG
- Caldo estremo (>30°C): fatica → -3% xG
- Freddo estremo (<0°C): rigidità muscolare → -2% xG

Configurazione:
  - OPENWEATHER_API_KEY: via .env o variabile d'ambiente
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_LOG_W = logging.getLogger("exchange.weather")

# Mapping squadra → città per geolocalizzazione
# Caricato da file JSON esterno per manutenibilità.
# Il file si trova in data/team_city_map.json
import json as _json
_TEAM_CITY_DATA_PATH = Path(__file__).parent.parent / "data" / "team_city_map.json"

def _load_team_city_map() -> dict[str, str]:
    """Carica il mapping squadra→città dal file JSON esterno."""
    try:
        if _TEAM_CITY_DATA_PATH.exists():
            with open(_TEAM_CITY_DATA_PATH, encoding="utf-8") as _f:
                return _json.load(_f)
    except (json.JSONDecodeError, OSError) as _e:
        _LOG_W.warning("Failed to load team_city_map.json: %s", _e)
    return {}

TEAM_CITY_MAP: dict[str, str] = _load_team_city_map()


@dataclass
class WeatherData:
    """Dati meteo per una partita."""

    condition: str = ""         # es. "Clear", "Rain", "Clouds"
    description: str = ""       # es. "light rain", "overcast clouds"
    temp_celsius: int = 0       # temperatura in °C
    feels_like: int = 0         # temperatura percepita
    humidity: int = 0           # umidità %
    wind_speed: float = 0.0     # velocità vento m/s
    wind_direction: int = 0     # direzione vento gradi
    pressure: int = 0           # pressione hPa
    visibility: int = 0         # visibilità metri
    clouds: int = 0             # copertura nuvolosa %
    rain_1h: float = 0.0        # pioggia ultima ora mm
    snow_1h: float = 0.0        # neve ultima ora mm

    # Impatto calcolato sul xG
    xg_impact: float = 0.0      # -0.05 = -5% xG, +0.02 = +2% xG
    impact_reason: str = ""     # motivazione dell'impatto

    # Metadati estrazione
    city_used: str = ""         # città usata per la query
    extraction_success: bool = False
    error_message: str = ""


def _get_openweather_api_key() -> str | None:
    """Ottiene la API key OpenWeather da environment, .env file o Streamlit secrets."""
    # 1. Prima prova dalla variabile d'ambiente
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if api_key:
        return api_key

    # 2. Prova a caricare dal file .env nella directory corrente
    try:
        # Cerca il file .env in diverse posizioni
        env_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
            os.path.join(os.path.dirname(__file__), ".env"),
        ]
        for env_path in env_paths:
            if os.path.isfile(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENWEATHER_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if key:
                                return key
    except Exception as _wke:
        _LOG_W.debug("OpenWeather .env file not found or unreadable: %s", _wke)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENWEATHER_API_KEY" in st.secrets:
            return st.secrets["OPENWEATHER_API_KEY"]
    except ImportError:
        pass

    return None


def get_city_for_team(team_name: str) -> str | None:
    """
    Trova la città associata a una squadra.
    
    Args:
        team_name: Nome della squadra (case-insensitive)
        
    Returns:
        Nome della città o None se non trovato
    """
    if not team_name:
        return None

    # Normalizza il nome
    team_lower = team_name.lower().strip()

    # Cerca nel mapping
    city = TEAM_CITY_MAP.get(team_lower)
    if city:
        return city

    # Prova con partial match
    for team_key, city_name in TEAM_CITY_MAP.items():
        if team_key in team_lower or team_lower in team_key:
            return city_name

    # Se il nome della squadra è già una città, usalo
    # Rimuovi suffissi comuni
    for suffix in [" fc", " cf", " ac", " sc", " fc.", " afc", " club", " united", " city",
                   " rovers", " town", " wanderers", " athletic", " dynamo", " spartak",
                   " sporting", " real", " atletico", " athletico", " deportivo"]:
        if team_lower.endswith(suffix):
            potential_city = team_lower[:-len(suffix)].strip()
            if potential_city in TEAM_CITY_MAP:
                return TEAM_CITY_MAP[potential_city]
            # Prova a capitalizzare
            return potential_city.title()

    # Fallback: usa il nome come città
    # Ma solo se sembra un nome di città (non troppo lungo)
    if len(team_lower) < 20 and " " not in team_lower:
        return team_name.title()

    return None


def get_weather_for_city(city: str, api_key: str | None = None) -> WeatherData:
    """
    Ottiene il meteo attuale per una città usando OpenWeather API.
    
    Args:
        city: Nome della città
        api_key: API key OpenWeather (se None, usa variabile d'ambiente)
        
    Returns:
        WeatherData con condizioni meteo e impatto xG
    """
    if api_key is None:
        api_key = _get_openweather_api_key()

    if not api_key:
        return WeatherData(
            extraction_success=False,
            error_message="OpenWeather API key non configurata",
        )

    if not city:
        return WeatherData(
            extraction_success=False,
            error_message="Città non specificata",
        )

    # URL API OpenWeather
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    url = f"{base_url}?q={city}&appid={api_key}&units=metric"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return WeatherData(extraction_success=False, error_message="API key non valida")
        elif e.code == 404:
            return WeatherData(extraction_success=False, error_message=f"Città '{city}' non trovata")
        else:
            return WeatherData(extraction_success=False, error_message=f"HTTP error {e.code}")
    except Exception as e:
        return WeatherData(extraction_success=False, error_message=str(e))

    # Parsing risposta
    try:
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        visibility = data.get("visibility", 0)
        rain = data.get("rain", {})
        snow = data.get("snow", {})

        condition = weather.get("main", "")
        description = weather.get("description", "")
        temp = int(main.get("temp", 0))
        feels_like = int(main.get("feels_like", 0))
        humidity = int(main.get("humidity", 0))
        pressure = int(main.get("pressure", 0))
        wind_speed = float(wind.get("speed", 0))
        wind_direction = int(wind.get("deg", 0))
        cloud_cover = int(clouds.get("all", 0))
        rain_1h = float(rain.get("1h", 0))
        snow_1h = float(snow.get("1h", 0))

        # Calcola impatto xG
        xg_impact = 0.0
        impact_reasons = []

        # Pioggia
        if condition == "Rain":
            if "heavy" in description.lower() or "storm" in description.lower():
                xg_impact -= 0.10
                impact_reasons.append("pioggia forte (-10%)")
            else:
                xg_impact -= 0.05
                impact_reasons.append("pioggia (-5%)")
        elif rain_1h > 0:
            xg_impact -= 0.03
            impact_reasons.append("pioggia leggera (-3%)")

        # Neve
        if condition == "Snow":
            if "heavy" in description.lower():
                xg_impact -= 0.12
                impact_reasons.append("neve abbondante (-12%)")
            else:
                xg_impact -= 0.07
                impact_reasons.append("neve (-7%)")
        elif snow_1h > 0:
            xg_impact -= 0.05
            impact_reasons.append("nevischio (-5%)")

        # Temporali
        if condition == "Thunderstorm":
            xg_impact -= 0.08
            impact_reasons.append("temporale (-8%)")

        # Vento forte
        if wind_speed > 15:  # > 15 m/s = ~54 km/h
            xg_impact -= 0.05
            impact_reasons.append("vento molto forte (-5%)")
        elif wind_speed > 10:  # > 10 m/s = ~36 km/h
            xg_impact -= 0.03
            impact_reasons.append("vento forte (-3%)")

        # Temperature estreme
        if temp > 35:
            xg_impact -= 0.05
            impact_reasons.append("caldo estremo (-5%)")
        elif temp > 30:
            xg_impact -= 0.03
            impact_reasons.append("caldo (-3%)")
        elif temp < -5:
            xg_impact -= 0.04
            impact_reasons.append("freddo estremo (-4%)")
        elif temp < 0:
            xg_impact -= 0.02
            impact_reasons.append("freddo (-2%)")

        # Nebbia
        if condition == "Fog" or condition == "Mist":
            if visibility < 1000:
                xg_impact -= 0.03
                impact_reasons.append("nebbia fitta (-3%)")

        # Cap impatto massimo
        xg_impact = max(-0.15, min(0.02, xg_impact))

        return WeatherData(
            condition=condition,
            description=description,
            temp_celsius=temp,
            feels_like=feels_like,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            pressure=pressure,
            visibility=visibility,
            clouds=cloud_cover,
            rain_1h=rain_1h,
            snow_1h=snow_1h,
            xg_impact=xg_impact,
            impact_reason=", ".join(impact_reasons) if impact_reasons else "condizioni normali",
            city_used=city,
            extraction_success=True,
        )

    except Exception as e:
        return WeatherData(
            extraction_success=False,
            error_message=f"Errore parsing risposta: {e}",
        )


def get_weather_for_match(
    home_team: str,
    away_team: str | None = None,
    league: str | None = None,
) -> WeatherData:
    """
    Ottiene il meteo per una partita.
    
    Prova prima con la squadra di casa, poi con la trasferta.
    
    Args:
        home_team: Nome squadra casa
        away_team: Nome squadra trasferta (opzionale)
        league: Nome lega (opzionale, per log futura)
        
    Returns:
        WeatherData con condizioni meteo
    """
    # Prova prima con la squadra di casa
    city = get_city_for_team(home_team)
    if city:
        weather = get_weather_for_city(city)
        if weather.extraction_success:
            return weather

    # Fallback: prova con la trasferta
    if away_team:
        city = get_city_for_team(away_team)
        if city:
            weather = get_weather_for_city(city)
            if weather.extraction_success:
                return weather

    # Se nessuna città trovata
    return WeatherData(
        extraction_success=False,
        error_message=f"Città non trovata per {home_team}" + (f" o {away_team}" if away_team else ""),
    )
