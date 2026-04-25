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

_LOG_W = logging.getLogger("exchange.weather")

# Mapping squadra → città per geolocalizzazione
# Aggiornare con le squadre più comuni
TEAM_CITY_MAP: dict[str, str] = {
    # Italia - Serie A
    "juventus": "Turin",
    "juve": "Turin",
    "torino": "Turin",
    "inter": "Milan",
    "milan": "Milan",
    "roma": "Rome",
    "lazio": "Rome",
    "napoli": "Naples",
    "fiorentina": "Florence",
    "atalanta": "Bergamo",
    "atalanta bergamo": "Bergamo",
    "lazio roma": "Rome",
    "as roma": "Rome",
    "ac milan": "Milan",
    "fc inter": "Milan",
    "ssc napoli": "Naples",
    "bologna": "Bologna",
    "bologna fc": "Bologna",
    "genoa": "Genoa",
    "sampdoria": "Genoa",
    "udinese": "Udine",
    "cagliari": "Cagliari",
    "verona": "Verona",
    "hellas verona": "Verona",
    "sassuolo": "Sassuolo",
    "lecce": "Lecce",
    "empoli": "Empoli",
    "monza": "Monza",
    "frosinone": "Frosinone",
    "salernitana": "Salerno",
    "cremonese": "Cremona",
    "spezia": "La Spezia",
    "venezia": "Venice",
    "parma": "Parma",
    "palermo": "Palermo",
    "brescia": "Brescia",
    "como": "Como",

    # Inghilterra - Premier League
    "manchester city": "Manchester",
    "man city": "Manchester",
    "manchester united": "Manchester",
    "man utd": "Manchester",
    "liverpool": "Liverpool",
    "chelsea": "London",
    "arsenal": "London",
    "tottenham": "London",
    "spurs": "London",
    "newcastle": "Newcastle upon Tyne",
    "brighton": "Brighton",
    "aston villa": "Birmingham",
    "west ham": "London",
    "everton": "Liverpool",
    "fulham": "London",
    "crystal palace": "London",
    "brentford": "London",
    "wolves": "Wolverhampton",
    "wolverhampton": "Wolverhampton",
    "bournemouth": "Bournemouth",
    "nottingham forest": "Nottingham",
    "leicester": "Leicester",
    "leeds": "Leeds",
    "southampton": "Southampton",
    "burnley": "Burnley",
    "luton": "Luton",
    "sheffield": "Sheffield",
    "ipswich": "Ipswich",

    # Spagna - La Liga
    "real madrid": "Madrid",
    "barcelona": "Barcelona",
    "barca": "Barcelona",
    "atletico madrid": "Madrid",
    "atletico": "Madrid",
    "athletico": "Madrid",
    "sevilla": "Seville",
    "seville": "Seville",
    "real betis": "Seville",
    "betis": "Seville",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
    "real sociedad": "San Sebastian",
    "sociedad": "San Sebastian",
    "athletic bilbao": "Bilbao",
    "athletic": "Bilbao",
    "getafe": "Getafe",
    "celta vigo": "Vigo",
    "celta": "Vigo",
    "osasuna": "Pamplona",
    "mallorca": "Palma de Mallorca",
    "rayo vallecano": "Madrid",
    "rayo": "Madrid",
    "cadiz": "Cadiz",
    "alaves": "Vitoria-Gasteiz",
    "girona": "Girona",
    "las palmas": "Las Palmas",
    "almeria": "Almeria",
    "granada": "Granada",

    # Germania - Bundesliga
    "bayern munich": "Munich",
    "bayern": "Munich",
    "borussia dortmund": "Dortmund",
    "dortmund": "Dortmund",
    "rb leipzig": "Leipzig",
    "leipzig": "Leipzig",
    "leverkusen": "Leverkusen",
    "bayer leverkusen": "Leverkusen",
    "stuttgart": "Stuttgart",
    "frankfurt": "Frankfurt",
    "eintracht frankfurt": "Frankfurt",
    "wolfsburg": "Wolfsburg",
    "freiburg": "Freiburg",
    "mainz": "Mainz",
    "borussia moenchengladbach": "Monchengladbach",
    "gladbach": "Monchengladbach",
    "hoffenheim": "Sinsheim",
    "werder bremen": "Bremen",
    "bremen": "Bremen",
    "augsburg": "Augsburg",
    "bochum": "Bochum",
    "heidenheim": "Heidenheim",
    "darmstadt": "Darmstadt",
    "koln": "Cologne",
    "cologne": "Cologne",
    "union berlin": "Berlin",
    "hertha": "Berlin",
    "hamburg": "Hamburg",
    "schalke": "Gelsenkirchen",

    # Francia - Ligue 1
    "psg": "Paris",
    "paris saint germain": "Paris",
    "paris": "Paris",
    "marseille": "Marseille",
    "lyon": "Lyon",
    "monaco": "Monaco",
    "lille": "Lille",
    "nice": "Nice",
    "lens": "Lens",
    "rennes": "Rennes",
    "montpellier": "Montpellier",
    "nantes": "Nantes",
    "reims": "Reims",
    "toulouse": "Toulouse",
    "strasbourg": "Strasbourg",
    "brest": "Brest",
    "metz": "Metz",
    "le havre": "Le Havre",
    "nancy": "Nancy",
    "bordeaux": "Bordeaux",
    "saint etienne": "Saint-Etienne",
    "auxerre": "Auxerre",
    "angers": "Angers",

    # Portogallo - Primeira Liga
    "benfica": "Lisbon",
    "porto": "Porto",
    "sporting": "Lisbon",
    "sporting lisbon": "Lisbon",
    "braga": "Braga",
    "guimaraes": "Guimaraes",
    "famalicao": "Famalicao",
    "rio ave": "Vila do Conde",
    "farense": "Faro",
    "arouca": "Arouca",
    "moreirense": "Moreira de Cónegos",
    "estoril": "Estoril",
    "boavista": "Porto",
    "casa pia": "Lisbon",
    "gil vicente": "Barcelos",
    "estrela": "Lisbon",
    "nacional": "Funchal",
    "aves": "Vila das Aves",

    # Olanda - Eredivisie
    "ajax": "Amsterdam",
    "psv": "Eindhoven",
    "feyenoord": "Rotterdam",
    "az alkmaar": "Alkmaar",
    "az": "Alkmaar",
    "twente": "Enschede",
    "utrecht": "Utrecht",
    "vitesse": "Arnhem",
    "groningen": "Groningen",
    "heerenveen": "Heerenveen",
    "go ahead eagles": "Deventer",
    "sparta rotterdam": "Rotterdam",
    "sparta": "Rotterdam",
    "fortuna sittard": "Sittard",
    "nec nijmegen": "Nijmegen",
    "nec": "Nijmegen",
    "willem ii": "Tilburg",
    "alkmaar": "Alkmaar",
    "heracles": "Almelo",
    "pec zwolle": "Zwolle",
    "rkc waalwijk": "Waalwijk",
    "excelsior": "Rotterdam",
    "camuur": "Alkmaar",

    # Belgio - Pro League
    "club brugge": "Bruges",
    "brugge": "Bruges",
    "anderlecht": "Brussels",
    "gent": "Ghent",
    "genk": "Genk",
    "antwerp": "Antwerp",
    "standard liege": "Liege",
    "charleroi": "Charleroi",
    "mechelen": "Mechelen",
    "gent": "Ghent",
    "kortrijk": "Kortrijk",
    "sint-truiden": "Sint-Truiden",
    "eupen": "Eupen",
    "cercle brugge": "Bruges",
    "oh leuven": "Leuven",
    "westerlo": "Westerlo",
    "beerschot": "Antwerp",
    "lommel": "Lommel",

    # Turchia - Super Lig
    "galatasaray": "Istanbul",
    "fenerbahce": "Istanbul",
    "besiktas": "Istanbul",
    "trabzonspor": "Trabzon",
    "istanbul basaksehir": "Istanbul",
    "basaksehir": "Istanbul",
    "antalyaspor": "Antalya",
    "konyaspor": "Konya",
    "kasimpasa": "Istanbul",
    "alanyaspor": "Alanya",
    "fatih karagumruk": "Istanbul",
    "sivasspor": "Sivas",
    "hatayspor": "Hatay",
    "gaziantep": "Gaziantep",
    "rizespor": "Rize",
    "pendikspor": "Istanbul",
    "samsunspor": "Samsun",
    "adana demirspor": "Adana",
    "bodrum": "Bodrum",
    "eyupspor": "Istanbul",
    "giresunspor": "Giresun",
    "umraniyespor": "Istanbul",
    "boluspor": "Bolu",

    # Grecia - Super League
    "olympiacos": "Piraeus",
    "panathinaikos": "Athens",
    "paok": "Thessaloniki",
    "aek athens": "Athens",
    "aek": "Athens",
    "panionios": "Athens",
    "aris": "Thessaloniki",
    "ofi": "Heraklion",
    "asteras tripolis": "Tripoli",
    "atromitos": "Athens",
    "volos": "Volos",
    "lamia": "Lamia",
    "panetolikos": "Agrinio",
    "levadeiakos": "Livadeia",
    "giannina": "Ioannina",

    # Argentina - Primera Division
    "boca juniors": "Buenos Aires",
    "boca": "Buenos Aires",
    "river plate": "Buenos Aires",
    "river": "Buenos Aires",
    "racing club": "Avellaneda",
    "racing": "Avellaneda",
    "independiente": "Avellaneda",
    "san lorenzo": "Buenos Aires",
    "huracan": "Buenos Aires",
    "velez sarsfield": "Buenos Aires",
    "velez": "Buenos Aires",
    "estudiantes": "La Plata",
    "gimnasia": "La Plata",
    "lanus": "Lanus",
    "tigre": "Victoria",
    "argentinos juniors": "Buenos Aires",
    "talleres": "Cordoba",
    "belgrano": "Cordoba",
    "rosario central": "Rosario",
    "newells": "Rosario",
    "newell's": "Rosario",
    "central cordoba": "Santiago del Estero",
    "godoy cruz": "Mendoza",
    "defensa y justicia": "Flores",
    "banfield": "Banfield",
    "colon": "Santa Fe",
    "atletico tucuman": "San Miguel de Tucuman",
    "tucuman": "San Miguel de Tucuman",
    "aldosivi": "Mar del Plata",
    "patronato": "Parana",
    "sarmiento": "Junin",
    "platenese": "Buenos Aires",
    "instituto": "Cordoba",
    "rifaela": "Rafaela",

    # Brasile - Brasileirao
    "flamengo": "Rio de Janeiro",
    "palmeiras": "Sao Paulo",
    "sao paulo": "Sao Paulo",
    "corinthians": "Sao Paulo",
    "santos": "Santos",
    "gremio": "Porto Alegre",
    "internacional": "Porto Alegre",
    "atletico mineiro": "Belo Horizonte",
    "atletico mg": "Belo Horizonte",
    "cruzeiro": "Belo Horizonte",
    "fluminense": "Rio de Janeiro",
    "botafogo": "Rio de Janeiro",
    "vasco da gama": "Rio de Janeiro",
    "vasco": "Rio de Janeiro",
    "bahia": "Salvador",
    "fortaleza": "Fortaleza",
    "ceara": "Fortaleza",
    "sport": "Recife",
    "athletico paranaense": "Curitiba",
    "atletico pr": "Curitiba",
    "coritiba": "Curitiba",
    "goias": "Goiania",
    "cuiaba": "Cuiaba",
    "juventude": "Caxias do Sul",
    "criciuma": "Criciuma",
    "vitoria": "Salvador",
    "bragantino": "Braganca Paulista",
    "red bull bragantino": "Braganca Paulista",
    "america mineiro": "Belo Horizonte",
    "america mg": "Belo Horizonte",
    "chapecoense": "Chapeco",
    "avai": "Florianopolis",
    "parana": "Curitiba",
    "figueirense": "Florianopolis",
    "operario": "Ponta Grossa",
    "ponte preta": "Campinas",
    "luverdense": "Lucas do Rio Verde",

    # USA - MLS
    "la galaxy": "Los Angeles",
    "los angeles fc": "Los Angeles",
    "lafc": "Los Angeles",
    "seattle sounders": "Seattle",
    "portland timbers": "Portland",
    "atlanta united": "Atlanta",
    "inter miami": "Miami",
    "new york city": "New York",
    "nycfc": "New York",
    "new york red bulls": "Harrison",
    "red bulls": "Harrison",
    "toronto fc": "Toronto",
    "toronto": "Toronto",
    "vancouver whitecaps": "Vancouver",
    "whitecaps": "Vancouver",
    "montreal": "Montreal",
    "cf montreal": "Montreal",
    "philadelphia union": "Chester",
    "philadelphia": "Chester",
    "columbus crew": "Columbus",
    "columbus": "Columbus",
    "fc dallas": "Dallas",
    "dallas": "Dallas",
    "houston dynamo": "Houston",
    "houston": "Houston",
    "sporting kansas city": "Kansas City",
    "sporting kc": "Kansas City",
    "real salt lake": "Sandy",
    "salt lake": "Sandy",
    "colorado rapids": "Commerce City",
    "colorado": "Commerce City",
    "minnesota united": "Saint Paul",
    "minnesota": "Saint Paul",
    "orlando city": "Orlando",
    "orlando": "Orlando",
    "new england revolution": "Foxborough",
    "new england": "Foxborough",
    "chicago fire": "Chicago",
    "chicago": "Chicago",
    "dc united": "Washington",
    "dc": "Washington",
    "san jose earthquakes": "San Jose",
    "san jose": "San Jose",
    "austin fc": "Austin",
    "austin": "Austin",
    "charlotte fc": "Charlotte",
    "charlotte": "Charlotte",
    "st. louis city": "St. Louis",
    "st louis": "St. Louis",
    "nashville": "Nashville",
    "fc cincinnati": "Cincinnati",
    "cincinnati": "Cincinnati",

    # Messico - Liga MX
    "club america": "Mexico City",
    "america": "Mexico City",
    "chivas": "Guadalajara",
    "guadalajara": "Guadalajara",
    "cruz azul": "Mexico City",
    "monterrey": "Monterrey",
    "tigres uanl": "Monterrey",
    "tigres": "Monterrey",
    "pumas": "Mexico City",
    "puebla": "Puebla",
    "leon": "Leon",
    "toluca": "Toluca",
    "santos laguna": "Torreon",
    "atlhetico san luis": "San Luis Potosi",
    "san luis": "San Luis Potosi",
    "queretaro": "Queretaro",
    "mazatlan": "Mazatlan",
    "fc juarez": "Juarez",
    "juarez": "Juarez",
    "necaxa": "Aguascalientes",
    "pachuca": "Pachuca",
    "tijuana": "Tijuana",
    "atlas": "Guadalajara",

    # Asia - J-League, K-League, etc.
    "urawa reds": "Saitama",
    "urawa": "Saitama",
    "kawasaki frontale": "Kawasaki",
    "kawasaki": "Kawasaki",
    "yokohama f. marinos": "Yokohama",
    "yokohama marinos": "Yokohama",
    "yokohama": "Yokohama",
    "kashima antlers": "Kashima",
    "kashima": "Kashima",
    "gamba osaka": "Osaka",
    "gamba": "Osaka",
    "cerezo osaka": "Osaka",
    "cerezo": "Osaka",
    "fc tokyo": "Tokyo",
    "tokyo": "Tokyo",
    "shimizu s-pulse": "Shimizu",
    "shimizu": "Shimizu",
    "nagoya grampus": "Nagoya",
    "nagoya": "Nagoya",
    "jeonbuk hyundai": "Jeonju",
    "jeonbuk": "Jeonju",
    "suwon samsung": "Suwon",
    "suwon": "Suwon",
    "seoul": "Seoul",
    "fc seoul": "Seoul",
    "ulsan hyundai": "Ulsan",
    "ulsan": "Ulsan",
    "pohang steelers": "Pohang",
    "pohang": "Pohang",
    "daegu": "Daegu",
    "incheon united": "Incheon",
    "incheon": "Incheon",
    "gangwon": "Chuncheon",
    "shanghai shenhua": "Shanghai",
    "shanghai": "Shanghai",
    "beijing guoan": "Beijing",
    "beijing": "Beijing",
    "guangzhou evergrande": "Guangzhou",
    "guangzhou": "Guangzhou",
    "shandong taishan": "Jinan",
    "shandong": "Jinan",
    "suwon fc": "Suwon",
    "daegu fc": "Daegu",
    "jeju united": "Jeju",
    "gwangju": "Gwangju",
    "gangwon fc": "Chuncheon",

    # Australia - A-League
    "sydney fc": "Sydney",
    "sydney": "Sydney",
    "melbourne city": "Melbourne",
    "melbourne victory": "Melbourne",
    "melbourne": "Melbourne",
    "western sydney": "Sydney",
    "wanderers": "Sydney",
    "adelaide united": "Adelaide",
    "adelaide": "Adelaide",
    "perth glory": "Perth",
    "perth": "Perth",
    "wellington phoenix": "Wellington",
    "wellington": "Wellington",
    "central coast": "Gosford",
    "mariners": "Gosford",
    "newcastle jets": "Newcastle",
    "newcastle australia": "Newcastle",
    "brisbane roar": "Brisbane",
    "brisbane": "Brisbane",
    "macarthur": "Sydney",
    "western united": "Melbourne",

    # Nazionali
    "italy": "Rome",
    "england": "London",
    "spain": "Madrid",
    "germany": "Berlin",
    "france": "Paris",
    "brazil": "Brasilia",
    "argentina": "Buenos Aires",
    "portugal": "Lisbon",
    "netherlands": "Amsterdam",
    "belgium": "Brussels",
    "turkey": "Ankara",
    "greece": "Athens",
    "usa": "Washington",
    "mexico": "Mexico City",
    "japan": "Tokyo",
    "south korea": "Seoul",
    "korea republic": "Seoul",
    "korea": "Seoul",
    "china": "Beijing",
    "australia": "Canberra",
    "colombia": "Bogota",
    "chile": "Santiago",
    "uruguay": "Montevideo",
    "peru": "Lima",
    "ecuador": "Quito",
    "venezuela": "Caracas",
    "paraguay": "Asuncion",
    "bolivia": "La Paz",
    "costa rica": "San Jose",
    "panama": "Panama City",
    "honduras": "Tegucigalpa",
    "jamaica": "Kingston",
    "trinidad": "Port of Spain",
    "haiti": "Port-au-Prince",
    "canada": "Ottawa",
    "wales": "Cardiff",
    "scotland": "Glasgow",
    "ireland": "Dublin",
    "northern ireland": "Belfast",
    "austria": "Vienna",
    "switzerland": "Bern",
    "poland": "Warsaw",
    "czech republic": "Prague",
    "czechia": "Prague",
    "hungary": "Budapest",
    "romania": "Bucharest",
    "serbia": "Belgrade",
    "croatia": "Zagreb",
    "slovenia": "Ljubljana",
    "slovakia": "Bratislava",
    "ukraine": "Kyiv",
    "russia": "Moscow",
    "sweden": "Stockholm",
    "norway": "Oslo",
    "denmark": "Copenhagen",
    "finland": "Helsinki",
    "iceland": "Reykjavik",
    "israel": "Tel Aviv",
    "egypt": "Cairo",
    "morocco": "Rabat",
    "algeria": "Algiers",
    "tunisia": "Tunis",
    "senegal": "Dakar",
    "nigeria": "Abuja",
    "ghana": "Accra",
    "cameroon": "Yaounde",
    "ivory coast": "Yamoussoukro",
    "cote d'ivoire": "Yamoussoukro",
    "mali": "Bamako",
    "congo": "Kinshasa",
    "south africa": "Pretoria",
    "saudi arabia": "Riyadh",
    "iran": "Tehran",
    "iraq": "Baghdad",
    "united arab emirates": "Abu Dhabi",
    "uae": "Abu Dhabi",
    "qatar": "Doha",
    "uzbekistan": "Tashkent",
    "thailand": "Bangkok",
    "vietnam": "Hanoi",
    "indonesia": "Jakarta",
    "malaysia": "Kuala Lumpur",
    "singapore": "Singapore",
    "philippines": "Manila",
    "india": "New Delhi",

    # Nordici
    "malta": "Valletta",
    "cyprus": "Nicosia",
    "luxembourg": "Luxembourg",
    "andorra": "Andorra la Vella",
    "liechtenstein": "Vaduz",
    "san marino": "San Marino",
    "gibraltar": "Gibraltar",
    "faroe islands": "Torshavn",
    "estonia": "Tallinn",
    "latvia": "Riga",
    "lithuania": "Vilnius",
    "belarus": "Minsk",
    "moldova": "Chisinau",
    "georgia": "Tbilisi",
    "armenia": "Yerevan",
    "azerbaijan": "Baku",
    "kazakhstan": "Astana",
    "kyrgyzstan": "Bishkek",
    "tajikistan": "Dushanbe",
    "turkmenistan": "Ashgabat",
    "bosnia": "Sarajevo",
    "bosnia and herzegovina": "Sarajevo",
    "montenegro": "Podgorica",
    "north macedonia": "Skopje",
    "macedonia": "Skopje",
    "albania": "Tirana",
    "bulgaria": "Sofia",
    "macedonia fyr": "Skopje",
    "north macedonia": "Skopje",
    "gibraltar": "Gibraltar",

    # Africa
    "guinea": "Conakry",
    "burkina faso": "Ouagadougou",
    "sierra leone": "Freetown",
    "liberia": "Monrovia",
    "benin": "Porto-Novo",
    "togo": "Lome",
    "gabon": "Libreville",
    "congo dr": "Kinshasa",
    "democratic congo": "Kinshasa",
    "zambia": "Lusaka",
    "zimbabwe": "Harare",
    "kenya": "Nairobi",
    "uganda": "Kampala",
    "ethiopia": "Addis Ababa",
    "sudan": "Khartoum",
    "libya": "Tripoli",
    "angola": "Luanda",
    "mozambique": "Maputo",
    "tanzania": "Dodoma",
    "rwanda": "Kigali",
    "burundi": "Bujumbura",
    "malawi": "Lilongwe",
    "botswana": "Gaborone",
    "namibia": "Windhoek",
    "madagascar": "Antananarivo",
    "mauritania": "Nouakchott",
    "niger": "Niamey",
    "chad": "N'Djamena",
    "gambia": "Banjul",
    "guinea-bissau": "Bissau",
    "equatorial guinea": "Malabo",
    "central african republic": "Bangui",
    "south sudan": "Juba",
    "lesotho": "Maseru",
    "eswatini": "Mbabane",
    "swaziland": "Mbabane",
    "comoros": "Moroni",
    "cape verde": "Praia",
    "cabo verde": "Praia",
    "mauritius": "Port Louis",
    "seychelles": "Victoria",

    # Centro America
    "guatemala": "Guatemala City",
    "el salvador": "San Salvador",
    "nicaragua": "Managua",
    "belize": "Belmopan",

    # Caraibi
    "cuba": "Havana",
    "dominican republic": "Santo Domingo",
    "puerto rico": "San Juan",
    "barbados": "Bridgetown",
    "bahamas": "Nassau",
    "trinidad and tobago": "Port of Spain",
    "suriname": "Paramaribo",
    "guyana": "Georgetown",
    "curacao": "Willemstad",
    "martinique": "Fort-de-France",
    "guadeloupe": "Basse-Terre",
    "french guiana": "Cayenne",

    # Oceania
    "fiji": "Suva",
    "new caledonia": "Noumea",
    "papua new guinea": "Port Moresby",
    "solomon islands": "Honiara",
    "vanuatu": "Port Vila",
    "samoa": "Apia",
    "tonga": "Nuku'alofa",
    "cook islands": "Avarua",
    "tahiti": "Papeete",
}


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
