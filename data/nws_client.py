"""
NWS weather client — api.weather.gov (free, no key, US-only).

Flow per request:
  1. (cached forever) GET /points/{lat},{lon} → forecast grid pointer
  2. GET the forecastHourly URL → list of hourly periods
  3. Pick the period whose startTime spans the game's first pitch

Indoor parks short-circuit to a constant (no HTTP call). NWS coverage gaps:
TOR (Rogers Centre) is in Canada — also short-circuited as indoor.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import aiohttp


NWS_BASE = "https://api.weather.gov"
USER_AGENT = "(mlb-v3 baseball analytics, rodkazazi@gmail.com)"


# Roof-closed / indoor parks — never call NWS for these.
INDOOR_VENUES = {
    "TB", "HOU", "MIL", "AZ", "MIA", "TOR",
    # Mariners (T-Mobile Park) and Rangers (Globe Life Field) have retractable
    # roofs but are usually open; treat as outdoor.
}

INDOOR_DEFAULT = {
    "temperature": 72,
    "wind_speed": 0,
    "wind_direction": "indoor",
    "humidity": 45,
    "precip_pct": 0.0,
    "indoor": True,
}

OUTDOOR_FALLBACK = {
    "temperature": 70,
    "wind_speed": 5,
    "wind_direction": "out to CF",
    "humidity": 55,
    "precip_pct": 0.05,
    "indoor": False,
}

# (latitude, longitude) of each home stadium. Sourced from public records.
STADIUM_LATLON = {
    "ATL": (33.8907, -84.4677),    # Truist Park
    "ATH": (37.7515, -122.2005),   # Oakland Coliseum (still 2024-2026)
    "BAL": (39.2840, -76.6217),    # Camden Yards
    "BOS": (42.3467, -71.0972),    # Fenway
    "CHC": (41.9484, -87.6553),    # Wrigley
    "CIN": (39.0975, -84.5063),    # Great American
    "CLE": (41.4962, -81.6852),    # Progressive Field
    "COL": (39.7561, -104.9942),   # Coors
    "CWS": (41.8299, -87.6338),    # Rate Field
    "DET": (42.3390, -83.0485),    # Comerica
    "KC":  (39.0517, -94.4803),    # Kauffman
    "LAA": (33.8003, -117.8827),   # Angel Stadium
    "LAD": (34.0739, -118.2400),   # Dodger Stadium
    "MIN": (44.9817, -93.2776),    # Target Field
    "NYM": (40.7571, -73.8458),    # Citi Field
    "NYY": (40.8296, -73.9262),    # Yankee Stadium
    "PHI": (39.9061, -75.1665),    # Citizens Bank
    "PIT": (40.4469, -80.0057),    # PNC Park
    "SD":  (32.7073, -117.1566),   # Petco
    "SEA": (47.5914, -122.3325),   # T-Mobile (open most of year)
    "SF":  (37.7786, -122.3893),   # Oracle Park
    "STL": (38.6226, -90.1928),    # Busch
    "TEX": (32.7475, -97.0826),    # Globe Life (open most of year)
    "WSH": (38.8730, -77.0074),    # Nationals Park
    # Indoor — listed for completeness but never queried
    "TB":  (27.7682, -82.6534),    # Tropicana
    "HOU": (29.7572, -95.3555),    # Minute Maid
    "MIL": (43.0280, -87.9712),    # American Family Field
    "AZ":  (33.4453, -112.0667),   # Chase Field
    "MIA": (25.7781, -80.2196),    # loanDepot
    "TOR": (43.6414, -79.3894),    # Rogers Centre (Canada — indoor in NWS terms)
}


def _parse_wind_speed(s: str) -> int:
    """Convert NWS '8 mph' or '5 to 10 mph' to integer mph (high end)."""
    if not s:
        return 0
    parts = s.split()
    nums = [p for p in parts if p.replace(".", "").isdigit()]
    if not nums:
        return 0
    return int(float(nums[-1]))


def _pick_period_for(periods: list, game_dt: datetime) -> Optional[Dict]:
    """Return the hourly period containing game_dt, or the closest if none."""
    best = None
    best_delta = None
    for p in periods:
        try:
            start = datetime.fromisoformat(p["startTime"])
        except (KeyError, ValueError):
            continue
        # Strip tz for comparison if needed
        start_naive = start.replace(tzinfo=None)
        delta = abs((start_naive - game_dt.replace(tzinfo=None)).total_seconds())
        if best_delta is None or delta < best_delta:
            best = p
            best_delta = delta
    return best


class NWSClient:
    """Async NWS client with one cached `/points` lookup per stadium."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _grid_cache: Dict[str, str] = {}  # team_abbr → forecastHourly URL

    def __init__(self, cache=None):
        self.cache = cache
        self._disk = Path("cache")
        self._disk.mkdir(parents=True, exist_ok=True)
        self._grid_file = self._disk / "nws_grid.json"
        if self._grid_file.exists() and not type(self)._grid_cache:
            try:
                type(self)._grid_cache = json.loads(self._grid_file.read_text())
            except Exception:
                pass

    @classmethod
    async def _session(cls) -> aiohttp.ClientSession:
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": USER_AGENT,
                         "Accept": "application/geo+json"},
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def _get_hourly_url(self, team: str) -> Optional[str]:
        """Return the cached forecastHourly URL for a team's stadium."""
        if team in type(self)._grid_cache:
            return type(self)._grid_cache[team]

        coords = STADIUM_LATLON.get(team)
        if not coords:
            return None

        session = await self._session()
        url = f"{NWS_BASE}/points/{coords[0]},{coords[1]}"
        async with session.get(url) as r:
            if r.status != 200:
                return None
            data = await r.json()

        hourly = data.get("properties", {}).get("forecastHourly")
        if hourly:
            type(self)._grid_cache[team] = hourly
            self._grid_file.write_text(json.dumps(type(self)._grid_cache))
        return hourly

    async def get_forecast(self, team: str,
                           game_dt_iso: Optional[str] = None) -> Dict:
        """
        Return forecast at game time for a home team's stadium. Indoor parks
        return INDOOR_DEFAULT without an HTTP call. On any failure or if the
        forecast horizon is past (e.g. backtesting old games), returns
        OUTDOOR_FALLBACK so downstream agents always see a numeric dict.
        """
        if team in INDOOR_VENUES:
            return dict(INDOOR_DEFAULT)

        hourly_url = await self._get_hourly_url(team)
        if not hourly_url:
            return dict(OUTDOOR_FALLBACK)

        try:
            session = await self._session()
            async with session.get(hourly_url) as r:
                if r.status != 200:
                    return dict(OUTDOOR_FALLBACK)
                feed = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return dict(OUTDOOR_FALLBACK)

        periods = feed.get("properties", {}).get("periods", [])
        if not periods:
            return dict(OUTDOOR_FALLBACK)

        # Resolve game datetime — default to "next 19:00 local" if not given
        if game_dt_iso:
            try:
                game_dt = datetime.fromisoformat(game_dt_iso.replace("Z", "+00:00"))
            except ValueError:
                game_dt = datetime.fromisoformat(periods[0]["startTime"])
        else:
            game_dt = datetime.fromisoformat(periods[0]["startTime"])

        period = _pick_period_for(periods, game_dt)
        if not period:
            return dict(OUTDOOR_FALLBACK)

        precip_val = (period.get("probabilityOfPrecipitation") or {}).get("value")
        return {
            "temperature": period.get("temperature", 70),
            "wind_speed": _parse_wind_speed(period.get("windSpeed", "")),
            "wind_direction": period.get("windDirection", "N/A"),
            "humidity": (period.get("relativeHumidity") or {}).get("value", 55),
            "precip_pct": (precip_val / 100) if precip_val is not None else 0.0,
            "short_forecast": period.get("shortForecast", ""),
            "indoor": False,
        }
