"""
Agent 5: Park & Weather Agent
Real-time weather, park factors, dome detection,
wind direction/speed impact on Over/Under
"""

import asyncio
from typing import Dict
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("park_weather_agent")

# Park factor database — confirmed v2.4 values
PARK_FACTORS = {
    # AL West
    "SEA": {"runs": 0.85, "hr": 0.90, "name": "T-Mobile Park", "dome": True},
    "LAA": {"runs": 0.95, "hr": 0.92, "name": "Angel Stadium", "dome": False},
    "HOU": {"runs": 0.97, "hr": 0.95, "name": "Minute Maid Park", "dome": True},
    "OAK": {"runs": 0.93, "hr": 0.88, "name": "Sutter Health Park", "dome": False},
    "ATH": {"runs": 0.93, "hr": 0.88, "name": "Sutter Health Park", "dome": False},
    "TEX": {"runs": 1.02, "hr": 1.05, "name": "Globe Life Field", "dome": True},

    # AL Central
    "CLE": {"runs": 0.97, "hr": 0.95, "name": "Progressive Field", "dome": False},
    "KC":  {"runs": 1.01, "hr": 1.00, "name": "Kauffman Stadium", "dome": False},
    "DET": {"runs": 0.97, "hr": 0.93, "name": "Comerica Park", "dome": False},
    "CWS": {"runs": 0.98, "hr": 1.00, "name": "Guaranteed Rate Field", "dome": False},
    "MIN": {"runs": 0.96, "hr": 0.94, "name": "Target Field", "dome": False},

    # AL East
    "NYY": {"runs": 0.99, "hr": 1.16, "name": "Yankee Stadium", "dome": False},
    "BOS": {"runs": 1.04, "hr": 1.08, "name": "Fenway Park", "dome": False},
    "TOR": {"runs": 1.02, "hr": 1.03, "name": "Rogers Centre", "dome": True},
    "BAL": {"runs": 1.01, "hr": 1.02, "name": "Camden Yards", "dome": False},
    "TB":  {"runs": 0.94, "hr": 0.96, "name": "Tropicana Field", "dome": True},

    # NL West
    "LAD": {"runs": 0.96, "hr": 0.92, "name": "Dodger Stadium", "dome": False},
    "SF":  {"runs": 0.93, "hr": 0.88, "name": "Oracle Park", "dome": False},
    "SD":  {"runs": 0.93, "hr": 0.91, "name": "Petco Park", "dome": False},
    "AZ":  {"runs": 1.01, "hr": 1.03, "name": "Chase Field", "dome": True},
    "COL": {"runs": 1.16, "hr": 1.28, "name": "Coors Field", "dome": False},

    # NL Central
    "CHC": {"runs": 0.99, "hr": 1.01, "name": "Wrigley Field", "dome": False},
    "MIL": {"runs": 1.02, "hr": 1.04, "name": "American Family Field", "dome": True},
    "STL": {"runs": 0.99, "hr": 0.97, "name": "Busch Stadium", "dome": False},
    "CIN": {"runs": 1.10, "hr": 1.18, "name": "Great American Ball Park", "dome": False},
    "PIT": {"runs": 0.96, "hr": 0.93, "name": "PNC Park", "dome": False},

    # NL East
    "ATL": {"runs": 1.01, "hr": 1.03, "name": "Truist Park", "dome": False},
    "NYM": {"runs": 0.96, "hr": 0.92, "name": "Citi Field", "dome": False},
    "PHI": {"runs": 1.04, "hr": 1.09, "name": "Citizens Bank Park", "dome": False},
    "MIA": {"runs": 0.94, "hr": 0.90, "name": "loanDepot Park", "dome": True},
    "WSH": {"runs": 1.00, "hr": 1.00, "name": "Nationals Park", "dome": False},
}

# Coors special rules
COORS_RISP_GATE = 0.28  # Both teams must have RISP > 28% to activate Over rule


class ParkWeatherAgent:
    """Fetches real-time weather and calculates park-adjusted environment"""

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        home_team = game_data.get("home_team")
        date = game_data.get("date")

        # Get park data
        park = PARK_FACTORS.get(home_team, {"runs": 1.00, "hr": 1.00,
                                              "name": "Unknown Park", "dome": False})

        # Get weather (skip for dome games)
        weather = {}
        if not park.get("dome"):
            try:
                weather = await self.mlb_api.get_weather(home_team, date)
            except Exception as e:
                logger.warning(f"Weather fetch failed: {e}")
                weather = {}

        # Calculate environment
        environment = self._calculate_environment(park, weather)

        # Coors special handling
        coors_active = home_team == "COL"
        coors_rules = self._apply_coors_rules(game_data) if coors_active else {}

        return {
            "park": park,
            "weather": weather,
            "environment": environment,
            "coors_active": coors_active,
            "coors_rules": coors_rules,
            "rain_concern": weather.get("precip_pct", 0) >= 0.50,
            "rain_hold": weather.get("precip_pct", 0) >= 0.30
        }

    def _calculate_environment(self, park: Dict, weather: Dict) -> Dict:
        """Calculate combined park + weather scoring environment"""
        base_pf = park.get("runs", 1.00)
        hr_pf = park.get("hr", 1.00)
        dome = park.get("dome", False)

        # Weather adjustments (only for outdoor parks)
        wind_adj = 0.0
        temp_adj = 0.0
        total_adj = 0.0

        if not dome and weather:
            # Wind direction impact
            wind_speed = weather.get("wind_speed", 0)
            wind_dir = weather.get("wind_direction", "").lower()
            temp = weather.get("temperature", 70)

            # Wind in = suppressive, wind out = Over lean
            if "out" in wind_dir:
                wind_adj = min(wind_speed * 0.008, 0.12)  # Max +12% at gale force
            elif "in" in wind_dir:
                wind_adj = -min(wind_speed * 0.008, 0.10)  # Max -10% suppression

            # Temperature impact
            if temp <= 45:
                temp_adj = -0.08  # Cold = Under lean
            elif temp <= 55:
                temp_adj = -0.04
            elif temp >= 85:
                temp_adj = +0.05  # Hot + humid = slight Over lean

            total_adj = wind_adj + temp_adj

        # Final environment score
        over_lean = (base_pf - 1.0) + total_adj
        environment_type = (
            "extreme_over" if over_lean >= 0.20 else
            "over_lean" if over_lean >= 0.08 else
            "slight_over" if over_lean >= 0.03 else
            "neutral" if abs(over_lean) < 0.03 else
            "slight_under" if over_lean >= -0.08 else
            "under_lean" if over_lean >= -0.15 else
            "extreme_under"
        )

        # Run projection adjustment (applied to total)
        run_proj_adj = over_lean * 9.0  # Convert to runs

        return {
            "base_park_factor": base_pf,
            "hr_park_factor": hr_pf,
            "wind_adj": round(wind_adj, 3),
            "temp_adj": round(temp_adj, 3),
            "total_adj": round(total_adj, 3),
            "over_lean": round(over_lean, 3),
            "environment_type": environment_type,
            "run_proj_adj": round(run_proj_adj, 2),
            "dome": dome,
            "wind_speed": weather.get("wind_speed", 0),
            "wind_direction": weather.get("wind_direction", "N/A"),
            "temperature": weather.get("temperature", 70),
            "humidity": weather.get("humidity", 50),
            "precip_pct": weather.get("precip_pct", 0)
        }

    def _apply_coors_rules(self, game_data: Dict) -> Dict:
        """
        Special Coors Field Over rules (v2.4)
        Both teams must have RISP > 28% to confirm Over
        ATL road average: 6.27 RPG = automatic RISP gate pass
        """
        away = game_data.get("away_team")

        # Teams with confirmed high RISP rates (auto-pass gate)
        HIGH_RISP_TEAMS = {"ATL", "LAD", "NYY", "HOU", "MIL", "TOR"}
        auto_pass = away in HIGH_RISP_TEAMS

        return {
            "risp_gate_required": True,
            "risp_threshold": COORS_RISP_GATE,
            "auto_pass_away": auto_pass,
            "over_boost": 0.08,  # Coors base Over boost
            "altitude_walk_penalty": 0.015,  # Walks score at higher rate
            "note": "Coors rules active — verify RISP gate before Over bet"
        }
