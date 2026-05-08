"""
Agent 8: Umpire Modeling Agent
HP umpire K rate, Over/Under record, squeeze tendency,
run expectancy adjustment — lowest hanging fruit, highest impact on totals
"""

import asyncio
from typing import Dict, Optional
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("umpire_agent")

# Umpire database — 2026 season data
# Source: UmpScores.com + Baseball Savant
# Format: name: {k_rate_adj, runs_per_game_adj, over_pct, squeeze_tendency}
UMPIRE_DATABASE = {
    # Expansive umpires (more walks, bigger strike zone issues = more runs)
    "angel hernandez":    {"k_adj": -0.08, "runs_adj": +1.20, "over_pct": 0.58, "squeeze": False},
    "cb bucknor":         {"k_adj": -0.06, "runs_adj": +1.00, "over_pct": 0.56, "squeeze": False},
    "joe west":           {"k_adj": -0.04, "runs_adj": +0.80, "over_pct": 0.54, "squeeze": False},
    "john tumpane":       {"k_adj": -0.03, "runs_adj": +0.60, "over_pct": 0.54, "squeeze": False},
    "dan iassogna":       {"k_adj": -0.03, "runs_adj": +0.50, "over_pct": 0.53, "squeeze": False},

    # Squeeze umpires (tight zone = more Ks, fewer walks = fewer runs)
    "ted barrett":        {"k_adj": +0.06, "runs_adj": -1.10, "over_pct": 0.42, "squeeze": True},
    "mike everitt":       {"k_adj": +0.05, "runs_adj": -0.90, "over_pct": 0.43, "squeeze": True},
    "paul emmel":         {"k_adj": +0.04, "runs_adj": -0.80, "over_pct": 0.44, "squeeze": True},
    "bill miller":        {"k_adj": +0.04, "runs_adj": -0.70, "over_pct": 0.44, "squeeze": True},
    "quinn wolcott":      {"k_adj": +0.03, "runs_adj": -0.60, "over_pct": 0.45, "squeeze": True},
    "stu scheurwater":    {"k_adj": +0.03, "runs_adj": -0.55, "over_pct": 0.46, "squeeze": True},

    # Slightly expansive
    "alan porter":        {"k_adj": -0.02, "runs_adj": +0.40, "over_pct": 0.52, "squeeze": False},
    "brian gorman":       {"k_adj": -0.02, "runs_adj": +0.35, "over_pct": 0.52, "squeeze": False},
    "mark ripperger":     {"k_adj": -0.01, "runs_adj": +0.25, "over_pct": 0.51, "squeeze": False},

    # Average umpires
    "jim wolf":           {"k_adj": 0.00, "runs_adj": +0.10, "over_pct": 0.50, "squeeze": False},
    "chris segal":        {"k_adj": 0.00, "runs_adj": +0.05, "over_pct": 0.50, "squeeze": False},
    "david rackley":      {"k_adj": 0.00, "runs_adj": 0.00, "over_pct": 0.50, "squeeze": False},
    "mike muchlinski":    {"k_adj": +0.01, "runs_adj": -0.10, "over_pct": 0.49, "squeeze": False},
    "chad fairchild":     {"k_adj": +0.01, "runs_adj": -0.15, "over_pct": 0.49, "squeeze": False},

    # Slightly squeeze
    "marty foster":       {"k_adj": +0.02, "runs_adj": -0.30, "over_pct": 0.48, "squeeze": False},
    "andy fletcher":      {"k_adj": +0.02, "runs_adj": -0.30, "over_pct": 0.47, "squeeze": False},
    "todd tichenor":      {"k_adj": +0.02, "runs_adj": -0.35, "over_pct": 0.47, "squeeze": False},
}

# League average umpire (when unknown)
LEAGUE_AVG_UMPIRE = {
    "k_adj": 0.00,
    "runs_adj": 0.00,
    "over_pct": 0.50,
    "squeeze": False,
    "name": "Unknown"
}


class UmpireAgent:
    """
    Models home plate umpire impact on game scoring
    
    Key insight: Best vs worst umpires create 1.2-1.8 run differential
    This directly impacts Over/Under edge calculations
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        """Pull and analyze HP umpire for this game"""

        home = game_data.get("home_team")
        away = game_data.get("away_team")
        date = game_data.get("date")

        # Get HP umpire assignment
        umpire_name = await self.mlb_api.get_hp_umpire(home, away, date)

        # Look up umpire data
        ump_data = self._get_umpire_data(umpire_name)

        # Calculate impact
        impact = self._calculate_impact(ump_data, game_data)

        return {
            "umpire_name": umpire_name or "TBD",
            "umpire_data": ump_data,
            "impact": impact,
            "total_proj_adj": impact["runs_adj"],
            "over_lean": impact["over_lean"],
            "walk_risk_amplifier": impact["walk_risk_amplifier"],
            "squeeze_note": self._generate_note(ump_data, game_data)
        }

    def _get_umpire_data(self, name: Optional[str]) -> Dict:
        """Look up umpire from database"""
        if not name:
            return {**LEAGUE_AVG_UMPIRE}

        name_lower = name.lower().strip()

        # Direct match
        if name_lower in UMPIRE_DATABASE:
            data = dict(UMPIRE_DATABASE[name_lower])
            data["name"] = name
            data["found"] = True
            return data

        # Partial match (last name)
        last_name = name_lower.split()[-1] if name_lower else ""
        for ump_key, ump_data in UMPIRE_DATABASE.items():
            if last_name and last_name in ump_key:
                data = dict(ump_data)
                data["name"] = name
                data["found"] = True
                return data

        # Unknown umpire — use league average
        return {**LEAGUE_AVG_UMPIRE, "name": name, "found": False}

    def _calculate_impact(self, ump: Dict, game_data: Dict) -> Dict:
        """Calculate umpire impact on game"""

        runs_adj = ump.get("runs_adj", 0.0)
        k_adj = ump.get("k_adj", 0.0)
        over_pct = ump.get("over_pct", 0.50)
        squeeze = ump.get("squeeze", False)

        # Over/Under lean from umpire
        # Expansive ump = Over lean, Squeeze ump = Under lean
        over_lean = runs_adj / 9.0  # Normalize to per-game factor

        # Walk risk amplifier
        # Squeeze ump neutralizes walk-prone SP (fewer walks granted)
        # Expansive ump amplifies walk-prone SP (more walks granted)
        if squeeze:
            walk_risk_amplifier = 0.60  # Reduces walk-risk impact by 40%
        else:
            walk_risk_amplifier = 1.20 if runs_adj >= 0.80 else 1.00

        # K rate impact on SP projection
        # Higher k_adj = more Ks → SP can go deeper → fewer BP innings
        sp_depth_adj = k_adj * 0.5  # Each 0.01 K% = 0.005 innings pitched adj

        return {
            "runs_adj": runs_adj,
            "k_adj": k_adj,
            "over_pct": over_pct,
            "over_lean": round(over_lean, 3),
            "walk_risk_amplifier": walk_risk_amplifier,
            "sp_depth_adj": sp_depth_adj,
            "squeeze": squeeze,
            "total_impact_label": (
                "🔥 Over lean" if runs_adj >= 0.80 else
                "Slight Over lean" if runs_adj >= 0.30 else
                "⬇️ Under lean" if runs_adj <= -0.60 else
                "Slight Under lean" if runs_adj <= -0.25 else
                "Neutral"
            )
        }

    def _generate_note(self, ump: Dict, game_data: Dict) -> str:
        """Generate actionable umpire note"""
        name = ump.get("name", "Unknown")
        runs_adj = ump.get("runs_adj", 0)
        squeeze = ump.get("squeeze", False)

        if not ump.get("found"):
            return f"Umpire {name} not in database — using league average"

        if abs(runs_adj) < 0.25:
            return f"{name} — neutral umpire, no adjustment"

        if squeeze:
            note = f"{name} — squeeze umpire ({runs_adj:+.1f} RPG)"
            # Check if walk-prone SP is pitching
            home_sp = game_data.get("home_sp_preview", {})
            away_sp = game_data.get("away_sp_preview", {})
            if (home_sp.get("bb9", 0) >= 4.5 or
                    away_sp.get("bb9", 0) >= 4.5):
                note += " — AMPLIFIES walk-prone SP squeeze"
            return note
        else:
            note = f"{name} — expansive umpire ({runs_adj:+.1f} RPG)"
            return note
