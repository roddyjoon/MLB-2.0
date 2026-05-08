"""
Agent 13: First Inning Model + Catcher Framing Tiers
First inning run expectancy, F5 line modeling,
full catcher framing tier system
"""

import asyncio
from typing import Dict, List, Optional
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("first_inning_agent")

# Full catcher framing tier database — 2026
CATCHER_FRAMING = {
    # Elite framers (Tier 1)
    "realmuto":    {"tier": 1, "runs_saved": 1.50, "team": "PHI"},
    "murphy":      {"tier": 1, "runs_saved": 1.40, "team": "ATL"},
    "raleigh":     {"tier": 1, "runs_saved": 1.20, "team": "SEA"},
    "contreras_w": {"tier": 1, "runs_saved": 1.10, "team": "MIL"},

    # Above average framers (Tier 2)
    "barnhart":    {"tier": 2, "runs_saved": 0.80, "team": "CHC"},
    "stallings":   {"tier": 2, "runs_saved": 0.75, "team": "MIA"},
    "d_arnaud":    {"tier": 2, "runs_saved": 0.70, "team": "LAA"},
    "wong_c":      {"tier": 2, "runs_saved": 0.65, "team": "BOS"},
    "naylor_b":    {"tier": 2, "runs_saved": 0.60, "team": "CLE"},
    "langeliers":  {"tier": 2, "runs_saved": 0.55, "team": "ATH"},
    "ramos":       {"tier": 2, "runs_saved": 0.55, "team": "MIA"},

    # Average framers (Tier 3) — 0 adjustment
    # (default for unlisted catchers)

    # Below average framers (Tier 4)
    "perez_sal":   {"tier": 4, "runs_saved": -0.60, "team": "KC"},
    "narvaez":     {"tier": 4, "runs_saved": -0.65, "team": "NYM"},
    "vazquez":     {"tier": 4, "runs_saved": -0.70, "team": "MIN"},

    # Poor framers (Tier 5)
    "stubbs":      {"tier": 5, "runs_saved": -1.20, "team": "PHI"},
    "quero":       {"tier": 5, "runs_saved": -0.90, "team": "CWS"},
}

# First inning ERA by SP (2026 season)
# Identifies "slow starters" vs "fast starters"
SP_FIRST_INNING = {
    # Slow starters — higher first inning ERA than season ERA
    "flaherty": {"f1_era": 8.10, "season_era": 5.90, "slow_start": True},
    "wacha":    {"f1_era": 5.40, "season_era": 3.13, "slow_start": True},
    "kay":      {"f1_era": 7.20, "season_era": 6.12, "slow_start": True},

    # Fast starters — dominant first inning
    "schlittler": {"f1_era": 0.00, "season_era": 0.95, "slow_start": False},
    "fried":      {"f1_era": 0.50, "season_era": 1.90, "slow_start": False},
    "skubal":     {"f1_era": 1.20, "season_era": 2.10, "slow_start": False},
    "glasnow":    {"f1_era": 1.00, "season_era": 2.45, "slow_start": False},
    "soriano":    {"f1_era": 0.00, "season_era": 0.24, "slow_start": False},
}


class FirstInningCatcherAgent:
    """
    Two models in one:
    1. First inning run expectancy → F5 line accuracy
    2. Catcher framing tiers → precise run suppression value
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        """Full first inning and catcher analysis"""

        home_sp_name = game_data.get("home_sp_name", "").lower()
        away_sp_name = game_data.get("away_sp_name", "").lower()
        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")

        # First inning analysis
        home_f1 = self._analyze_first_inning(home_sp_name, home_team)
        away_f1 = self._analyze_first_inning(away_sp_name, away_team)

        # Catcher framing
        home_catcher = await self._get_catcher(home_team)
        away_catcher = await self._get_catcher(away_team)

        home_framing = self._get_framing_value(home_catcher)
        away_framing = self._get_framing_value(away_catcher)

        # F5 line projection
        f5_projection = self._project_f5(home_f1, away_f1,
                                          home_framing, away_framing)

        # Combined WP adjustments
        wp_adj = self._calculate_wp_adj(home_f1, away_f1,
                                         home_framing, away_framing)

        return {
            "home_first_inning": home_f1,
            "away_first_inning": away_f1,
            "home_catcher": home_catcher,
            "away_catcher": away_catcher,
            "home_framing": home_framing,
            "away_framing": away_framing,
            "f5_projection": f5_projection,
            "wp_adjustments": wp_adj,
            "total_framing_adj": home_framing.get("runs_saved", 0) +
                                  away_framing.get("runs_saved", 0),
            "notes": self._generate_notes(
                home_f1, away_f1, home_framing, away_framing,
                home_team, away_team
            )
        }

    def _analyze_first_inning(self, sp_name: str, team: str) -> Dict:
        """Analyze SP first inning tendencies"""

        # Look up by partial name match
        f1_data = None
        for key, data in SP_FIRST_INNING.items():
            if key in sp_name:
                f1_data = data
                break

        if not f1_data:
            return {
                "sp_name": sp_name,
                "f1_era": None,
                "slow_start": False,
                "wp_adj": 0.0,
                "note": "No first inning data"
            }

        f1_era = f1_data["f1_era"]
        season_era = f1_data["season_era"]
        slow_start = f1_data["slow_start"]

        # F1 ERA delta
        era_delta = f1_era - season_era

        # WP impact in first inning
        if slow_start and era_delta >= 3.0:
            wp_adj = -0.020  # SP is much worse early
            note = f"Slow starter: F1 ERA {f1_era:.2f} vs season {season_era:.2f}"
        elif not slow_start and era_delta <= -1.0:
            wp_adj = +0.015  # SP is better early
            note = f"Fast starter: F1 ERA {f1_era:.2f} vs season {season_era:.2f}"
        else:
            wp_adj = 0.0
            note = "Normal first inning tendencies"

        return {
            "sp_name": sp_name,
            "f1_era": f1_era,
            "season_era": season_era,
            "era_delta": round(era_delta, 2),
            "slow_start": slow_start,
            "wp_adj": wp_adj,
            "note": note
        }

    async def _get_catcher(self, team: str) -> Dict:
        """Get starting catcher for team"""
        try:
            lineup = await self.mlb_api.get_confirmed_lineup(team)
            for player in lineup:
                if player.get("position") == "C":
                    return {
                        "name": player.get("name", ""),
                        "team": team
                    }
        except Exception:
            pass

        # Fall back to known starters
        STARTING_CATCHERS_2026 = {
            "PHI": "realmuto", "ATL": "murphy", "SEA": "raleigh",
            "MIL": "contreras_w", "CLE": "naylor_b", "LAA": "d_arnaud",
            "BOS": "wong_c", "MIA": "ramos", "ATH": "langeliers",
            "KC": "perez_sal", "MIN": "vazquez", "CWS": "quero",
        }
        name = STARTING_CATCHERS_2026.get(team, "unknown")
        return {"name": name, "team": team}

    def _get_framing_value(self, catcher: Dict) -> Dict:
        """Get catcher framing value from tier database"""
        name = catcher.get("name", "").lower()

        # Look up by partial name match
        for key, data in CATCHER_FRAMING.items():
            if key in name or name in key:
                return {
                    "name": catcher["name"],
                    "tier": data["tier"],
                    "runs_saved": data["runs_saved"],
                    "label": (
                        "🔥 Elite framer" if data["tier"] == 1 else
                        "Above avg framer" if data["tier"] == 2 else
                        "Average framer" if data["tier"] == 3 else
                        "Below avg framer" if data["tier"] == 4 else
                        "Poor framer"
                    )
                }

        # Unknown — assume average
        return {
            "name": catcher.get("name", "Unknown"),
            "tier": 3,
            "runs_saved": 0.0,
            "label": "Average framer (unknown)"
        }

    def _project_f5(self, home_f1: Dict, away_f1: Dict,
                     home_framing: Dict, away_framing: Dict) -> Dict:
        """Project first 5 innings total"""

        # Base projection: ~55% of game total in first 5
        game_total_base = 8.5
        f5_base = game_total_base * 0.55  # ~4.7 runs

        # SP first inning adjustments
        home_f1_adj = home_f1.get("wp_adj", 0) * 5  # Scale to 5 innings
        away_f1_adj = away_f1.get("wp_adj", 0) * 5

        # Framing adjustments (applied per 9 innings, scale to 5)
        home_frame_adj = home_framing.get("runs_saved", 0) * (5/9)
        away_frame_adj = away_framing.get("runs_saved", 0) * (5/9)

        total_adj = home_f1_adj + away_f1_adj + home_frame_adj + away_frame_adj
        f5_projection = f5_base + total_adj

        return {
            "f5_projection": round(f5_projection, 1),
            "f5_base": f5_base,
            "adjustments": round(total_adj, 2),
            "f5_over_lean": f5_projection > f5_base + 0.3,
            "f5_under_lean": f5_projection < f5_base - 0.3
        }

    def _calculate_wp_adj(self, home_f1: Dict, away_f1: Dict,
                           home_framing: Dict, away_framing: Dict) -> Dict:
        """Calculate WP adjustments"""

        # Framing: better home framing = home SP gets more called strikes
        # = fewer walks = fewer runs = better home WP
        home_frame_runs = home_framing.get("runs_saved", 0)
        away_frame_runs = away_framing.get("runs_saved", 0)

        # Net framing advantage (positive = home team benefits)
        frame_net = home_frame_runs - away_frame_runs

        # Convert runs saved to WP (each run ≈ 10% WP roughly)
        framing_wp_adj = frame_net * 0.008

        # First inning adjustments
        # Home SP slow start = away team scores first = away WP boost
        home_f1_adj = home_f1.get("wp_adj", 0)
        away_f1_adj = away_f1.get("wp_adj", 0)

        net_f1_adj = home_f1_adj - away_f1_adj

        return {
            "framing_wp_adj": round(framing_wp_adj, 3),
            "f1_wp_adj": round(net_f1_adj, 3),
            "total_wp_adj": round(framing_wp_adj + net_f1_adj, 3)
        }

    def _generate_notes(self, home_f1: Dict, away_f1: Dict,
                         home_framing: Dict, away_framing: Dict,
                         home: str, away: str) -> List[str]:
        """Generate key notes"""
        notes = []

        if home_f1.get("slow_start"):
            notes.append(f"⚠️ {home} SP slow starter: {home_f1['note']}")
        if away_f1.get("slow_start"):
            notes.append(f"⚠️ {away} SP slow starter: {away_f1['note']}")
        if home_f1.get("wp_adj", 0) >= 0.015:
            notes.append(f"🔥 {home} SP fast starter: {home_f1['note']}")
        if away_f1.get("wp_adj", 0) >= 0.015:
            notes.append(f"🔥 {away} SP fast starter: {away_f1['note']}")

        # Framing notes
        if home_framing.get("tier") == 1:
            notes.append(f"🔥 {home}: {home_framing['name']} elite framer "
                        f"(+{home_framing['runs_saved']} runs saved)")
        if away_framing.get("tier") == 1:
            notes.append(f"🔥 {away}: {away_framing['name']} elite framer "
                        f"(+{away_framing['runs_saved']} runs saved)")
        if home_framing.get("tier") == 5:
            notes.append(f"🚨 {home}: {home_framing['name']} poor framer "
                        f"({home_framing['runs_saved']} runs)")
        if away_framing.get("tier") == 5:
            notes.append(f"🚨 {away}: {away_framing['name']} poor framer "
                        f"({away_framing['runs_saved']} runs)")

        return notes
