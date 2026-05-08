"""
Agent 2: Lineup & Offense Agent
wRC+, xwOBA splits vs LHP/RHP, hot/cold streaks, BvP data
"""

import asyncio
from typing import Dict, List
from data.mlb_api import MLBDataAPI
from data.savant_api import SavantAPI
from core.logger import ModelLogger

logger = ModelLogger("lineup_offense_agent")


class LineupOffenseAgent:
    """Pulls lineup data, offensive metrics, BvP matchups"""

    def __init__(self):
        self.mlb_api = MLBDataAPI()
        self.savant = SavantAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        home = game_data.get("home_team")
        away = game_data.get("away_team")
        home_sp_id = game_data.get("home_sp_id")
        away_sp_id = game_data.get("away_sp_id")

        # Pull all data in parallel
        (
            home_lineup, away_lineup,
            home_team_stats, away_team_stats,
            home_bvp, away_bvp
        ) = await asyncio.gather(
            self.mlb_api.get_confirmed_lineup(home),
            self.mlb_api.get_confirmed_lineup(away),
            self.mlb_api.get_team_offense_stats(home),
            self.mlb_api.get_team_offense_stats(away),
            self.mlb_api.get_bvp_data(home, away_sp_id) if away_sp_id else asyncio.sleep(0),
            self.mlb_api.get_bvp_data(away, home_sp_id) if home_sp_id else asyncio.sleep(0),
            return_exceptions=True
        )

        # Safe defaults
        home_lineup = home_lineup if not isinstance(home_lineup, Exception) else []
        away_lineup = away_lineup if not isinstance(away_lineup, Exception) else []
        home_team_stats = home_team_stats if not isinstance(home_team_stats, Exception) else {}
        away_team_stats = away_team_stats if not isinstance(away_team_stats, Exception) else {}
        home_bvp = home_bvp if not isinstance(home_bvp, Exception) else {}
        away_bvp = away_bvp if not isinstance(away_bvp, Exception) else {}

        # Analyze lineups
        home_analysis = self._analyze_lineup(home_lineup, home_team_stats, home_bvp)
        away_analysis = self._analyze_lineup(away_lineup, away_team_stats, away_bvp)

        # wRC+ gap
        home_wrc = home_analysis.get("team_wrc_plus", 100)
        away_wrc = away_analysis.get("team_wrc_plus", 100)
        wrc_gap = home_wrc - away_wrc

        return {
            "home": home_analysis,
            "away": away_analysis,
            "wrc_gap": round(wrc_gap, 1),
            "wrc_gap_favor": "home" if wrc_gap > 0 else "away",
            "wrc_wp_impact": self._wrc_to_wp(wrc_gap)
        }

    def _analyze_lineup(self, lineup: List, team_stats: Dict, bvp: Dict) -> Dict:
        """Full lineup analysis"""

        # Team-level offensive stats
        wrc_plus = team_stats.get("wrc_plus", 100)
        ops = team_stats.get("ops", 0.720)
        obp = team_stats.get("obp", 0.320)
        slg = team_stats.get("slg", 0.400)
        ba = team_stats.get("ba", 0.245)
        rpg = team_stats.get("rpg", 4.5)
        k_pct = team_stats.get("k_pct", 0.22)
        bb_pct = team_stats.get("bb_pct", 0.08)
        hr = team_stats.get("hr", 40)

        # Hot/cold streak detection
        last_7 = team_stats.get("last_7_games", {})
        last_14 = team_stats.get("last_14_games", {})
        streak = self._detect_streak(last_7, last_14)

        # Individual hot bats from lineup
        hot_bats = []
        for player in lineup:
            player_streak = player.get("recent_ops", player.get("ops", 0))
            if player_streak >= 0.900:
                hot_bats.append({
                    "name": player.get("name"),
                    "recent_ops": player_streak,
                    "hr": player.get("hr", 0),
                    "rbi": player.get("rbi", 0)
                })

        # BvP threats
        bvp_threats = self._extract_bvp_threats(bvp)

        # Platoon advantage check
        # (applied in WP formula using SP handedness data)
        handedness_breakdown = self._get_handedness_breakdown(lineup)

        return {
            "team_wrc_plus": wrc_plus,
            "ops": ops,
            "obp": obp,
            "slg": slg,
            "ba": ba,
            "rpg": rpg,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "hr_total": hr,
            "streak": streak,
            "hot_bats": hot_bats[:5],  # Top 5 hot bats
            "bvp_threats": bvp_threats,
            "handedness": handedness_breakdown,
            "lineup_confirmed": len(lineup) > 0,
            "lineup": lineup[:9]
        }

    def _detect_streak(self, last_7: Dict, last_14: Dict) -> Dict:
        """Detect team offensive streak"""
        recent_ops = last_7.get("ops", 0.720)
        season_ops = last_14.get("ops", 0.720)
        recent_rpg = last_7.get("rpg", 4.5)

        if recent_ops >= 0.850 or recent_rpg >= 6.0:
            return {"type": "hot", "strength": "strong",
                    "recent_ops": recent_ops, "recent_rpg": recent_rpg}
        elif recent_ops >= 0.780 or recent_rpg >= 5.0:
            return {"type": "hot", "strength": "mild",
                    "recent_ops": recent_ops, "recent_rpg": recent_rpg}
        elif recent_ops <= 0.620 or recent_rpg <= 3.0:
            return {"type": "cold", "strength": "strong",
                    "recent_ops": recent_ops, "recent_rpg": recent_rpg}
        elif recent_ops <= 0.680 or recent_rpg <= 3.8:
            return {"type": "cold", "strength": "mild",
                    "recent_ops": recent_ops, "recent_rpg": recent_rpg}
        else:
            return {"type": "neutral", "recent_ops": recent_ops, "recent_rpg": recent_rpg}

    def _extract_bvp_threats(self, bvp: Dict) -> List:
        """Extract significant BvP matchups"""
        threats = []
        for player_id, data in bvp.items():
            pa = data.get("pa", 0)
            ops = data.get("ops", 0)
            if pa >= 5 and ops >= 0.900:
                threats.append({
                    "name": data.get("name"),
                    "pa": pa,
                    "ops": ops,
                    "hr": data.get("hr", 0),
                    "ba": data.get("ba", 0),
                    "significance": "elite" if ops >= 1.200 else
                                   "strong" if ops >= 1.000 else "moderate"
                })
            elif pa >= 5 and ops <= 0.400:
                threats.append({
                    "name": data.get("name"),
                    "pa": pa,
                    "ops": ops,
                    "significance": "futile"  # Specific weakness
                })
        return sorted(threats, key=lambda x: x["ops"], reverse=True)

    def _get_handedness_breakdown(self, lineup: List) -> Dict:
        """Get L/R/S breakdown for platoon analysis"""
        lhb = sum(1 for p in lineup if p.get("bats") == "L")
        rhb = sum(1 for p in lineup if p.get("bats") == "R")
        switch = sum(1 for p in lineup if p.get("bats") == "S")
        total = len(lineup) or 9

        return {
            "lhb": lhb,
            "rhb": rhb,
            "switch": switch,
            "lhb_pct": round(lhb / total, 2),
            "rhb_pct": round(rhb / total, 2)
        }

    def _wrc_to_wp(self, wrc_gap: float) -> float:
        """Convert wRC+ gap to WP impact"""
        # +20 wRC+ gap ≈ +4% WP
        return round(wrc_gap * 0.002, 3)
