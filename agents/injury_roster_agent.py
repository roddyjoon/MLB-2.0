"""
Agent 4: Injury & Roster Intelligence Agent
IL tracking, day-to-day designations, lineup confirmation,
SP change monitoring — the most critical real-time agent
"""

import asyncio
from typing import Dict, List
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("injury_roster_agent")


class InjuryRosterAgent:
    """
    Monitors injury reports, IL moves, lineup changes, SP changes
    This agent is the primary defense against wrong-pitcher bets
    """

    IMPACT_POSITIONS = {
        "SP": 10,   # Starting pitcher change = maximum impact
        "C": 3,     # Catcher (framing value)
        "SS": 2,    # Shortstop
        "2B": 2,
        "CF": 2,
        "1B": 1,
        "3B": 1,
        "LF": 1,
        "RF": 1,
        "DH": 1,
        "CL": 3     # Closer
    }

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        home = game_data.get("home_team")
        away = game_data.get("away_team")
        date = game_data.get("date")

        # Pull all injury/roster data in parallel
        (
            home_il, away_il,
            home_dtd, away_dtd,
            sp_confirmation
        ) = await asyncio.gather(
            self.mlb_api.get_injury_list(home),
            self.mlb_api.get_injury_list(away),
            self.mlb_api.get_day_to_day(home),
            self.mlb_api.get_day_to_day(away),
            self.mlb_api.confirm_starters(home, away, date),
            return_exceptions=True
        )

        home_il = home_il if not isinstance(home_il, Exception) else []
        away_il = away_il if not isinstance(away_il, Exception) else []
        home_dtd = home_dtd if not isinstance(home_dtd, Exception) else []
        away_dtd = away_dtd if not isinstance(away_dtd, Exception) else []
        sp_confirmation = sp_confirmation if not isinstance(sp_confirmation, Exception) else {}

        # Analyze impact
        home_impact = self._calculate_injury_impact(home_il, home_dtd)
        away_impact = self._calculate_injury_impact(away_il, away_dtd)

        # SP change detection
        sp_changes = self._detect_sp_changes(sp_confirmation, game_data)

        # Closer availability
        home_closer = self._check_closer_availability(home_il, home_dtd)
        away_closer = self._check_closer_availability(away_il, away_dtd)

        # Catcher framing impact
        home_framing = self._calculate_framing_impact(home_il, home_dtd, home)
        away_framing = self._calculate_framing_impact(away_il, away_dtd, away)

        return {
            "home_il": home_il,
            "away_il": away_il,
            "home_dtd": home_dtd,
            "away_dtd": away_dtd,
            "home_impact": home_impact,
            "away_impact": away_impact,
            "sp_confirmation": sp_confirmation,
            "sp_changes": sp_changes,
            "home_closer": home_closer,
            "away_closer": away_closer,
            "home_framing_adj": home_framing,
            "away_framing_adj": away_framing,
            "alerts": self._generate_alerts(sp_changes, home_impact, away_impact)
        }

    def _calculate_injury_impact(self, il_list: List, dtd_list: List) -> Dict:
        """Calculate total roster impact from injuries"""
        total_impact = 0
        key_missing = []

        all_injured = il_list + [p for p in dtd_list
                                  if p.get("status") == "out_today"]

        for player in all_injured:
            position = player.get("position", "")
            impact_score = self.IMPACT_POSITIONS.get(position, 1)
            total_impact += impact_score

            if impact_score >= 2:
                key_missing.append({
                    "name": player.get("name"),
                    "position": position,
                    "status": player.get("status"),
                    "impact": impact_score,
                    "il_type": player.get("il_type", "IL")
                })

        # Normalize to WP impact (every 10 impact points ≈ 1% WP)
        wp_impact = -(total_impact / 1000)

        return {
            "total_impact": total_impact,
            "key_missing": sorted(key_missing,
                                   key=lambda x: x["impact"], reverse=True),
            "wp_adj": round(wp_impact, 3),
            "severity": "severe" if total_impact >= 15 else
                       "significant" if total_impact >= 8 else
                       "moderate" if total_impact >= 4 else "minimal"
        }

    def _detect_sp_changes(self, confirmation: Dict, game_data: Dict) -> Dict:
        """
        Detect SP changes from expected lineup
        THIS IS THE MOST IMPORTANT FUNCTION — wrong SP = wrong bet
        """
        expected_away = game_data.get("away_sp_name", "").lower()
        expected_home = game_data.get("home_sp_name", "").lower()

        confirmed_away = confirmation.get("away_sp", {}).get("name", "").lower()
        confirmed_home = confirmation.get("home_sp", {}).get("name", "").lower()

        away_changed = (expected_away and confirmed_away and
                        expected_away != confirmed_away)
        home_changed = (expected_home and confirmed_home and
                        expected_home != confirmed_home)

        changes = {}
        if away_changed:
            logger.warning(f"🚨 SP CHANGE DETECTED: Away — "
                          f"Expected {expected_away}, Confirmed {confirmed_away}")
            changes["away"] = {
                "expected": expected_away,
                "confirmed": confirmed_away,
                "impact": "RECALCULATE REQUIRED"
            }
        if home_changed:
            logger.warning(f"🚨 SP CHANGE DETECTED: Home — "
                          f"Expected {expected_home}, Confirmed {confirmed_home}")
            changes["home"] = {
                "expected": expected_home,
                "confirmed": confirmed_home,
                "impact": "RECALCULATE REQUIRED"
            }

        return {
            "changes_detected": bool(changes),
            "changes": changes,
            "away_confirmed": confirmed_away,
            "home_confirmed": confirmed_home,
            "recalculate_required": bool(changes)
        }

    def _check_closer_availability(self, il_list: List, dtd_list: List) -> Dict:
        """Check if team's closer is available"""
        for player in il_list + dtd_list:
            if player.get("role") == "closer" or player.get("position") == "CL":
                return {
                    "available": False,
                    "name": player.get("name"),
                    "status": player.get("status")
                }
        return {"available": True}

    def _calculate_framing_impact(self, il_list: List,
                                   dtd_list: List, team: str) -> float:
        """Calculate catcher framing adjustment"""
        # Elite framers: Realmuto (+1.5), Raleigh (+1.2), Murphy (+1.4)
        ELITE_FRAMERS = {
            "realmuto": 1.5,
            "raleigh": 1.2,
            "murphy": 1.4,
            "barnhart": 0.8,
            "stallings": 0.7
        }

        for player in il_list + dtd_list:
            name_lower = player.get("name", "").lower()
            for framer, value in ELITE_FRAMERS.items():
                if framer in name_lower:
                    return -value  # Negative = missing framing value
        return 0.0

    def _generate_alerts(self, sp_changes: Dict,
                          home_impact: Dict, away_impact: Dict) -> List:
        """Generate actionable alerts"""
        alerts = []

        if sp_changes.get("changes_detected"):
            alerts.append({
                "type": "SP_CHANGE",
                "severity": "CRITICAL",
                "message": "Starting pitcher changed — model recalculation required",
                "details": sp_changes["changes"]
            })

        if home_impact.get("severity") in ["severe", "significant"]:
            alerts.append({
                "type": "HOME_INJURIES",
                "severity": "HIGH",
                "message": f"Home team injuries: {home_impact['severity']}",
                "key_missing": home_impact["key_missing"][:3]
            })

        if away_impact.get("severity") in ["severe", "significant"]:
            alerts.append({
                "type": "AWAY_INJURIES",
                "severity": "HIGH",
                "message": f"Away team injuries: {away_impact['severity']}",
                "key_missing": away_impact["key_missing"][:3]
            })

        return alerts
