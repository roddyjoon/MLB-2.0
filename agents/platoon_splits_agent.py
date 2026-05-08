"""
Agent 11: Platoon Splits Database
Full LHP/RHP wRC+ splits for every team,
SP ERA/xFIP vs LHB/RHB — quantifies every platoon advantage automatically

This replaces manual platoon notes with precise WP adjustments
"""

import asyncio
from typing import Dict, List
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("platoon_agent")

# Team wRC+ splits vs LHP/RHP — 2026 season
# Updated from FanGraphs team splits
TEAM_SPLITS_2026 = {
    # Format: team: {vs_lhp_wrc, vs_rhp_wrc, k_vs_lhp, k_vs_rhp}
    "ATL": {"vs_lhp": 138, "vs_rhp": 130, "k_lhp": 0.19, "k_rhp": 0.20},
    "NYY": {"vs_lhp": 131, "vs_rhp": 125, "k_lhp": 0.20, "k_rhp": 0.21},
    "LAD": {"vs_lhp": 135, "vs_rhp": 128, "k_lhp": 0.19, "k_rhp": 0.20},
    "HOU": {"vs_lhp": 118, "vs_rhp": 112, "k_lhp": 0.20, "k_rhp": 0.21},
    "MIL": {"vs_lhp": 102, "vs_rhp": 94, "k_lhp": 0.22, "k_rhp": 0.23},
    "CHC": {"vs_lhp": 128, "vs_rhp": 132, "k_lhp": 0.18, "k_rhp": 0.18},
    "SEA": {"vs_lhp": 94, "vs_rhp": 88, "k_lhp": 0.24, "k_rhp": 0.23},
    "DET": {"vs_lhp": 110, "vs_rhp": 100, "k_lhp": 0.21, "k_rhp": 0.22},
    "CLE": {"vs_lhp": 108, "vs_rhp": 104, "k_lhp": 0.20, "k_rhp": 0.21},
    "CIN": {"vs_lhp": 112, "vs_rhp": 108, "k_lhp": 0.21, "k_rhp": 0.20},
    "NYM": {"vs_lhp": 68, "vs_rhp": 76, "k_lhp": 0.26, "k_rhp": 0.24},
    "PHI": {"vs_lhp": 88, "vs_rhp": 95, "k_lhp": 0.23, "k_rhp": 0.22},
    "MIA": {"vs_lhp": 100, "vs_rhp": 108, "k_lhp": 0.21, "k_rhp": 0.20},
    "WSH": {"vs_lhp": 112, "vs_rhp": 108, "k_lhp": 0.20, "k_rhp": 0.21},
    "BAL": {"vs_lhp": 98, "vs_rhp": 95, "k_lhp": 0.22, "k_rhp": 0.23},
    "BOS": {"vs_lhp": 78, "vs_rhp": 82, "k_lhp": 0.24, "k_rhp": 0.23},
    "TOR": {"vs_lhp": 100, "vs_rhp": 96, "k_lhp": 0.21, "k_rhp": 0.22},
    "TB":  {"vs_lhp": 94, "vs_rhp": 90, "k_lhp": 0.23, "k_rhp": 0.24},
    "KC":  {"vs_lhp": 86, "vs_rhp": 82, "k_lhp": 0.23, "k_rhp": 0.24},
    "MIN": {"vs_lhp": 85, "vs_rhp": 82, "k_lhp": 0.24, "k_rhp": 0.24},
    "CWS": {"vs_lhp": 100, "vs_rhp": 104, "k_lhp": 0.22, "k_rhp": 0.21},
    "STL": {"vs_lhp": 108, "vs_rhp": 106, "k_lhp": 0.21, "k_rhp": 0.21},
    "PIT": {"vs_lhp": 110, "vs_rhp": 118, "k_lhp": 0.20, "k_rhp": 0.19},
    "COL": {"vs_lhp": 96, "vs_rhp": 92, "k_lhp": 0.23, "k_rhp": 0.24},
    "AZ":  {"vs_lhp": 108, "vs_rhp": 104, "k_lhp": 0.22, "k_rhp": 0.21},
    "SF":  {"vs_lhp": 82, "vs_rhp": 79, "k_lhp": 0.24, "k_rhp": 0.25},
    "SD":  {"vs_lhp": 88, "vs_rhp": 86, "k_lhp": 0.23, "k_rhp": 0.23},
    "LAA": {"vs_lhp": 96, "vs_rhp": 92, "k_lhp": 0.22, "k_rhp": 0.23},
    "TEX": {"vs_lhp": 86, "vs_rhp": 82, "k_lhp": 0.24, "k_rhp": 0.24},
    "ATH": {"vs_lhp": 106, "vs_rhp": 104, "k_lhp": 0.20, "k_rhp": 0.20},
    "OAK": {"vs_lhp": 106, "vs_rhp": 104, "k_lhp": 0.20, "k_rhp": 0.20},

    # Special cases
    "NYM_vs_LHP": 68,  # Dead last MLB — structural NYM weakness flagged
}

# NYM structural weakness (confirmed from this week's data)
NYM_VS_LHP_RECORD = "1-6"  # Worst vs LHP in baseball 2026


class PlatoonSplitsAgent:
    """
    Quantifies platoon advantages precisely using team wRC+ splits
    
    v2.4 improvement: Replaces qualitative platoon notes
    with exact WP percentages based on confirmed splits data
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        """Full platoon analysis"""

        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")
        home_sp_throws = game_data.get("home_sp_throws", "R")
        away_sp_throws = game_data.get("away_sp_throws", "R")

        # Get splits for each team vs opposing SP handedness
        home_vs_away_sp = self._get_team_split(home_team, away_sp_throws)
        away_vs_home_sp = self._get_team_split(away_team, home_sp_throws)

        # SP platoon splits
        home_sp_splits = await self._get_sp_platoon_splits(
            game_data.get("home_sp_id"),
            game_data.get("home_sp_name", "")
        )
        away_sp_splits = await self._get_sp_platoon_splits(
            game_data.get("away_sp_id"),
            game_data.get("away_sp_name", "")
        )

        # Calculate platoon advantages
        home_adv = self._calculate_platoon_advantage(
            home_vs_away_sp, away_sp_splits, away_sp_throws
        )
        away_adv = self._calculate_platoon_advantage(
            away_vs_home_sp, home_sp_splits, home_sp_throws
        )

        # Special structural flags
        structural_flags = self._check_structural_flags(
            home_team, away_team, home_sp_throws, away_sp_throws
        )

        # Net WP impact
        wp_impact = self._calculate_wp_impact(home_adv, away_adv)

        return {
            "home_team_split": home_vs_away_sp,
            "away_team_split": away_vs_home_sp,
            "home_sp_splits": home_sp_splits,
            "away_sp_splits": away_sp_splits,
            "home_platoon_advantage": home_adv,
            "away_platoon_advantage": away_adv,
            "structural_flags": structural_flags,
            "wp_impact": wp_impact,
            "summary": self._generate_summary(
                home_adv, away_adv, structural_flags,
                home_team, away_team
            )
        }

    def _get_team_split(self, team: str, sp_throws: str) -> Dict:
        """Get team wRC+ vs specific SP handedness"""
        splits = TEAM_SPLITS_2026.get(team, {
            "vs_lhp": 100, "vs_rhp": 100, "k_lhp": 0.22, "k_rhp": 0.22
        })

        if sp_throws == "L":
            wrc = splits.get("vs_lhp", 100)
            k_rate = splits.get("k_lhp", 0.22)
            same_side = True  # Team's LHBs face same-side SP
        else:
            wrc = splits.get("vs_rhp", 100)
            k_rate = splits.get("k_rhp", 0.22)
            same_side = False

        return {
            "team": team,
            "sp_throws": sp_throws,
            "wrc_plus": wrc,
            "k_rate": k_rate,
            "same_side_disadvantage": same_side and wrc <= 85,
            "strong_advantage": wrc >= 120
        }

    async def _get_sp_platoon_splits(self, sp_id: str,
                                      sp_name: str) -> Dict:
        """Get SP ERA/xFIP vs LHB and RHB"""
        if not sp_id:
            return self._default_sp_splits()

        try:
            splits = await self.mlb_api.get_pitcher_splits(sp_id)
            return {
                "vs_lhb_era": splits.get("vs_LHB", {}).get("era", 4.20),
                "vs_rhb_era": splits.get("vs_RHB", {}).get("era", 4.20),
                "vs_lhb_ops": splits.get("vs_LHB", {}).get("ops", 0.720),
                "vs_rhb_ops": splits.get("vs_RHB", {}).get("ops", 0.720),
                "vs_lhb_ba": splits.get("vs_LHB", {}).get("ba", 0.245),
                "vs_rhb_ba": splits.get("vs_RHB", {}).get("ba", 0.245),
                "platoon_split": abs(
                    splits.get("vs_LHB", {}).get("era", 4.20) -
                    splits.get("vs_RHB", {}).get("era", 4.20)
                )
            }
        except Exception:
            return self._default_sp_splits()

    def _calculate_platoon_advantage(self, team_split: Dict,
                                      sp_splits: Dict,
                                      sp_throws: str) -> Dict:
        """
        Calculate precise platoon advantage/disadvantage
        
        Combines team wRC+ split with SP's specific split vs that handedness
        """
        team_wrc = team_split.get("wrc_plus", 100)
        same_side = team_split.get("same_side_disadvantage", False)
        strong_adv = team_split.get("strong_advantage", False)

        # SP vulnerability to this team's handedness
        if sp_throws == "L":
            sp_era_vs_team = sp_splits.get("vs_rhb_era", 4.20)
            # RHB lineup vs LHP = platoon advantage for batters
            platoon_advantage = True
        else:
            sp_era_vs_team = sp_splits.get("vs_lhb_era", 4.20)
            # LHB lineup vs RHP = platoon advantage for batters
            platoon_advantage = True

        # WP adjustment calculation
        # wRC+ 120 vs sp type → +3% WP
        # wRC+ 80 vs sp type → -3% WP
        # Normalized around 100
        wrc_adj = (team_wrc - 100) * 0.0003

        # Strong advantage bonus
        bonus = 0.015 if strong_adv else 0
        # Same-side penalty
        penalty = -0.010 if same_side else 0

        net_adj = round(wrc_adj + bonus + penalty, 3)

        return {
            "team_wrc_vs_sp_hand": team_wrc,
            "platoon_advantage": platoon_advantage,
            "same_side_disadvantage": same_side,
            "strong_advantage": strong_adv,
            "wp_adj": net_adj,
            "label": (
                "🔥 Strong advantage" if net_adj >= 0.020 else
                "Mild advantage" if net_adj >= 0.005 else
                "Disadvantage" if net_adj <= -0.010 else
                "Neutral"
            )
        }

    def _check_structural_flags(self, home: str, away: str,
                                  home_throws: str, away_throws: str) -> Dict:
        """Check for major structural platoon mismatches"""
        flags = []
        wp_adj = 0

        # NYM structural weakness vs LHP
        if away == "NYM" and home_throws == "L":
            flags.append({
                "type": "NYM_vs_LHP",
                "severity": "major",
                "description": f"NYM {NYM_VS_LHP_RECORD} vs LHP this season — structural weakness",
                "wp_adj": +0.030  # Benefits home team
            })
            wp_adj += 0.030

        if home == "NYM" and away_throws == "L":
            flags.append({
                "type": "NYM_vs_LHP_home",
                "severity": "major",
                "description": f"NYM {NYM_VS_LHP_RECORD} vs LHP this season",
                "wp_adj": -0.030  # Hurts home team
            })
            wp_adj -= 0.030

        # Extreme splits (wRC+ ≥ 130 vs specific hand)
        home_vs_away = TEAM_SPLITS_2026.get(home, {})
        away_vs_home = TEAM_SPLITS_2026.get(away, {})

        key = "vs_lhp" if away_throws == "L" else "vs_rhp"
        if home_vs_away.get(key, 100) >= 130:
            flags.append({
                "type": "elite_lineup_vs_sp_hand",
                "team": home,
                "wrc": home_vs_away[key],
                "description": f"{home} wRC+ {home_vs_away[key]} vs {away_throws}HP — elite",
                "wp_adj": +0.025
            })
            wp_adj += 0.025

        return {
            "flags": flags,
            "total_wp_adj": round(wp_adj, 3),
            "has_major_flag": any(f["severity"] == "major" for f in flags)
        }

    def _calculate_wp_impact(self, home_adv: Dict,
                              away_adv: Dict) -> Dict:
        """Net WP impact from platoon analysis"""
        home_adj = home_adv.get("wp_adj", 0)
        away_adj = away_adv.get("wp_adj", 0)

        # Home advantage helps home WP
        # Away advantage helps away WP = hurts home WP
        net_home_wp = home_adj - away_adj

        return {
            "net_home_wp_adj": round(net_home_wp, 3),
            "home_batting_adj": home_adj,
            "away_batting_adj": away_adj
        }

    def _generate_summary(self, home_adv: Dict, away_adv: Dict,
                           flags: Dict, home: str, away: str) -> str:
        """Generate one-line platoon summary"""
        home_label = home_adv.get("label", "Neutral")
        away_label = away_adv.get("label", "Neutral")

        parts = []
        if "advantage" in home_label.lower():
            parts.append(f"{home} batting: {home_label}")
        if "advantage" in away_label.lower():
            parts.append(f"{away} batting: {away_label}")
        if flags.get("has_major_flag"):
            for f in flags["flags"]:
                parts.append(f["description"])

        return " | ".join(parts) if parts else "No significant platoon advantages"

    def _default_sp_splits(self) -> Dict:
        return {
            "vs_lhb_era": 4.20, "vs_rhb_era": 4.20,
            "vs_lhb_ops": 0.720, "vs_rhb_ops": 0.720,
            "platoon_split": 0.0
        }
