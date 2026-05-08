"""
Agent 6: Advanced Metrics Synthesis Agent
ERA mirage detection, walk-risk flags, bimodal detection,
platoon advantages, manager tendencies
"""

from typing import Dict
from core.logger import ModelLogger

logger = ModelLogger("advanced_metrics_agent")


class AdvancedMetricsAgent:
    """
    Synthesizes advanced metrics from other agents into WP modifiers
    ERA mirage detection, walk-risk automation, bimodal flags
    """

    # Manager quick-hook tendencies (affects SP IP expectations)
    MANAGER_HOOKS = {
        "Roberts": {"hook_threshold": 85, "quick_hook": True},   # LAD
        "Snitker": {"hook_threshold": 105, "quick_hook": False},  # ATL
        "Mattingly": {"hook_threshold": 90, "quick_hook": False}, # PHI
        "Bell": {"hook_threshold": 95, "quick_hook": False},      # CLE
    }

    def __init__(self):
        pass

    async def analyze(self, game_data: Dict) -> Dict:
        """
        Synthesizes metrics from game data
        Note: This agent receives pre-pulled data from other agents
        It focuses on cross-agent synthesis
        """

        away_sp_data = game_data.get("away_sp_preview", {})
        home_sp_data = game_data.get("home_sp_preview", {})

        away_team = game_data.get("away_team")
        home_team = game_data.get("home_team")

        # Platoon analysis
        platoon = self._analyze_platoon(
            game_data.get("away_lineup_preview", []),
            game_data.get("home_lineup_preview", []),
            away_sp_data, home_sp_data
        )

        # ERA mirage severity ranking
        away_mirage = self._classify_era_mirage(away_sp_data)
        home_mirage = self._classify_era_mirage(home_sp_data)

        # Walk-risk impact on offense
        away_walk_risk = self._walk_risk_impact(away_sp_data)
        home_walk_risk = self._walk_risk_impact(home_sp_data)

        # Manager tendencies
        away_manager = self._get_manager_tendency(away_team)
        home_manager = self._get_manager_tendency(home_team)

        # Debut/first-exposure penalties
        first_exposure = self._first_exposure_flags(
            away_sp_data, home_sp_data,
            game_data.get("away_bvp_preview", {}),
            game_data.get("home_bvp_preview", {})
        )

        # Over/Under structural analysis
        ou_structure = self._ou_structural_analysis(
            away_sp_data, home_sp_data,
            game_data
        )

        return {
            "platoon": platoon,
            "away_era_mirage": away_mirage,
            "home_era_mirage": home_mirage,
            "away_walk_risk": away_walk_risk,
            "home_walk_risk": home_walk_risk,
            "away_manager": away_manager,
            "home_manager": home_manager,
            "first_exposure": first_exposure,
            "ou_structure": ou_structure,
            "combined_wp_adjustments": self._compile_wp_adjustments(
                platoon, away_mirage, home_mirage,
                away_walk_risk, home_walk_risk,
                first_exposure
            )
        }

    def _analyze_platoon(self, away_lineup: list, home_lineup: list,
                          away_sp: Dict, home_sp: Dict) -> Dict:
        """
        Analyze platoon advantages
        SP handedness vs lineup handedness
        """
        away_sp_hand = away_sp.get("throws", "R")
        home_sp_hand = home_sp.get("throws", "R")

        # Home lineup vs away SP
        home_advantage = self._calc_platoon_advantage(home_lineup, away_sp_hand)
        # Away lineup vs home SP
        away_advantage = self._calc_platoon_advantage(away_lineup, home_sp_hand)

        return {
            "home_lineup_vs_away_sp": home_advantage,
            "away_lineup_vs_home_sp": away_advantage,
            "home_platoon_wp": home_advantage.get("wp_adj", 0),
            "away_platoon_wp": away_advantage.get("wp_adj", 0)
        }

    def _calc_platoon_advantage(self, lineup: list, sp_throws: str) -> Dict:
        """Calculate platoon advantage for lineup vs pitcher"""
        if not lineup:
            return {"wp_adj": 0, "note": "lineup unknown"}

        opposite_hand_batters = 0
        same_hand_batters = 0

        for player in lineup:
            bats = player.get("bats", "R")
            if bats == "S":  # Switch hitter - always advantaged
                opposite_hand_batters += 0.5
            elif sp_throws == "L" and bats == "R":
                opposite_hand_batters += 1
            elif sp_throws == "R" and bats == "L":
                opposite_hand_batters += 1
            else:
                same_hand_batters += 1

        total = len(lineup)
        opp_pct = opposite_hand_batters / total if total > 0 else 0.5

        # Heavy platoon advantage = lineup is >65% opposite hand
        if opp_pct >= 0.65:
            return {"wp_adj": 0.020, "strength": "strong",
                    "opp_hand_pct": round(opp_pct, 2)}
        elif opp_pct >= 0.55:
            return {"wp_adj": 0.010, "strength": "mild",
                    "opp_hand_pct": round(opp_pct, 2)}
        elif opp_pct <= 0.35:
            return {"wp_adj": -0.015, "strength": "disadvantage",
                    "opp_hand_pct": round(opp_pct, 2)}
        else:
            return {"wp_adj": 0.0, "strength": "neutral",
                    "opp_hand_pct": round(opp_pct, 2)}

    def _classify_era_mirage(self, sp_data: Dict) -> Dict:
        """Classify ERA mirage magnitude and direction"""
        era = sp_data.get("era")
        xfip = sp_data.get("xfip")
        siera = sp_data.get("siera")

        if not era or not xfip:
            return {"active": False}

        magnitude = era - xfip  # Positive = ERA overstates badness (pitcher better than appears)

        return {
            "active": abs(magnitude) >= 1.00,
            "direction": "positive" if magnitude > 0 else "negative",
            "magnitude": round(magnitude, 2),
            "severity": (
                "extreme" if abs(magnitude) >= 3.00 else
                "large" if abs(magnitude) >= 2.00 else
                "moderate" if abs(magnitude) >= 1.50 else
                "mild" if abs(magnitude) >= 1.00 else
                "minimal"
            ),
            "interpretation": (
                "ERA dramatically overstates badness — true talent much better" if magnitude >= 2.0 else
                "ERA slightly overstates badness" if magnitude >= 1.0 else
                "ERA understates badness — regression incoming" if magnitude <= -2.0 else
                "ERA slightly understates badness" if magnitude <= -1.0 else
                "ERA near true talent"
            )
        }

    def _walk_risk_impact(self, sp_data: Dict) -> Dict:
        """Calculate walk risk impact on offense"""
        bb9 = sp_data.get("bb9")
        if not bb9:
            return {"active": False, "kelly_adj": 1.0}

        if bb9 >= 5.0:
            return {
                "active": True,
                "bb9": bb9,
                "severity": "maximum",
                "kelly_adj": 0.50,  # Half-Kelly when walk-risk confirmed
                "wp_boost_vs_patient_lineup": 0.025,
                "total_proj_boost": 0.8,
                "note": "BB/9 ≥ 5.0 — walk-risk maximum flag"
            }
        elif bb9 >= 4.0:
            return {
                "active": True,
                "bb9": bb9,
                "severity": "elevated",
                "kelly_adj": 0.75,
                "wp_boost_vs_patient_lineup": 0.015,
                "total_proj_boost": 0.4,
                "note": "BB/9 ≥ 4.0 — elevated walk risk"
            }
        else:
            return {
                "active": False,
                "bb9": bb9,
                "kelly_adj": 1.0
            }

    def _get_manager_tendency(self, team: str) -> Dict:
        """Get manager hook tendencies"""
        TEAM_MANAGERS = {
            "LAD": "Roberts",
            "ATL": "Snitker",
            "PHI": "Mattingly",
            "CLE": "Bell"
        }
        manager_name = TEAM_MANAGERS.get(team, "Unknown")
        return self.MANAGER_HOOKS.get(manager_name,
                                       {"hook_threshold": 95, "quick_hook": False,
                                        "name": manager_name})

    def _first_exposure_flags(self, away_sp: Dict, home_sp: Dict,
                               away_bvp: Dict, home_bvp: Dict) -> Dict:
        """
        First-exposure penalty when lineup has never faced this SP
        Reduces scoring expectation for that half of the game
        """
        away_debut = away_sp.get("flags", {}).get("debut_flag", False)
        home_debut = home_sp.get("flags", {}).get("debut_flag", False)

        away_games_vs_home_lineup = home_bvp.get("games_vs_sp", 99)
        home_games_vs_away_lineup = away_bvp.get("games_vs_sp", 99)

        return {
            "home_lineup_first_exposure": home_games_vs_away_lineup <= 0,
            "away_lineup_first_exposure": away_games_vs_home_lineup <= 0,
            "away_sp_debut": away_debut,
            "home_sp_debut": home_debut,
            "home_first_exp_wp_adj": -0.015 if home_games_vs_away_lineup <= 0 else 0,
            "away_first_exp_wp_adj": -0.015 if away_games_vs_home_lineup <= 0 else 0
        }

    def _ou_structural_analysis(self, away_sp: Dict, home_sp: Dict,
                                  game_data: Dict) -> Dict:
        """
        Determine structural Over/Under lean
        v2.4 Lesson #39: Under requires at least ONE elite SP (SIERA ≤ 3.00)
        """
        away_siera = away_sp.get("siera", 4.5)
        home_siera = home_sp.get("siera", 4.5)
        combined_siera = away_siera + home_siera

        # Both SPs below average = correlated risk for Under
        both_below_average = away_siera >= 4.50 and home_siera >= 4.50

        # Under anchor check — one elite SP required for high-confidence Under
        has_elite_anchor = away_siera <= 3.00 or home_siera <= 3.00
        has_above_avg_anchor = away_siera <= 3.50 or home_siera <= 3.50

        # Both SPs bad = Over structural lean
        both_bad = away_siera >= 5.00 and home_siera >= 5.00

        return {
            "combined_siera": round(combined_siera, 2),
            "both_below_average": both_below_average,
            "has_elite_anchor": has_elite_anchor,
            "has_above_avg_anchor": has_above_avg_anchor,
            "both_bad": both_bad,
            "structural_lean": (
                "strong_over" if both_bad else
                "over" if both_below_average and not has_above_avg_anchor else
                "under" if has_elite_anchor else
                "neutral"
            ),
            "under_confidence": (
                "high" if has_elite_anchor else
                "moderate" if has_above_avg_anchor else
                "low"  # Lesson #39 — risky Under without elite anchor
            )
        }

    def _compile_wp_adjustments(self, platoon: Dict, away_mirage: Dict,
                                  home_mirage: Dict, away_walk: Dict,
                                  home_walk: Dict, first_exp: Dict) -> Dict:
        """Compile all WP adjustments from advanced metrics"""
        adjustments = {}

        # Platoon
        adjustments["home_platoon"] = platoon.get("home_platoon_wp", 0)
        adjustments["away_platoon"] = platoon.get("away_platoon_wp", 0)

        # ERA mirage corrections
        if away_mirage.get("active"):
            mag = away_mirage.get("magnitude", 0)
            # Positive mirage: away pitcher better than ERA shows → suppress away run scoring
            adjustments["away_sp_correction"] = -(mag * 0.005)

        if home_mirage.get("active"):
            mag = home_mirage.get("magnitude", 0)
            adjustments["home_sp_correction"] = -(mag * 0.005)

        # Walk-risk boosts offense of patient lineup
        if home_walk.get("active"):
            adjustments["home_batting_walk_boost"] = home_walk.get(
                "wp_boost_vs_patient_lineup", 0)

        if away_walk.get("active"):
            adjustments["away_batting_walk_boost"] = away_walk.get(
                "wp_boost_vs_patient_lineup", 0)

        # First exposure
        adjustments["home_first_exp"] = first_exp.get("home_first_exp_wp_adj", 0)
        adjustments["away_first_exp"] = first_exp.get("away_first_exp_wp_adj", 0)

        return adjustments
