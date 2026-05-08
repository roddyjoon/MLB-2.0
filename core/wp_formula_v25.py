"""
WP Formula v2.5 — Integrates all 13 agent outputs
IRON RULE STILL ENFORCED: Market never enters WP calculation
New inputs: bullpen, umpire, arsenal, platoon, regression, f1/framing
"""

from typing import Dict
from core.logger import ModelLogger

logger = ModelLogger("wp_formula_v25")


class WPFormulaV25:
    """
    v2.5 Win Probability Calculator
    
    New steps vs v2.4:
    3b. Catcher framing tier adjustment (replaces binary flag)
    5b. Umpire run expectancy adjustment
    6b. Bullpen quality adjustment
    6c. Pitch arsenal matchup adjustment
    6d. Platoon splits database adjustment
    6e. Regression detection adjustment
    6f. First inning model adjustment
    """

    HOME_FIELD_BASE = 0.540

    def calculate(self, inputs: Dict) -> Dict:
        """
        Calculate WP with all 13 agent inputs
        Returns home_wp, away_wp, total_projection, under_wp
        """

        sp = inputs.get("sp", {})
        lineup = inputs.get("lineup", {})
        trends = inputs.get("trends", {})
        injuries = inputs.get("injuries", {})
        park = inputs.get("park", {})
        metrics = inputs.get("metrics", {})

        # v2.5 additions
        bullpen = inputs.get("bullpen", {})
        umpire = inputs.get("umpire", {})
        arsenal = inputs.get("arsenal", {})
        platoon_splits = inputs.get("platoon", {})
        regression = inputs.get("regression", {})
        first_inning = inputs.get("first_inning", {})

        home_wp = self.HOME_FIELD_BASE
        reasoning = {"home_field_base": home_wp}

        # ============================================================
        # STEP 1: SP gap (same as v2.4)
        # ============================================================
        fip_gap = sp.get("fip_gap", 0)
        fip_favor = sp.get("fip_gap_favor", "home")
        sp_wp_adj = fip_gap * 0.05
        if fip_favor == "home":
            home_wp += sp_wp_adj
        else:
            home_wp -= abs(sp_wp_adj)
        reasoning["sp_gap"] = sp_wp_adj if fip_favor == "home" else -abs(sp_wp_adj)

        # ============================================================
        # STEP 2: Recency adjustment
        # ============================================================
        home_sp = sp.get("home_sp", {})
        away_sp = sp.get("away_sp", {})
        home_trend = home_sp.get("recent_form", {}).get("trend", "average")
        away_trend = away_sp.get("recent_form", {}).get("trend", "average")
        trend_map = {"hot": 0.025, "average": 0.0, "cold": -0.025}
        home_wp += trend_map.get(home_trend, 0)
        home_wp -= trend_map.get(away_trend, 0)
        reasoning["recency"] = (trend_map.get(home_trend, 0) -
                                 trend_map.get(away_trend, 0))

        # ============================================================
        # STEP 3: Catcher framing (v2.5 tier system)
        # ============================================================
        f1_wp = first_inning.get("wp_adjustments", {})
        framing_adj = f1_wp.get("framing_wp_adj", 0)
        home_wp += framing_adj
        reasoning["framing_tier"] = framing_adj

        # ============================================================
        # STEP 4: Walk-risk flag
        # ============================================================
        home_walk = metrics.get("home_walk_risk", {})
        away_walk = metrics.get("away_walk_risk", {})
        if home_walk.get("active"):
            home_wp -= 0.020
        if away_walk.get("active"):
            home_wp += 0.020
        reasoning["walk_risk"] = (
            (-0.020 if home_walk.get("active") else 0) +
            (0.020 if away_walk.get("active") else 0)
        )

        # ============================================================
        # STEP 5: Umpire run expectancy (v2.5 new)
        # ============================================================
        ump_impact = umpire.get("impact", {})
        walk_amplifier = ump_impact.get("walk_risk_amplifier", 1.0)

        # Apply walk amplifier to walk-risk adjustments
        if (home_walk.get("active") or away_walk.get("active")):
            walk_correction = reasoning["walk_risk"] * (walk_amplifier - 1.0)
            home_wp += walk_correction * 0.5
            reasoning["umpire_walk_amp"] = walk_correction * 0.5

        # ============================================================
        # STEP 6: Behavioral (v2.4 + new)
        # ============================================================
        situational = trends.get("situational_flags", {})

        # Back-to-back blowout (Lesson #30)
        if situational.get("lesson_30_active"):
            bounce = situational.get("bounce_back_team")
            if bounce == "home":
                home_wp += 0.040
                reasoning["lesson_30"] = 0.040
            elif bounce == "away":
                home_wp -= 0.040
                reasoning["lesson_30"] = -0.040

        # Momentum caps
        home_cap = situational.get("momentum_home_cap", 1.0)
        away_cap = situational.get("momentum_away_cap", 1.0)
        if home_cap < 1.0:
            excess = max(home_wp - 0.50, 0)
            home_wp -= excess * (1 - home_cap)
        if away_cap < 1.0:
            excess = max((1 - home_wp) - 0.50, 0)
            home_wp += excess * (1 - away_cap)

        if situational.get("home_dominant_at_home"):
            home_wp += 0.015
        if situational.get("away_brutal_road"):
            home_wp += 0.020
        if situational.get("home_overachieving"):
            home_wp -= 0.010
        if situational.get("away_overachieving"):
            home_wp += 0.010

        # ============================================================
        # STEP 6b: Bullpen quality (v2.5 new)
        # ============================================================
        home_bp_adj = bullpen.get("home_bp_wp_adj", 0)
        away_bp_adj = bullpen.get("away_bp_wp_adj", 0)
        home_wp += home_bp_adj
        home_wp -= away_bp_adj
        reasoning["bullpen"] = home_bp_adj - away_bp_adj

        # ============================================================
        # STEP 6c: Pitch arsenal matchup (v2.5 new)
        # ============================================================
        arsenal_adj = arsenal.get("wp_adjustments", {})
        home_wp += arsenal_adj.get("home_wp_adj", 0)
        reasoning["arsenal"] = arsenal_adj.get("home_wp_adj", 0)

        # ============================================================
        # STEP 6d: Platoon splits database (v2.5 new)
        # ============================================================
        platoon_wp = platoon_splits.get("wp_impact", {})
        home_wp += platoon_wp.get("net_home_wp_adj", 0)

        # Structural flags (e.g., NYM vs LHP)
        structural = platoon_splits.get("structural_flags", {})
        home_wp += structural.get("total_wp_adj", 0)
        reasoning["platoon_splits"] = (platoon_wp.get("net_home_wp_adj", 0) +
                                        structural.get("total_wp_adj", 0))

        # ============================================================
        # STEP 6e: Regression detection (v2.5 new)
        # ============================================================
        reg_adj = regression.get("wp_adjustments", {})
        home_wp += reg_adj.get("net_home_wp", 0)
        reasoning["regression"] = reg_adj.get("net_home_wp", 0)

        # ============================================================
        # STEP 6f: First inning model (v2.5 new)
        # ============================================================
        f1_adj = f1_wp.get("f1_wp_adj", 0)
        home_wp += f1_adj
        reasoning["first_inning"] = f1_adj

        # ============================================================
        # STEP 7: wRC+ differential
        # ============================================================
        home_lineup = lineup.get("home", {})
        away_lineup = lineup.get("away", {})
        home_wrc = home_lineup.get("team_wrc_plus", 100)
        away_wrc = away_lineup.get("team_wrc_plus", 100)
        wrc_gap = home_wrc - away_wrc
        wrc_wp_adj = wrc_gap * 0.002
        home_wp += wrc_wp_adj
        reasoning["wrc_gap"] = wrc_wp_adj

        # Streak adjustments
        home_streak = situational.get("home_streak", {})
        away_streak = situational.get("away_streak", {})
        if home_streak.get("type") == "W" and home_streak.get("count", 0) >= 4:
            home_wp += 0.010
        elif home_streak.get("type") == "L" and home_streak.get("count", 0) >= 4:
            home_wp -= 0.015
        if away_streak.get("type") == "W" and away_streak.get("count", 0) >= 4:
            home_wp -= 0.010
        elif away_streak.get("type") == "L" and away_streak.get("count", 0) >= 4:
            home_wp += 0.015

        # ============================================================
        # STEP 8: Injury adjustments
        # ============================================================
        home_inj = injuries.get("home_impact", {}).get("wp_adj", 0)
        away_inj = injuries.get("away_impact", {}).get("wp_adj", 0)
        home_wp += home_inj
        home_wp -= away_inj
        reasoning["injuries"] = home_inj - away_inj

        # ============================================================
        # STEP 9: Park factor
        # ============================================================
        pf = park.get("park", {}).get("runs", 1.00)
        if pf >= 1.08:
            home_wp += 0.010
        elif pf <= 0.90:
            home_wp += 0.015
        reasoning["park"] = park.get("environment", {}).get("over_lean", 0)

        # ============================================================
        # FINALIZE
        # ============================================================
        home_wp = max(0.20, min(0.82, home_wp))
        away_wp = 1.0 - home_wp

        reasoning["final_home_wp"] = round(home_wp, 3)
        reasoning["final_away_wp"] = round(away_wp, 3)

        # Total projection (v2.5 enhanced)
        total = self._project_total_v25(
            sp, lineup, park, bullpen, umpire, arsenal, first_inning
        )

        under_wp = self._calc_under_wp(total)

        return {
            "home_wp": round(home_wp, 3),
            "away_wp": round(away_wp, 3),
            "total_projection": round(total, 1),
            "under_wp": round(under_wp, 3),
            "over_wp": round(1 - under_wp, 3),
            "reasoning": reasoning,
            "version": "v2.5"
        }

    def _project_total_v25(self, sp: Dict, lineup: Dict,
                            park: Dict, bullpen: Dict,
                            umpire: Dict, arsenal: Dict,
                            first_inning: Dict) -> float:
        """Enhanced total projection using all new data streams"""

        home_sp = sp.get("home_sp", {})
        away_sp = sp.get("away_sp", {})
        home_siera = home_sp.get("siera", 4.50)
        away_siera = away_sp.get("siera", 4.50)

        # SP contribution (6 innings)
        home_sp_runs = away_siera * (6/9)
        away_sp_runs = home_siera * (6/9)

        # Lineup RPG
        home_rpg = lineup.get("home", {}).get("rpg", 4.5)
        away_rpg = lineup.get("away", {}).get("rpg", 4.5)

        home_proj = (home_sp_runs * 0.40) + (home_rpg * 0.60)
        away_proj = (away_sp_runs * 0.40) + (away_rpg * 0.60)

        base_total = home_proj + away_proj

        # Park factor
        pf = park.get("park", {}).get("runs", 1.00)
        park_adj = base_total * pf

        # Weather
        env = park.get("environment", {})
        weather_adj = env.get("run_proj_adj", 0)

        # Umpire run expectancy (v2.5)
        ump_adj = umpire.get("total_proj_adj", 0)

        # Bullpen quality (v2.5) — better BPs save runs in innings 7-9
        bp_adj = bullpen.get("total_proj_adj", 0)

        # Arsenal matchup (v2.5) — better arsenal matchup = fewer runs
        arsenal_adj = arsenal.get("wp_adjustments", {}).get("total_proj_adj", 0)

        # Framing (v2.5) — better framing = fewer runs
        framing_adj = -first_inning.get("total_framing_adj", 0) * 0.5

        total = (park_adj + weather_adj + ump_adj +
                 bp_adj + arsenal_adj + framing_adj)

        return max(3.0, min(22.0, total))

    def _calc_under_wp(self, projection: float,
                        std_dev: float = 1.8) -> float:
        """Base under probability — compared to actual line in edge calc"""
        return 0.50  # Actual edge from line comparison in EdgeCalculator
