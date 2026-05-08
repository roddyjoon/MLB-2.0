"""
Agent 7: Bullpen Quality Agent
Individual reliever xFIP, closer availability, high-leverage quality,
BP ERA last 7 days — the single biggest blind spot in v2.4
"""

import asyncio
from typing import Dict, List, Optional
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("bullpen_agent")

# Elite closer database — updated weekly
ELITE_CLOSERS = {
    "miller_mason": {"team": "SD", "era": 0.00, "xfip": 1.80, "scoreless_streak": 34},
    "clase_emmanuel": {"team": "CLE", "era": 1.20, "xfip": 2.10, "saves": 12},
    "hader_josh": {"team": "HOU", "era": 1.80, "xfip": 2.40},
    "diaz_edwin": {"team": "NYM", "era": 2.10, "xfip": 2.80},
    "orze_eric": {"team": "MIN", "era": 3.40, "xfip": 3.80, "blown_saves": 3},
    "obrien_riley": {"team": "PIT", "era": 0.80, "xfip": 2.20, "saves": 8},
}

# Bullpen tier thresholds
BP_TIERS = {
    "elite":    {"max_era": 3.00, "wp_boost": 0.030},
    "above_avg":{"max_era": 3.80, "wp_boost": 0.015},
    "average":  {"max_era": 4.40, "wp_boost": 0.000},
    "below_avg":{"max_era": 5.20, "wp_boost": -0.015},
    "poor":     {"max_era": 99.0, "wp_boost": -0.030},
}


class BullpenQualityAgent:
    """
    Models bullpen quality at the individual reliever level
    
    Key insight: SP pitches ~6 innings, BP pitches ~3
    Total projection accuracy requires knowing BOTH
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        home = game_data.get("home_team")
        away = game_data.get("away_team")

        # Pull both bullpens in parallel
        home_bp_task = self._analyze_bullpen(home)
        away_bp_task = self._analyze_bullpen(away)

        home_bp, away_bp = await asyncio.gather(
            home_bp_task, away_bp_task, return_exceptions=True
        )

        home_bp = home_bp if not isinstance(home_bp, Exception) else self._default_bp()
        away_bp = away_bp if not isinstance(away_bp, Exception) else self._default_bp()

        # BP quality impact on WP and totals
        home_bp_wp = self._bp_to_wp_adj(home_bp)
        away_bp_wp = self._bp_to_wp_adj(away_bp)

        # Late-game hold probability (affects close game WP)
        home_hold_prob = self._calculate_hold_prob(home_bp)
        away_hold_prob = self._calculate_hold_prob(away_bp)

        # Total projection adjustment
        # Better BP = fewer runs allowed in innings 7-9
        home_bp_runs_saved = self._bp_runs_saved(home_bp)
        away_bp_runs_saved = self._bp_runs_saved(away_bp)

        return {
            "home_bp": home_bp,
            "away_bp": away_bp,
            "home_bp_wp_adj": home_bp_wp,
            "away_bp_wp_adj": away_bp_wp,
            "home_hold_prob": home_hold_prob,
            "away_hold_prob": away_hold_prob,
            "home_bp_runs_saved": home_bp_runs_saved,
            "away_bp_runs_saved": away_bp_runs_saved,
            "total_proj_adj": -(home_bp_runs_saved + away_bp_runs_saved),
            "matchup_notes": self._generate_bp_notes(home_bp, away_bp, home, away)
        }

    async def _analyze_bullpen(self, team: str) -> Dict:
        """Full bullpen analysis for a team"""

        # Get bullpen stats from API
        bp_stats = await self.mlb_api.get_bullpen_stats(team)
        relievers = await self.mlb_api.get_reliever_list(team)
        closer_avail = await self.mlb_api.get_closer_availability(team)

        # Season ERA and recent ERA
        season_era = bp_stats.get("season_era", 4.20)
        last_7_era = bp_stats.get("last_7_era", season_era)
        xfip = bp_stats.get("xfip", 4.20)
        k9 = bp_stats.get("k9", 9.0)
        bb9 = bp_stats.get("bb9", 3.5)
        hr9 = bp_stats.get("hr9", 1.2)
        holds = bp_stats.get("holds", 0)
        blown_saves = bp_stats.get("blown_saves", 0)
        save_pct = bp_stats.get("save_pct", 0.70)

        # Identify closer
        closer = self._identify_closer(team, relievers, closer_avail)

        # High-leverage relievers
        high_lev = self._identify_high_leverage(relievers)

        # Tier classification
        tier = self._classify_tier(last_7_era, xfip)

        # Availability flags
        depleted = bp_stats.get("high_usage_last_2", False)
        unavailable = closer_avail.get("unavailable_arms", [])

        return {
            "team": team,
            "season_era": season_era,
            "last_7_era": last_7_era,
            "xfip": xfip,
            "k9": k9,
            "bb9": bb9,
            "hr9": hr9,
            "save_pct": save_pct,
            "blown_saves": blown_saves,
            "tier": tier,
            "closer": closer,
            "high_leverage": high_lev,
            "depleted": depleted,
            "unavailable": unavailable,
            "quality_score": self._bp_quality_score(
                last_7_era, xfip, save_pct, depleted, closer
            )
        }

    def _identify_closer(self, team: str, relievers: List,
                          avail: Dict) -> Dict:
        """Identify and assess closer availability"""
        # Check known elite closers
        for key, data in ELITE_CLOSERS.items():
            if data["team"] == team:
                is_available = avail.get("closer_available", True)
                return {
                    "name": key.replace("_", " ").title(),
                    "era": data["era"],
                    "xfip": data["xfip"],
                    "available": is_available,
                    "elite": data["xfip"] <= 2.50,
                    "blown_saves": data.get("blown_saves", 0)
                }

        # Generic closer from roster
        for r in relievers:
            if r.get("role") == "closer":
                return {
                    "name": r.get("name", "Unknown"),
                    "era": r.get("era", 3.50),
                    "xfip": r.get("xfip", 3.80),
                    "available": True,
                    "elite": False
                }

        return {"name": "Unknown", "era": 3.50, "xfip": 3.80,
                "available": True, "elite": False}

    def _identify_high_leverage(self, relievers: List) -> List:
        """Identify high-leverage relievers"""
        high_lev = []
        for r in relievers:
            if r.get("leverage_index", 1.0) >= 1.5:
                high_lev.append({
                    "name": r.get("name"),
                    "era": r.get("era"),
                    "xfip": r.get("xfip"),
                    "li": r.get("leverage_index")
                })
        return sorted(high_lev, key=lambda x: x.get("li", 0), reverse=True)[:3]

    def _classify_tier(self, era: float, xfip: float) -> str:
        """Classify bullpen tier"""
        # Use blend of ERA and xFIP
        blend = 0.5 * era + 0.5 * xfip
        for tier_name, limits in BP_TIERS.items():
            if blend <= limits["max_era"]:
                return tier_name
        return "poor"

    def _bp_to_wp_adj(self, bp: Dict) -> float:
        """Convert BP quality to WP adjustment"""
        tier = bp.get("tier", "average")
        base_adj = BP_TIERS.get(tier, {}).get("wp_boost", 0)

        # Additional adjustments
        depleted_penalty = -0.015 if bp.get("depleted") else 0
        no_closer_penalty = -0.010 if not bp.get("closer", {}).get("available") else 0
        elite_closer_bonus = 0.010 if bp.get("closer", {}).get("elite") else 0

        return round(base_adj + depleted_penalty +
                     no_closer_penalty + elite_closer_bonus, 3)

    def _calculate_hold_prob(self, bp: Dict) -> float:
        """Probability of holding a lead in 7th-9th innings"""
        save_pct = bp.get("save_pct", 0.70)
        tier = bp.get("tier", "average")

        # Base from save percentage
        base = save_pct

        # Tier adjustment
        tier_adj = {
            "elite": 0.05,
            "above_avg": 0.02,
            "average": 0.0,
            "below_avg": -0.05,
            "poor": -0.10
        }.get(tier, 0)

        depleted_adj = -0.05 if bp.get("depleted") else 0

        return round(min(0.95, max(0.50, base + tier_adj + depleted_adj)), 3)

    def _bp_runs_saved(self, bp: Dict) -> float:
        """
        Calculate runs saved by BP quality vs league average
        League avg BP ERA ~4.20 → ~1.40 runs per 3 innings
        """
        last_7_era = bp.get("last_7_era", 4.20)
        league_avg_era = 4.20

        era_diff = league_avg_era - last_7_era  # Positive = better than avg
        innings_pitched = 3.0  # BP pitches ~3 innings per game

        runs_saved = (era_diff / 9) * innings_pitched

        # Depleted BP penalty
        if bp.get("depleted"):
            runs_saved -= 0.30

        return round(runs_saved, 2)

    def _bp_quality_score(self, era: float, xfip: float,
                           save_pct: float, depleted: bool,
                           closer: Dict) -> int:
        """0-100 bullpen quality score"""
        score = 50

        # ERA component
        if era <= 2.50: score += 20
        elif era <= 3.20: score += 12
        elif era <= 3.80: score += 5
        elif era <= 4.50: score += 0
        elif era <= 5.20: score -= 8
        else: score -= 18

        # xFIP component
        if xfip <= 2.80: score += 15
        elif xfip <= 3.40: score += 8
        elif xfip <= 4.00: score += 2
        elif xfip <= 4.60: score -= 5
        else: score -= 12

        # Save percentage
        if save_pct >= 0.85: score += 8
        elif save_pct >= 0.75: score += 4
        elif save_pct <= 0.60: score -= 8

        # Closer bonus
        if closer.get("elite"): score += 10
        elif closer.get("available"): score += 3

        # Depletion penalty
        if depleted: score -= 12

        return max(0, min(100, score))

    def _generate_bp_notes(self, home_bp: Dict, away_bp: Dict,
                            home: str, away: str) -> List[str]:
        """Generate key BP matchup notes"""
        notes = []

        home_tier = home_bp.get("tier", "average")
        away_tier = away_bp.get("tier", "average")

        if home_tier == "elite":
            notes.append(f"{home} BP elite — leads project to wins")
        elif home_tier == "poor":
            notes.append(f"🚨 {home} BP poor — leads not safe")

        if away_tier == "elite":
            notes.append(f"{away} BP elite — holds leads on road")
        elif away_tier == "poor":
            notes.append(f"🚨 {away} BP poor — road leads vulnerable")

        if home_bp.get("depleted"):
            notes.append(f"⚠️ {home} BP depleted — high usage last 2 days")
        if away_bp.get("depleted"):
            notes.append(f"⚠️ {away} BP depleted — high usage last 2 days")

        # Closer notes
        home_closer = home_bp.get("closer", {})
        away_closer = away_bp.get("closer", {})

        if not home_closer.get("available"):
            notes.append(f"🚨 {home} closer unavailable")
        if away_closer.get("blown_saves", 0) >= 3:
            notes.append(f"⚠️ {away} closer {away_closer.get('blown_saves')} blown saves")

        return notes

    def _default_bp(self) -> Dict:
        """Default BP when data unavailable"""
        return {
            "season_era": 4.20, "last_7_era": 4.20,
            "xfip": 4.20, "tier": "average",
            "save_pct": 0.70, "depleted": False,
            "quality_score": 50,
            "closer": {"available": True, "elite": False}
        }
