"""
Agent 9: Pitch Arsenal Matchup Matrix
SP pitch-type xwOBA vs LHB/RHB, lineup vulnerability detection,
specific pitch exploitation — turns platoon data from qualitative to quantitative
"""

import asyncio
from typing import Dict, List, Optional
from data.mlb_api import MLBDataAPI
from data.savant_api import SavantAPI
from core.logger import ModelLogger

logger = ModelLogger("arsenal_agent")


class PitchArsenalAgent:
    """
    Models SP pitch arsenal vs opposing lineup handedness
    
    Key insight: Overall xwOBA misses pitch-type specific vulnerabilities
    Example: Baz four-seam .414 wOBA vs LHB specifically
    → Facing 70% LHB lineup = amplified run risk beyond overall xwOBA
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()
        self.savant = SavantAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        """Full pitch arsenal matchup analysis"""

        home_sp_id = game_data.get("home_sp_id")
        away_sp_id = game_data.get("away_sp_id")

        # Pull arsenal data for both SPs
        home_arsenal_task = self._get_arsenal(home_sp_id)
        away_arsenal_task = self._get_arsenal(away_sp_id)

        # Pull lineup handedness
        home_lineup_task = self.mlb_api.get_confirmed_lineup(
            game_data.get("home_team"))
        away_lineup_task = self.mlb_api.get_confirmed_lineup(
            game_data.get("away_team"))

        (home_arsenal, away_arsenal,
         home_lineup, away_lineup) = await asyncio.gather(
            home_arsenal_task, away_arsenal_task,
            home_lineup_task, away_lineup_task,
            return_exceptions=True
        )

        home_arsenal = home_arsenal if not isinstance(
            home_arsenal, Exception) else {}
        away_arsenal = away_arsenal if not isinstance(
            away_arsenal, Exception) else {}
        home_lineup = home_lineup if not isinstance(
            home_lineup, Exception) else []
        away_lineup = away_lineup if not isinstance(
            away_lineup, Exception) else []

        # Calculate matchup scores
        # Away SP faces home lineup
        home_vulnerability = self._calc_lineup_vulnerability(
            away_arsenal, home_lineup)
        # Home SP faces away lineup
        away_vulnerability = self._calc_lineup_vulnerability(
            home_arsenal, away_lineup)

        # Identify specific exploits and concerns
        home_exploits = self._find_exploits(away_arsenal, home_lineup)
        away_exploits = self._find_exploits(home_arsenal, away_lineup)

        return {
            "home_vs_away_sp": home_vulnerability,
            "away_vs_home_sp": away_vulnerability,
            "home_lineup_exploits": home_exploits,
            "away_lineup_exploits": away_exploits,
            "home_arsenal_summary": self._summarize_arsenal(home_arsenal),
            "away_arsenal_summary": self._summarize_arsenal(away_arsenal),
            "wp_adjustments": self._calculate_wp_adj(
                home_vulnerability, away_vulnerability)
        }

    async def _get_arsenal(self, pitcher_id: Optional[str]) -> Dict:
        """Get pitch arsenal data from Savant"""
        if not pitcher_id:
            return {}

        try:
            raw = await self.savant.get_pitcher_arsenal(pitcher_id)
            return self._parse_arsenal(raw)
        except Exception as e:
            logger.warning(f"Arsenal fetch failed for {pitcher_id}: {e}")
            return {}

    def _parse_arsenal(self, raw: Dict) -> Dict:
        """Parse and structure arsenal data"""
        if not raw:
            return {}

        arsenal = {}
        for pitch_type, data in raw.items():
            arsenal[pitch_type] = {
                "usage_pct": data.get("usage_pct", 0),
                "velo": data.get("velo", 92),
                "whiff_rate": data.get("whiff_rate", 0.25),
                "xwoba_vs_lhb": data.get("xwoba_vs_lhb", 0.320),
                "xwoba_vs_rhb": data.get("xwoba_vs_rhb", 0.320),
                "xwoba_overall": data.get("xwoba_overall", 0.320),
                "hh_pct": data.get("hh_pct", 0.35),
                "put_away_rate": data.get("put_away_rate", 0.15),
                "is_primary": data.get("usage_pct", 0) >= 0.20,
                "is_elite": data.get("xwoba_overall", 0.320) <= 0.250,
                "is_vulnerable": data.get("xwoba_overall", 0.320) >= 0.400
            }
        return arsenal

    def _calc_lineup_vulnerability(self, sp_arsenal: Dict,
                                    lineup: List) -> Dict:
        """
        Calculate how vulnerable a lineup is to this SP's arsenal
        
        Key calculation: weighted xwOBA by pitch usage × lineup handedness
        """
        if not sp_arsenal or not lineup:
            return {"score": 0.320, "confidence": "low", "details": {}}

        # Get lineup handedness breakdown
        lhb_count = sum(1 for p in lineup if p.get("bats") == "L")
        rhb_count = sum(1 for p in lineup if p.get("bats") in ["R", "S"])
        total = max(len(lineup), 9)
        lhb_pct = lhb_count / total
        rhb_pct = rhb_count / total

        # Calculate weighted xwOBA for lineup
        weighted_xwoba = 0
        total_usage = 0

        for pitch_type, pitch_data in sp_arsenal.items():
            usage = pitch_data.get("usage_pct", 0)
            xwoba_lhb = pitch_data.get("xwoba_vs_lhb", 0.320)
            xwoba_rhb = pitch_data.get("xwoba_vs_rhb", 0.320)

            # Weighted by handedness breakdown
            blended_xwoba = (lhb_pct * xwoba_lhb + rhb_pct * xwoba_rhb)
            weighted_xwoba += usage * blended_xwoba
            total_usage += usage

        if total_usage > 0:
            final_xwoba = weighted_xwoba / total_usage
        else:
            final_xwoba = 0.320

        # Vulnerability score
        vulnerability = self._score_vulnerability(final_xwoba)

        return {
            "weighted_xwoba": round(final_xwoba, 3),
            "lhb_pct": round(lhb_pct, 2),
            "rhb_pct": round(rhb_pct, 2),
            "vulnerability": vulnerability,
            "confidence": "high" if len(sp_arsenal) >= 3 else "medium"
        }

    def _score_vulnerability(self, xwoba: float) -> str:
        """Score how vulnerable lineup is"""
        if xwoba <= 0.220: return "dominated"
        elif xwoba <= 0.260: return "suppressed"
        elif xwoba <= 0.300: return "below_average"
        elif xwoba <= 0.340: return "average"
        elif xwoba <= 0.380: return "exploitable"
        elif xwoba <= 0.420: return "vulnerable"
        else: return "dominated_by_lineup"

    def _find_exploits(self, sp_arsenal: Dict, lineup: List) -> List:
        """
        Find specific pitch-lineup exploits
        Both vulnerabilities (SP weakness) and strengths (SP dominance)
        """
        exploits = []

        lhb_count = sum(1 for p in lineup if p.get("bats") == "L")
        rhb_count = sum(1 for p in lineup if p.get("bats") in ["R", "S"])
        total = max(len(lineup), 9)

        for pitch_type, data in sp_arsenal.items():
            usage = data.get("usage_pct", 0)
            if usage < 0.10:
                continue  # Skip rarely-used pitches

            xwoba_lhb = data.get("xwoba_vs_lhb", 0.320)
            xwoba_rhb = data.get("xwoba_vs_rhb", 0.320)

            # Specific vulnerability: heavy LHB lineup vs LHB-vulnerable pitch
            if lhb_count >= 5 and xwoba_lhb >= 0.400:
                exploits.append({
                    "type": "lineup_exploits_sp",
                    "pitch": pitch_type,
                    "detail": f"{pitch_type} vs LHB: .{int(xwoba_lhb*1000)} xwOBA — vulnerable",
                    "handedness": "vs_LHB",
                    "severity": "high" if xwoba_lhb >= 0.440 else "moderate",
                    "run_impact": +0.40 if xwoba_lhb >= 0.440 else +0.20
                })

            # SP dominates lineup with this pitch
            if data.get("is_elite") and (
                    (xwoba_lhb <= 0.220 and lhb_count >= 5) or
                    (xwoba_rhb <= 0.220 and rhb_count >= 5)):
                exploits.append({
                    "type": "sp_dominates_lineup",
                    "pitch": pitch_type,
                    "detail": f"{pitch_type}: elite put-away vs this lineup",
                    "severity": "high",
                    "run_impact": -0.35
                })

            # High whiff vs lineup's K-prone bats
            if data.get("whiff_rate", 0) >= 0.45 and usage >= 0.20:
                exploits.append({
                    "type": "strikeout_weapon",
                    "pitch": pitch_type,
                    "detail": f"{pitch_type}: {data['whiff_rate']*100:.0f}% whiff rate",
                    "severity": "moderate",
                    "run_impact": -0.15
                })

        return sorted(exploits,
                      key=lambda x: abs(x.get("run_impact", 0)),
                      reverse=True)[:5]

    def _summarize_arsenal(self, arsenal: Dict) -> Dict:
        """Summarize arsenal quality"""
        if not arsenal:
            return {}

        elite_pitches = [p for p, d in arsenal.items() if d.get("is_elite")]
        vulnerable_pitches = [p for p, d in arsenal.items()
                               if d.get("is_vulnerable")]
        primary_pitches = [p for p, d in arsenal.items() if d.get("is_primary")]

        overall_xwoba = sum(
            d.get("xwoba_overall", 0.320) * d.get("usage_pct", 0)
            for d in arsenal.values()
        )
        total_usage = sum(d.get("usage_pct", 0) for d in arsenal.values())
        avg_xwoba = overall_xwoba / total_usage if total_usage > 0 else 0.320

        return {
            "pitch_count": len(arsenal),
            "elite_pitches": elite_pitches,
            "vulnerable_pitches": vulnerable_pitches,
            "primary_pitches": primary_pitches,
            "arsenal_xwoba": round(avg_xwoba, 3),
            "has_elite_weapon": len(elite_pitches) > 0,
            "has_vulnerability": len(vulnerable_pitches) > 0
        }

    def _calculate_wp_adj(self, home_vuln: Dict,
                           away_vuln: Dict) -> Dict:
        """
        Calculate WP adjustments from arsenal matchup
        
        Higher home lineup xwOBA vs away SP = more runs for home = higher home WP
        """
        home_xwoba = home_vuln.get("weighted_xwoba", 0.320)
        away_xwoba = away_vuln.get("weighted_xwoba", 0.320)

        league_avg = 0.320

        # Home lineup advantage vs away SP
        home_offense_adj = (home_xwoba - league_avg) * 2.0
        # Away lineup advantage vs home SP
        away_offense_adj = (away_xwoba - league_avg) * 2.0

        # Net WP impact
        home_wp_adj = home_offense_adj - away_offense_adj

        # Total projection impact
        total_proj_adj = (home_xwoba + away_xwoba - 2 * league_avg) * 3.0

        return {
            "home_wp_adj": round(home_wp_adj, 3),
            "total_proj_adj": round(total_proj_adj, 2),
            "home_offense_adj": round(home_offense_adj, 3),
            "away_offense_adj": round(away_offense_adj, 3)
        }
