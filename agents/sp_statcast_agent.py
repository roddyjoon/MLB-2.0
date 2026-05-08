"""
Agent 1: SP Statcast Agent
Deep-dives xwOBA, EV, HH%, Barrel%, whiff rates by pitch type,
recent form (last 3 starts weighted), platoon splits
"""

import asyncio
from typing import Dict, Optional
from data.savant_api import SavantAPI
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("sp_statcast_agent")

class SPStatcastAgent:
    """
    Pulls comprehensive SP Statcast data from Baseball Savant
    Calculates xFIP/SIERA blend, comp score, walk-risk flags
    """
    
    def __init__(self):
        self.savant = SavantAPI()
        self.mlb_api = MLBDataAPI()
    
    async def analyze(self, game_data: Dict) -> Dict:
        """Full SP Statcast analysis for both starters"""
        
        away_sp_id = game_data.get("away_sp_id")
        home_sp_id = game_data.get("home_sp_id")
        away_sp_name = game_data.get("away_sp_name", "TBD")
        home_sp_name = game_data.get("home_sp_name", "TBD")
        
        # Pull both SPs in parallel
        away_task = self._analyze_pitcher(away_sp_id, away_sp_name, is_home=False)
        home_task = self._analyze_pitcher(home_sp_id, home_sp_name, is_home=True)
        
        away_sp, home_sp = await asyncio.gather(away_task, home_task)
        
        # Calculate FIP gap
        fip_gap = (home_sp.get("adjusted_blend", 4.5) - 
                   away_sp.get("adjusted_blend", 4.5))
        
        return {
            "away_sp": away_sp,
            "home_sp": home_sp,
            "fip_gap": round(fip_gap, 2),
            "fip_gap_favor": "home" if fip_gap > 0 else "away",
            "primary_threshold_met": abs(fip_gap) >= 1.50,
            "flags": self._extract_flags(away_sp, home_sp)
        }
    
    async def _analyze_pitcher(self, pitcher_id: Optional[str], 
                                name: str, is_home: bool) -> Dict:
        """Full analysis for single pitcher"""
        
        if not pitcher_id:
            return self._debut_proxy(name)
        
        # Pull Statcast data
        statcast_task = self.savant.get_pitcher_statcast(pitcher_id)
        season_task = self.mlb_api.get_pitcher_season_stats(pitcher_id)
        gamelogs_task = self.mlb_api.get_pitcher_gamelogs(pitcher_id, last_n=5)
        splits_task = self.mlb_api.get_pitcher_splits(pitcher_id)
        
        statcast, season, gamelogs, splits = await asyncio.gather(
            statcast_task, season_task, gamelogs_task, splits_task,
            return_exceptions=True
        )
        
        if isinstance(statcast, Exception):
            statcast = {}
        if isinstance(season, Exception):
            season = {}
        if isinstance(gamelogs, Exception):
            gamelogs = []
        if isinstance(splits, Exception):
            splits = {}
        
        # Core Statcast metrics
        xwoba = statcast.get("xwoba", None)
        ev = statcast.get("exit_velocity", None)
        hh_pct = statcast.get("hard_hit_pct", None)
        barrel_pct = statcast.get("barrel_pct", None)
        k_pct = statcast.get("k_pct", None)
        bb_pct = statcast.get("bb_pct", None)
        
        # ERA / FIP
        era = season.get("era", None)
        fip = season.get("fip", None)
        xera = statcast.get("xera", None)
        ip = season.get("ip", 0)
        
        # Pitch arsenal whiff rates
        arsenal = statcast.get("arsenal", {})
        
        # Calculate blended metrics
        xfip = self._estimate_xfip(statcast, season)
        siera = self._estimate_siera(statcast, season)
        blend = self._calculate_blend(xfip, siera, ip)
        
        # Road/home adjustment
        road_home_adj = 0.20 if is_home else -0.20
        adjusted_blend = blend + road_home_adj
        
        # Recent form (last 3 starts weighted)
        recent_form = self._analyze_recent_form(gamelogs)
        
        # Walk risk flag
        bb9 = self._calculate_bb9(season)
        walk_risk = bb9 >= 5.0
        
        # Bimodal flag
        bimodal = self._detect_bimodal(gamelogs)
        
        # ERA mirage check
        era_mirage = self._check_era_mirage(era, xfip, siera)
        
        # Platoon splits
        vs_lhb = splits.get("vs_LHB", {})
        vs_rhb = splits.get("vs_RHB", {})
        
        # Debut/uncertainty flag
        debut_flag = ip < 20
        
        # v2.4 comp score (0-100)
        comp = self._calculate_comp(xwoba, xfip, siera, k_pct, bb9, 
                                     hh_pct, barrel_pct, recent_form)
        
        return {
            "name": name,
            "pitcher_id": pitcher_id,
            "era": era,
            "xera": xera,
            "fip": fip,
            "xfip": xfip,
            "siera": siera,
            "blend": round(blend, 2),
            "adjusted_blend": round(adjusted_blend, 2),
            "road_home_adj": road_home_adj,
            "xwoba": xwoba,
            "ev": ev,
            "hh_pct": hh_pct,
            "barrel_pct": barrel_pct,
            "k_pct": k_pct,
            "bb9": round(bb9, 2) if bb9 else None,
            "ip": ip,
            "arsenal": arsenal,
            "recent_form": recent_form,
            "splits": {
                "vs_lhb": vs_lhb,
                "vs_rhb": vs_rhb
            },
            "comp": comp,
            "flags": {
                "walk_risk": walk_risk,
                "bimodal": bimodal,
                "debut_flag": debut_flag,
                "era_mirage": era_mirage,
                "era_mirage_direction": "positive" if era_mirage and 
                    (era or 5) > (xfip or 5) else "negative" if era_mirage else None
            },
            "era_mirage_magnitude": round(abs((era or 0) - (xfip or 0)), 2) if era and xfip else None
        }
    
    def _estimate_xfip(self, statcast: Dict, season: Dict) -> float:
        """Estimate xFIP from available metrics"""
        # Use direct xFIP if available
        if statcast.get("xfip"):
            return statcast["xfip"]
        
        # Estimate from components
        k_pct = statcast.get("k_pct", 0.22)
        bb_pct = statcast.get("bb_pct", 0.08)
        xwoba = statcast.get("xwoba", 0.320)
        
        # Simplified xFIP estimation
        # Lower xwOBA → lower xFIP
        xwoba_factor = (xwoba - 0.320) * 8.0
        k_factor = -(k_pct - 0.22) * 12.0
        bb_factor = (bb_pct - 0.08) * 18.0
        
        return round(3.80 + xwoba_factor + k_factor + bb_factor, 2)
    
    def _estimate_siera(self, statcast: Dict, season: Dict) -> float:
        """Estimate SIERA from available metrics"""
        if statcast.get("siera"):
            return statcast["siera"]
        
        xfip = self._estimate_xfip(statcast, season)
        gb_pct = statcast.get("gb_pct", 0.44)
        
        # SIERA correlates with xFIP but penalizes walks less when GB rate high
        gb_adj = -(gb_pct - 0.44) * 0.80
        return round(xfip + gb_adj + 0.15, 2)
    
    def _calculate_blend(self, xfip: float, siera: float, ip: float) -> float:
        """60% SIERA + 40% xFIP blend (v2.4 rule)"""
        if ip < 20:
            return xfip  # Use xFIP only for small samples
        return round(0.60 * siera + 0.40 * xfip, 2)
    
    def _analyze_recent_form(self, gamelogs: list) -> Dict:
        """Analyze last 3 starts weighted"""
        if not gamelogs:
            return {"available": False}
        
        last_3 = gamelogs[:3]
        
        er_list = [g.get("er", 0) for g in last_3]
        ip_list = [g.get("ip", 0) for g in last_3]
        
        # Weight recent starts more heavily (3, 2, 1)
        weights = [3, 2, 1][:len(last_3)]
        total_weight = sum(weights)
        
        weighted_er = sum(er * w for er, w in zip(er_list, weights)) / total_weight
        weighted_ip = sum(ip * w for ip, w in zip(ip_list, weights)) / total_weight
        
        weighted_era = (weighted_er / weighted_ip * 9) if weighted_ip > 0 else 0
        
        return {
            "available": True,
            "last_3_starts": last_3[:3],
            "weighted_era": round(weighted_era, 2),
            "trend": "hot" if weighted_era < 2.50 else 
                     "cold" if weighted_era > 5.00 else "average",
            "consecutive_quality": self._count_quality_starts(last_3)
        }
    
    def _count_quality_starts(self, gamelogs: list) -> int:
        """Count consecutive quality starts (6+ IP, ≤3 ER)"""
        count = 0
        for g in gamelogs:
            if g.get("ip", 0) >= 6.0 and g.get("er", 99) <= 3:
                count += 1
            else:
                break
        return count
    
    def _calculate_bb9(self, season: Dict) -> Optional[float]:
        """Calculate BB/9 from season stats"""
        bb = season.get("bb", 0)
        ip = season.get("ip", 0)
        if ip == 0:
            return None
        return round((bb / ip) * 9, 2)
    
    def _detect_bimodal(self, gamelogs: list) -> bool:
        """Detect bimodal distribution (elite starts + disasters)"""
        if len(gamelogs) < 4:
            return False
        
        er_list = [g.get("era_this_start", g.get("er", 0) / max(g.get("ip", 1), 1) * 9) 
                   for g in gamelogs]
        
        great = sum(1 for e in er_list if e <= 2.00)
        terrible = sum(1 for e in er_list if e >= 6.00)
        
        return great >= 2 and terrible >= 2
    
    def _check_era_mirage(self, era: Optional[float], 
                           xfip: Optional[float], 
                           siera: Optional[float]) -> bool:
        """Detect significant ERA mirage (1.5+ point difference)"""
        if not era or not xfip:
            return False
        return abs(era - xfip) >= 1.50
    
    def _calculate_comp(self, xwoba, xfip, siera, k_pct, bb9, 
                         hh_pct, barrel_pct, recent_form) -> int:
        """
        v2.4 Comp score (0-100)
        100 = best pitchers in baseball (Glasnow/Fried/Skubal tier)
        0   = catastrophic (Mahle/Lorenzen tier)
        """
        score = 50  # baseline
        
        # xwOBA impact (most important)
        if xwoba:
            if xwoba <= 0.220: score += 30
            elif xwoba <= 0.250: score += 20
            elif xwoba <= 0.280: score += 12
            elif xwoba <= 0.310: score += 5
            elif xwoba <= 0.340: score += 0
            elif xwoba <= 0.370: score -= 8
            elif xwoba <= 0.400: score -= 18
            else: score -= 30
        
        # xFIP impact
        if xfip:
            if xfip <= 2.50: score += 15
            elif xfip <= 3.00: score += 10
            elif xfip <= 3.50: score += 5
            elif xfip <= 4.00: score += 0
            elif xfip <= 4.50: score -= 5
            elif xfip <= 5.00: score -= 12
            else: score -= 20
        
        # K rate bonus
        if k_pct:
            if k_pct >= 0.30: score += 8
            elif k_pct >= 0.25: score += 4
            elif k_pct <= 0.16: score -= 8
        
        # Walk risk penalty
        if bb9:
            if bb9 >= 5.0: score -= 12
            elif bb9 >= 4.0: score -= 6
            elif bb9 <= 2.0: score += 6
        
        # Hard hit penalty
        if hh_pct:
            if hh_pct >= 50: score -= 8
            elif hh_pct <= 30: score += 6
        
        # Barrel penalty
        if barrel_pct:
            if barrel_pct >= 15: score -= 10
            elif barrel_pct <= 5: score += 5
        
        # Recent form adjustment
        if recent_form.get("available"):
            trend = recent_form.get("trend")
            if trend == "hot": score += 5
            elif trend == "cold": score -= 5
            
            quality_starts = recent_form.get("consecutive_quality", 0)
            score += min(quality_starts * 3, 9)
        
        return max(0, min(100, score))
    
    def _debut_proxy(self, name: str) -> Dict:
        """Return proxy metrics for debut/unknown pitcher"""
        return {
            "name": name,
            "pitcher_id": None,
            "era": None,
            "xfip": 4.50,
            "siera": 4.65,
            "blend": 4.56,
            "adjusted_blend": 4.56,
            "xwoba": 0.340,
            "comp": 45,
            "ip": 0,
            "flags": {
                "walk_risk": False,
                "bimodal": False,
                "debut_flag": True,
                "era_mirage": False
            },
            "note": "DEBUT/UNKNOWN — proxy metrics applied"
        }
    
    def _extract_flags(self, away_sp: Dict, home_sp: Dict) -> Dict:
        """Extract combined SP flags for WP formula"""
        return {
            "away_walk_risk": away_sp.get("flags", {}).get("walk_risk", False),
            "home_walk_risk": home_sp.get("flags", {}).get("walk_risk", False),
            "away_bimodal": away_sp.get("flags", {}).get("bimodal", False),
            "home_bimodal": home_sp.get("flags", {}).get("bimodal", False),
            "away_debut": away_sp.get("flags", {}).get("debut_flag", False),
            "home_debut": home_sp.get("flags", {}).get("debut_flag", False),
            "away_era_mirage": away_sp.get("flags", {}).get("era_mirage", False),
            "home_era_mirage": home_sp.get("flags", {}).get("era_mirage", False)
        }
