"""
Agent 12: Regression Detection System
Automatic ERA mirage severity ranking, SP regression candidates,
lineup hot streak regression, HR/FB rate regression
"""

from typing import Dict, List
from core.logger import ModelLogger

logger = ModelLogger("regression_agent")

# Regression thresholds
ERA_MIRAGE_THRESHOLD = 1.50    # ERA vs xFIP gap for positive mirage
NEGATIVE_MIRAGE_THRESHOLD = 1.50  # ERA better than xFIP = negative mirage
HOT_STREAK_REGRESSION_THRESHOLD = 0.850  # OPS last 7 days
LOB_REGRESSION_THRESHOLD = 0.85   # LOB% above this = unsustainable
HR_FB_REGRESSION_THRESHOLD = 0.18  # HR/FB above this = regression due
BABIP_SP_HIGH = 0.330              # SP BABIP above this = luck-based good ERA
BABIP_SP_LOW = 0.240               # SP BABIP below this = luck-based bad ERA


class RegressionDetectionAgent:
    """
    Automatically detects regression candidates on both sides
    
    Key insight: Markets lag on regression — pricing Wacha's 3.13 ERA
    when xFIP says 4.40 is exactly the edge we exploit
    """

    def __init__(self):
        pass

    async def analyze(self, game_data: Dict) -> Dict:
        """Full regression analysis"""

        # SP regression candidates
        home_sp = game_data.get("home_sp_preview", {})
        away_sp = game_data.get("away_sp_preview", {})

        home_sp_regression = self._analyze_sp_regression(
            home_sp, game_data.get("home_sp_name", ""))
        away_sp_regression = self._analyze_sp_regression(
            away_sp, game_data.get("away_sp_name", ""))

        # Lineup regression
        home_lineup_reg = self._analyze_lineup_regression(
            game_data.get("home_lineup_preview", {}))
        away_lineup_reg = self._analyze_lineup_regression(
            game_data.get("away_lineup_preview", {}))

        # Team-level regression flags
        home_team_reg = self._analyze_team_regression(
            game_data.get("home_team_stats", {}),
            game_data.get("home_team"))
        away_team_reg = self._analyze_team_regression(
            game_data.get("away_team_stats", {}),
            game_data.get("away_team"))

        # Compile all signals
        all_signals = self._compile_signals(
            home_sp_regression, away_sp_regression,
            home_lineup_reg, away_lineup_reg,
            home_team_reg, away_team_reg
        )

        # WP adjustments from regression
        wp_adj = self._calculate_wp_adj(all_signals)

        return {
            "home_sp_regression": home_sp_regression,
            "away_sp_regression": away_sp_regression,
            "home_lineup_regression": home_lineup_reg,
            "away_lineup_regression": away_lineup_reg,
            "home_team_regression": home_team_reg,
            "away_team_regression": away_team_reg,
            "all_signals": all_signals,
            "wp_adjustments": wp_adj,
            "key_flags": self._extract_key_flags(all_signals)
        }

    def _analyze_sp_regression(self, sp_data: Dict, name: str) -> Dict:
        """Analyze SP for regression candidates"""
        if not sp_data:
            return {"available": False}

        era = sp_data.get("era")
        xfip = sp_data.get("xfip") or sp_data.get("xfip_est")
        siera = sp_data.get("siera") or sp_data.get("siera_est")
        babip = sp_data.get("babip")
        lob_pct = sp_data.get("lob_pct")
        hr_fb = sp_data.get("hr_fb")
        ip = sp_data.get("ip", 0)

        signals = []

        # ERA mirage detection (positive = ERA overstates badness)
        if era and xfip:
            gap = era - xfip
            if gap >= ERA_MIRAGE_THRESHOLD:
                severity = "extreme" if gap >= 3.0 else "large" if gap >= 2.0 else "moderate"
                signals.append({
                    "type": "positive_era_mirage",
                    "severity": severity,
                    "gap": round(gap, 2),
                    "direction": "ERA_will_fall",
                    "description": f"ERA {era:.2f} vs xFIP {xfip:.2f} (+{gap:.2f}) — ERA will drop",
                    "wp_adj": 0.010 * (gap / 1.5),  # Benefits the team starting this SP
                    "betting_implication": "SP better than ERA shows — fade ERA-based public"
                })

            elif gap <= -ERA_MIRAGE_THRESHOLD:
                abs_gap = abs(gap)
                severity = "extreme" if abs_gap >= 3.0 else "large" if abs_gap >= 2.0 else "moderate"
                signals.append({
                    "type": "negative_era_mirage",
                    "severity": severity,
                    "gap": round(gap, 2),
                    "direction": "ERA_will_rise",
                    "description": f"ERA {era:.2f} vs xFIP {xfip:.2f} ({gap:.2f}) — ERA will rise",
                    "wp_adj": -0.010 * (abs_gap / 1.5),
                    "betting_implication": "SP worse than ERA shows — fade this SP"
                })

        # LOB% regression
        if lob_pct and lob_pct >= LOB_REGRESSION_THRESHOLD:
            signals.append({
                "type": "lob_regression",
                "severity": "moderate",
                "value": lob_pct,
                "description": f"LOB% {lob_pct*100:.0f}% unsustainable — ERA will rise",
                "wp_adj": -0.008,
                "betting_implication": "Strand rate will normalize — more runs incoming"
            })

        # BABIP regression
        if babip:
            if babip <= BABIP_SP_LOW:
                signals.append({
                    "type": "babip_low_regression",
                    "severity": "moderate",
                    "value": babip,
                    "description": f"BABIP .{int(babip*1000)} — luck-based good ERA",
                    "wp_adj": -0.006,
                    "betting_implication": "Hits will normalize — ERA will rise"
                })
            elif babip >= BABIP_SP_HIGH:
                signals.append({
                    "type": "babip_high_regression",
                    "severity": "moderate",
                    "value": babip,
                    "description": f"BABIP .{int(babip*1000)} — bad luck ERA",
                    "wp_adj": +0.006,
                    "betting_implication": "Hits will normalize — ERA will fall"
                })

        # Recent form regression (Wacha: 6.75 last 2 starts)
        recent_era = sp_data.get("recent_form", {}).get("weighted_era")
        if recent_era and era:
            recent_vs_season = recent_era - era
            if recent_vs_season >= 2.5:
                signals.append({
                    "type": "recent_collapse",
                    "severity": "high",
                    "description": f"Recent ERA {recent_era:.2f} vs season {era:.2f} — collapse",
                    "wp_adj": -0.015,
                    "betting_implication": "SP in decline — fade now"
                })
            elif recent_vs_season <= -2.0:
                signals.append({
                    "type": "recent_surge",
                    "severity": "moderate",
                    "description": f"Recent ERA {recent_era:.2f} vs season {era:.2f} — surging",
                    "wp_adj": +0.010,
                    "betting_implication": "SP peaking — back now"
                })

        return {
            "name": name,
            "available": True,
            "signals": signals,
            "regression_score": self._calc_regression_score(signals),
            "net_wp_adj": round(sum(s.get("wp_adj", 0) for s in signals), 3),
            "key_signal": signals[0] if signals else None
        }

    def _analyze_lineup_regression(self, lineup_data: Dict) -> Dict:
        """Detect lineup hot/cold streak regression"""
        if not lineup_data:
            return {"available": False}

        signals = []
        recent_ops = lineup_data.get("streak", {}).get("recent_ops", 0.720)
        streak_type = lineup_data.get("streak", {}).get("type", "neutral")

        # Hot streak regression
        if streak_type == "hot" and recent_ops >= HOT_STREAK_REGRESSION_THRESHOLD:
            signals.append({
                "type": "hot_streak_regression_due",
                "severity": "moderate",
                "recent_ops": recent_ops,
                "description": f"OPS {recent_ops:.3f} last 7 days — regression likely",
                "wp_adj": -0.008,
                "betting_implication": "Lineup due for cooling off"
            })

        # Cold streak continuation risk
        if streak_type == "cold" and recent_ops <= 0.620:
            signals.append({
                "type": "cold_streak_continuation",
                "severity": "moderate",
                "recent_ops": recent_ops,
                "description": f"OPS {recent_ops:.3f} — cold streak may continue",
                "wp_adj": -0.008,
                "betting_implication": "Offense still cold"
            })

        # HR/FB regression
        hr_fb = lineup_data.get("hr_fb_rate")
        if hr_fb and hr_fb >= HR_FB_REGRESSION_THRESHOLD:
            signals.append({
                "type": "hr_fb_regression",
                "severity": "moderate",
                "value": hr_fb,
                "description": f"HR/FB {hr_fb*100:.0f}% — HR rate will regress",
                "wp_adj": -0.006,
                "betting_implication": "Reduce HR projection for this lineup"
            })

        return {
            "available": True,
            "signals": signals,
            "net_wp_adj": round(sum(s.get("wp_adj", 0) for s in signals), 3)
        }

    def _analyze_team_regression(self, team_stats: Dict,
                                   team: str) -> Dict:
        """Team-level regression analysis"""
        if not team_stats:
            return {"available": False}

        signals = []

        # Run differential vs win percentage
        wins = team_stats.get("wins", 0)
        losses = team_stats.get("losses", 0)
        total_games = wins + losses
        run_diff = team_stats.get("run_differential", 0)

        if total_games > 10:
            rdpg = run_diff / total_games
            win_pct = wins / total_games

            # Overachieving: winning more than run diff suggests
            pythag_wins = self._calc_pythagorean(team_stats)
            if pythag_wins and (wins - pythag_wins) >= 3:
                signals.append({
                    "type": "overachieving",
                    "severity": "moderate",
                    "actual_wins": wins,
                    "expected_wins": round(pythag_wins, 1),
                    "description": f"{team} {wins}W vs {pythag_wins:.0f}W expected — overachieving",
                    "wp_adj": -0.010,
                    "betting_implication": "Team overperforming — expect regression"
                })

            # Underachieving
            elif pythag_wins and (pythag_wins - wins) >= 3:
                signals.append({
                    "type": "underachieving",
                    "severity": "moderate",
                    "description": f"{team} {wins}W vs {pythag_wins:.0f}W expected — underachieving",
                    "wp_adj": +0.010,
                    "betting_implication": "Team underperforming — positive regression due"
                })

        return {
            "team": team,
            "available": True,
            "signals": signals,
            "net_wp_adj": round(sum(s.get("wp_adj", 0) for s in signals), 3)
        }

    def _calc_pythagorean(self, team_stats: Dict) -> float:
        """Pythagorean win expectation"""
        rs = team_stats.get("runs_scored", 0)
        ra = team_stats.get("runs_allowed", 0)
        games = team_stats.get("games_played", 1)

        if not rs or not ra:
            return None

        exp = (rs ** 1.83) / (rs ** 1.83 + ra ** 1.83)
        return exp * games

    def _compile_signals(self, *signal_groups) -> List:
        """Compile all regression signals"""
        all_signals = []
        for group in signal_groups:
            if isinstance(group, dict) and group.get("available"):
                all_signals.extend(group.get("signals", []))
        return sorted(all_signals,
                      key=lambda x: abs(x.get("wp_adj", 0)),
                      reverse=True)

    def _calculate_wp_adj(self, signals: List) -> Dict:
        """Net WP adjustments from regression signals"""
        home_adj = 0
        away_adj = 0

        for sig in signals:
            # Signals tagged with which team they apply to
            team = sig.get("team_side", "")
            adj = sig.get("wp_adj", 0)
            if team == "home":
                home_adj += adj
            elif team == "away":
                away_adj += adj

        return {
            "home_wp_adj": round(home_adj, 3),
            "away_wp_adj": round(away_adj, 3),
            "net_home_wp": round(home_adj - away_adj, 3)
        }

    def _calc_regression_score(self, signals: List) -> int:
        """0-100 regression risk score (higher = more regression due)"""
        if not signals:
            return 0
        total = sum(abs(s.get("wp_adj", 0)) * 100 for s in signals)
        return min(100, int(total))

    def _extract_key_flags(self, signals: List) -> List:
        """Extract top 3 most important regression flags"""
        high_impact = [s for s in signals if abs(s.get("wp_adj", 0)) >= 0.008]
        return high_impact[:3]
