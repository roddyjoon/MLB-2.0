"""
Agent 3: Trends & Situational Agent
Over/Under trends, home/road splits, series momentum,
back-to-back blowout corrections, day/night splits
"""

import asyncio
from typing import Dict
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("trends_agent")


class TrendsSituationalAgent:
    """Pulls and analyzes team trends, situational patterns, momentum"""

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        home = game_data.get("home_team")
        away = game_data.get("away_team")
        date = game_data.get("date")

        # Pull all trend data in parallel
        (
            home_trends, away_trends,
            series_history,
            last_10_matchups
        ) = await asyncio.gather(
            self.mlb_api.get_team_trends(home),
            self.mlb_api.get_team_trends(away),
            self.mlb_api.get_recent_series(home, away, date),
            self.mlb_api.get_last_n_matchups(home, away, n=10),
            return_exceptions=True
        )

        home_trends = home_trends if not isinstance(home_trends, Exception) else {}
        away_trends = away_trends if not isinstance(away_trends, Exception) else {}
        series_history = series_history if not isinstance(series_history, Exception) else []
        last_10_matchups = last_10_matchups if not isinstance(last_10_matchups, Exception) else []

        # Series momentum + back-to-back corrections
        momentum = self._analyze_series_momentum(series_history)

        # Historical H2H trends
        h2h_trends = self._analyze_h2h(last_10_matchups)

        # Individual team trends
        home_team_trends = self._analyze_team_trends(home_trends, is_home=True)
        away_team_trends = self._analyze_team_trends(away_trends, is_home=False)

        return {
            "home_trends": home_team_trends,
            "away_trends": away_team_trends,
            "series_momentum": momentum,
            "h2h_trends": h2h_trends,
            "situational_flags": self._extract_situational_flags(
                home_team_trends, away_team_trends, momentum, h2h_trends
            )
        }

    def _analyze_series_momentum(self, series_history: list) -> Dict:
        """
        Analyze current series momentum and apply back-to-back corrections
        
        v2.4 Rules:
        - Win by 5+ runs: 50% momentum cap for winner
        - Win by 9+ runs (blowout): 25% cap for winner
        - Two consecutive blowouts: Lesson #30 max bounce-back signal
        """
        if not series_history:
            return {"available": False}

        recent_games = series_history[:3]  # Last 3 games of series

        results = []
        for g in recent_games:
            margin = abs(g.get("home_score", 0) - g.get("away_score", 0))
            winner = g.get("winner")
            results.append({
                "winner": winner,
                "margin": margin,
                "blowout": margin >= 9,
                "large_win": margin >= 5
            })

        # Calculate momentum caps
        momentum_home = 1.0
        momentum_away = 1.0

        consecutive_blowouts = {
            "home": 0,
            "away": 0
        }

        for r in results[:2]:  # Check last 2 games
            winner = r["winner"]
            if winner == "home":
                if r["blowout"]:
                    momentum_home = min(momentum_home, 0.25)
                    consecutive_blowouts["home"] += 1
                elif r["large_win"]:
                    momentum_home = min(momentum_home, 0.50)
            elif winner == "away":
                if r["blowout"]:
                    momentum_away = min(momentum_away, 0.25)
                    consecutive_blowouts["away"] += 1
                elif r["large_win"]:
                    momentum_away = min(momentum_away, 0.50)

        # Back-to-back blowout = max bounce-back signal (Lesson #30)
        lesson_30_active = consecutive_blowouts["home"] >= 2 or consecutive_blowouts["away"] >= 2
        bounce_back_team = None
        if consecutive_blowouts["home"] >= 2:
            bounce_back_team = "away"  # Away team bounces back
        elif consecutive_blowouts["away"] >= 2:
            bounce_back_team = "home"  # Home team bounces back

        # Series record
        home_wins = sum(1 for g in recent_games if g.get("winner") == "home")
        away_wins = sum(1 for g in recent_games if g.get("winner") == "away")

        return {
            "available": True,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "momentum_home_cap": momentum_home,
            "momentum_away_cap": momentum_away,
            "lesson_30_active": lesson_30_active,
            "bounce_back_team": bounce_back_team,
            "recent_results": results,
            "sweep_attempt": (home_wins >= 2 or away_wins >= 2) and len(recent_games) >= 2
        }

    def _analyze_h2h(self, matchups: list) -> Dict:
        """Analyze last 10 H2H matchup trends"""
        if not matchups:
            return {"available": False}

        totals = [g.get("home_score", 0) + g.get("away_score", 0) for g in matchups]
        avg_total = sum(totals) / len(totals) if totals else 8.5

        overs = sum(1 for g in matchups
                    if (g.get("home_score", 0) + g.get("away_score", 0)) >
                    g.get("total_line", 8.5))
        unders = len(matchups) - overs

        return {
            "available": True,
            "games_played": len(matchups),
            "avg_total": round(avg_total, 1),
            "overs": overs,
            "unders": unders,
            "over_pct": round(overs / len(matchups), 2) if matchups else 0.5,
            "structural_over": overs >= 7,
            "structural_under": unders >= 7
        }

    def _analyze_team_trends(self, trends: Dict, is_home: bool) -> Dict:
        """Analyze individual team trends"""
        split_key = "home" if is_home else "road"

        # Record splits
        home_road_record = trends.get(f"{split_key}_record", "")
        home_road_wins = trends.get(f"{split_key}_wins", 0)
        home_road_losses = trends.get(f"{split_key}_losses", 0)
        total_games = home_road_wins + home_road_losses
        win_pct = home_road_wins / total_games if total_games > 0 else 0.5

        # Over/Under trends
        over_pct = trends.get("over_pct", 0.50)
        under_pct = trends.get("under_pct", 0.50)
        over_games = trends.get("over_count", 0)
        total_ou = trends.get("total_ou_games", 0)

        # Day/night splits
        night_win_pct = trends.get("night_win_pct", 0.50)
        day_win_pct = trends.get("day_win_pct", 0.50)

        # Winning/losing streaks
        current_streak = trends.get("current_streak", 0)
        streak_type = trends.get("streak_type", "W")

        # Last 10 overall
        last_10_wins = trends.get("last_10_wins", 5)
        last_10 = f"{last_10_wins}-{10 - last_10_wins}"

        # Run differential
        run_diff = trends.get("run_differential", 0)
        run_diff_per_game = run_diff / max(total_games, 1)

        return {
            "split_key": split_key,
            "record": home_road_record,
            "win_pct": round(win_pct, 3),
            "over_pct": over_pct,
            "under_pct": under_pct,
            "current_streak": current_streak,
            "streak_type": streak_type,
            "last_10": last_10,
            "last_10_wins": last_10_wins,
            "run_differential": run_diff,
            "run_diff_per_game": round(run_diff_per_game, 2),
            "overachieving": run_diff_per_game < -0.5,  # Win% >> run diff
            "structural_over": over_pct >= 0.65,
            "structural_under": under_pct >= 0.65,
            "dominant_home": is_home and win_pct >= 0.650,
            "brutal_road": not is_home and win_pct <= 0.300
        }

    def _extract_situational_flags(self, home_trends: Dict,
                                    away_trends: Dict,
                                    momentum: Dict,
                                    h2h: Dict) -> Dict:
        """Extract key situational flags for WP formula"""
        return {
            # Momentum
            "lesson_30_active": momentum.get("lesson_30_active", False),
            "bounce_back_team": momentum.get("bounce_back_team"),
            "momentum_home_cap": momentum.get("momentum_home_cap", 1.0),
            "momentum_away_cap": momentum.get("momentum_away_cap", 1.0),
            "sweep_attempt": momentum.get("sweep_attempt", False),

            # H2H structural
            "structural_over_h2h": h2h.get("structural_over", False),
            "structural_under_h2h": h2h.get("structural_under", False),
            "h2h_avg_total": h2h.get("avg_total", 8.5),

            # Team-level
            "home_dominant_at_home": home_trends.get("dominant_home", False),
            "away_brutal_road": away_trends.get("brutal_road", False),
            "home_structural_over": home_trends.get("structural_over", False),
            "away_structural_over": away_trends.get("structural_over", False),
            "home_overachieving": home_trends.get("overachieving", False),
            "away_overachieving": away_trends.get("overachieving", False),

            # Streak info
            "home_streak": {
                "count": home_trends.get("current_streak", 0),
                "type": home_trends.get("streak_type", "W")
            },
            "away_streak": {
                "count": away_trends.get("current_streak", 0),
                "type": away_trends.get("streak_type", "W")
            }
        }
