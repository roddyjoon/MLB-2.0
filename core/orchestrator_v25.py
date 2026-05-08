"""
V2.5 Orchestrator — 13-agent system
Adds: Bullpen, Umpire, Arsenal, Line Movement,
      Platoon Splits, Regression Detection, F1/Framing
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Original 6 agents
from agents.sp_statcast_agent import SPStatcastAgent
from agents.lineup_offense_agent import LineupOffenseAgent
from agents.trends_situational_agent import TrendsSituationalAgent
from agents.injury_roster_agent import InjuryRosterAgent
from agents.park_weather_agent import ParkWeatherAgent
from agents.advanced_metrics_agent import AdvancedMetricsAgent

# New 7 agents (v2.5)
from agents.bullpen_quality_agent import BullpenQualityAgent
from agents.umpire_agent import UmpireAgent
from agents.pitch_arsenal_agent import PitchArsenalAgent
from agents.line_movement_agent import LineMovementAgent
from agents.platoon_splits_agent import PlatoonSplitsAgent
from agents.regression_detection_agent import RegressionDetectionAgent
from agents.first_inning_catcher_agent import FirstInningCatcherAgent

from core.wp_formula_v25 import WPFormulaV25
from core.edge_calculator import EdgeCalculator, KellySizer
from core.logger import ModelLogger
from data.mlb_api import MLBDataAPI
from data import mlb_api as mlb_api_module

logger = ModelLogger("orchestrator_v25")


class V25Orchestrator:
    """
    v2.5 Master Orchestrator — 13 agents
    
    New capabilities:
    - Bullpen quality at individual reliever level
    - Umpire run expectancy model
    - Pitch arsenal vs lineup handedness matrix
    - Sharp line movement with Kelly adjustments
    - Full platoon splits database
    - Automatic regression detection
    - First inning model + catcher framing tiers
    """

    def __init__(self, api: Optional[MLBDataAPI] = None, mode: str = "live",
                 as_of_date: Optional[str] = None):
        """
        v2.5 orchestrator.

        Args:
            api: optional preconfigured MLBDataAPI (mainly for testing).
                If omitted, the orchestrator instantiates one and configures
                the data.mlb_api module so agents that self-instantiate
                MLBDataAPI() pick up the same as_of_date / mode.
            mode: "live" or "backtest". In backtest, market/edge/Kelly are
                skipped (historical odds aren't available for free).
            as_of_date: ISO date string for backtest snapshots. None = today.
        """
        self.mode = mode
        self.as_of_date = as_of_date

        # Configure the data.mlb_api module so agents that call MLBDataAPI()
        # internally pick up the same as_of_date and mode.
        mlb_api_module.configure(as_of_date=as_of_date, mode=mode)

        # Original 6
        self.sp_agent = SPStatcastAgent()
        self.lineup_agent = LineupOffenseAgent()
        self.trends_agent = TrendsSituationalAgent()
        self.injury_agent = InjuryRosterAgent()
        self.park_agent = ParkWeatherAgent()
        self.metrics_agent = AdvancedMetricsAgent()

        # New 7 (v2.5)
        self.bullpen_agent = BullpenQualityAgent()
        self.umpire_agent = UmpireAgent()
        self.arsenal_agent = PitchArsenalAgent()
        self.line_movement_agent = LineMovementAgent()
        self.platoon_agent = PlatoonSplitsAgent()
        self.regression_agent = RegressionDetectionAgent()
        self.f1_catcher_agent = FirstInningCatcherAgent()

        self.wp_formula = WPFormulaV25()
        self.edge_calc = EdgeCalculator()
        self.kelly = KellySizer()
        self.mlb_api = api or MLBDataAPI(as_of_date=as_of_date, mode=mode)

    async def generate_daily_card(self, date: str) -> Dict:
        """Generate full daily card with all 13 agents"""
        logger.info(f"v2.5 — Generating daily card for {date}")

        games = await self.mlb_api.get_games_for_date(date)
        logger.info(f"Found {len(games)} games")

        # Run all games in parallel
        tasks = [self.simulate_game_v25(g, date) for g in games]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in results if not isinstance(r, Exception)]

        primaries, secondaries, passes = [], [], []
        for r in valid:
            play = r.get("best_play")
            if play:
                if play["edge"] >= 0.06:
                    primaries.append(play)
                elif play["edge"] >= 0.04:
                    secondaries.append(play)
                else:
                    passes.append(r["game"])
            else:
                passes.append(r.get("game", {}))

        primaries.sort(key=lambda x: x["edge"], reverse=True)
        secondaries.sort(key=lambda x: x["edge"], reverse=True)

        return {
            "version": "v2.5",
            "date": date,
            "generated_at": datetime.now().isoformat(),
            "agents_used": 13,
            "total_games": len(games),
            "primaries": primaries,
            "secondaries": secondaries,
            "passes": passes,
            "total_primary_risk": sum(p.get("kelly_size", 0)
                                       for p in primaries),
            "game_details": valid
        }

    async def simulate_game_v25(self, game_data: Dict, date: str) -> Dict:
        """
        Full 13-agent simulation
        
        Sequence (enforced):
        1-6: Original agents (market-blind)
        7-13: New agents (market-blind except line movement)
        WP calculated from all agent outputs
        Market pulled last
        Edge calculated
        Kelly adjusted by line movement signals
        """
        game_id = game_data.get("game_id",
            f"{game_data.get('away_team')}@{game_data.get('home_team')}")
        logger.info(f"v2.5 simulating {game_id}")

        # Inject `date` into game_data so the 5 agents that read it
        # (park_weather, trends, line_movement, injury, umpire) can find it.
        # v2.5 left this implicit, which caused those agents to silently
        # get None and error out under gather(return_exceptions=True).
        if "date" not in game_data:
            game_data = {**game_data, "date": date}

        try:
            # ============================================================
            # PHASE 1: Run all market-blind agents in parallel
            # ============================================================
            (sp_data, lineup_data, trends_data, injury_data,
             park_data, metrics_data,
             bullpen_data, umpire_data, arsenal_data,
             platoon_data, regression_data, f1_data) = await asyncio.gather(
                self.sp_agent.analyze(game_data),
                self.lineup_agent.analyze(game_data),
                self.trends_agent.analyze(game_data),
                self.injury_agent.analyze(game_data),
                self.park_agent.analyze(game_data),
                self.metrics_agent.analyze(game_data),
                self.bullpen_agent.analyze(game_data),
                self.umpire_agent.analyze(game_data),
                self.arsenal_agent.analyze(game_data),
                self.platoon_agent.analyze(game_data),
                self.regression_agent.analyze(game_data),
                self.f1_catcher_agent.analyze(game_data),
                return_exceptions=True
            )

            # Safe defaults for any agent failures
            def safe(x): return x if not isinstance(x, Exception) else {}
            sp_data = safe(sp_data)
            lineup_data = safe(lineup_data)
            trends_data = safe(trends_data)
            injury_data = safe(injury_data)
            park_data = safe(park_data)
            metrics_data = safe(metrics_data)
            bullpen_data = safe(bullpen_data)
            umpire_data = safe(umpire_data)
            arsenal_data = safe(arsenal_data)
            platoon_data = safe(platoon_data)
            regression_data = safe(regression_data)
            f1_data = safe(f1_data)

            # ============================================================
            # PHASE 2: Pure WP calculation — NO MARKET DATA
            # ============================================================
            wp_inputs = {
                "game": game_data,
                "sp": sp_data,
                "lineup": lineup_data,
                "trends": trends_data,
                "injuries": injury_data,
                "park": park_data,
                "metrics": metrics_data,
                # v2.5 additions
                "bullpen": bullpen_data,
                "umpire": umpire_data,
                "arsenal": arsenal_data,
                "platoon": platoon_data,
                "regression": regression_data,
                "first_inning": f1_data
            }

            wp_result = self.wp_formula.calculate(wp_inputs)

            # ============================================================
            # BACKTEST MODE: skip phases 3-6 (market/edges/Kelly)
            # Historical odds aren't available for free, so v3 backtest
            # validates WP calibration only. Edge/Kelly only run live.
            # ============================================================
            if self.mode == "backtest":
                return {
                    "game": game_data,
                    "version": "v2.5",
                    "mode": "backtest",
                    "agents": {
                        "sp": sp_data, "lineup": lineup_data,
                        "trends": trends_data, "injuries": injury_data,
                        "park": park_data, "metrics": metrics_data,
                        "bullpen": bullpen_data, "umpire": umpire_data,
                        "arsenal": arsenal_data, "platoon": platoon_data,
                        "regression": regression_data,
                        "first_inning": f1_data
                    },
                    "wp_result": wp_result,
                    "market": None,
                    "edges": None,
                    "best_play": None,
                    "sharp_signals": []
                }

            # ============================================================
            # PHASE 3: Pull market odds (ONLY HERE)
            # ============================================================
            market = await self.mlb_api.get_odds(
                game_data["home_team"],
                game_data["away_team"],
                date
            )

            # ============================================================
            # PHASE 4: Line movement (reads market — sizing only)
            # ============================================================
            line_movement = await self.line_movement_agent.analyze({
                **game_data,
                "current_market": market
            })

            # ============================================================
            # PHASE 5: Calculate edges
            # ============================================================
            edges = self.edge_calc.calculate_all(wp_result, market)

            # ============================================================
            # PHASE 6: Kelly sizing with line movement adjustments
            # ============================================================
            kelly_adjs = (line_movement.get("kelly_adjustments", {})
                          if not isinstance(line_movement, Exception)
                          else {})

            best_play = self._select_best_play_v25(
                edges, wp_result, market, game_data, kelly_adjs
            )

            return {
                "game": game_data,
                "version": "v2.5",
                "agents": {
                    "sp": sp_data,
                    "lineup": lineup_data,
                    "trends": trends_data,
                    "injuries": injury_data,
                    "park": park_data,
                    "metrics": metrics_data,
                    "bullpen": bullpen_data,
                    "umpire": umpire_data,
                    "arsenal": arsenal_data,
                    "platoon": platoon_data,
                    "regression": regression_data,
                    "first_inning": f1_data,
                    "line_movement": line_movement
                },
                "wp_result": wp_result,
                "market": market,
                "edges": edges,
                "best_play": best_play,
                "sharp_signals": (line_movement.get("sharp_signals", [])
                                  if not isinstance(line_movement, Exception)
                                  else [])
            }

        except Exception as e:
            logger.error(f"Error in v2.5 sim {game_id}: {e}")
            return {"game": game_data, "error": str(e), "version": "v2.5"}

    def _select_best_play_v25(self, edges: Dict, wp: Dict,
                               market: Dict, game: Dict,
                               kelly_adjs: Dict) -> Optional[Dict]:
        """Select best play with v2.5 Kelly adjustments"""
        all_plays = []
        home = game.get("home_team")
        away = game.get("away_team")
        matchup = f"{away} @ {home}"

        play_configs = [
            ("home_ml", home, f"{home} ML",
             market.get("home_ml_odds", -110),
             edges.get("home_ml_edge", 0),
             wp.get("home_wp", 0.5),
             kelly_adjs.get("home_ml_adj", 1.0)),
            ("away_ml", away, f"{away} ML",
             market.get("away_ml_odds", +110),
             edges.get("away_ml_edge", 0),
             wp.get("away_wp", 0.5),
             kelly_adjs.get("away_ml_adj", 1.0)),
        ]

        for (ptype, team, label, odds, edge,
             model_wp, kelly_adj) in play_configs:
            if edge > 0.03:
                base_kelly = self.kelly.calculate(model_wp, odds, "ml")
                adjusted_kelly = min(int(base_kelly * kelly_adj), 220)
                all_plays.append({
                    "type": ptype,
                    "team": team,
                    "matchup": matchup,
                    "label": label,
                    "odds": odds,
                    "model_wp": model_wp,
                    "implied_prob": self.edge_calc._odds_to_implied(odds),
                    "edge": edge,
                    "kelly_size": adjusted_kelly,
                    "kelly_adj": kelly_adj,
                    "tier": "primary" if edge >= 0.06 else "secondary"
                })

        # Totals
        for total_type in ["over", "under"]:
            edge = edges.get(f"{total_type}_edge", 0)
            if edge > 0.03:
                model_wp = wp.get(f"{total_type}_wp", 0.5)
                odds = market.get(f"{total_type}_odds", -110)
                k_adj = kelly_adjs.get(f"{total_type}_adj", 1.0)
                base_kelly = self.kelly.calculate(model_wp, odds, "total")
                adj_kelly = min(int(base_kelly * k_adj), 220)
                line = market.get("total_line", 8.5)

                all_plays.append({
                    "type": total_type,
                    "matchup": matchup,
                    "label": f"{'Over' if total_type == 'over' else 'Under'} {line}",
                    "odds": odds,
                    "model_wp": model_wp,
                    "projected_total": wp.get("total_projection"),
                    "edge": edge,
                    "kelly_size": adj_kelly,
                    "kelly_adj": k_adj,
                    "tier": "primary" if edge >= 0.06 else "secondary"
                })

        if not all_plays:
            return None
        return max(all_plays, key=lambda x: x["edge"])

    def print_card(self, card: Dict):
        """Print formatted v2.5 card"""
        print(f"\n{'='*65}")
        print(f"MLB MODEL v2.5 — {card['date']} — {card['agents_used']} AGENTS")
        print(f"{'='*65}")
        print(f"Games: {card['total_games']} | "
              f"Primary risk: ${card['total_primary_risk']:.0f}")

        if card["primaries"]:
            print(f"\n🔥 PRIMARIES ({len(card['primaries'])})")
            print("-" * 80)
            for i, p in enumerate(card["primaries"], 1):
                sharp = "⚡" if p.get("kelly_adj", 1.0) > 1.0 else " "
                matchup = p.get("matchup", "?")
                print(f"P{i:<2}{sharp}| {matchup:<14} | {p['label']:<14} | "
                      f"{p['odds']:>6} | ${p['kelly_size']:>4} | "
                      f"{p['edge']*100:.1f}%")

        if card["secondaries"]:
            print(f"\n⚡ SECONDARIES ({len(card['secondaries'])})")
            print("-" * 80)
            for i, s in enumerate(card["secondaries"], 1):
                matchup = s.get("matchup", "?")
                print(f"S{i:<2} | {matchup:<14} | {s['label']:<14} | "
                      f"{s['odds']:>6} | ${s['kelly_size']:>4} | "
                      f"{s['edge']*100:.1f}%")

        print(f"\n⏭  PASSES: {len(card['passes'])}")
        print(f"{'='*65}\n")
