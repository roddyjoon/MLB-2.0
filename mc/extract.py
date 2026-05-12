"""
Extract Monte Carlo inputs from an orchestrator simulate_game_v25 result.

The orchestrator runs all 13 agents and returns a dict with agent outputs.
We pull the numbers MC needs (offense rates, SP/BP rates, park, weather,
BvP counts) and shape them for mc.simulator.simulate_game.
"""

from typing import Dict


LEAGUE_AVG_RPG = 4.5
LEAGUE_AVG_RUNS_PER_9 = 4.20


def extract_inputs(sim_result: Dict) -> Dict:
    """Convert orchestrator sim result → mc.simulator.simulate_game kwargs."""
    agents = sim_result.get("agents", {})
    sp = agents.get("sp", {})
    lineup = agents.get("lineup", {})
    bullpen = agents.get("bullpen", {})
    park = agents.get("park", {})

    home_sp = sp.get("home_sp", {}) or {}
    away_sp = sp.get("away_sp", {}) or {}
    home_pane = lineup.get("home", {}) or {}
    away_pane = lineup.get("away", {}) or {}
    home_bp = bullpen.get("home_bp", {}) or {}
    away_bp = bullpen.get("away_bp", {}) or {}
    env = park.get("environment", {}) or {}

    home_bvp_threats = home_pane.get("bvp_threats", []) or []
    away_bvp_threats = away_pane.get("bvp_threats", []) or []
    # Count only elite/strong/homer_history threats (≥5 AB, ≥1.000 OPS or HR)
    def _count_elite(threats):
        return sum(1 for t in threats
                   if t.get("significance") in
                   ("elite", "strong", "homer_history"))

    return {
        "home_team_rpg": float(home_pane.get("rpg", LEAGUE_AVG_RPG)),
        "away_team_rpg": float(away_pane.get("rpg", LEAGUE_AVG_RPG)),
        "home_sp_rate": float(home_sp.get("adjusted_blend",
                                          home_sp.get("xfip",
                                                      LEAGUE_AVG_RUNS_PER_9))),
        "away_sp_rate": float(away_sp.get("adjusted_blend",
                                          away_sp.get("xfip",
                                                      LEAGUE_AVG_RUNS_PER_9))),
        "home_bp_rate": float(home_bp.get("season_era", LEAGUE_AVG_RUNS_PER_9)),
        "away_bp_rate": float(away_bp.get("season_era", LEAGUE_AVG_RUNS_PER_9)),
        "park_runs_factor": float(park.get("park", {}).get("runs", 1.00)),
        "weather_run_adj": float(env.get("run_proj_adj", 0)),
        "home_bvp_elite_count": _count_elite(home_bvp_threats),
        "away_bvp_elite_count": _count_elite(away_bvp_threats),
    }
