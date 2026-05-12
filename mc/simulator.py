"""
Per-inning Monte Carlo MLB game simulator.

For each trial:
  for each of 9 innings:
    away half-inning runs ~ Poisson(λ_away_vs_home_pitcher)
    home half-inning runs ~ Poisson(λ_home_vs_away_pitcher)
  → winner = max(away_total, home_total)

λ derivation per half-inning:
  base_lambda = (offense_RPG * pitcher_runs_per_9) / LEAGUE_RPG / 9
  with park + weather + BvP modifiers

Pitcher swap: innings 1-6 use SP rate; innings 7-9 use bullpen ERA.

Industry-standard simplification — runs per inning are over-dispersed
(zero-inflated) in reality, but Poisson is sufficient for game-level WP.
The 95% CI on totals is wider with Poisson than with negative-binomial,
which is the conservative direction for projection bands.
"""

import math
import random
from typing import Dict, List, Tuple


# Calibration constants — 2024-2025 MLB league averages
LEAGUE_AVG_RPG = 4.5            # runs/team/game
LEAGUE_AVG_RUNS_PER_9 = 4.20    # for pitchers (FIP/ERA scale)
LEAGUE_AVG_WRC_PLUS = 100


def _poisson(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm — exact for small λ (most innings: λ ∈ [0.1, 1.5])."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def half_inning_lambda(
    offense_rpg: float,
    pitcher_runs_per_9: float,
    park_runs_factor: float,
    weather_per_game_adj: float,
    bvp_threat_count: int = 0,
) -> float:
    """
    Combine offense + pitcher + park + weather + BvP into a per-half-inning
    Poisson rate.

    Uses a multiplicative log5-style blend:
      effective_rate = offense_RPG * (pitcher_rate / league) * park * (1 + weather_adj/league)
    then / 9 for per-inning.

    BvP threats slightly amplify offense — each "elite" (career OPS ≥ 1.000
    vs this pitcher, ≥ 5 AB) adds ~1% to that side's offense.
    """
    pitcher_factor = pitcher_runs_per_9 / LEAGUE_AVG_RUNS_PER_9
    effective_rpg = offense_rpg * pitcher_factor * park_runs_factor
    bvp_factor = 1.0 + 0.01 * bvp_threat_count
    effective_rpg *= bvp_factor

    lam = effective_rpg / 9
    # Weather adjustment is a per-game runs delta; distribute across innings.
    lam += (weather_per_game_adj / 18)  # half across 18 half-innings
    return max(0.05, lam)


def simulate_game(
    inputs: Dict,
    n_trials: int = 7000,
    rng: random.Random = None,
) -> Dict:
    """
    Run Monte Carlo for one game. `inputs` shape:
      home_team_rpg, away_team_rpg          (float, RPG of each team)
      home_sp_rate, away_sp_rate            (float, FIP/SIERA blend per 9)
      home_bp_rate, away_bp_rate            (float, bullpen ERA per 9)
      park_runs_factor                      (float, ~0.93 to 1.16)
      weather_run_adj                       (float, runs/game weather delta)
      home_bvp_elite_count, away_bvp_elite_count  (int)

    Returns:
      home_wp, away_wp,
      projected_total_mean, projected_total_ci95_lo, projected_total_ci95_hi,
      home_runline_win_wp  (home wins by >=2),
      away_runline_win_wp  (away wins by >=2),
      pushes,
      run_distribution_mean_home, run_distribution_mean_away
    """
    rng = rng or random.Random()

    home_offense = inputs["home_team_rpg"]
    away_offense = inputs["away_team_rpg"]
    home_sp = inputs["home_sp_rate"]
    away_sp = inputs["away_sp_rate"]
    home_bp = inputs["home_bp_rate"]
    away_bp = inputs["away_bp_rate"]
    park = inputs["park_runs_factor"]
    wx = inputs["weather_run_adj"]
    home_bvp_count = inputs.get("home_bvp_elite_count", 0)
    away_bvp_count = inputs.get("away_bvp_elite_count", 0)

    # Precompute the four per-inning lambdas (don't recompute per trial)
    # Home batting (vs away SP for innings 1-6, vs away BP for 7-9)
    lam_home_vs_away_sp = half_inning_lambda(
        home_offense, away_sp, park, wx, home_bvp_count)
    lam_home_vs_away_bp = half_inning_lambda(
        home_offense, away_bp, park, wx, home_bvp_count)
    lam_away_vs_home_sp = half_inning_lambda(
        away_offense, home_sp, park, wx, away_bvp_count)
    lam_away_vs_home_bp = half_inning_lambda(
        away_offense, home_bp, park, wx, away_bvp_count)

    home_wins = 0
    pushes = 0
    home_runline_wins = 0
    away_runline_wins = 0
    total_samples: List[int] = []
    home_runs_sum = 0
    away_runs_sum = 0

    MAX_EXTRA_INNINGS = 5  # cap at 14 innings to bound runtime
    for _ in range(n_trials):
        home_runs = 0
        away_runs = 0
        # Regulation 1-9
        for inning in range(1, 10):
            if inning <= 6:
                away_runs += _poisson(lam_away_vs_home_sp, rng)
                home_runs += _poisson(lam_home_vs_away_sp, rng)
            else:
                away_runs += _poisson(lam_away_vs_home_bp, rng)
                home_runs += _poisson(lam_home_vs_away_bp, rng)
        # Extra innings if tied — MLB has no regulation ties.
        # In real life the ghost-runner rule applies; for game-level WP
        # purposes we use the same per-inning λ (rough but bounded).
        extras = 0
        while home_runs == away_runs and extras < MAX_EXTRA_INNINGS:
            away_runs += _poisson(lam_away_vs_home_bp, rng)
            home_runs += _poisson(lam_home_vs_away_bp, rng)
            extras += 1

        if home_runs > away_runs:
            home_wins += 1
            if home_runs - away_runs >= 2:
                home_runline_wins += 1
        elif away_runs > home_runs:
            if away_runs - home_runs >= 2:
                away_runline_wins += 1
        else:
            # Still tied after MAX_EXTRA_INNINGS — extremely rare; count as
            # a push but it'll be a tiny fraction of trials
            pushes += 1

        total_samples.append(home_runs + away_runs)
        home_runs_sum += home_runs
        away_runs_sum += away_runs

    total_samples.sort()
    ci_lo = total_samples[int(0.025 * n_trials)]
    ci_hi = total_samples[int(0.975 * n_trials)]

    return {
        "home_wp": round(home_wins / n_trials, 4),
        "away_wp": round((n_trials - home_wins - pushes) / n_trials, 4),
        "pushes": pushes,
        "projected_total_mean": round(sum(total_samples) / n_trials, 2),
        "projected_total_ci95_lo": ci_lo,
        "projected_total_ci95_hi": ci_hi,
        "home_runline_win_wp": round(home_runline_wins / n_trials, 4),
        "away_runline_win_wp": round(away_runline_wins / n_trials, 4),
        "mean_home_runs": round(home_runs_sum / n_trials, 2),
        "mean_away_runs": round(away_runs_sum / n_trials, 2),
        # Echo back the inputs used for transparency / drivers
        "inputs": {
            "home_team_rpg": home_offense,
            "away_team_rpg": away_offense,
            "home_sp_rate": home_sp,
            "away_sp_rate": away_sp,
            "home_bp_rate": home_bp,
            "away_bp_rate": away_bp,
            "park_runs_factor": park,
            "weather_run_adj": wx,
            "home_bvp_elite_count": home_bvp_count,
            "away_bvp_elite_count": away_bvp_count,
        },
    }
