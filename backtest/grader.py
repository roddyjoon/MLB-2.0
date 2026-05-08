"""
Backtest grader — flatten orchestrator outputs into CSV rows and join with
final scores.

Backtest mode: market/edges/Kelly are skipped (no historical odds for free).
We grade WP-only: predicted home_wp vs actual home_won, projected total vs
actual total runs.
"""

from typing import Dict, List, Optional


CSV_COLUMNS = [
    "date", "game_pk", "home", "away",
    "home_sp", "away_sp",
    "home_wp", "away_wp", "total_proj", "over_wp", "under_wp",
    "fip_gap", "wrc_gap",
    "park_runs_factor", "park_hr_factor",
    "ump_name", "weather_temp", "weather_wind_speed",
    "home_score", "away_score", "home_won", "total_actual", "over_8_5_hit",
    "agents_succeeded",
    "error",
]


def _safe(d: Dict, *path, default=None):
    """Walk a dict tree safely; return `default` on any missing key."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def flatten_prediction(date: str, sim_result: Dict) -> Dict:
    """Convert orchestrator simulate_game_v25 output to a flat CSV row."""
    game = sim_result.get("game", {})
    agents = sim_result.get("agents", {})
    wp = sim_result.get("wp_result") or {}

    # Count agents that returned non-empty dicts (failed agents → {} via gather)
    agents_succeeded = sum(1 for v in agents.values()
                           if isinstance(v, dict) and v)

    return {
        "date": date,
        "game_pk": game.get("game_id", ""),
        "home": game.get("home_team", ""),
        "away": game.get("away_team", ""),
        "home_sp": game.get("home_sp_name", ""),
        "away_sp": game.get("away_sp_name", ""),
        "home_wp": wp.get("home_wp"),
        "away_wp": wp.get("away_wp"),
        "total_proj": wp.get("total_projection"),
        "over_wp": wp.get("over_wp"),
        "under_wp": wp.get("under_wp"),
        "fip_gap": _safe(agents, "sp", "fip_gap"),
        "wrc_gap": _safe(agents, "lineup", "wrc_gap"),
        "park_runs_factor": _safe(agents, "park", "park", "runs"),
        "park_hr_factor": _safe(agents, "park", "park", "hr"),
        "ump_name": _safe(agents, "umpire", "umpire_name"),
        "weather_temp": _safe(agents, "park", "weather", "temperature"),
        "weather_wind_speed": _safe(agents, "park", "weather", "wind_speed"),
        # Final-score columns filled by `attach_finals` after grading.
        "home_score": None,
        "away_score": None,
        "home_won": None,
        "total_actual": None,
        "over_8_5_hit": None,
        "agents_succeeded": agents_succeeded,
        "error": sim_result.get("error"),
    }


def attach_finals(row: Dict, finals_by_pk: Dict[str, Dict]) -> Dict:
    """Join one prediction row with the final-score dict by game_pk."""
    final = finals_by_pk.get(str(row["game_pk"]))
    if not final:
        return row
    h = int(final["home_score"])
    a = int(final["away_score"])
    row["home_score"] = h
    row["away_score"] = a
    row["home_won"] = 1 if h > a else 0
    row["total_actual"] = h + a
    row["over_8_5_hit"] = 1 if (h + a) > 8.5 else 0
    return row


def join_predictions_with_finals(
    rows: List[Dict],
    finals: List[Dict],
) -> List[Dict]:
    by_pk = {str(f.get("game_pk", "")): f for f in finals}
    return [attach_finals(r, by_pk) for r in rows]
