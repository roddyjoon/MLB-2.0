#!/usr/bin/env python3
"""
MLB Model v3 — entry point.

Commands:
  sim <away> <home> [date]          Simulate one game (live mode by default)
  card [date]                       Generate full daily card (live mode)
  backtest --start <YYYY-MM-DD> --end <YYYY-MM-DD>
                                    Run WP-only backtest across a date range
  calibrate --start <YYYY-MM-DD> --end <YYYY-MM-DD>
                                    Compute Brier/log-loss/reliability over
                                    an existing backtest run
  monitor                           SP-change monitor (live, runs forever)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from core.orchestrator_v25 import V25Orchestrator
from core.scheduler import DailyScheduler
from core.logger import ModelLogger
from data.mlb_api import MLBDataAPI, SavantAPI

logger = ModelLogger("main")


async def run_sim(away: str, home: str, date_str: str = None):
    """Simulate a single game in live mode."""
    date = date_str or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"v3 sim — {away} @ {home} on {date}")

    orch = V25Orchestrator(mode="live", as_of_date=date)
    games = await orch.mlb_api.get_games_for_date(date)
    game = next((g for g in games
                 if g["home_team"] == home and g["away_team"] == away), None)
    if not game:
        print(f"No game found for {away} @ {home} on {date}")
        # Fall back to a synthetic game_data so the smoke test exercises the
        # full agent pipeline even on a date with no scheduled matchup.
        game = {
            "game_id": f"synthetic_{away}_{home}_{date}",
            "home_team": home, "away_team": away,
            "home_team_id": None, "away_team_id": None,
            "home_sp_name": "", "away_sp_name": "",
            "home_sp_id": "", "away_sp_id": "",
            "game_time": f"{date}T19:05:00Z",
            "venue": "", "status": "Scheduled"
        }

    result = await orch.simulate_game_v25(game, date)
    print(json.dumps(result, indent=2, default=str))
    return result


async def run_card(date_str: str = None):
    """Generate daily card in live mode."""
    date = date_str or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"v3 card — {date}")

    orch = V25Orchestrator(mode="live", as_of_date=date)
    card = await orch.generate_daily_card(date)

    output_dir = Path("outputs/cards")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"card_{date}.json"
    with open(out, "w") as f:
        json.dump(card, f, indent=2, default=str)
    orch.print_card(card)
    logger.info(f"Card saved to {out}")
    return card


async def run_backtest(start: str, end: str):
    """Run WP-only backtest across [start, end]. Step 6+ implementation."""
    try:
        from backtest.harness import run_backtest as harness_run
    except ImportError:
        print("backtest harness not yet implemented (planned for step 6).")
        print(f"Would have run: {start} → {end}")
        return
    await harness_run(start, end)


async def run_calibrate(start: str, end: str):
    """Compute calibration metrics over an existing backtest run."""
    try:
        from backtest.calibration import run_calibration
    except ImportError:
        print("calibration not yet implemented (planned for step 8).")
        print(f"Would have calibrated: {start} → {end}")
        return
    await run_calibration(start, end)


async def run_monitor():
    """SP-change monitor (live)."""
    scheduler = DailyScheduler()
    await scheduler.monitor_sp_changes()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlb-v3",
        description="MLB v3 — multi-agent betting analytics")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="Simulate one game")
    s.add_argument("away")
    s.add_argument("home")
    s.add_argument("date", nargs="?", default=None)

    c = sub.add_parser("card", help="Generate daily card")
    c.add_argument("date", nargs="?", default=None)

    b = sub.add_parser("backtest", help="Run WP-only backtest")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)

    cal = sub.add_parser("calibrate", help="Compute calibration metrics")
    cal.add_argument("--start", required=True)
    cal.add_argument("--end", required=True)

    sub.add_parser("monitor", help="SP-change monitor (live, runs forever)")

    return p


async def _close_sessions():
    """Close shared aiohttp sessions on shutdown to avoid 'Unclosed session'."""
    from data.fangraphs_client import FanGraphsClient
    from data.nws_client import NWSClient
    from data.umpscores_client import UmpScoresClient
    from data.lineup_fetcher import LineupFetcher
    from data.odds_scraper import OddsScraper
    await MLBDataAPI.close()
    await SavantAPI.close()
    await FanGraphsClient.close()
    await NWSClient.close()
    await UmpScoresClient.close()
    await LineupFetcher.close()
    await OddsScraper.close()


def _run(coro):
    """Run an async entry point and ensure HTTP sessions are closed."""
    async def wrapper():
        try:
            return await coro
        finally:
            await _close_sessions()
    return asyncio.run(wrapper())


def main():
    args = build_parser().parse_args()
    if args.cmd == "sim":
        _run(run_sim(args.away, args.home, args.date))
    elif args.cmd == "card":
        _run(run_card(args.date))
    elif args.cmd == "backtest":
        _run(run_backtest(args.start, args.end))
    elif args.cmd == "calibrate":
        _run(run_calibrate(args.start, args.end))
    elif args.cmd == "monitor":
        _run(run_monitor())
    else:
        print(f"Unknown command: {args.cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
