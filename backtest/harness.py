"""
Backtest harness.

Iterates a date range, runs V25Orchestrator in backtest mode per day, joins
to final scores, writes one CSV per day to outputs/backtest/.
"""

import asyncio
import csv
from datetime import date as DateT, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from core.orchestrator_v25 import V25Orchestrator
from core.logger import ModelLogger
from backtest.grader import (
    CSV_COLUMNS,
    flatten_prediction,
    join_predictions_with_finals,
)

logger = ModelLogger("backtest.harness")


def daterange(start: str, end: str) -> Iterable[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        raise ValueError(f"end {end} is before start {start}")
    cur = s
    while cur <= e:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


async def run_one_day(date_str: str, out_dir: Path,
                      game_concurrency: int = 4) -> List[dict]:
    """
    Backtest a single date. Returns the list of CSV rows (also written to disk).
    Skips games not in 'Final' status (in-progress / postponed / spring training).
    """
    orch = V25Orchestrator(mode="backtest", as_of_date=date_str)
    games = await orch.mlb_api.get_games_for_date(date_str)
    finals_only = [g for g in games
                   if g.get("status") == "Final"]
    if not finals_only:
        logger.info(f"{date_str}: no completed games, skipping")
        return []

    sem = asyncio.Semaphore(game_concurrency)

    async def sim(g):
        # Make sure date is on the game_data dict (5 agents read it from there)
        g_with_date = {**g, "date": date_str}
        async with sem:
            return await orch.simulate_game_v25(g_with_date, date_str)

    sims = await asyncio.gather(*[sim(g) for g in finals_only],
                                 return_exceptions=True)
    rows = []
    for sim_result in sims:
        if isinstance(sim_result, Exception):
            logger.error(f"sim failed: {sim_result}")
            continue
        rows.append(flatten_prediction(date_str, sim_result))

    finals = await orch.mlb_api.get_final_scores(date_str)
    rows = join_predictions_with_finals(rows, finals)

    out_path = out_dir / f"predictions_{date_str}.csv"
    _write_csv(out_path, rows)
    logger.info(f"{date_str}: wrote {len(rows)} rows → {out_path}")
    return rows


async def run_backtest(start: str, end: str,
                       day_concurrency: int = 1,
                       game_concurrency: int = 4) -> None:
    """
    Backtest [start, end] inclusive. Days run sequentially by default
    (day_concurrency=1) to keep the FanGraphs/Savant rate footprint small;
    games within a day run in parallel (game_concurrency=4).
    """
    out_dir = Path("outputs/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(day_concurrency)

    async def one(d):
        async with sem:
            return await run_one_day(d, out_dir, game_concurrency)

    dates = list(daterange(start, end))
    logger.info(f"Backtesting {len(dates)} dates: {start} → {end}")
    for d in dates:
        await one(d)
