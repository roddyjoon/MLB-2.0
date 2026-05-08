"""
Baseball Savant custom-leaderboard CSV client.

Replaces v2.5's broken statcast_search/csv flow. The original
SavantAPI._parse_savant_csv (mlb_api.py:413-452) declared k_count, bb_count,
and pa_count but never incremented them, so K%/BB% always returned 0.

Endpoint: /leaderboard/custom?year=YYYY&type=pitcher&filter=&min=q
          &selections=k_percent,bb_percent,xwoba,exit_velocity_avg,
                      hard_hit_percent,barrel_batted_rate,whiff_percent
          &chartType=beeswarm&csv=true

Strategy: one fetch per (year, min_threshold), cached in-memory and on disk.
Looks up by player_id from the cached league-wide CSV.

NOTE on backtest as-of-date precision: the leaderboard URL ignores
start_dt/end_dt parameters — it always returns season-to-date stats as of
the time of fetch. For backtest in 2026 against a 2025 game, this returns
end-of-2025 numbers (slight forward-looking bias, but pitcher peripherals
stabilize over a season so impact on WP calibration is small). A precise
fix uses statcast_search/csv with game_date_gt/lt + group_by=name aggregation
— deferred until calibration shows it matters.
"""

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Dict, Optional

import aiohttp


SAVANT = "https://baseballsavant.mlb.com"

# Pitcher columns we request from the custom leaderboard. Names verified
# against the live endpoint on 2026-05-05; do not paraphrase.
SELECTIONS = ",".join([
    "k_percent",
    "bb_percent",
    "xwoba",
    "exit_velocity_avg",
    "hard_hit_percent",
    "barrel_batted_rate",
    "whiff_percent",
])


def _strip_bom(s: str) -> str:
    return s.lstrip("﻿")


def _to_float(v: str, scale: float = 1.0) -> Optional[float]:
    """Parse a Savant CSV value. Empty / blank → None. Scales (e.g. 24.6 → 0.246)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s) * scale
    except ValueError:
        return None


class StatcastLeaderboard:
    """Year-keyed cache of the Savant custom pitcher leaderboard."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _mem: Dict[int, Dict[str, Dict]] = {}  # {year: {player_id: row_dict}}

    def __init__(self, cache=None, as_of: str = ""):
        self.cache = cache
        self.as_of = as_of
        self._disk = Path("cache/statcast")
        self._disk.mkdir(parents=True, exist_ok=True)

    @classmethod
    async def _session(cls) -> aiohttp.ClientSession:
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "mlb-v3 (analytics; rodkazazi@gmail.com)"}
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def _fetch_year(self, year: int, min_threshold: str = "10") -> Dict[str, Dict]:
        """Fetch (or load from disk cache) the full league CSV for one year."""
        if year in type(self)._mem:
            return type(self)._mem[year]

        cache_file = self._disk / f"pitcher_leaderboard_{year}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            type(self)._mem[year] = data
            return data

        url = (f"{SAVANT}/leaderboard/custom"
               f"?year={year}&type=pitcher&filter=&min={min_threshold}"
               f"&selections={SELECTIONS}&chartType=beeswarm&csv=true")

        session = await self._session()
        async with session.get(url) as resp:
            if resp.status != 200:
                return {}
            text = _strip_bom(await resp.text())

        rows = list(csv.DictReader(io.StringIO(text)))
        by_id: Dict[str, Dict] = {}
        for row in rows:
            pid = (row.get("player_id") or "").strip()
            if not pid:
                continue
            by_id[pid] = {
                "name": row.get("last_name, first_name", "").strip(),
                "year": year,
                # K%/BB% come back as percentages (24.6 → 0.246)
                "k_pct": _to_float(row.get("k_percent"), 0.01),
                "bb_pct": _to_float(row.get("bb_percent"), 0.01),
                "xwoba": _to_float(row.get("xwoba")),
                "exit_velocity": _to_float(row.get("exit_velocity_avg")),
                "hard_hit_pct": _to_float(row.get("hard_hit_percent"), 0.01),
                "barrel_pct": _to_float(row.get("barrel_batted_rate"), 0.01),
                "whiff_pct": _to_float(row.get("whiff_percent"), 0.01),
            }

        with open(cache_file, "w") as f:
            json.dump(by_id, f)
        type(self)._mem[year] = by_id
        return by_id

    async def get_pitcher(self, pitcher_id: str, as_of: str = "") -> Dict:
        """
        Return Statcast metrics for one pitcher as of the season containing
        as_of. Empty dict if pitcher not in the leaderboard (insufficient PA).
        """
        if not pitcher_id:
            return {}
        year = int((as_of or self.as_of)[:4]) if (as_of or self.as_of) else 0
        if not year:
            from datetime import datetime
            year = datetime.now().year

        league = await self._fetch_year(year)
        row = league.get(str(pitcher_id))
        if not row:
            return {}

        # Drop None values so callers can `.get(k, default)` cleanly
        return {k: v for k, v in row.items() if v is not None}

    async def get_pitcher_arsenal(self, pitcher_id: str,
                                  as_of: str = "") -> Dict:
        """Pitch-arsenal data — TODO: separate Savant endpoint, deferred."""
        return {}
