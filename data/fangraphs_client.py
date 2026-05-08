"""
FanGraphs leaderboard client.

FanGraphs is a Next.js app: their team-leaderboard page is HTML with the data
embedded in a `<script id="__NEXT_DATA__">` tag. This client fetches the page,
extracts the JSON, and indexes the 30 team rows by abbreviation.

Endpoints used:
  /leaders/major-league?stats=bat&season=YYYY&team=0,ts&type=8&qual=0
  /leaders/major-league?stats=rel&season=YYYY&team=0,ts&type=8&qual=0

Cached per (season, stats) — one HTTP request gets all 30 teams.

NOTE on backtest as-of-date precision: FanGraphs leaderboards default to
season-to-date as of fetch time. For backtest in 2026 against a 2025-06-15
game, this returns end-of-2025 stats. URL params `month`, `startdate`,
`enddate` exist but are inconsistently honored — verify per-call before
relying. For the wRC+ blender's rolling-11 component, this client returns
None and the blender falls back to season-only with a warning.

FanGraphs abbreviations differ slightly from MLB Stats API for 7 teams:
  MLB CWS → FG CHW; MLB SD → SDP; MLB SF → SFG; MLB KC → KCR;
  MLB TB → TBR; MLB WSH → WSN; MLB AZ → ARI.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Optional

import aiohttp


FG_BASE = "https://www.fangraphs.com"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)

# 7 teams with different abbreviations on FanGraphs vs MLB Stats API.
MLB_TO_FG = {
    "CWS": "CHW",
    "SD": "SDP",
    "SF": "SFG",
    "KC": "KCR",
    "TB": "TBR",
    "WSH": "WSN",
    "AZ": "ARI",
    # OAK/ATH share id 133 in MLB; Athletics renamed to "ATH" — FG uses ATH too.
}


def to_fg_abbr(mlb_abbr: str) -> str:
    return MLB_TO_FG.get(mlb_abbr, mlb_abbr)


class FanGraphsClient:
    """Year-keyed cache of FanGraphs team leaderboards (batting + bullpen)."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _mem: Dict[tuple, Dict[str, Dict]] = {}  # {(year, stats): {fg_abbr: row}}

    def __init__(self, cache=None, as_of: str = ""):
        self.cache = cache
        self.as_of = as_of
        self._disk = Path("cache/fangraphs")
        self._disk.mkdir(parents=True, exist_ok=True)

    @classmethod
    async def _session(cls) -> aiohttp.ClientSession:
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (mlb-v3 analytics)"}
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def _fetch_leaderboard(self, year: int, stats: str) -> Dict[str, Dict]:
        """Fetch + cache one FanGraphs team leaderboard. stats: 'bat' or 'rel'."""
        key = (year, stats)
        if key in type(self)._mem:
            return type(self)._mem[key]

        cache_file = self._disk / f"{stats}_{year}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            type(self)._mem[key] = data
            return data

        url = (f"{FG_BASE}/leaders/major-league"
               f"?stats={stats}&season={year}&season1={year}"
               f"&pos=all&qual=0&type=8&team=0%2Cts&ind=0")

        session = await self._session()
        async with session.get(url) as resp:
            if resp.status != 200:
                return {}
            html = await resp.text()

        m = NEXT_DATA_RE.search(html)
        if not m:
            return {}

        try:
            payload = json.loads(m.group(1))
            rows = (payload["props"]["pageProps"]["dehydratedState"]
                    ["queries"][0]["state"]["data"]["data"])
        except (KeyError, IndexError, json.JSONDecodeError):
            return {}

        by_abbr: Dict[str, Dict] = {}
        for row in rows:
            abbr = row.get("TeamNameAbb")
            if abbr:
                by_abbr[abbr] = row

        with open(cache_file, "w") as f:
            json.dump(by_abbr, f)
        type(self)._mem[key] = by_abbr
        return by_abbr

    @staticmethod
    def _season_for(as_of: str) -> int:
        try:
            return int(as_of[:4])
        except (ValueError, TypeError):
            from datetime import datetime
            return datetime.now().year

    async def get_team_batting(self, team: str, as_of: str) -> Dict:
        """
        Returns season-to-date batting stats for one team:
        {wrc_plus, k_pct, bb_pct, iso, ops, ba, obp, slg, hr, rpg, woba, xwoba}.
        Empty dict if team not found.
        """
        year = self._season_for(as_of or self.as_of)
        league = await self._fetch_leaderboard(year, "bat")
        row = league.get(to_fg_abbr(team))
        if not row:
            return {}

        # FanGraphs team-aggregate G is the sum of *player* games, not team
        # games (NYY 2025 G=2401 vs ~162 team games). Estimate team games
        # from PA: a full lineup averages ~38 PA per team-game.
        pa = row.get("PA") or 0
        team_games = pa / 38 if pa else 0
        runs = row.get("R") or 0
        rpg = round(runs / team_games, 2) if team_games else 0.0
        return {
            "wrc_plus": row.get("wRC+"),
            "k_pct": row.get("K%"),
            "bb_pct": row.get("BB%"),
            "iso": row.get("ISO"),
            "ops": row.get("OPS"),
            "ba": row.get("AVG"),
            "obp": row.get("OBP"),
            "slg": row.get("SLG"),
            "hr": row.get("HR"),
            "rpg": rpg,
            "woba": row.get("wOBA"),
            "xwoba": row.get("xwOBA"),
        }

    async def get_team_rolling11(self, team: str, as_of: str) -> Optional[Dict]:
        """
        Last-11-day batting line for the wRC+ blender.

        Currently returns None — FanGraphs URL date-range filtering is
        inconsistent and an extra HTTP call per team isn't justified until
        the backtest demonstrates rolling-11 wRC+ moves WP calibration.
        wrc_blender.WRCBlender.blend handles None gracefully (season-only).
        """
        return None

    async def get_bullpen(self, team: str, as_of: str) -> Dict:
        """
        Returns season-to-date bullpen stats:
        {season_era, xfip, fip, k9, bb9, hr9, war, ip, sv, bs, hld}.
        Empty dict if team not found.
        """
        year = self._season_for(as_of or self.as_of)
        league = await self._fetch_leaderboard(year, "rel")
        row = league.get(to_fg_abbr(team))
        if not row:
            return {}

        return {
            "season_era": row.get("ERA"),
            "fip": row.get("FIP"),
            "xfip": row.get("xFIP"),
            "k9": row.get("K/9"),
            "bb9": row.get("BB/9"),
            "hr9": row.get("HR/9"),
            "ip": row.get("IP"),
            "sv": row.get("SV"),
            "bs": row.get("BS"),
            "hld": row.get("HLD"),
            "war": row.get("WAR"),
        }
