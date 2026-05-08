"""
HP umpire client.

Despite the name, we don't scrape UmpScores.com — MLB Stats API itself returns
umpire assignments via the live game feed:
  GET /api/v1.1/game/{gamePk}/feed/live  →  liveData.boxscore.officials

The 'officials' list is empty until shortly before first pitch (live mode for
unstarted games returns None — agent falls back to league average). For
completed games (backtest), the list is always populated.

Per-ump tendencies (K-rate adj, runs/game adj, over%, squeeze) come from the
hardcoded `agents/umpire_agent.py:UMPIRE_DATABASE`. This client only resolves
the name; the agent does the lookup.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional

import aiohttp


MLB_API = "https://statsapi.mlb.com/api/v1"
MLB_API_V11 = "https://statsapi.mlb.com/api/v1.1"


class UmpScoresClient:
    """HP umpire resolver. Cached per gamePk."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _mem: Dict[str, Optional[str]] = {}  # gamePk → ump name (or None)

    def __init__(self, cache=None, as_of: str = ""):
        self.cache = cache
        self.as_of = as_of
        self._disk = Path("cache")
        self._disk.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._disk / "umpires.json"
        if self._cache_file.exists() and not type(self)._mem:
            try:
                type(self)._mem = json.loads(self._cache_file.read_text())
            except Exception:
                pass

    @classmethod
    async def _session(cls) -> aiohttp.ClientSession:
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def _resolve_game_pk(self, home: str, away: str,
                               date: Optional[str]) -> Optional[str]:
        if not date:
            return None
        session = await self._session()
        url = f"{MLB_API}/schedule"
        params = {"sportId": 1, "date": date}
        async with session.get(url, params=params) as r:
            if r.status != 200:
                return None
            sched = await r.json()
        # Reverse-lookup team abbr from id (lazy import to avoid cycle)
        from data.mlb_api import ID_TO_ABBR
        for date_data in sched.get("dates", []):
            for game in date_data.get("games", []):
                teams = game.get("teams", {})
                h_id = teams.get("home", {}).get("team", {}).get("id")
                a_id = teams.get("away", {}).get("team", {}).get("id")
                if (ID_TO_ABBR.get(h_id) == home
                        and ID_TO_ABBR.get(a_id) == away):
                    return str(game.get("gamePk"))
        return None

    async def get_hp_umpire(self, home: str, away: str,
                            date: str) -> Optional[str]:
        """Return HP umpire name or None if not yet assigned / not findable."""
        game_pk = await self._resolve_game_pk(home, away, date)
        if not game_pk:
            return None

        if game_pk in type(self)._mem:
            return type(self)._mem[game_pk]

        try:
            session = await self._session()
            url = f"{MLB_API_V11}/game/{game_pk}/feed/live"
            async with session.get(url) as r:
                if r.status != 200:
                    return None
                feed = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

        officials = (feed.get("liveData", {}).get("boxscore", {})
                     .get("officials", []))
        for o in officials:
            if o.get("officialType") == "Home Plate":
                name = o.get("official", {}).get("fullName")
                if name:
                    type(self)._mem[game_pk] = name
                    self._cache_file.write_text(json.dumps(type(self)._mem))
                    return name

        # Officials list exists but empty (game not yet started). Don't cache
        # the negative — try again next time the agent asks.
        return None
