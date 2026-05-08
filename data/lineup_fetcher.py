"""
Confirmed-lineup fetcher — MLB Stats API live feed.

Endpoint:
  GET /api/v1.1/game/{gamePk}/feed/live
  → liveData.boxscore.teams.{home,away}.{battingOrder, players}

In live mode, lineups post ~90 min pre-game; before that, battingOrder is
empty (returns []). In backtest mode, completed-game boxscores always have
the actual lineup.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp


MLB_API_V11 = "https://statsapi.mlb.com/api/v1.1"


class LineupFetcher:
    """Resolves confirmed batting orders from MLB Stats API."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _mem: Dict[str, Dict[str, List[Dict]]] = {}  # gamePk → {home, away}

    def __init__(self, cache=None):
        self.cache = cache
        self._disk = Path("cache")
        self._disk.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._disk / "lineups.json"
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

    async def _fetch(self, game_pk: str) -> Dict[str, List[Dict]]:
        """Fetch lineups for one game. Returns {'home': [...], 'away': [...]}."""
        if game_pk in type(self)._mem:
            return type(self)._mem[game_pk]

        session = await self._session()
        url = f"{MLB_API_V11}/game/{game_pk}/feed/live"
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    return {"home": [], "away": []}
                feed = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {"home": [], "away": []}

        teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
        result = {}
        for side in ("home", "away"):
            t = teams.get(side, {})
            order = t.get("battingOrder", [])
            players = t.get("players", {})
            lineup = []
            for pid in order:
                p = players.get(f"ID{pid}", {})
                person = p.get("person", {})
                pos = p.get("position", {})
                lineup.append({
                    "player_id": str(pid),
                    "name": person.get("fullName", ""),
                    "position": pos.get("abbreviation", ""),
                    "bats": person.get("batSide", {}).get("code", ""),
                })
            result[side] = lineup

        # Only cache if at least one side is populated (live feeds may have
        # empty battingOrder pre-game; we want to retry then).
        if result["home"] or result["away"]:
            type(self)._mem[game_pk] = result
            try:
                self._cache_file.write_text(json.dumps(type(self)._mem))
            except OSError:
                pass
        return result

    async def get_lineups_for_game(self, game_pk: str) -> Dict[str, List[Dict]]:
        """Return both lineups: {'home': [...], 'away': [...]}."""
        if not game_pk:
            return {"home": [], "away": []}
        return await self._fetch(game_pk)
