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

    async def get_probable_lineup(self, team_abbr: str,
                                  as_of: str,
                                  lookback: int = 7) -> List[Dict]:
        """
        Predict today's starting 9 by aggregating the team's confirmed
        batting orders from its last `lookback` completed games.

        Returns up to 9 players ordered by their average batting-slot
        position (cleanup hitters near slot 4, leadoff near 1, etc.).
        Each player dict carries `"projected": True` so downstream consumers
        can flag the lineup as estimated rather than confirmed.

        Caveats:
        - Doesn't filter by opposing-SP handedness (would improve accuracy
          on platoon-heavy teams but adds API calls; skipped for now).
        - Players returning from IL or recent call-ups won't appear until
          they've started at least one of the lookback games.
        """
        from datetime import datetime, timedelta
        from data.mlb_api import TEAM_IDS, MLB_STATS_API

        team_id = TEAM_IDS.get(team_abbr)
        if not team_id:
            return []

        try:
            end_dt = datetime.strptime(as_of, "%Y-%m-%d")
        except ValueError:
            return []
        start_dt = end_dt - timedelta(days=lookback + 3)  # extra slack for off days

        session = await self._session()
        params = {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_dt.strftime("%Y-%m-%d"),
            "endDate": (end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        try:
            async with session.get(f"{MLB_STATS_API}/schedule",
                                    params=params) as r:
                if r.status != 200:
                    return []
                data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []

        # Collect (game_pk, side) pairs for completed games in chronological
        # order (we'll iterate latest first when capping at `lookback`).
        completed: List[tuple] = []  # [(date_str, game_pk, side), ...]
        for date_data in data.get("dates", []):
            for g in date_data.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                teams = g.get("teams", {})
                h_id = teams.get("home", {}).get("team", {}).get("id")
                a_id = teams.get("away", {}).get("team", {}).get("id")
                pk = str(g.get("gamePk"))
                date = (g.get("officialDate")
                        or g.get("gameDate", "")[:10])
                if h_id == team_id:
                    completed.append((date, pk, "home"))
                elif a_id == team_id:
                    completed.append((date, pk, "away"))

        completed.sort(reverse=True)  # most recent first
        completed = completed[:lookback]

        if not completed:
            return []

        # Aggregate batting orders. player_counts = how many of the last N
        # games each player started; player_slots = their batting positions.
        player_counts: Dict[str, int] = {}
        player_slots: Dict[str, List[int]] = {}
        player_info: Dict[str, Dict] = {}

        for _date, pk, side in completed:
            lineups = await self._fetch(pk)
            team_lineup = lineups.get(side, [])
            for idx, batter in enumerate(team_lineup):
                pid = batter.get("player_id")
                if not pid:
                    continue
                player_counts[pid] = player_counts.get(pid, 0) + 1
                player_slots.setdefault(pid, []).append(idx + 1)
                player_info[pid] = {
                    "player_id": pid,
                    "name": batter.get("name", ""),
                    "position": batter.get("position", ""),
                    "bats": batter.get("bats", ""),
                }

        if not player_counts:
            return []

        # Top 9 by start count, then sort by avg batting slot
        ranked = sorted(player_counts.items(), key=lambda x: x[1],
                        reverse=True)[:9]
        ranked.sort(key=lambda x: sum(player_slots[x[0]]) / len(player_slots[x[0]]))

        return [
            {**player_info[pid],
             "projected": True,
             "starts_recent": cnt,
             "avg_slot": round(sum(player_slots[pid]) / len(player_slots[pid]), 1)}
            for pid, cnt in ranked
        ]
