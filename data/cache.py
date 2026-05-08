"""
SQLite cache for the data layer.

STEP-1 SCAFFOLD: schema + thin async API. Backtest writes are immutable
(cache key includes as_of); live writes use TTLs. Step 2+ wires the clients
to actually read/write.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

try:
    import aiosqlite
except ImportError:
    aiosqlite = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    ttl        REAL
);
CREATE TABLE IF NOT EXISTS team_stats (
    team       TEXT, as_of DATE, source TEXT,
    payload    TEXT, fetched_at REAL,
    PRIMARY KEY(team, as_of, source)
);
CREATE TABLE IF NOT EXISTS pitcher_stats (
    pitcher_id TEXT, as_of DATE, group_name TEXT,
    payload    TEXT, fetched_at REAL,
    PRIMARY KEY(pitcher_id, as_of, group_name)
);
CREATE TABLE IF NOT EXISTS pitcher_gamelogs (
    pitcher_id TEXT, game_date DATE,
    payload    TEXT, fetched_at REAL,
    PRIMARY KEY(pitcher_id, game_date)
);
CREATE TABLE IF NOT EXISTS bvp (
    batter_id  TEXT, pitcher_id TEXT, as_of DATE,
    payload    TEXT, fetched_at REAL,
    PRIMARY KEY(batter_id, pitcher_id, as_of)
);
CREATE TABLE IF NOT EXISTS weather (
    venue      TEXT, date DATE, payload TEXT, fetched_at REAL,
    PRIMARY KEY(venue, date)
);
CREATE TABLE IF NOT EXISTS umpires (
    date DATE, game_id TEXT, ump_name TEXT,
    PRIMARY KEY(date, game_id)
);
CREATE TABLE IF NOT EXISTS lineups (
    game_pk TEXT, team TEXT, payload TEXT,
    PRIMARY KEY(game_pk, team)
);
CREATE TABLE IF NOT EXISTS final_scores (
    game_pk TEXT PRIMARY KEY, date DATE, payload TEXT
);
"""


class SQLiteCache:
    def __init__(self, path: str = "cache/mlb.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure(self):
        if self._initialized:
            return
        if aiosqlite is None:
            raise RuntimeError(
                "aiosqlite not installed — `pip install aiosqlite`")
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            await db.commit()
        self._initialized = True

    async def get(self, key: str) -> Optional[Any]:
        """Get from kv table, honoring TTL."""
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            row = await (await db.execute(
                "SELECT value, fetched_at, ttl FROM kv WHERE key = ?",
                (key,))).fetchone()
            if not row:
                return None
            value, fetched_at, ttl = row
            if ttl is not None and time.time() - fetched_at > ttl:
                return None
            return json.loads(value)

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO kv (key, value, fetched_at, ttl) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), time.time(), ttl))
            await db.commit()
