"""
Live-odds scraper.

Initial design called for direct DraftKings/FanDuel scraping, but DK responds
with Akamai 403 to non-browser fingerprints and FD ships an SPA shell. The
Action Network scoreboard endpoint returns a clean JSON aggregate of the
major US sportsbooks (DK, FanDuel, BetMGM, Caesars, Pinnacle, etc.) and is
not behind a bot wall. We use it as the primary source.

Endpoint:
  GET https://api.actionnetwork.com/web/v1/scoreboard/mlb?bookIds=15,30,68,69,71,75,79,123,247,972

Each game has an `odds` list with one entry per (book_id, type) pair. We
filter to type='game' and prefer DraftKings (book_id=75); fall back to
FanDuel (71), then any book.

NOTE: this module is imported only in live mode. The backtest harness never
calls it (historical odds aren't available for free per project decisions).
"""

import asyncio
import random
from typing import Dict, List, Optional

import aiohttp


AN_BASE = "https://api.actionnetwork.com/web/v1"
DEFAULT_BOOK_IDS = [15, 30, 68, 69, 71, 75, 79, 123, 247, 972]

# Action Network → MLB Stats API team abbreviation translation. The 7
# mismatches are the same ones FanGraphs uses (CWS↔CHW, etc.) plus a couple
# AN-specific ones.
AN_TO_MLB = {
    "CHW": "CWS",
    "SDP": "SD",
    "SFG": "SF",
    "KCR": "KC",
    "TBR": "TB",
    "WSN": "WSH",
    "ARI": "AZ",
    "OAK": "ATH",
}


def _to_mlb_abbr(an_abbr: str) -> str:
    return AN_TO_MLB.get(an_abbr, an_abbr)


def _odds_to_implied(american_odds: int) -> float:
    """Plus money: 100/(odds+100). Minus: |odds|/(|odds|+100)."""
    if american_odds is None:
        return 0.0
    if american_odds >= 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


class OddsScraper:
    """Action-Network-backed live odds. Live mode only."""

    _shared_session: Optional[aiohttp.ClientSession] = None
    _scoreboard_cache: Dict[str, dict] = {}  # date → scoreboard payload

    def __init__(self):
        # Mild jitter on the shared session's first connect
        self._jitter = random.uniform(0.0, 1.5)

    @classmethod
    async def _session(cls) -> aiohttp.ClientSession:
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={
                    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X "
                                   "10_15_7) AppleWebKit/537.36 (KHTML, like "
                                   "Gecko) Chrome/124.0.0.0 Safari/537.36"),
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.actionnetwork.com/",
                },
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def _scoreboard(self, date: str) -> dict:
        """Fetch the MLB scoreboard for `date`, cached per-date in-process."""
        if date in type(self)._scoreboard_cache:
            return type(self)._scoreboard_cache[date]
        if self._jitter > 0:
            await asyncio.sleep(self._jitter)
            self._jitter = 0
        session = await self._session()
        # AN expects date as YYYYMMDD (no dashes); ISO YYYY-MM-DD silently
        # returns 0 games.
        an_date = date.replace("-", "")
        url = (f"{AN_BASE}/scoreboard/mlb"
               f"?bookIds={','.join(map(str, DEFAULT_BOOK_IDS))}"
               f"&date={an_date}")
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    return {}
                data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {}
        type(self)._scoreboard_cache[date] = data
        return data

    @staticmethod
    def _select_game_odds(game: dict,
                          preferred_books: List[int] = (75, 71, 30, 68)
                          ) -> Optional[dict]:
        """Pick best-available `type=game` odds entry by preferred-book order."""
        game_odds = [o for o in game.get("odds", []) if o.get("type") == "game"]
        for book in preferred_books:
            for o in game_odds:
                if o.get("book_id") == book:
                    return o
        return game_odds[0] if game_odds else None

    @staticmethod
    def _find_game(scoreboard: dict, home: str, away: str) -> Optional[dict]:
        for g in scoreboard.get("games", []):
            try:
                a_team = next(t for t in g["teams"]
                              if t["id"] == g["away_team_id"])
                h_team = next(t for t in g["teams"]
                              if t["id"] == g["home_team_id"])
            except (StopIteration, KeyError):
                continue
            if (_to_mlb_abbr(h_team.get("abbr", "")) == home
                    and _to_mlb_abbr(a_team.get("abbr", "")) == away):
                return g
        return None

    async def get_current_lines(self, home: str, away: str,
                                date: str) -> Dict:
        """
        Return current lines from the preferred sportsbook (DK > FD > others).
        Empty dict if game not found or no odds posted yet.
        """
        sb = await self._scoreboard(date)
        if not sb:
            return {}
        game = self._find_game(sb, home, away)
        if not game:
            return {}
        odds = self._select_game_odds(game)
        if not odds:
            return {}

        ml_home = odds.get("ml_home")
        ml_away = odds.get("ml_away")
        return {
            "home_ml_odds": ml_home,
            "away_ml_odds": ml_away,
            "total_line": odds.get("total"),
            "over_odds": odds.get("over"),
            "under_odds": odds.get("under"),
            "home_implied": round(_odds_to_implied(ml_home), 4)
                            if ml_home is not None else None,
            "away_implied": round(_odds_to_implied(ml_away), 4)
                            if ml_away is not None else None,
            "spread_home": odds.get("spread_home"),
            "spread_away": odds.get("spread_away"),
            "book_id": odds.get("book_id"),
        }

    async def get_opening_lines(self, home: str, away: str,
                                date: str) -> Dict:
        """
        Action Network exposes inserted/openline via a separate endpoint we
        haven't wired yet. For now, return current as a best-effort.
        Line-movement agent gracefully degrades when opening==current.
        """
        return await self.get_current_lines(home, away, date)

    async def get_public_pcts(self, home: str, away: str,
                              date: str) -> Dict:
        """Public-betting % from Action Network ml_home_public/total_over_public."""
        sb = await self._scoreboard(date)
        if not sb:
            return {}
        game = self._find_game(sb, home, away)
        if not game:
            return {}
        odds = self._select_game_odds(game)
        if not odds:
            return {}

        def _pct(v):
            return float(v) / 100 if v is not None else 0.50

        return {
            "home_ml_pct": _pct(odds.get("ml_home_public")),
            "away_ml_pct": _pct(odds.get("ml_away_public")),
            "over_pct": _pct(odds.get("total_over_public")),
            "under_pct": _pct(odds.get("total_under_public")),
        }
