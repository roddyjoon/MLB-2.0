"""
Live-odds scraper with two-tier fallback.

PRIMARY: Action Network MLB scoreboard
  GET https://api.actionnetwork.com/web/v1/scoreboard/mlb?bookIds=...&date=YYYYMMDD
  Clean JSON aggregate of DK/FanDuel/BetMGM/Caesars/Pinnacle etc.
  Quirk: sometimes hasn't populated today's games at 8 AM PT.

FALLBACK: SportsBookReview MLB odds page
  GET https://www.sportsbookreview.com/betting-odds/mlb-baseball/
  Next.js page with odds embedded in __NEXT_DATA__ script tag.
  6 books incl DraftKings (id 42), FanDuel (34), BetMGM (28), Caesars (41).
  Lines posted earlier than AN in practice (often by ~4 AM ET).

Direct DraftKings scraping was attempted first but DK responds with Akamai
403 to non-browser fingerprints; Odds Trader fetches lines client-side and
would require a headless browser. SBR + AN is the right combo for HTTP-only
scraping.

NOTE: this module is imported only in live mode. The backtest harness never
calls it (historical odds aren't available for free).
"""

import asyncio
import json
import random
import re
from typing import Dict, List, Optional

import aiohttp


AN_BASE = "https://api.actionnetwork.com/web/v1"
DEFAULT_BOOK_IDS = [15, 30, 68, 69, 71, 75, 79, 123, 247, 972]

# SBR identifies books by lowercase string slug in oddsViews[i].sportsbook
# (verified 2026-05-16 — `sportsbookId` field is null in the live page).
SBR_BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "caesars",
                      "bet365", "fanatics"]
SBR_MLB_URL = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
SBR_MLB_TOTALS_URL = (
    "https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/"
)
SBR_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)

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
        """
        Fetch the MLB scoreboard for `date`, cached per-date in-process.

        AN's MLB feed may not have today's games populated at 8 AM PT
        (the first morning card fire is right when books are still
        posting lines). Retry with backoff if we get 0 games: 60s, 90s,
        120s — cumulative ~4.5 min, well under launchd's tolerance.
        """
        if date in type(self)._scoreboard_cache:
            return type(self)._scoreboard_cache[date]
        if self._jitter > 0:
            await asyncio.sleep(self._jitter)
            self._jitter = 0

        an_date = date.replace("-", "")
        url = (f"{AN_BASE}/scoreboard/mlb"
               f"?bookIds={','.join(map(str, DEFAULT_BOOK_IDS))}"
               f"&date={an_date}")
        session = await self._session()

        # Retry pattern: try immediately, then wait 60s, 90s, 120s if empty.
        delays = [0, 60, 90, 120]
        data: dict = {}
        for attempt, delay in enumerate(delays, start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                async with session.get(url) as r:
                    if r.status == 200:
                        data = await r.json()
                    else:
                        data = {}
            except (aiohttp.ClientError, asyncio.TimeoutError):
                data = {}
            if data.get("games"):
                break

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
        Return current lines from the preferred sportsbook. Primary source
        is Action Network; falls back to SportsBookReview if AN has no
        odds for this game (e.g. early morning before AN populates).
        Empty dict if both sources fail.
        """
        # Tier 1: Action Network
        result = await self._lines_from_an(home, away, date)
        if result.get("home_ml_odds") is not None:
            return result

        # Tier 2: SportsBookReview (only useful for today's date — the
        # SBR page doesn't honor a date query param)
        sbr = await self._lines_from_sbr(home, away)
        if sbr.get("home_ml_odds") is not None:
            return sbr

        # Both empty
        return {}

    async def _lines_from_an(self, home: str, away: str,
                              date: str) -> Dict:
        """Fetch from Action Network primary source."""
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
            "source": "action_network",
        }

    # ---- SBR fallback ----
    # In-process caches keyed nothing (date is implicit: today's MLB page).
    _sbr_ml_cache: Optional[List[Dict]] = None
    _sbr_totals_cache: Optional[List[Dict]] = None

    async def _sbr_fetch(self, url: str) -> List[Dict]:
        """Fetch + parse one SBR /betting-odds page → gameRows list."""
        session = await self._session()
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    return []
                html = await r.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []
        m = SBR_NEXT_DATA_RE.search(html)
        if not m:
            return []
        try:
            data = json.loads(m.group(1))
            return (data["props"]["pageProps"]["oddsTables"][0]
                    ["oddsTableModel"]["gameRows"])
        except (KeyError, IndexError, json.JSONDecodeError):
            return []

    async def _sbr_games(self) -> List[Dict]:
        """Moneyline page — primary SBR fetch."""
        if type(self)._sbr_ml_cache is not None:
            return type(self)._sbr_ml_cache
        games = await self._sbr_fetch(SBR_MLB_URL)
        type(self)._sbr_ml_cache = games
        return games

    async def _sbr_totals_games(self) -> List[Dict]:
        """Totals page — separate fetch on SBR (only includes over/under odds)."""
        if type(self)._sbr_totals_cache is not None:
            return type(self)._sbr_totals_cache
        games = await self._sbr_fetch(SBR_MLB_TOTALS_URL)
        type(self)._sbr_totals_cache = games
        return games

    @staticmethod
    def _select_sbr_book(odds_views: List,
                          required_field: str = "homeOdds") -> Optional[Dict]:
        """
        Pick the preferred book's currentLine; falls through to any.

        `required_field` controls which field must be non-None for the entry
        to count — use 'homeOdds' for the moneyline page, 'total' for the
        totals page (where homeOdds is null).
        """
        for slug in SBR_BOOK_PRIORITY:
            for ov in odds_views or []:
                if not ov:
                    continue
                if (ov.get("sportsbook") or "").lower() == slug:
                    line = ov.get("currentLine") or {}
                    if line.get(required_field) is not None:
                        return {"line": line, "book_id": slug}
        # Fallback: any book with a usable line
        for ov in odds_views or []:
            if not ov:
                continue
            line = ov.get("currentLine") or {}
            if line.get(required_field) is not None:
                return {"line": line, "book_id": ov.get("sportsbook")}
        return None

    @staticmethod
    def _match_sbr_game(games: List[Dict], home: str, away: str
                        ) -> Optional[Dict]:
        """Find the matching SBR game row by 3-letter MLB abbreviation."""
        for g in games:
            gv = g.get("gameView") or {}
            h = (gv.get("homeTeam") or {}).get("shortName") or ""
            a = (gv.get("awayTeam") or {}).get("shortName") or ""
            if h == home and a == away:
                return g
        return None

    async def _lines_from_sbr(self, home: str, away: str) -> Dict:
        """Fetch from SportsBookReview (ML + totals merged)."""
        ml_games = await self._sbr_games()
        ml_game = self._match_sbr_game(ml_games, home, away)
        if not ml_game:
            return {}
        ml_pick = self._select_sbr_book(ml_game.get("oddsViews") or [])
        if not ml_pick:
            return {}
        ml = ml_pick["line"]

        # Merge in totals from the separate /totals/ page (where currentLine
        # has total + overOdds + underOdds populated, with homeOdds null).
        total_line = total_over = total_under = None
        totals_games = await self._sbr_totals_games()
        t_game = self._match_sbr_game(totals_games, home, away)
        if t_game:
            t_pick = self._select_sbr_book(t_game.get("oddsViews") or [],
                                            required_field="total")
            if t_pick:
                t_line = t_pick["line"]
                total_line = t_line.get("total")
                total_over = t_line.get("overOdds")
                total_under = t_line.get("underOdds")

        ml_home = ml.get("homeOdds")
        ml_away = ml.get("awayOdds")
        return {
            "home_ml_odds": ml_home,
            "away_ml_odds": ml_away,
            "total_line": total_line,
            "over_odds": total_over,
            "under_odds": total_under,
            "home_implied": round(_odds_to_implied(ml_home), 4)
                            if ml_home is not None else None,
            "away_implied": round(_odds_to_implied(ml_away), 4)
                            if ml_away is not None else None,
            "spread_home": ml.get("homeSpread"),
            "spread_away": ml.get("awaySpread"),
            "book_id": ml_pick["book_id"],
            "source": "sportsbookreview",
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
