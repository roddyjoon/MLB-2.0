"""
MLB Data API (v3) — facade over MLB Stats API + Savant + free public sources.

Step 1 (scaffold): ports v2.5 behavior so the smoke test runs end-to-end with
default fallbacks. Real data sources (FanGraphs, NWS, UmpScores, sportsbook
scraping, fixed Statcast parser) are wired in step 2+.

The agents in v2.5 self-instantiate MLBDataAPI(). To keep agent code unchanged
while still routing as_of_date / mode for backtest, this module exposes a
module-level configure() that the orchestrator calls before running agents.
Subsequent MLBDataAPI() calls inherit the configured defaults.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from core.logger import ModelLogger

logger = ModelLogger("mlb_api")

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"


# Module-level configuration — set by orchestrator before agents run so that
# agents that self-instantiate MLBDataAPI() inherit the correct as_of/mode.
_GLOBAL_AS_OF: Optional[str] = None
_GLOBAL_MODE: str = "live"


def configure(as_of_date: Optional[str] = None, mode: str = "live") -> None:
    """Set global as_of_date and mode for subsequently-instantiated APIs."""
    global _GLOBAL_AS_OF, _GLOBAL_MODE
    _GLOBAL_AS_OF = as_of_date
    _GLOBAL_MODE = mode


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class MLBDataAPI:
    """
    Async wrapper for MLB Stats API + delegated specialist clients.

    Sessions are class-level (shared across all instances) so that the 13
    agents that self-instantiate MLBDataAPI() don't each leak their own
    aiohttp session.
    """

    _shared_session: Optional[aiohttp.ClientSession] = None

    def __init__(self, as_of_date: Optional[str] = None,
                 mode: Optional[str] = None):
        self.as_of = as_of_date or _GLOBAL_AS_OF or _today_iso()
        self.mode = mode or _GLOBAL_MODE

    @property
    def _session(self) -> Optional[aiohttp.ClientSession]:
        """Backward-compat: some callers (and SavantAPI) reference _session."""
        return type(self)._shared_session

    @property
    def season(self) -> int:
        """Year of the as-of date (used for season-scoped MLB API queries)."""
        try:
            return int(self.as_of[:4])
        except (ValueError, TypeError):
            return datetime.now().year

    async def _get_session(self) -> aiohttp.ClientSession:
        cls = type(self)
        if not cls._shared_session or cls._shared_session.closed:
            cls._shared_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return cls._shared_session

    @classmethod
    async def close(cls) -> None:
        """Close the shared aiohttp session. Call once at process shutdown."""
        if cls._shared_session and not cls._shared_session.closed:
            await cls._shared_session.close()
            cls._shared_session = None

    async def get(self, url: str, params: Dict = None) -> Dict:
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning(f"API returned {resp.status} for {url}")
                return {}
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return {}

    async def get_games_for_date(self, date: str) -> List[Dict]:
        url = f"{MLB_STATS_API}/schedule"
        params = {
            "sportId": 1,
            "date": date,
            "hydrate": "probablePitcher,linescore,game(content(summary,media(epg)))"
        }
        data = await self.get(url, params)
        games = []
        for date_data in data.get("dates", []):
            for game in date_data.get("games", []):
                games.append(self._parse_game(game))
        return games

    def _parse_game(self, game: Dict) -> Dict:
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        home_pitcher = home.get("probablePitcher", {})
        away_pitcher = away.get("probablePitcher", {})
        home_id = home.get("team", {}).get("id")
        away_id = away.get("team", {}).get("id")
        return {
            "game_id": str(game.get("gamePk")),
            "home_team": ID_TO_ABBR.get(home_id, ""),
            "away_team": ID_TO_ABBR.get(away_id, ""),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_sp_name": home_pitcher.get("fullName", ""),
            "away_sp_name": away_pitcher.get("fullName", ""),
            "home_sp_id": str(home_pitcher.get("id", "")),
            "away_sp_id": str(away_pitcher.get("id", "")),
            "game_time": game.get("gameDate", ""),
            "venue": game.get("venue", {}).get("name", ""),
            "status": game.get("status", {}).get("detailedState", "")
        }

    async def get_confirmed_starters(self, home: str, away: str,
                                     date: str) -> Dict:
        games = await self.get_games_for_date(date)
        for g in games:
            if g["home_team"] == home and g["away_team"] == away:
                return {
                    "home_sp_name": g.get("home_sp_name"),
                    "home_sp_id": g.get("home_sp_id"),
                    "away_sp_name": g.get("away_sp_name"),
                    "away_sp_id": g.get("away_sp_id")
                }
        return {}

    async def confirm_starters(self, home: str, away: str, date: str) -> Dict:
        return await self.get_confirmed_starters(home, away, date)

    async def get_pitcher_season_stats(self, pitcher_id: str) -> Dict:
        if not pitcher_id:
            return {}
        url = f"{MLB_STATS_API}/people/{pitcher_id}/stats"
        params = {"stats": "season", "group": "pitching", "season": self.season}
        data = await self.get(url, params)
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                return {
                    "era": float(s.get("era", 0) or 0),
                    "ip": float(s.get("inningsPitched", 0) or 0),
                    "fip": float(s.get("fip", 0) or 0),
                    "k": int(s.get("strikeOuts", 0) or 0),
                    "bb": int(s.get("baseOnBalls", 0) or 0),
                    "hr": int(s.get("homeRuns", 0) or 0),
                    "whip": float(s.get("whip", 0) or 0),
                    "wins": int(s.get("wins", 0) or 0),
                    "losses": int(s.get("losses", 0) or 0)
                }
        return {}

    async def get_pitcher_gamelogs(self, pitcher_id: str,
                                   last_n: int = 5) -> List[Dict]:
        if not pitcher_id:
            return []
        url = f"{MLB_STATS_API}/people/{pitcher_id}/stats"
        params = {"stats": "gameLog", "group": "pitching",
                  "season": self.season}
        data = await self.get(url, params)
        logs = []
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                # As-of-date guard: drop game logs after as_of
                date_str = split.get("date", "")
                if date_str and date_str > self.as_of:
                    continue
                s = split.get("stat", {})
                logs.append({
                    "date": date_str,
                    "er": int(s.get("earnedRuns", 0) or 0),
                    "ip": float(s.get("inningsPitched", 0) or 0),
                    "hits": int(s.get("hits", 0) or 0),
                    "bb": int(s.get("baseOnBalls", 0) or 0),
                    "k": int(s.get("strikeOuts", 0) or 0),
                    "opponent": split.get("opponent", {}).get(
                        "abbreviation", "")
                })
        return list(reversed(logs))[:last_n]

    async def get_pitcher_splits(self, pitcher_id: str) -> Dict:
        if not pitcher_id:
            return {"vs_LHB": {}, "vs_RHB": {}}
        url = f"{MLB_STATS_API}/people/{pitcher_id}/stats"
        params = {"stats": "statSplits", "group": "pitching",
                  "season": self.season, "sitCodes": "vl,vr"}
        data = await self.get(url, params)
        splits = {"vs_LHB": {}, "vs_RHB": {}}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                code = split.get("split", {}).get("code", "")
                s = split.get("stat", {})
                payload = {
                    "ba": float(s.get("avg", 0) or 0),
                    "ops": float(s.get("ops", 0) or 0),
                    "era": float(s.get("era", 0) or 0)
                }
                if code == "vl":
                    splits["vs_LHB"] = payload
                elif code == "vr":
                    splits["vs_RHB"] = payload
        return splits

    async def get_team_offense_stats(self, team: str) -> Dict:
        """
        Team offensive stats — wRC+, K%, BB%, ISO, OPS, BA, OBP, SLG, HR, RPG.
        Delegates to FanGraphs (step 3); falls back to MLB Stats API basic
        line if FanGraphs is unavailable.
        """
        from data.fangraphs_client import FanGraphsClient
        fg = FanGraphsClient(as_of=self.as_of)
        result = await fg.get_team_batting(team, self.as_of)
        if result:
            return result

        # Fallback: MLB Stats API basic line (no wRC+, no K%/BB%).
        team_id = await self._get_team_id(team)
        if not team_id:
            return {}
        url = f"{MLB_STATS_API}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "hitting", "season": self.season}
        data = await self.get(url, params)
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                games = int(s.get("gamesPlayed", 1) or 1)
                runs = int(s.get("runs", 0) or 0)
                return {
                    "ba": float(s.get("avg", 0) or 0),
                    "obp": float(s.get("obp", 0) or 0),
                    "slg": float(s.get("slg", 0) or 0),
                    "ops": float(s.get("ops", 0) or 0),
                    "hr": int(s.get("homeRuns", 0) or 0),
                    "rpg": round(runs / games, 2) if games else 0,
                    "wrc_plus": 100,   # fallback only
                    "k_pct": 0.22,
                    "bb_pct": 0.08,
                }
        return {}

    async def get_confirmed_lineup(self, team: str) -> List[Dict]:
        """
        Confirmed batting order for `team` on self.as_of. Resolves the team's
        game on that date, fetches both lineups, returns the matching side.
        Empty list if game not found or lineup not yet posted (~90 min pre-game).
        """
        from data.lineup_fetcher import LineupFetcher
        games = await self.get_games_for_date(self.as_of)
        target = next((g for g in games
                       if g["home_team"] == team or g["away_team"] == team),
                      None)
        if not target:
            return []

        fetcher = LineupFetcher()
        lineups = await fetcher.get_lineups_for_game(target["game_id"])
        side = "home" if target["home_team"] == team else "away"
        return lineups.get(side, [])

    async def get_team_trends(self, team: str) -> Dict:
        """
        Team trends — overall record, last-10 record, home/road splits,
        run differential. Pulled from MLB standings endpoint. `over_pct` is
        not in standings; left at 0.50 (computing it would require iterating
        all gamelogs — deferred).
        """
        team_id = await self._get_team_id(team)
        if not team_id:
            return {}

        url = f"{MLB_STATS_API}/standings"
        params = {"leagueId": "103,104", "season": self.season}
        data = await self.get(url, params)
        target_record = None
        for league in data.get("records", []):
            for tr in league.get("teamRecords", []):
                if tr.get("team", {}).get("id") == team_id:
                    target_record = tr
                    break
            if target_record:
                break

        if not target_record:
            return {"over_pct": 0.50, "last_10_wins": 5}

        last_10_wins = 5
        last_10_losses = 5
        home_wins = away_wins = home_losses = away_losses = 0
        for sr in target_record.get("records", {}).get("splitRecords", []):
            t = sr.get("type")
            if t == "lastTen":
                last_10_wins = sr.get("wins", 5)
                last_10_losses = sr.get("losses", 5)
            elif t == "home":
                home_wins = sr.get("wins", 0)
                home_losses = sr.get("losses", 0)
            elif t == "away":
                away_wins = sr.get("wins", 0)
                away_losses = sr.get("losses", 0)

        runs_scored = int(target_record.get("runsScored", 0) or 0)
        runs_allowed = int(target_record.get("runsAllowed", 0) or 0)
        gp = (target_record.get("wins", 0) + target_record.get("losses", 0)) or 1

        return {
            "wins": target_record.get("wins"),
            "losses": target_record.get("losses"),
            "win_pct": float(target_record.get("winningPercentage", 0) or 0),
            "last_10_wins": last_10_wins,
            "last_10_losses": last_10_losses,
            "home_wins": home_wins,
            "home_losses": home_losses,
            "away_wins": away_wins,
            "away_losses": away_losses,
            "runs_scored": runs_scored,
            "runs_allowed": runs_allowed,
            "run_diff": runs_scored - runs_allowed,
            "run_diff_per_game": round((runs_scored - runs_allowed) / gp, 2),
            "streak": target_record.get("streak", {}).get("streakCode"),
            "over_pct": 0.50,  # would require gamelog iteration
        }

    async def get_recent_series(self, home: str, away: str,
                                date: str) -> List[Dict]:
        """Most recent completed series between these two teams."""
        matchups = await self.get_last_n_matchups(home, away, n=10)
        if not matchups:
            return []
        # Group consecutive same-day-or-next-day games as a series
        matchups.sort(key=lambda g: g.get("date", ""))
        # Return the last 3 games (typical series length)
        return matchups[-3:]

    async def get_last_n_matchups(self, home: str, away: str,
                                  n: int = 10) -> List[Dict]:
        """
        Last N completed head-to-head games, most recent first.
        Looks back across the current and prior seasons to fill n if needed.
        """
        home_id = await self._get_team_id(home)
        away_id = await self._get_team_id(away)
        if not home_id or not away_id:
            return []

        # Search back from as_of through prior season for completed H2H games.
        seasons = [self.season, self.season - 1]
        results: List[Dict] = []
        for season in seasons:
            url = f"{MLB_STATS_API}/schedule"
            params = {
                "sportId": 1,
                "teamId": home_id,
                "opponentId": away_id,
                "startDate": f"{season}-03-01",
                "endDate": f"{season}-11-30",
            }
            data = await self.get(url, params)
            for date_data in data.get("dates", []):
                for g in date_data.get("games", []):
                    if g.get("status", {}).get("detailedState") != "Final":
                        continue
                    teams = g.get("teams", {})
                    h_team = teams.get("home", {})
                    a_team = teams.get("away", {})
                    g_date = g.get("officialDate") or g.get("gameDate", "")[:10]
                    if g_date and g_date > self.as_of:
                        continue
                    results.append({
                        "date": g_date,
                        "home_team": ID_TO_ABBR.get(
                            h_team.get("team", {}).get("id"), ""),
                        "away_team": ID_TO_ABBR.get(
                            a_team.get("team", {}).get("id"), ""),
                        "home_score": h_team.get("score", 0),
                        "away_score": a_team.get("score", 0),
                        "total_runs": (h_team.get("score", 0)
                                       + a_team.get("score", 0)),
                        "winner": (
                            "home" if h_team.get("isWinner")
                            else "away" if a_team.get("isWinner")
                            else None),
                    })
            if len(results) >= n:
                break

        results.sort(key=lambda g: g.get("date", ""), reverse=True)
        return results[:n]

    async def get_bvp_data(self, team: str, pitcher_id: str) -> Dict:
        """
        Batter-vs-pitcher career splits for `team`'s lineup against
        `pitcher_id`. Returns {batter_name: {ab, h, hr, ba, ops}, ...}.
        Requires the lineup to be confirmed; empty dict otherwise.
        """
        if not pitcher_id or not team:
            return {}

        lineup = await self.get_confirmed_lineup(team)
        if not lineup:
            return {}

        async def fetch_one(batter):
            bid = batter.get("player_id")
            if not bid:
                return None
            url = f"{MLB_STATS_API}/people/{bid}/stats"
            params = {
                "stats": "vsPlayer",
                "group": "hitting",
                "opposingPlayerId": pitcher_id,
                "season": self.season,
                # 'sportId' isn't required; the API filters by player anyway
            }
            data = await self.get(url, params)
            best = None
            for sg in data.get("stats", []):
                for sp in sg.get("splits", []):
                    s = sp.get("stat", {})
                    ab = int(s.get("atBats", 0) or 0)
                    if best is None or ab > best.get("ab", 0):
                        best = {
                            "ab": ab,
                            "h": int(s.get("hits", 0) or 0),
                            "hr": int(s.get("homeRuns", 0) or 0),
                            "bb": int(s.get("baseOnBalls", 0) or 0),
                            "k": int(s.get("strikeOuts", 0) or 0),
                            "ba": float(s.get("avg", 0) or 0),
                            "ops": float(s.get("ops", 0) or 0),
                        }
            return (batter.get("name"), best) if best else None

        results = await asyncio.gather(
            *[fetch_one(b) for b in lineup], return_exceptions=True)
        bvp = {}
        for r in results:
            if isinstance(r, tuple) and r[1] and r[1]["ab"] > 0:
                bvp[r[0]] = r[1]
        return bvp

    async def get_injury_list(self, team: str) -> List[Dict]:
        team_id = await self._get_team_id(team)
        if not team_id:
            return []
        url = f"{MLB_STATS_API}/teams/{team_id}/roster"
        params = {"rosterType": "injuries"}
        data = await self.get(url, params)
        return [{
            "name": p.get("person", {}).get("fullName", ""),
            "position": p.get("position", {}).get("abbreviation", ""),
            "status": p.get("status", {}).get("description", ""),
            "il_type": "IL"
        } for p in data.get("roster", [])]

    async def get_day_to_day(self, team: str) -> List[Dict]:
        return []

    async def get_odds(self, home: str, away: str, date: str) -> Dict:
        """
        Current market odds. Backtest: returns {} (orchestrator skips edge/Kelly).
        Live: delegates to OddsScraper (Action Network MLB scoreboard,
        prefers DK > FD > BetMGM).
        """
        if self.mode == "backtest":
            return {}
        from data.odds_scraper import OddsScraper
        scraper = OddsScraper()
        result = await scraper.get_current_lines(home, away, date)
        if result:
            logger.info(f"Odds {away}@{home} {date}: ML {result.get('home_ml_odds')}/"
                        f"{result.get('away_ml_odds')} total {result.get('total_line')} "
                        f"book={result.get('book_id')}")
        return result

    async def get_final_scores(self, date: str) -> List[Dict]:
        url = f"{MLB_STATS_API}/schedule"
        params = {"sportId": 1, "date": date, "hydrate": "linescore"}
        data = await self.get(url, params)
        scores = []
        for date_data in data.get("dates", []):
            for game in date_data.get("games", []):
                if game.get("status", {}).get("detailedState") == "Final":
                    teams = game.get("teams", {})
                    scores.append({
                        "game_pk": str(game.get("gamePk")),
                        "home_team": teams.get("home", {}).get(
                            "team", {}).get("abbreviation"),
                        "away_team": teams.get("away", {}).get(
                            "team", {}).get("abbreviation"),
                        "home_score": teams.get("home", {}).get("score", 0),
                        "away_score": teams.get("away", {}).get("score", 0)
                    })
        return scores

    async def get_bullpen_stats(self, team: str) -> Dict:
        """
        Team bullpen stats — ERA, FIP, xFIP, K/9, BB/9, HR/9, IP, SV, BS, HLD.
        Delegates to FanGraphs (step 3) with a v2.5-shaped fallback dict.
        `last_7_era` and `high_usage_last_2` aren't in the FG team-level feed
        and are returned as `None` — agent handles None.
        """
        from data.fangraphs_client import FanGraphsClient
        fg = FanGraphsClient(as_of=self.as_of)
        bp = await fg.get_bullpen(team, self.as_of)
        if not bp:
            return {"season_era": 4.20, "last_7_era": 4.20, "xfip": 4.20,
                    "k9": 9.0, "bb9": 3.5, "hr9": 1.2, "save_pct": 0.70,
                    "holds": 10, "blown_saves": 3, "high_usage_last_2": False}

        sv = bp.get("sv") or 0
        bs = bp.get("bs") or 0
        save_pct = sv / (sv + bs) if (sv + bs) > 0 else 0.70
        season_era = bp.get("season_era")
        return {
            "season_era": season_era,
            # FG team-level feed has no rolling 7-day. Use season ERA as a
            # proxy so downstream tier classification (which compares numeric
            # last_7_era to xfip) doesn't blow up. A real rolling window
            # requires per-game gamelog aggregation — deferred.
            "last_7_era": season_era,
            "xfip": bp.get("xfip"),
            "fip": bp.get("fip"),
            "k9": bp.get("k9"),
            "bb9": bp.get("bb9"),
            "hr9": bp.get("hr9"),
            "save_pct": save_pct,
            "holds": bp.get("hld") or 0,
            "blown_saves": bp.get("bs") or 0,
            "ip": bp.get("ip"),
            "war": bp.get("war"),
            "high_usage_last_2": False,  # requires per-game tracking
        }

    async def get_reliever_list(self, team: str) -> List:
        return []

    async def get_closer_availability(self, team: str) -> Dict:
        return {"closer_available": True, "unavailable_arms": []}

    async def get_hp_umpire(self, home: str, away: str,
                            date: str) -> Optional[str]:
        """
        HP umpire name (or None if not yet assigned). Sourced from MLB Stats
        API game feed officials. None triggers league-average fallback in
        the umpire_agent.
        """
        from data.umpscores_client import UmpScoresClient
        client = UmpScoresClient(as_of=self.as_of)
        return await client.get_hp_umpire(home, away, date)

    async def get_opening_lines(self, home: str, away: str,
                                date: str) -> Dict:
        if self.mode == "backtest":
            return {}
        from data.odds_scraper import OddsScraper
        return await OddsScraper().get_opening_lines(home, away, date)

    async def get_current_lines(self, home: str, away: str,
                                date: str) -> Dict:
        if self.mode == "backtest":
            return {}
        from data.odds_scraper import OddsScraper
        return await OddsScraper().get_current_lines(home, away, date)

    async def get_public_betting_pcts(self, home: str, away: str,
                                      date: str) -> Dict:
        if self.mode == "backtest":
            return {"home_ml_pct": 0.50, "away_ml_pct": 0.50,
                    "over_pct": 0.50, "under_pct": 0.50}
        from data.odds_scraper import OddsScraper
        result = await OddsScraper().get_public_pcts(home, away, date)
        return result or {"home_ml_pct": 0.50, "away_ml_pct": 0.50,
                          "over_pct": 0.50, "under_pct": 0.50}

    async def get_weather(self, team: str, date: str) -> Dict:
        """
        Weather forecast for the home team's stadium at game time. Indoor
        venues short-circuit to constants. NWS is US-only — TOR is treated
        as indoor for fallback purposes.
        """
        from data.nws_client import NWSClient
        nws = NWSClient()
        # We don't have the precise game start time here — pass the date and
        # let NWSClient pick the nearest 19:00-ish hourly period.
        game_dt = f"{date}T19:00:00"
        return await nws.get_forecast(team, game_dt)

    async def _get_team_id(self, abbreviation: str) -> Optional[int]:
        return TEAM_IDS.get(abbreviation)


# Module-level so _parse_game can do the reverse lookup. The MLB Stats API
# /schedule endpoint returns team {id, name, link} but no abbreviation, so we
# map id → abbr ourselves.
TEAM_IDS = {
    "ATL": 144, "MIA": 146, "NYM": 121, "PHI": 143, "WSH": 120,
    "CHC": 112, "CIN": 113, "MIL": 158, "PIT": 134, "STL": 138,
    "AZ": 109, "COL": 115, "LAD": 119, "SD": 135, "SF": 137,
    "BAL": 110, "BOS": 111, "NYY": 147, "TB": 139, "TOR": 141,
    "CWS": 145, "CLE": 114, "DET": 116, "KC": 118, "MIN": 142,
    "HOU": 117, "LAA": 108, "ATH": 133, "OAK": 133,
    "SEA": 136, "TEX": 140,
}
ID_TO_ABBR = {v: k for k, v in TEAM_IDS.items() if k != "OAK"}
ID_TO_ABBR[133] = "ATH"  # OAK and ATH share id 133; canonical is ATH for 2026


class SavantAPI:
    """
    Baseball Savant Statcast data.

    Step 2: delegates to data/statcast_leaderboard.py which fetches Savant's
    custom pitcher leaderboard CSV (year-keyed, cached on disk + in memory).
    Replaces v2.5's broken statcast_search/csv parser that declared
    k_count/bb_count/pa_count counters but never incremented them.

    Inherits as_of_date / mode from data.mlb_api module-level globals
    (set by V25Orchestrator before agents run).
    """

    def __init__(self):
        from data.statcast_leaderboard import StatcastLeaderboard
        self._leaderboard = StatcastLeaderboard(as_of=_GLOBAL_AS_OF or "")

    @classmethod
    async def close(cls) -> None:
        from data.statcast_leaderboard import StatcastLeaderboard
        await StatcastLeaderboard.close()

    async def get_pitcher_statcast(self, pitcher_id: str) -> Dict:
        """Returns {xwoba, exit_velocity, hard_hit_pct, barrel_pct, k_pct, bb_pct, whiff_pct}."""
        as_of = _GLOBAL_AS_OF or _today_iso()
        return await self._leaderboard.get_pitcher(pitcher_id, as_of)

    async def get_pitcher_arsenal(self, pitcher_id: str) -> Dict:
        """
        Pitch-type arsenal (xwOBA per pitch type vs LHB/RHB).
        Was missing entirely from v2.5 SavantAPI. Step 2 stubs to {} — Savant
        pitch-arsenal endpoint requires a different fetch path than the
        leaderboard, deferred until pitch_arsenal_agent's WP impact justifies
        the work. Agent already handles {} gracefully.
        """
        return {}
