"""
SP Change Monitor — Runs every 30 minutes, catches SP changes before betting
This is the most important real-time feature of the system
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("scheduler")


class DailyScheduler:
    """
    Monitors SP changes every 30 minutes
    Sends alerts when confirmed starter differs from expected
    """

    CHECK_INTERVAL = 1800  # 30 minutes in seconds

    def __init__(self):
        self.mlb_api = MLBDataAPI()
        self.known_starters = {}  # Cache of last known starters
        self.alerts_sent = set()  # Avoid duplicate alerts

    async def monitor_sp_changes(self):
        """Run continuous SP change monitoring"""
        logger.info("SP Monitor started — checking every 30 minutes")

        while True:
            try:
                await self._check_sp_changes()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_sp_changes(self):
        """Check for SP changes for today's games"""
        today = datetime.now().strftime("%Y-%m-%d")
        games = await self.mlb_api.get_games_for_date(today)

        changes_detected = []

        for game in games:
            game_id = game.get("game_id")
            home_sp = game.get("home_sp_name", "")
            away_sp = game.get("away_sp_name", "")

            # Compare with cached starters
            if game_id in self.known_starters:
                prev = self.known_starters[game_id]

                # Check for changes
                if (prev["home_sp"] and home_sp and
                        prev["home_sp"].lower() != home_sp.lower()):
                    change = {
                        "game": f"{game['away_team']} @ {game['home_team']}",
                        "side": "HOME",
                        "from": prev["home_sp"],
                        "to": home_sp,
                        "time": datetime.now().isoformat()
                    }
                    changes_detected.append(change)

                if (prev["away_sp"] and away_sp and
                        prev["away_sp"].lower() != away_sp.lower()):
                    change = {
                        "game": f"{game['away_team']} @ {game['home_team']}",
                        "side": "AWAY",
                        "from": prev["away_sp"],
                        "to": away_sp,
                        "time": datetime.now().isoformat()
                    }
                    changes_detected.append(change)

            # Update cache
            self.known_starters[game_id] = {
                "home_sp": home_sp,
                "away_sp": away_sp,
                "updated": datetime.now().isoformat()
            }

        # Alert on changes
        for change in changes_detected:
            alert_key = f"{change['game']}_{change['side']}_{change['to']}"
            if alert_key not in self.alerts_sent:
                await self._send_alert(change)
                self.alerts_sent.add(alert_key)

        if not changes_detected:
            logger.debug(f"SP check complete — no changes detected ({len(games)} games)")

    async def _send_alert(self, change: Dict):
        """Send SP change alert"""
        msg = (f"\n🚨 SP CHANGE DETECTED 🚨\n"
               f"Game: {change['game']}\n"
               f"Side: {change['side']}\n"
               f"Was:  {change['from']}\n"
               f"Now:  {change['to']}\n"
               f"Time: {change['time']}\n"
               f"ACTION: Model recalculation required!\n")

        logger.warning(msg)
        print(msg)

        # Save to alerts file
        alerts_file = Path("outputs/sp_alerts.json")
        alerts = []
        if alerts_file.exists():
            with open(alerts_file) as f:
                alerts = json.load(f)

        alerts.append(change)

        with open(alerts_file, "w") as f:
            json.dump(alerts, f, indent=2)

    async def run_morning_setup(self):
        """
        Run morning setup — pull day's games, probable starters,
        generate initial card before lines move
        """
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Morning setup — {today}")

        games = await self.mlb_api.get_games_for_date(today)

        setup_data = {
            "date": today,
            "games": len(games),
            "probable_starters": {},
            "setup_time": datetime.now().isoformat()
        }

        for game in games:
            game_id = game.get("game_id")
            setup_data["probable_starters"][game_id] = {
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "away_sp": game.get("away_sp_name", "TBD"),
                "home_sp": game.get("home_sp_name", "TBD"),
                "game_time": game.get("game_time", "")
            }
            # Initialize cache
            self.known_starters[game_id] = {
                "home_sp": game.get("home_sp_name", ""),
                "away_sp": game.get("away_sp_name", ""),
                "updated": datetime.now().isoformat()
            }

        # Save morning setup
        setup_file = Path(f"outputs/morning_setup_{today}.json")
        with open(setup_file, "w") as f:
            json.dump(setup_data, f, indent=2)

        logger.info(f"Morning setup complete — {len(games)} games loaded")
        return setup_data
