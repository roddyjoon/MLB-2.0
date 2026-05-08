"""
Agent 10: Sharp Line Movement Tracker
Opening vs current line movement, sharp money signals,
Kelly size adjustments based on market confirmation
"""

import asyncio
from typing import Dict, List, Optional
from data.mlb_api import MLBDataAPI
from core.logger import ModelLogger

logger = ModelLogger("line_movement_agent")

# Sharp movement thresholds
SHARP_ML_THRESHOLD = 10    # 10+ cent ML move = sharp signal
SHARP_TOTAL_THRESHOLD = 0.5  # 0.5+ run line move = sharp signal


class LineMovementAgent:
    """
    Tracks opening vs current line movement
    
    KEY RULE: Sharp movement CONFIRMS or WARNS model edge
    It NEVER changes the WP calculation — only adjusts Kelly sizing
    
    Model edge ≥ 6% + sharp movement SAME direction = increase Kelly 25%
    Model edge ≥ 6% + sharp movement OPPOSITE direction = reduce Kelly 25%
    Model edge ≥ 6% + no movement = standard Kelly
    """

    def __init__(self):
        self.mlb_api = MLBDataAPI()

    async def analyze(self, game_data: Dict) -> Dict:
        """Pull opening and current lines, detect sharp movement"""

        home = game_data.get("home_team")
        away = game_data.get("away_team")
        date = game_data.get("date")

        # Pull opening and current lines
        opening = await self.mlb_api.get_opening_lines(home, away, date)
        current = await self.mlb_api.get_current_lines(home, away, date)

        if not opening or not current:
            return {"available": False, "kelly_adj": 1.0}

        # Detect movement
        ml_movement = self._detect_ml_movement(opening, current)
        total_movement = self._detect_total_movement(opening, current)

        # Public betting percentages
        public = await self.mlb_api.get_public_betting_pcts(home, away, date)

        # Reverse line movement detection
        rlm = self._detect_reverse_line_movement(ml_movement, public)

        return {
            "available": True,
            "opening": opening,
            "current": current,
            "ml_movement": ml_movement,
            "total_movement": total_movement,
            "public_pcts": public,
            "reverse_line_movement": rlm,
            "sharp_signals": self._compile_sharp_signals(
                ml_movement, total_movement, rlm),
            "kelly_adjustments": self._calculate_kelly_adj(
                ml_movement, total_movement, rlm)
        }

    def _detect_ml_movement(self, opening: Dict, current: Dict) -> Dict:
        """Detect ML line movement"""
        open_home = opening.get("home_ml", 0)
        curr_home = current.get("home_ml", 0)
        open_away = opening.get("away_ml", 0)
        curr_away = current.get("away_ml", 0)

        if not open_home or not curr_home:
            return {"movement": 0, "direction": "none", "sharp": False}

        # For minus-money: moving from -120 to -135 = 15 cents toward home
        home_movement = self._calc_movement(open_home, curr_home)

        return {
            "home_open": open_home,
            "home_current": curr_home,
            "away_open": open_away,
            "away_current": curr_away,
            "home_movement_cents": home_movement,
            "direction": "home" if home_movement > 0 else
                        "away" if home_movement < 0 else "none",
            "sharp": abs(home_movement) >= SHARP_ML_THRESHOLD,
            "magnitude": abs(home_movement)
        }

    def _detect_total_movement(self, opening: Dict, current: Dict) -> Dict:
        """Detect total line movement"""
        open_total = opening.get("total_line", 0)
        curr_total = current.get("total_line", 0)
        open_over = opening.get("over_odds", -110)
        curr_over = current.get("over_odds", -110)

        if not open_total or not curr_total:
            return {"movement": 0, "direction": "none", "sharp": False}

        line_move = curr_total - open_total
        juice_move = self._calc_movement(open_over, curr_over)

        # Line moved Up = Over signal, Down = Under signal
        # Juice moved toward Over = Over sharp money
        direction = (
            "over" if line_move >= SHARP_TOTAL_THRESHOLD else
            "under" if line_move <= -SHARP_TOTAL_THRESHOLD else
            "over" if juice_move <= -SHARP_ML_THRESHOLD else
            "under" if juice_move >= SHARP_ML_THRESHOLD else
            "none"
        )

        return {
            "open_total": open_total,
            "current_total": curr_total,
            "line_move": round(line_move, 1),
            "juice_move": juice_move,
            "direction": direction,
            "sharp": (abs(line_move) >= SHARP_TOTAL_THRESHOLD or
                      abs(juice_move) >= SHARP_ML_THRESHOLD),
            "magnitude": abs(line_move)
        }

    def _detect_reverse_line_movement(self, ml_movement: Dict,
                                       public: Dict) -> Dict:
        """
        Reverse line movement: line moves AGAINST public money
        = Sharp money on minority side
        
        Example: 70% public on home team but line moves toward away
        = Sharp money on away team despite public backing home
        """
        if not public:
            return {"detected": False}

        home_public_pct = public.get("home_ml_pct", 0.50)
        ml_direction = ml_movement.get("direction", "none")

        # RLM: Public on home but line moves toward away
        rlm_away = (home_public_pct >= 0.60 and ml_direction == "away")
        # RLM: Public on away but line moves toward home
        rlm_home = (home_public_pct <= 0.40 and ml_direction == "home")

        return {
            "detected": rlm_away or rlm_home,
            "sharp_side": "away" if rlm_away else "home" if rlm_home else None,
            "public_pct_home": home_public_pct,
            "ml_direction": ml_direction,
            "note": (
                f"⚡ RLM: {int(home_public_pct*100)}% public on home "
                f"but line moving {ml_direction}" if rlm_away or rlm_home
                else "No RLM detected"
            )
        }

    def _calc_movement(self, opening: int, current: int) -> int:
        """
        Calculate cents of movement
        Moving from -120 to -130 = -10 cents (toward favorite)
        Moving from -120 to -110 = +10 cents (toward underdog)
        """
        if not opening or not current:
            return 0

        # Convert to implied probability for comparison
        def to_prob(ml):
            if ml > 0:
                return 100 / (100 + ml)
            return abs(ml) / (abs(ml) + 100)

        prob_diff = to_prob(current) - to_prob(opening)
        # Convert to approximate cents
        return int(prob_diff * 200)

    def _compile_sharp_signals(self, ml_move: Dict,
                                total_move: Dict, rlm: Dict) -> List:
        """Compile all sharp signals"""
        signals = []

        if ml_move.get("sharp"):
            direction = ml_move.get("direction")
            mag = ml_move.get("magnitude", 0)
            signals.append({
                "type": "ml_movement",
                "direction": direction,
                "magnitude": mag,
                "description": f"ML moved {mag} cents toward {direction}"
            })

        if total_move.get("sharp"):
            direction = total_move.get("direction")
            signals.append({
                "type": "total_movement",
                "direction": direction,
                "line_move": total_move.get("line_move"),
                "description": f"Total moved {total_move.get('line_move'):+.1f} ({direction})"
            })

        if rlm.get("detected"):
            signals.append({
                "type": "reverse_line_movement",
                "direction": rlm.get("sharp_side"),
                "description": rlm.get("note")
            })

        return signals

    def _calculate_kelly_adj(self, ml_move: Dict,
                              total_move: Dict, rlm: Dict) -> Dict:
        """
        Calculate Kelly size adjustments
        
        CRITICAL: These adjustments apply AFTER WP and edge are calculated
        They modify sizing only — never the WP calculation itself
        """
        home_ml_adj = 1.0
        away_ml_adj = 1.0
        over_adj = 1.0
        under_adj = 1.0

        # ML adjustments from sharp movement
        if ml_move.get("sharp"):
            direction = ml_move.get("direction")
            mag = ml_move.get("magnitude", 0)
            boost = 1.25 if mag >= 20 else 1.15

            if direction == "home":
                home_ml_adj *= boost
                away_ml_adj *= 0.85  # Reduce opposite side
            elif direction == "away":
                away_ml_adj *= boost
                home_ml_adj *= 0.85

        # Total adjustments from sharp movement
        if total_move.get("sharp"):
            direction = total_move.get("direction")
            if direction == "over":
                over_adj *= 1.20
                under_adj *= 0.85
            elif direction == "under":
                under_adj *= 1.20
                over_adj *= 0.85

        # RLM provides strongest confirmation signal
        if rlm.get("detected"):
            sharp_side = rlm.get("sharp_side")
            if sharp_side == "home":
                home_ml_adj *= 1.30
            elif sharp_side == "away":
                away_ml_adj *= 1.30

        return {
            "home_ml_adj": round(min(home_ml_adj, 1.50), 2),
            "away_ml_adj": round(min(away_ml_adj, 1.50), 2),
            "over_adj": round(min(over_adj, 1.50), 2),
            "under_adj": round(min(under_adj, 1.50), 2),
            "note": ("Adjustments are sizing only — WP unchanged" if any([
                home_ml_adj != 1.0, away_ml_adj != 1.0,
                over_adj != 1.0, under_adj != 1.0
            ]) else "No sizing adjustments")
        }


# Fix missing List import
from typing import List
