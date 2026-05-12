"""
Edge Calculator — Market enters HERE and ONLY HERE
Corrected v2.4 edge formula: Edge = Model WP% − Market Implied %
"""

import math
from typing import Dict
from core.logger import ModelLogger

logger = ModelLogger("edge_calculator")


class EdgeCalculator:
    """
    IRON RULE: Market odds NEVER influence WP calculation
    WP is calculated purely from SP metrics, lineup, park, weather, behavioral data
    Market enters ONLY at this step
    """

    def calculate_all(self, wp_result: Dict, market: Dict) -> Dict:
        """Calculate edges for all markets"""

        home_wp = wp_result.get("home_wp", 0.50)
        away_wp = wp_result.get("away_wp", 0.50)
        total_proj = wp_result.get("total_projection", 8.5)

        home_ml = market.get("home_ml_odds", -110)
        away_ml = market.get("away_ml_odds", +110)
        total_line = market.get("total_line", 8.5)
        over_odds = market.get("over_odds", -110)
        under_odds = market.get("under_odds", -110)

        # Convert odds to implied probability
        home_implied = self._odds_to_implied(home_ml)
        away_implied = self._odds_to_implied(away_ml)
        over_implied = self._odds_to_implied(over_odds)
        under_implied = self._odds_to_implied(under_odds)

        # Calculate Over/Under WP from projection vs line
        over_wp = self._calc_total_wp(total_proj, total_line, "over")
        under_wp = self._calc_total_wp(total_proj, total_line, "under")

        # CORRECTED EDGE FORMULA (v2.4 Iron Rule)
        # Edge = Model WP% - Market Implied%
        home_ml_edge = home_wp - home_implied
        away_ml_edge = away_wp - away_implied
        over_edge = over_wp - over_implied
        under_edge = under_wp - under_implied

        edges = {
            "home_ml_edge": round(home_ml_edge, 4),
            "away_ml_edge": round(away_ml_edge, 4),
            "over_edge": round(over_edge, 4),
            "under_edge": round(under_edge, 4),
            "home_implied": round(home_implied, 4),
            "away_implied": round(away_implied, 4),
            "over_implied": round(over_implied, 4),
            "under_implied": round(under_implied, 4),
            "over_wp": round(over_wp, 4),
            "under_wp": round(under_wp, 4),
            "total_projection": total_proj,
            "total_line": total_line,
            "total_cushion": round(total_proj - total_line, 2)
        }

        # Log any large negative EV plays (market traps to avoid)
        for market_name, edge in [
            ("Home ML", home_ml_edge),
            ("Away ML", away_ml_edge),
            ("Over", over_edge),
            ("Under", under_edge)
        ]:
            if edge <= -0.08:
                logger.warning(f"⚠️  Deeply negative EV: {market_name} = {edge*100:.1f}%")
            elif edge >= 0.10:
                logger.info(f"🔥 Strong edge: {market_name} = {edge*100:.1f}%")

        return edges

    def _odds_to_implied(self, odds: int) -> float:
        """
        Convert American odds to implied probability
        
        CORRECTED FORMULA (v2.4 Iron Rule):
        Plus-money dog: Implied = 100 / (100 + dog_odds)
        Minus-money fav: Implied = |fav_odds| / (|fav_odds| + 100)
        """
        if odds is None:
            return 0.50

        if odds > 0:
            # Plus money (underdog)
            return 100 / (100 + odds)
        else:
            # Minus money (favorite)
            return abs(odds) / (abs(odds) + 100)

    def _calc_total_wp(self, projection: float, line: float,
                        direction: str, std_dev: float = 3.0) -> float:
        """
        Calculate Over/Under probability using normal distribution.

        std_dev=3.0 reflects observed MLB game-total dispersion (was 1.8,
        which produced over-confident 80% probabilities at modest gaps).
        Cap at 0.65 reflects empirical realization: the 6-day grading sample
        showed model-implied 80% under-WPs realized as 61% — model is
        directionally right but magnitude-overconfident. Capping at 0.65
        prevents Kelly from over-betting on inflated probabilities.

        WP(Over line X) = 1 - CDF(X, projection, std_dev)
        WP(Under line X) = CDF(X, projection, std_dev)
        """

        # Z-score: how many std devs is the line from projection
        z = (line - projection) / std_dev

        # Standard normal CDF approximation
        cdf = self._normal_cdf(z)

        if direction == "over":
            return max(0.35, min(0.65, 1.0 - cdf))
        else:  # under
            return max(0.35, min(0.65, cdf))

    def _normal_cdf(self, z: float) -> float:
        """Standard normal CDF approximation"""
        # Abramowitz and Stegun approximation
        if z < -6:
            return 0.0
        if z > 6:
            return 1.0

        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 +
                    t * (-0.356563782 +
                    t * (1.781477937 +
                    t * (-1.821255978 +
                    t * 1.330274429))))

        pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf_pos = 1.0 - pdf * poly

        return cdf_pos if z >= 0 else 1.0 - cdf_pos

    def get_fair_moneyline(self, wp: float) -> int:
        """Convert WP to fair American moneyline"""
        if wp >= 0.50:
            # Favorite
            return -round((wp / (1 - wp)) * 100)
        else:
            # Underdog
            return round(((1 - wp) / wp) * 100)


class KellySizer:
    """Kelly criterion bet sizing with $220 cap"""

    MAX_BET = 220  # Hard cap per v2.4 rules
    BANKROLL = 10000  # Standard bankroll assumption
    KELLY_FRACTION = 0.25  # Quarter-Kelly for variance reduction

    def calculate(self, wp: float, odds: int, bet_type: str) -> int:
        """
        Calculate Kelly bet size
        Returns dollar amount (capped at $220)
        """
        if wp <= 0 or wp >= 1:
            return 0

        # Convert odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        b = decimal_odds - 1  # Net odds
        p = wp
        q = 1 - wp

        # Kelly fraction
        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0  # Negative EV — no bet

        # Apply quarter-Kelly
        fractional_kelly = kelly * self.KELLY_FRACTION

        # Convert to dollar amount
        bet_size = fractional_kelly * self.BANKROLL

        # Round to nearest $5
        bet_size = round(bet_size / 5) * 5

        # Apply $220 cap
        return min(int(bet_size), self.MAX_BET)
