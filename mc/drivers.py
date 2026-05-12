"""
Top-3 drivers per game — quantify which inputs are pushing the WP most.

We don't run ablation (counterfactual MC per driver) because that's 6×
runtime per game. Instead, we score each input by an analytic
"WP-equivalent impact" and rank by absolute magnitude.

The mapping from each driver to WP-impact units is calibrated against the
empirical sensitivities in `wp_formula_v25` — they're rough but consistent.
"""

from typing import Dict, List


def _signed_pct(x: float) -> str:
    """Format a WP-impact magnitude like '+5.2%' / '-2.8%'."""
    return f"{'+' if x >= 0 else ''}{x*100:.1f}%"


def compute_drivers(inputs: Dict, sim_result: Dict,
                     extras: Dict = None) -> List[Dict]:
    """
    Rank inputs by absolute WP-impact and return top 3.

    `inputs` = mc.simulator inputs dict
    `sim_result` = mc.simulator output (used for projected totals/context)
    `extras` = optional context: SP names, wRC+, weather, park, bvp lists

    Returns list of {name, impact_signed_pct, explanation} sorted by |impact|.
    """
    extras = extras or {}
    drivers = []

    # 1. SP gap — better away SP = home offense suppressed (worse for home)
    sp_gap = inputs["home_sp_rate"] - inputs["away_sp_rate"]
    # 1 ERA point ≈ 5% home_wp swing
    sp_impact = -sp_gap * 0.05
    home_sp_name = extras.get("home_sp_name", "?")
    away_sp_name = extras.get("away_sp_name", "?")
    drivers.append({
        "name": "SP matchup",
        "impact": sp_impact,
        "explanation": (
            f"{home_sp_name} {inputs['home_sp_rate']:.2f} "
            f"vs {away_sp_name} {inputs['away_sp_rate']:.2f} "
            f"({'home edge' if sp_impact > 0 else 'away edge'} "
            f"{abs(sp_gap):.2f} runs)"
        ),
    })

    # 2. wRC+ / RPG gap
    home_rpg = inputs["home_team_rpg"]
    away_rpg = inputs["away_team_rpg"]
    rpg_gap = home_rpg - away_rpg
    rpg_impact = rpg_gap * 0.04  # 1 RPG ≈ 4% home_wp swing
    home_wrc = extras.get("home_wrc_plus")
    away_wrc = extras.get("away_wrc_plus")
    expl = f"home RPG {home_rpg:.2f} vs away {away_rpg:.2f}"
    if home_wrc is not None and away_wrc is not None:
        expl = (f"home wRC+ {home_wrc:.0f} (RPG {home_rpg:.2f}) vs "
                f"away wRC+ {away_wrc:.0f} (RPG {away_rpg:.2f})")
    drivers.append({
        "name": "Lineup strength",
        "impact": rpg_impact,
        "explanation": expl,
    })

    # 3. Park factor (deviation from 1.00 = scoring tilt; doesn't directly
    # favor a side but shifts variance + totals)
    park = inputs["park_runs_factor"]
    park_dev = park - 1.00
    # Park doesn't favor home/away directionally for WP per se — it shifts
    # the total. Encode as "total environment" with sign indicating
    # over/under tilt.
    park_impact = abs(park_dev) * 0.5  # encode as magnitude only for ranking
    park_name = extras.get("park_name", "park")
    drivers.append({
        "name": "Park environment",
        "impact": park_impact if park_dev > 0 else -park_impact,
        "explanation": (
            f"{park_name}: runs factor {park:.2f} "
            f"({'+' if park_dev > 0 else ''}{park_dev*100:.0f}% vs league avg)"
        ),
    })

    # 4. Bullpen gap (matters for innings 7-9 ≈ 1/3 of game)
    bp_gap = inputs["home_bp_rate"] - inputs["away_bp_rate"]
    bp_impact = -bp_gap * 0.02  # 1 ERA point of BP gap ≈ 2% WP swing
    drivers.append({
        "name": "Bullpen depth",
        "impact": bp_impact,
        "explanation": (
            f"home BP {inputs['home_bp_rate']:.2f} ERA "
            f"vs away BP {inputs['away_bp_rate']:.2f}"
        ),
    })

    # 5. Weather (only matters in outdoor parks)
    wx_adj = inputs["weather_run_adj"]
    wx_impact = abs(wx_adj) * 0.02  # rough; weather mostly affects totals
    weather = extras.get("weather", {})
    if weather.get("indoor"):
        wx_expl = "indoor — no weather impact"
        wx_impact = 0
    else:
        temp = weather.get("temperature", 70)
        wind = weather.get("wind_speed", 0)
        wind_dir = weather.get("wind_direction", "?")
        wx_expl = (f"{temp}°F, {wind} mph {wind_dir} "
                   f"({_signed_pct(wx_adj * 0.02)} run env)")
    drivers.append({
        "name": "Weather",
        "impact": wx_impact if wx_adj > 0 else -wx_impact,
        "explanation": wx_expl,
    })

    # 6. BvP edge (home offense vs away SP elite threats)
    home_bvp = inputs.get("home_bvp_elite_count", 0)
    away_bvp = inputs.get("away_bvp_elite_count", 0)
    bvp_gap = home_bvp - away_bvp
    bvp_impact = bvp_gap * 0.005  # each elite threat ≈ 0.5% WP
    if home_bvp + away_bvp > 0:
        drivers.append({
            "name": "BvP history",
            "impact": bvp_impact,
            "explanation": (
                f"{home_bvp} elite home threat(s) vs {away_sp_name}, "
                f"{away_bvp} vs {home_sp_name}"
            ),
        })

    # Sort by absolute impact, return top 3 non-zero
    drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
    top = [d for d in drivers if abs(d["impact"]) > 0.001][:3]
    return [{
        "name": d["name"],
        "impact_pct": _signed_pct(d["impact"]),
        "explanation": d["explanation"],
    } for d in top]
