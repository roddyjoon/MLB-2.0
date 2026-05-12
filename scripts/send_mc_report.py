#!/usr/bin/env python3
"""
Daily Monte Carlo win-probability report.

For every game on the schedule:
  1. Run V25Orchestrator (live mode) to gather all agent outputs
  2. Extract MC inputs (offense, SP, BP, park, weather, BvP)
  3. Run 7000-trial per-inning Monte Carlo
  4. Compute top-3 drivers

Email contains, per game:
  - Home / away WP
  - Projected total runs: mean + 95% CI
  - Top 3 drivers
  - BvP highlights (5+ AB, elite or futile)
  - Run-line probability (home / away wins by ≥2)

Configured to fire daily at 08:05 PT via launchd (5-min stagger after
the daily betting card).
"""

import argparse
import asyncio
import base64
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import aiohttp


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESEND_ENDPOINT = "https://api.resend.com/emails"
MC_TRIALS = 7000


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


async def run_all_games(date_str: str) -> List[Dict]:
    """Run V25Orchestrator over every game on `date_str`. Returns list of
    dicts ready for MC + rendering."""
    from core.orchestrator_v25 import V25Orchestrator
    from data.mlb_api import MLBDataAPI, SavantAPI
    from data.fangraphs_client import FanGraphsClient
    from data.nws_client import NWSClient
    from data.umpscores_client import UmpScoresClient
    from data.lineup_fetcher import LineupFetcher
    from data.odds_scraper import OddsScraper
    from mc.simulator import simulate_game
    from mc.extract import extract_inputs
    from mc.drivers import compute_drivers

    try:
        orch = V25Orchestrator(mode="live", as_of_date=date_str)
        games = await orch.mlb_api.get_games_for_date(date_str)

        # Parallel sim per game (orchestrator already does internal agent
        # parallelism per game; we run games sequentially to avoid swamping
        # upstream APIs)
        results = []
        for g in games:
            g_with_date = {**g, "date": date_str}
            sim = await orch.simulate_game_v25(g_with_date, date_str)
            mc_inputs = extract_inputs(sim)

            # Seed RNG per-game so re-runs are reproducible
            rng = random.Random(int(g.get("game_id", "0")) or 42)
            mc_result = simulate_game(mc_inputs, n_trials=MC_TRIALS, rng=rng)

            # Extras for the drivers function
            sp = sim.get("agents", {}).get("sp", {})
            lineup = sim.get("agents", {}).get("lineup", {})
            park = sim.get("agents", {}).get("park", {})
            extras = {
                "home_sp_name": g.get("home_sp_name") or "TBD",
                "away_sp_name": g.get("away_sp_name") or "TBD",
                "home_wrc_plus": lineup.get("home", {}).get("team_wrc_plus"),
                "away_wrc_plus": lineup.get("away", {}).get("team_wrc_plus"),
                "park_name": park.get("park", {}).get("name"),
                "weather": park.get("weather", {}),
            }
            drivers = compute_drivers(mc_inputs, mc_result, extras)

            results.append({
                "game": g,
                "mc": mc_result,
                "drivers": drivers,
                "bvp_home_vs_away_sp": (lineup.get("home", {})
                                        .get("bvp_threats", []))[:3],
                "bvp_away_vs_home_sp": (lineup.get("away", {})
                                        .get("bvp_threats", []))[:3],
                "lineup_projected": _is_projected(lineup),
            })

        return results
    finally:
        await MLBDataAPI.close()
        await SavantAPI.close()
        await FanGraphsClient.close()
        await NWSClient.close()
        await UmpScoresClient.close()
        await LineupFetcher.close()
        await OddsScraper.close()


def _is_projected(lineup_data: Dict) -> bool:
    for side in ("home", "away"):
        for p in lineup_data.get(side, {}).get("lineup", []):
            if isinstance(p, dict) and p.get("projected"):
                return True
    return False


def _format_bvp(threats: List[Dict], vs: str) -> str:
    if not threats:
        return ""
    bits = []
    for t in threats[:3]:
        sig = t.get("significance", "")
        glyph = {"elite": "🔥", "strong": "●", "homer_history": "💣",
                 "moderate": "○", "futile": "✗"}.get(sig, "")
        bits.append(
            f"{glyph} {t.get('name','?')} {t.get('h',0)}/{t.get('ab',0)}"
            f"{' '+str(t.get('hr'))+'HR' if t.get('hr') else ''} "
            f"({t.get('ops',0):.3f})"
        )
    return f"vs <em>{vs}</em>: " + " · ".join(bits)


def render_email_html(date_str: str, results: List[Dict]) -> str:
    cards = []
    n_projected = sum(1 for r in results if r["lineup_projected"])
    for r in results:
        g = r["game"]
        mc = r["mc"]
        home = g.get("home_team", "?")
        away = g.get("away_team", "?")
        home_sp = g.get("home_sp_name") or "TBD"
        away_sp = g.get("away_sp_name") or "TBD"

        home_wp = mc["home_wp"]
        away_wp = mc["away_wp"]
        proj_total = mc["projected_total_mean"]
        ci_lo = mc["projected_total_ci95_lo"]
        ci_hi = mc["projected_total_ci95_hi"]
        rl_home = mc["home_runline_win_wp"]
        rl_away = mc["away_runline_win_wp"]

        # Highlight stronger side
        if home_wp > away_wp:
            fav_label = f"<strong>{home}</strong> {home_wp*100:.1f}%"
            dog_label = f"{away} {away_wp*100:.1f}%"
        else:
            fav_label = f"<strong>{away}</strong> {away_wp*100:.1f}%"
            dog_label = f"{home} {home_wp*100:.1f}%"

        drivers_html = "".join(
            f"<li><strong>{d['name']}</strong> ({d['impact_pct']}): "
            f"<span style='color:#555'>{d['explanation']}</span></li>"
            for d in r["drivers"]
        )

        bvp_lines = []
        if r["bvp_home_vs_away_sp"]:
            bvp_lines.append(_format_bvp(r["bvp_home_vs_away_sp"], away_sp))
        if r["bvp_away_vs_home_sp"]:
            bvp_lines.append(_format_bvp(r["bvp_away_vs_home_sp"], home_sp))
        bvp_html = ""
        if bvp_lines:
            proj_tag = (" 🔮" if r["lineup_projected"] else "")
            bvp_html = (f"<div style='color:#374151; font-size:12px; "
                        f"margin-top:6px;'>"
                        f"<strong>BvP{proj_tag}:</strong><br>"
                        + "<br>".join(bvp_lines) + "</div>")

        cards.append(f"""
<div style="border: 1px solid #e5e7eb; border-radius: 8px;
            padding: 12px 16px; margin: 10px 0; background: #fafafa;">
  <div style="display:flex; justify-content:space-between; align-items:baseline;">
    <div style="font-size:16px;">
      <strong>{away}</strong> @ <strong>{home}</strong>
    </div>
    <div style="font-size:12px; color:#666;">{away_sp} (A) vs {home_sp} (H)</div>
  </div>
  <div style="margin: 10px 0; font-variant-numeric: tabular-nums;">
    <div style="font-size:14px;">
      WP: {fav_label} &nbsp;|&nbsp; {dog_label}
    </div>
    <div style="font-size:13px; color:#374151; margin-top:4px;">
      Projected total: <strong>{proj_total:.1f}</strong>
      &nbsp;(95% CI: {ci_lo}–{ci_hi})
      &nbsp;|&nbsp;
      Run line: home -1.5 = {rl_home*100:.1f}%, away -1.5 = {rl_away*100:.1f}%
    </div>
  </div>
  <ul style="margin: 6px 0 0 18px; padding: 0; font-size:12px; color:#222;">
    {drivers_html}
  </ul>
  {bvp_html}
</div>""")

    games_html = "\n".join(cards)
    projected_note = ""
    if n_projected > 0:
        projected_note = (
            f"<div style='background:#fef3c7; border-left:4px solid #f59e0b; "
            f"padding:8px 12px; margin-bottom:12px; font-size:12px; "
            f"color:#92400e;'>🔮 {n_projected} of {len(results)} games use "
            f"projected lineups (today's not yet posted at 8 AM PT). "
            f"BvP analysis based on probable starters from last 7 games.</div>"
        )

    return f"""<!doctype html>
<html><body style="font-family: ui-sans-serif, system-ui, sans-serif;
                  max-width: 760px; margin: 0 auto; color: #111;">
  <h2 style="margin-bottom:4px;">MLB Win Probability Report &mdash; {date_str}</h2>
  <div style="color:#555; margin-bottom:12px; font-size:13px;">
    Monte Carlo: <strong>{MC_TRIALS:,} trials/game</strong> &middot;
    per-inning Poisson sampling &middot;
    {len(results)} games &middot;
    weighted wRC+ (70% season / 30% rolling)
  </div>
  {projected_note}
  {games_html}
  <hr style="margin-top:24px; border:none; border-top:1px solid #e5e7eb;">
  <div style="color:#888; font-size:11px;">
    Inputs: SIERA/xFIP blend per SP, FanGraphs RPG per offense,
    bullpen ERA for innings 7-9, park runs factor, NWS weather forecast,
    BvP from MLB Stats API (≥3 ABs).
    Reproducible: each game seeded with its game_id; same date re-run
    yields identical numbers.
  </div>
</body></html>"""


async def send_via_resend(api_key: str, to: str, subject: str,
                          html: str, attachment_bytes: bytes = None,
                          attachment_name: str = None) -> Dict:
    payload = {
        "from": os.environ.get("CARD_FROM_EMAIL",
                                "MLB v3 <onboarding@resend.dev>"),
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if attachment_bytes:
        payload["attachments"] = [{
            "filename": attachment_name,
            "content": base64.b64encode(attachment_bytes).decode(),
        }]
    headers = {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as s:
        async with s.post(RESEND_ENDPOINT, json=payload, headers=headers,
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            body = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"Resend {r.status}: {body[:500]}")
            return json.loads(body) if body else {}


async def main_async(date_str: str, dry_run: bool) -> int:
    _load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("CARD_RECIPIENT_EMAIL")
    if not dry_run and (not api_key or not to):
        print("error: RESEND_API_KEY and CARD_RECIPIENT_EMAIL must be set",
              file=sys.stderr)
        return 2

    print(f"Generating MC report for {date_str} ({MC_TRIALS} trials/game)...")
    results = await run_all_games(date_str)
    print(f"  {len(results)} games simulated")

    html = render_email_html(date_str, results)
    subject = f"MLB win probability — {date_str} (Monte Carlo)"

    # Write JSON snapshot for archive
    out_dir = REPO_ROOT / "outputs" / "mc_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"mc_{date_str}.json"
    json_path.write_text(json.dumps([
        {"game": r["game"], "mc": r["mc"], "drivers": r["drivers"],
         "lineup_projected": r["lineup_projected"]}
        for r in results
    ], indent=2, default=str))

    if dry_run:
        print("--- DRY RUN: HTML preview (first 1500 chars) ---")
        print(html[:1500])
        return 0

    result = await send_via_resend(
        api_key, to, subject, html,
        attachment_bytes=json_path.read_bytes(),
        attachment_name=f"mc_{date_str}.json",
    )
    print(f"Sent to {to} — Resend id: {result.get('id')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    return asyncio.run(main_async(date_str, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
