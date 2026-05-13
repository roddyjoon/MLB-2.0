#!/usr/bin/env python3
"""
Generate today's MLB v3 daily card and email it via Resend.

Reads RESEND_API_KEY and CARD_RECIPIENT_EMAIL from the environment (or from
a sibling .env file at v3/.env). Falls back to today's date if none given.

Usage:
  python scripts/send_daily_card.py                   # today
  python scripts/send_daily_card.py 2026-05-07        # explicit date
  python scripts/send_daily_card.py --dry-run         # build email, skip send

Designed to run from the v3 repo root (so relative paths resolve correctly).
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp


REPO_ROOT = Path(__file__).resolve().parent.parent
RESEND_ENDPOINT = "https://api.resend.com/emails"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Only sets vars not already in os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


async def generate_card(date_str: str) -> Dict:
    """Run V25Orchestrator (live mode) and return the card dict."""
    # Lazy import so the script doesn't fail on missing deps when run with
    # --dry-run from a stale venv. These imports require the v3 venv.
    from core.orchestrator_v25 import V25Orchestrator
    from data.mlb_api import MLBDataAPI, SavantAPI
    from data.fangraphs_client import FanGraphsClient
    from data.nws_client import NWSClient
    from data.umpscores_client import UmpScoresClient
    from data.lineup_fetcher import LineupFetcher
    from data.odds_scraper import OddsScraper

    try:
        orch = V25Orchestrator(mode="live", as_of_date=date_str)
        card = await orch.generate_daily_card(date_str)
        return card
    finally:
        await MLBDataAPI.close()
        await SavantAPI.close()
        await FanGraphsClient.close()
        await NWSClient.close()
        await UmpScoresClient.close()
        await LineupFetcher.close()
        await OddsScraper.close()


def _format_bvp_inline(threats: list, vs_pitcher: str) -> str:
    """Render top BvP threats for one side as a compact string."""
    if not threats:
        return ""
    bits = []
    for t in threats[:3]:
        name = t.get("name", "?")
        h = t.get("h", 0)
        ab = t.get("ab", 0)
        hr_str = f" {t['hr']}HR" if t.get("hr") else ""
        sig = t.get("significance", "")
        sig_dot = ("🔥 " if sig == "elite" else "● " if sig == "strong"
                   else "○ " if sig == "moderate" else "✗ " if sig == "futile"
                   else "")
        bits.append(f"{sig_dot}{name} {h}/{ab}{hr_str} ({t.get('ops', 0):.3f})")
    return f"vs <em>{vs_pitcher}</em>: " + " · ".join(bits)


def _format_play_row(p: Dict, prefix: str) -> str:
    odds = p.get("odds", "?")
    sign = "+" if isinstance(odds, (int, float)) and odds > 0 else ""
    edge_pct = (p.get("edge", 0) or 0) * 100
    kelly = p.get("kelly_size", 0)
    matchup = p.get("matchup", "?")
    home_sp = p.get("home_sp", "")
    away_sp = p.get("away_sp", "")

    sp_line = ""
    if home_sp or away_sp:
        sp_line = (f"<tr><td></td>"
                   f"<td colspan='5' style='color:#666; font-size:12px; "
                   f"padding:0 10px 4px 10px;'>"
                   f"<strong>{away_sp or '?'}</strong> (A) vs "
                   f"<strong>{home_sp or '?'}</strong> (H)"
                   f"</td></tr>")

    # BvP rows (only if there are threats to show)
    bvp_rows = ""
    bvp_home = p.get("bvp_home_vs_away_sp") or []
    bvp_away = p.get("bvp_away_vs_home_sp") or []
    proj_tag = (" <span style='color:#a16207;'>🔮 projected lineup</span>"
                if p.get("lineup_projected") else "")
    if bvp_home and away_sp:
        bvp_rows += (f"<tr><td></td><td colspan='5' "
                     f"style='color:#374151; font-size:11px; "
                     f"padding:0 10px 2px 22px;'>"
                     f"{_format_bvp_inline(bvp_home, away_sp)}{proj_tag}"
                     f"</td></tr>")
    if bvp_away and home_sp:
        bvp_rows += (f"<tr><td></td><td colspan='5' "
                     f"style='color:#374151; font-size:11px; "
                     f"padding:0 10px 8px 22px;'>"
                     f"{_format_bvp_inline(bvp_away, home_sp)}{proj_tag}"
                     f"</td></tr>")

    main = (f"<tr>"
            f"<td>{prefix}</td>"
            f"<td style='color:#555'>{matchup}</td>"
            f"<td><strong>{p.get('label', '?')}</strong></td>"
            f"<td style='text-align:right'>{sign}{odds}</td>"
            f"<td style='text-align:right'>${kelly}</td>"
            f"<td style='text-align:right'>{edge_pct:.1f}%</td>"
            f"</tr>")
    return main + sp_line + bvp_rows


def render_email_html(card: Dict) -> str:
    primaries = card.get("primaries", [])
    secondaries = card.get("secondaries", [])
    passes = card.get("passes", [])
    total_risk = card.get("total_primary_risk", 0)

    p_rows = "".join(_format_play_row(p, f"P{i+1}")
                     for i, p in enumerate(primaries)) or \
        "<tr><td colspan='6' style='color:#888'>(none)</td></tr>"
    s_rows = "".join(_format_play_row(s, f"S{i+1}")
                     for i, s in enumerate(secondaries)) or \
        "<tr><td colspan='6' style='color:#888'>(none)</td></tr>"
    p_lines = "".join(
        f"<li>{(g.get('away_team') or '?')} @ {(g.get('home_team') or '?')}</li>"
        for g in passes if isinstance(g, dict)
    )

    return f"""<!doctype html>
<html><body style="font-family: ui-sans-serif, system-ui, sans-serif;
                  max-width: 720px; margin: 0 auto; color: #111;">
  <h2 style="margin-bottom:4px;">MLB v3 daily card &mdash; {card.get('date')}</h2>
  <div style="color:#555; margin-bottom:16px;">
    Generated {card.get('generated_at', '')[:19].replace('T', ' ')}
    &middot; {card.get('total_games')} games
    &middot; {card.get('agents_used')} agents
    &middot; primary risk <strong>${total_risk:.0f}</strong>
  </div>

  <h3 style="border-bottom:2px solid #2563eb; padding-bottom:4px;">
    🔥 Primaries ({len(primaries)})
  </h3>
  <table style="border-collapse:collapse; width:100%;
                font-variant-numeric:tabular-nums;">
    <thead><tr style="background:#f3f4f6;">
      <th style="text-align:left; padding:6px 10px;">#</th>
      <th style="text-align:left; padding:6px 10px;">Game</th>
      <th style="text-align:left; padding:6px 10px;">Play</th>
      <th style="text-align:right; padding:6px 10px;">Odds</th>
      <th style="text-align:right; padding:6px 10px;">Kelly</th>
      <th style="text-align:right; padding:6px 10px;">Edge</th>
    </tr></thead>
    <tbody>{p_rows}</tbody>
  </table>

  <h3 style="border-bottom:2px solid #f59e0b; padding-bottom:4px;
             margin-top:24px;">⚡ Secondaries ({len(secondaries)})</h3>
  <table style="border-collapse:collapse; width:100%;
                font-variant-numeric:tabular-nums;">
    <tbody>{s_rows}</tbody>
  </table>

  <h3 style="margin-top:24px; color:#666;">⏭ Passes ({len(passes)})</h3>
  <ul style="color:#555; columns:2;">{p_lines}</ul>

  <hr style="margin-top:32px; border:none; border-top:1px solid #e5e7eb;">
  <div style="color:#888; font-size:12px;">
    Full structured output attached as JSON.
    Backtest calibration (n=2,832): Brier 0.2422, ECE 0.0186.
    Known biases: away-favorite over-confidence in [0.4–0.5)
    home_wp bucket (gap −7.5%); total under-projection (~−1.9 runs).
  </div>
</body></html>"""


async def send_via_resend(api_key: str, to: str, subject: str,
                          html: str, attachments: Optional[List[Dict]] = None
                          ) -> Dict:
    payload = {
        "from": os.environ.get("CARD_FROM_EMAIL",
                               "MLB v3 <onboarding@resend.dev>"),
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if attachments:
        payload["attachments"] = attachments

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(RESEND_ENDPOINT, json=payload,
                          headers=headers,
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            body = await r.text()
            if r.status >= 300:
                raise RuntimeError(
                    f"Resend returned {r.status}: {body[:500]}"
                )
            return json.loads(body) if body else {}


async def main_async(date_str: str, dry_run: bool) -> int:
    # Make sure the v3 package layout resolves when invoked from anywhere
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _load_dotenv(REPO_ROOT / ".env")

    # launchd fires this script as soon as the calendar interval hits, but
    # if the Mac was asleep Wi-Fi/DNS may take 30-60s to come up after wake.
    # Block until DNS resolves before doing any network work.
    if not dry_run:
        sys.path.insert(0, str(Path(__file__).parent))
        from _net_wait import wait_for_network
        wait_for_network(timeout_seconds=300, interval_seconds=10)

    api_key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("CARD_RECIPIENT_EMAIL")
    if not dry_run and (not api_key or not to):
        print("error: RESEND_API_KEY and CARD_RECIPIENT_EMAIL must be set "
              "in environment or v3/.env",
              file=sys.stderr)
        return 2

    print(f"Generating card for {date_str}...")
    card = await generate_card(date_str)
    print(f"  {card.get('total_games')} games, "
          f"{len(card.get('primaries', []))} primaries, "
          f"{len(card.get('secondaries', []))} secondaries, "
          f"primary risk ${card.get('total_primary_risk', 0):.0f}")

    # Save to disk (the orchestrator already does this, but be explicit)
    out_path = REPO_ROOT / "outputs" / "cards" / f"card_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2, default=str))

    html = render_email_html(card)

    # Differentiate morning (8 AM, projected lineups) from afternoon (3 PM,
    # confirmed lineups) so they don't thread in Gmail.
    hour = datetime.now().hour
    if hour < 12:
        tag = "forecast"  # ~8 AM, lineups mostly projected
    elif hour < 17:
        tag = "refresh"   # ~3 PM, lineups mostly confirmed
    else:
        tag = "late"
    subject = f"MLB v3 daily card — {date_str} ({tag})"

    if dry_run:
        print("--- DRY RUN: HTML preview (first 1200 chars) ---")
        print(html[:1200])
        print("...")
        return 0

    # Attach the JSON
    attachments = [{
        "filename": f"card_{date_str}.json",
        "content": base64.b64encode(out_path.read_bytes()).decode(),
    }]

    print(f"Sending to {to} via Resend...")
    result = await send_via_resend(api_key, to, subject, html, attachments)
    print(f"Resend message id: {result.get('id')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", nargs="?", default=None,
                    help="ISO date (default: today)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build email but don't send")
    args = ap.parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    return asyncio.run(main_async(date_str, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
