#!/usr/bin/env python3
"""
Daily grading report email.

Reads every existing `outputs/cards/card_<date>.json`, joins each play
against actual final scores from MLB Stats API, and emails a report:

  - Headline: all-time record, win rate, P&L (realized + $100-unit)
  - Yesterday in detail: each play with W/L, P&L
  - Day-by-day rolling table: last 14 days W/L/P&L
  - Breakdowns: by play type, by edge bucket
  - Attached CSV of every graded play for forensics

Reuses `scripts/grade_cards.py` logic (imported as a module).

Designed to fire daily at 08:10 PT via the GitHub Actions workflow,
~10 minutes after the daily card so the cards are all generated.
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

RESEND_ENDPOINT = "https://api.resend.com/emails"
UNIT = 100.0
PRIMARY_THRESHOLD = 0.10


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


async def grade_all_days(end_date: str) -> Dict:
    """Grade every card from outputs/cards/, return per-day rows + summaries."""
    from grade_cards import grade_day, fetch_finals

    cards_dir = REPO_ROOT / "outputs" / "cards"
    dates = sorted(p.stem.replace("card_", "")
                   for p in cards_dir.glob("card_*.json"))
    # Only grade dates STRICTLY BEFORE end_date — today's games aren't done
    dates = [d for d in dates if d < end_date]

    all_rows: List[Dict] = []
    day_summaries: List[Dict] = []
    for d in dates:
        finals = await fetch_finals(d)
        rows, summ = grade_day(d, finals)
        if summ:
            all_rows.extend(rows)
            day_summaries.append(summ)

    return {"all_rows": all_rows, "day_summaries": day_summaries,
            "dates": dates}


def _fmt_pnl(x: float) -> str:
    return f"<span style='color:{'#15803d' if x >= 0 else '#b91c1c'}'>${x:+.2f}</span>"


def _fmt_wr(wins: int, losses: int) -> str:
    if not (wins + losses):
        return "—"
    return f"{wins/(wins+losses)*100:.1f}%"


def _edge_bucket(edge: float) -> str:
    if edge < 0.05: return "<5%"
    if edge < 0.10: return "5-10%"
    if edge < 0.15: return "10-15%"
    if edge < 0.20: return "15-20%"
    return "20%+"


def render_email(end_date: str, grading: Dict) -> str:
    rows = grading["all_rows"]
    summaries = grading["day_summaries"]
    graded = [r for r in rows if r["outcome"] in ("WIN", "LOSS", "PUSH")]

    # Headline aggregate
    wins = sum(1 for r in graded if r["outcome"] == "WIN")
    losses = sum(1 for r in graded if r["outcome"] == "LOSS")
    pushes = sum(1 for r in graded if r["outcome"] == "PUSH")
    realized = sum(r["realized_pnl"] for r in graded)
    realized_risk = sum(r["kelly_size"] for r in graded if r["outcome"] != "PUSH")
    unit = sum(r["unit_pnl"] for r in graded)
    unit_risk = (wins + losses) * UNIT

    wr_pct = (wins / (wins + losses) * 100) if (wins + losses) else 0
    realized_roi = (realized / realized_risk * 100) if realized_risk else 0
    unit_roi = (unit / unit_risk * 100) if unit_risk else 0

    # Yesterday's plays
    yesterday_str = (datetime.strptime(end_date, "%Y-%m-%d")
                     - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_plays = [r for r in graded if r["date"] == yesterday_str]

    yesterday_html = ""
    if yesterday_plays:
        rows_html = ""
        for r in yesterday_plays:
            outcome = r["outcome"]
            color = ("#15803d" if outcome == "WIN" else
                     "#b91c1c" if outcome == "LOSS" else "#6b7280")
            rows_html += f"""<tr>
              <td>{r.get('matchup','?')}</td>
              <td><strong>{r.get('label','?')}</strong></td>
              <td style="text-align:right">{r['odds']:+d}</td>
              <td style="text-align:right">{(r['edge'] or 0)*100:.1f}%</td>
              <td style="text-align:center;color:{color}">{outcome}</td>
              <td style="text-align:right">{r.get('away_score','-')}-{r.get('home_score','-')}</td>
              <td style="text-align:right">{_fmt_pnl(r['unit_pnl'])}</td>
            </tr>"""
        y_wins = sum(1 for r in yesterday_plays if r["outcome"] == "WIN")
        y_losses = sum(1 for r in yesterday_plays if r["outcome"] == "LOSS")
        y_pnl = sum(r["unit_pnl"] for r in yesterday_plays)
        yesterday_html = f"""
<h3 style="border-bottom:2px solid #2563eb; padding-bottom:4px; margin-top:24px;">
  Yesterday ({yesterday_str}): {y_wins}-{y_losses} &middot; {_fmt_pnl(y_pnl)} unit
</h3>
<table style="border-collapse:collapse; width:100%; font-size:12px;
              font-variant-numeric:tabular-nums;">
  <thead><tr style="background:#f3f4f6;">
    <th style="text-align:left; padding:4px 8px;">Game</th>
    <th style="text-align:left; padding:4px 8px;">Play</th>
    <th style="text-align:right; padding:4px 8px;">Odds</th>
    <th style="text-align:right; padding:4px 8px;">Edge</th>
    <th style="text-align:center; padding:4px 8px;">Result</th>
    <th style="text-align:right; padding:4px 8px;">Score</th>
    <th style="text-align:right; padding:4px 8px;">Unit P&L</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

    # Day-by-day rolling table (last 14 days, most recent first)
    daily_html = "<table style='border-collapse:collapse; font-size:12px; " \
                 "font-variant-numeric:tabular-nums;'>" \
                 "<thead><tr style='background:#f3f4f6;'>" \
                 "<th style='text-align:left; padding:4px 10px;'>Date</th>" \
                 "<th style='text-align:right; padding:4px 10px;'>Plays</th>" \
                 "<th style='text-align:right; padding:4px 10px;'>W-L</th>" \
                 "<th style='text-align:right; padding:4px 10px;'>WR</th>" \
                 "<th style='text-align:right; padding:4px 10px;'>Realized</th>" \
                 "<th style='text-align:right; padding:4px 10px;'>Unit</th>" \
                 "</tr></thead><tbody>"
    for s in summaries[-14:][::-1]:
        wr = _fmt_wr(s["wins"], s["losses"])
        daily_html += f"""<tr>
          <td style="padding:4px 10px;">{s['date']}</td>
          <td style="text-align:right; padding:4px 10px;">{s['n_plays']}</td>
          <td style="text-align:right; padding:4px 10px;">{s['wins']}-{s['losses']}</td>
          <td style="text-align:right; padding:4px 10px;">{wr}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_pnl(s['realized_pnl'])}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_pnl(s['unit_pnl'])}</td>
        </tr>"""
    daily_html += "</tbody></table>"

    # Breakdowns
    type_groups: Dict[str, List[Dict]] = {}
    for r in graded:
        type_groups.setdefault(r["type"], []).append(r)
    type_rows = ""
    for ptype in sorted(type_groups.keys()):
        g = type_groups[ptype]
        gw = sum(1 for r in g if r["outcome"] == "WIN")
        gl = sum(1 for r in g if r["outcome"] == "LOSS")
        gp = sum(r["unit_pnl"] for r in g)
        type_rows += f"""<tr>
          <td style="padding:4px 10px;">{ptype}</td>
          <td style="text-align:right; padding:4px 10px;">{gw}-{gl}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_wr(gw,gl)}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_pnl(gp)}</td>
        </tr>"""

    bucket_groups: Dict[str, List[Dict]] = {}
    for r in graded:
        bucket_groups.setdefault(_edge_bucket(r["edge"] or 0), []).append(r)
    bucket_rows = ""
    for b in ["<5%", "5-10%", "10-15%", "15-20%", "20%+"]:
        if b not in bucket_groups: continue
        g = bucket_groups[b]
        gw = sum(1 for r in g if r["outcome"] == "WIN")
        gl = sum(1 for r in g if r["outcome"] == "LOSS")
        gp = sum(r["unit_pnl"] for r in g)
        bucket_rows += f"""<tr>
          <td style="padding:4px 10px;">{b}</td>
          <td style="text-align:right; padding:4px 10px;">{gw}-{gl}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_wr(gw,gl)}</td>
          <td style="text-align:right; padding:4px 10px;">{_fmt_pnl(gp)}</td>
        </tr>"""

    n_days = len(summaries)
    return f"""<!doctype html>
<html><body style="font-family: ui-sans-serif, system-ui, sans-serif;
                  max-width: 760px; margin: 0 auto; color: #111;">
  <h2 style="margin-bottom:4px;">MLB v3 grading report &mdash; through {(datetime.strptime(end_date,'%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d')}</h2>
  <div style="color:#555; font-size:13px; margin-bottom:16px;">
    <strong>{wins}-{losses}{'-'+str(pushes) if pushes else ''}</strong>
    &middot; WR <strong>{wr_pct:.1f}%</strong>
    (breakeven -110: 52.4%)
    &middot; {n_days} days, {len(graded)} graded plays
  </div>
  <div style="display:flex; gap:24px; margin-bottom:16px; font-size:13px;">
    <div style="background:#f3f4f6; padding:10px 16px; border-radius:6px;">
      <div style="color:#666; font-size:11px;">REALIZED (Kelly)</div>
      <div style="font-size:18px; font-variant-numeric:tabular-nums;">{_fmt_pnl(realized)}</div>
      <div style="color:#666; font-size:11px;">on ${realized_risk:.0f} risked &middot; ROI {realized_roi:+.1f}%</div>
    </div>
    <div style="background:#f3f4f6; padding:10px 16px; border-radius:6px;">
      <div style="color:#666; font-size:11px;">UNIT ($100/play)</div>
      <div style="font-size:18px; font-variant-numeric:tabular-nums;">{_fmt_pnl(unit)}</div>
      <div style="color:#666; font-size:11px;">on ${unit_risk:.0f} risked &middot; ROI {unit_roi:+.1f}%</div>
    </div>
  </div>

  {yesterday_html}

  <h3 style="border-bottom:2px solid #6b7280; padding-bottom:4px; margin-top:24px;">
    Daily rolling (last 14)
  </h3>
  {daily_html}

  <h3 style="border-bottom:2px solid #6b7280; padding-bottom:4px; margin-top:24px;">
    By play type
  </h3>
  <table style="border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums;">
    <thead><tr style="background:#f3f4f6;">
      <th style="text-align:left; padding:4px 10px;">Type</th>
      <th style="text-align:right; padding:4px 10px;">W-L</th>
      <th style="text-align:right; padding:4px 10px;">WR</th>
      <th style="text-align:right; padding:4px 10px;">Unit P&L</th>
    </tr></thead>
    <tbody>{type_rows}</tbody>
  </table>

  <h3 style="border-bottom:2px solid #6b7280; padding-bottom:4px; margin-top:24px;">
    By edge bucket
  </h3>
  <table style="border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums;">
    <thead><tr style="background:#f3f4f6;">
      <th style="text-align:left; padding:4px 10px;">Edge</th>
      <th style="text-align:right; padding:4px 10px;">W-L</th>
      <th style="text-align:right; padding:4px 10px;">WR</th>
      <th style="text-align:right; padding:4px 10px;">Unit P&L</th>
    </tr></thead>
    <tbody>{bucket_rows}</tbody>
  </table>

  <hr style="margin-top:24px; border:none; border-top:1px solid #e5e7eb;">
  <div style="color:#888; font-size:11px;">
    Realized P&L uses the Kelly stake on each play. Unit P&L is a counterfactual
    flat $100 per play, useful when many Kelly sizes are $0.
    Full per-play breakdown attached as CSV.
  </div>
</body></html>"""


def _rows_to_csv_bytes(rows: List[Dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


async def send_via_resend(api_key: str, to: str, subject: str,
                          html: str, csv_bytes: bytes,
                          csv_name: str) -> Dict:
    payload = {
        "from": os.environ.get("CARD_FROM_EMAIL",
                                "MLB v3 <onboarding@resend.dev>"),
        "to": [to],
        "subject": subject,
        "html": html,
        "attachments": [{
            "filename": csv_name,
            "content": base64.b64encode(csv_bytes).decode(),
        }] if csv_bytes else [],
    }
    headers = {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as s:
        async with s.post(RESEND_ENDPOINT, json=payload, headers=headers,
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            body = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"Resend {r.status}: {body[:500]}")
            return json.loads(body) if body else {}


async def main_async(end_date: str, dry_run: bool) -> int:
    _load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("CARD_RECIPIENT_EMAIL")
    if not dry_run and (not api_key or not to):
        print("error: RESEND_API_KEY and CARD_RECIPIENT_EMAIL must be set",
              file=sys.stderr)
        return 2

    # Block until DNS up (covers cold launchd wakes; harmless in cloud CI)
    if not dry_run:
        from _net_wait import wait_for_network
        wait_for_network(timeout_seconds=300, interval_seconds=10)

    print(f"Grading all cards through {end_date}...")
    grading = await grade_all_days(end_date)
    graded_count = sum(1 for r in grading["all_rows"]
                       if r["outcome"] in ("WIN", "LOSS", "PUSH"))
    print(f"  {len(grading['day_summaries'])} days, {graded_count} graded plays")

    html = render_email(end_date, grading)
    csv_bytes = _rows_to_csv_bytes(grading["all_rows"])

    yesterday = (datetime.strptime(end_date, "%Y-%m-%d")
                 - timedelta(days=1)).strftime("%Y-%m-%d")
    subject = f"MLB v3 grading — through {yesterday}"

    if dry_run:
        print("--- DRY RUN: HTML preview (first 1500 chars) ---")
        print(html[:1500])
        return 0

    result = await send_via_resend(
        api_key, to, subject, html, csv_bytes,
        f"plays_through_{yesterday}.csv",
    )
    print(f"Sent to {to} — Resend id: {result.get('id')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("end_date", nargs="?", default=None,
                    help="Grade all days strictly before this date (default: today)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    end = args.end_date or datetime.now().strftime("%Y-%m-%d")
    return asyncio.run(main_async(end, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
