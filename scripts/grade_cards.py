#!/usr/bin/env python3
"""
Grade live cards.

For each `outputs/cards/card_<date>.json`, join recommended plays against the
day's final scores (MLB Stats API) and compute:
  - Per-play: outcome (W/L/Push), realized P&L (based on Kelly size), unit P&L
    ($100 flat unit per play, for comparison since many Kelly sizes are $0)
  - Per-day aggregate
  - Cross-day rollup with breakdowns by play type and edge bucket

Usage:
  python scripts/grade_cards.py                # grade all completed days
  python scripts/grade_cards.py 2026-05-06 2026-05-07
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


UNIT = 100.0  # counterfactual "what if we bet $100 flat on every primary"

# v3 post-tuning sizing: WP cap 0.65 + quarter-Kelly + $220 hard cap. At any
# odds better than ~-150, the cap dominates → essentially $220 flat on every
# play. Threshold for "would have been primary under new rules" is 10%.
SIM_NEW_STAKE = 220.0
SIM_NEW_PRIMARY_THRESHOLD = 0.10


def american_payout(stake: float, odds: int) -> float:
    """Profit on a winning bet at American odds (loss returns -stake)."""
    if odds > 0:
        return stake * (odds / 100)
    return stake * (100 / abs(odds))


def parse_total_line(label: str) -> Optional[float]:
    """Pull the line out of 'Under 8.5' / 'Over 9' etc."""
    m = re.match(r"(?:Over|Under)\s+([\d.]+)", label or "")
    return float(m.group(1)) if m else None


def grade_play(play: Dict, finals_by_matchup: Dict[str, Dict]) -> Dict:
    matchup = play.get("matchup", "")
    final = finals_by_matchup.get(matchup)
    # Initialize ALL fields up-front so every grade row has the same schema —
    # downstream CSV writers use the first row's keys as fieldnames and a
    # missing sim_new_* (from the NO_GAME early-return path) crashed the
    # 2026-06-09 grade_launchd run.
    out = {
        "matchup": matchup,
        "label": play.get("label"),
        "type": play.get("type"),
        "odds": play.get("odds"),
        "model_wp": play.get("model_wp"),
        "edge": play.get("edge"),
        "kelly_size": play.get("kelly_size", 0),
        "tier": play.get("tier"),
        "home_score": None,
        "away_score": None,
        "total": None,
        "outcome": None,  # WIN / LOSS / PUSH / NO_GAME / UNGRADABLE
        "realized_pnl": 0.0,
        "unit_pnl": 0.0,
        "sim_new_stake": 0.0,
        "sim_new_pnl": 0.0,
    }
    if not final:
        out["outcome"] = "NO_GAME"
        return out

    h = int(final["home_score"])
    a = int(final["away_score"])
    out["home_score"] = h
    out["away_score"] = a
    total = h + a
    out["total"] = total

    ptype = play.get("type")
    if ptype == "home_ml":
        won = h > a
    elif ptype == "away_ml":
        won = a > h
    elif ptype in ("over", "under"):
        line = parse_total_line(play.get("label", ""))
        if line is None:
            out["outcome"] = "UNGRADABLE"
            return out
        if total == line:
            out["outcome"] = "PUSH"
            return out
        won = (total > line) if ptype == "over" else (total < line)
    else:
        out["outcome"] = "UNGRADABLE"
        return out

    out["outcome"] = "WIN" if won else "LOSS"
    odds = play.get("odds")
    kelly = float(play.get("kelly_size", 0) or 0)
    edge = float(play.get("edge", 0) or 0)
    if won:
        out["realized_pnl"] = american_payout(kelly, odds) if kelly else 0.0
        out["unit_pnl"] = american_payout(UNIT, odds)
    else:
        out["realized_pnl"] = -kelly
        out["unit_pnl"] = -UNIT

    # Simulated P&L under v3 post-tuning sizing: $220 flat on every play
    # with edge >= 10%. Below 10%, treated as a pass (no bet).
    if edge >= SIM_NEW_PRIMARY_THRESHOLD:
        out["sim_new_stake"] = SIM_NEW_STAKE
        if won:
            out["sim_new_pnl"] = american_payout(SIM_NEW_STAKE, odds)
        else:
            out["sim_new_pnl"] = -SIM_NEW_STAKE
    else:
        out["sim_new_stake"] = 0.0
        out["sim_new_pnl"] = 0.0
    return out


async def fetch_finals(date: str) -> Dict[str, Dict]:
    """Returns {'AWAY @ HOME': {...final score}} for completed games on `date`."""
    from data.mlb_api import MLBDataAPI
    api = MLBDataAPI(as_of_date=date, mode="backtest")
    try:
        finals = await api.get_final_scores(date)
    finally:
        await type(api).close()
    return {f"{f['away_team']} @ {f['home_team']}": f for f in finals}


def edge_bucket(edge: float) -> str:
    if edge < 0.05:
        return "<5%"
    if edge < 0.10:
        return "5-10%"
    if edge < 0.15:
        return "10-15%"
    if edge < 0.20:
        return "15-20%"
    return "20%+"


def _card_files_for_date(date: str) -> List[Tuple[str, Path]]:
    """
    Return list of (fire_tag, path) for all card files matching the date.

    Includes:
      - Tagged files: card_<date>_am.json, card_<date>_midday.json,
        card_<date>_pm.json (new convention, post 2026-07)
      - Legacy untagged: card_<date>.json (marked fire='legacy')

    We deliberately grade tagged files ALONGSIDE legacy ones so cards
    generated before the split-per-fire migration still count in totals.
    Legacy cards were written by whichever fire ran LAST on that day —
    usually the 3 PM refresh — so treat them as approximating "pm".
    """
    cards_dir = REPO_ROOT / "outputs" / "cards"
    files: List[Tuple[str, Path]] = []
    for tag in ("am", "midday", "pm", "late"):
        p = cards_dir / f"card_{date}_{tag}.json"
        if p.exists():
            files.append((tag, p))
    legacy = cards_dir / f"card_{date}.json"
    if legacy.exists():
        files.append(("legacy", legacy))
    return files


def grade_day(date: str, finals: Dict[str, Dict]) -> Tuple[List[Dict], Dict]:
    card_files = _card_files_for_date(date)
    if not card_files:
        return [], {}

    rows: List[Dict] = []
    per_fire_summary: Dict[str, Dict] = {}

    for fire_tag, card_path in card_files:
        card = json.loads(card_path.read_text())

        # Backfill matchup for older cards (pre-orchestrator-matchup-field).
        def _backfill_matchups(plays_list: List[Dict]) -> None:
            details = card.get("game_details", [])
            for play in plays_list:
                if play.get("matchup"):
                    continue
                for gd in details:
                    bp = gd.get("best_play")
                    if not bp:
                        continue
                    if (bp.get("label") == play.get("label")
                            and bp.get("odds") == play.get("odds")
                            and bp.get("edge") == play.get("edge")):
                        g = gd.get("game", {})
                        play["matchup"] = (f"{g.get('away_team', '?')} @ "
                                           f"{g.get('home_team', '?')}")
                        break

        primaries = card.get("primaries", [])
        secondaries = card.get("secondaries", [])
        _backfill_matchups(primaries)
        _backfill_matchups(secondaries)

        fire_rows: List[Dict] = []
        for p in primaries:
            r = grade_play(p, finals)
            r["date"] = date
            r["fire"] = fire_tag
            r["tier"] = "primary"
            fire_rows.append(r)
        for p in secondaries:
            r = grade_play(p, finals)
            r["date"] = date
            r["fire"] = fire_tag
            r["tier"] = "secondary"
            fire_rows.append(r)
        rows.extend(fire_rows)

        fg = [r for r in fire_rows if r["outcome"] in ("WIN", "LOSS", "PUSH")]
        fw = sum(1 for r in fg if r["outcome"] == "WIN")
        fl = sum(1 for r in fg if r["outcome"] == "LOSS")
        fpu = sum(1 for r in fg if r["outcome"] == "PUSH")
        per_fire_summary[fire_tag] = {
            "n_plays": len(fire_rows),
            "n_graded": len(fg),
            "wins": fw,
            "losses": fl,
            "pushes": fpu,
            "win_rate": fw / (fw + fl) if (fw + fl) else None,
            "realized_pnl": round(sum(r["realized_pnl"] for r in fg), 2),
            "unit_pnl": round(sum(r["unit_pnl"] for r in fg), 2),
        }

    # Aggregate summary across all fires
    graded = [r for r in rows if r["outcome"] in ("WIN", "LOSS", "PUSH")]
    wins = sum(1 for r in graded if r["outcome"] == "WIN")
    losses = sum(1 for r in graded if r["outcome"] == "LOSS")
    pushes = sum(1 for r in graded if r["outcome"] == "PUSH")
    summary = {
        "date": date,
        "n_plays": len(rows),
        "n_graded": len(graded),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / (wins + losses) if (wins + losses) else None,
        "realized_pnl": round(sum(r["realized_pnl"] for r in graded), 2),
        "unit_pnl": round(sum(r["unit_pnl"] for r in graded), 2),
        "per_fire": per_fire_summary,
    }
    return rows, summary


async def main_async(dates: Optional[List[str]]) -> int:
    cards_dir = REPO_ROOT / "outputs" / "cards"
    if not dates:
        # Extract YYYY-MM-DD from both `card_<date>.json` and
        # `card_<date>_<fire>.json` filename patterns.
        found = set()
        for p in cards_dir.glob("card_*.json"):
            stem = p.stem.replace("card_", "")
            # Trim fire suffix like "_am" / "_pm" / "_midday" / "_late"
            for suf in ("_am", "_pm", "_midday", "_late"):
                if stem.endswith(suf):
                    stem = stem[:-len(suf)]
                    break
            found.add(stem)
        dates = sorted(found)

    all_rows: List[Dict] = []
    day_summaries: List[Dict] = []

    print(f"Grading {len(dates)} days...\n")
    for d in dates:
        finals = await fetch_finals(d)
        rows, summ = grade_day(d, finals)
        if not summ:
            continue
        all_rows.extend(rows)
        day_summaries.append(summ)
        wr = (f"{summ['win_rate']*100:.1f}%" if summ['win_rate'] is not None
              else "n/a")
        print(f"  {d}: {summ['wins']:>2}W-{summ['losses']:>2}L"
              f"{'-'+str(summ['pushes'])+'P' if summ['pushes'] else '':>3}  "
              f"WR={wr:>6}  realized=${summ['realized_pnl']:>7.2f}  "
              f"unit=${summ['unit_pnl']:>8.2f}  (n_plays={summ['n_plays']})")

    # Aggregate
    graded = [r for r in all_rows if r["outcome"] in ("WIN", "LOSS", "PUSH")]
    wins = sum(1 for r in graded if r["outcome"] == "WIN")
    losses = sum(1 for r in graded if r["outcome"] == "LOSS")
    pushes = sum(1 for r in graded if r["outcome"] == "PUSH")
    realized = sum(r["realized_pnl"] for r in graded)
    unit = sum(r["unit_pnl"] for r in graded)
    n_units_risked = (wins + losses)  # one unit per graded play
    n_dollars_risked_realized = sum(r["kelly_size"] for r in graded
                                     if r["outcome"] != "PUSH")

    print(f"\n{'='*72}")
    print(f"AGGREGATE  ({len(graded)} graded plays across {len(day_summaries)} days)")
    print(f"{'='*72}")
    if wins + losses:
        wr = wins / (wins + losses)
        print(f"  Record:           {wins}-{losses}"
              f"{'-'+str(pushes) if pushes else ''}  WR={wr*100:.1f}%")
        breakeven = abs((-110)) / (abs(-110) + 100)  # 52.4% @ -110
        print(f"  Breakeven (-110): {breakeven*100:.1f}%  "
              f"({'+' if wr > breakeven else ''}{(wr-breakeven)*100:.1f}% vs breakeven)")
    print(f"  Realized P&L:     ${realized:.2f}  on ${n_dollars_risked_realized:.0f} risked")
    if n_dollars_risked_realized:
        print(f"      ROI:          {realized/n_dollars_risked_realized*100:+.1f}%")
    print(f"  Unit ($100/play): ${unit:.2f}  on ${n_units_risked*UNIT:.0f} risked")
    if n_units_risked:
        print(f"      ROI:          {unit/(n_units_risked*UNIT)*100:+.1f}%")

    # Simulated P&L under v3 post-tuning sizing
    sim_plays = [r for r in graded if r.get("sim_new_stake", 0) > 0]
    sim_pnl = sum(r.get("sim_new_pnl", 0) for r in sim_plays)
    sim_risked = sum(r.get("sim_new_stake", 0) for r in sim_plays)
    sim_wins = sum(1 for r in sim_plays if r["outcome"] == "WIN")
    sim_losses = sum(1 for r in sim_plays if r["outcome"] == "LOSS")
    print(f"\n  NEW sizing simulation ($220 flat, edge >=10%):")
    print(f"    Record: {sim_wins}W-{sim_losses}L  "
          f"WR={sim_wins/(sim_wins+sim_losses)*100:.1f}%  "
          f"(filtered {len(graded) - len(sim_plays)} sub-10% plays)")
    print(f"    P&L:    ${sim_pnl:+.2f}  on ${sim_risked:.0f} risked  "
          f"ROI={sim_pnl/sim_risked*100:+.1f}%")

    # Breakdown by play type
    print(f"\n  Breakdown by play type (unit P&L, since most Kelly sizes are $0):")
    type_groups: Dict[str, List[Dict]] = {}
    for r in graded:
        type_groups.setdefault(r["type"], []).append(r)
    for ptype, group in sorted(type_groups.items()):
        gw = sum(1 for r in group if r["outcome"] == "WIN")
        gl = sum(1 for r in group if r["outcome"] == "LOSS")
        gpnl = sum(r["unit_pnl"] for r in group)
        gwr = gw / (gw + gl) if (gw + gl) else 0
        roi = gpnl / ((gw + gl) * UNIT) * 100 if (gw + gl) else 0
        print(f"    {ptype:<10} {gw:>3}W-{gl:>3}L  WR={gwr*100:>5.1f}%  "
              f"P&L=${gpnl:>+8.2f}  ROI={roi:>+6.1f}%")

    # Breakdown by edge bucket
    print(f"\n  Breakdown by edge bucket (unit P&L):")
    bucket_groups: Dict[str, List[Dict]] = {}
    for r in graded:
        bucket_groups.setdefault(edge_bucket(r["edge"] or 0), []).append(r)
    for bucket in ["<5%", "5-10%", "10-15%", "15-20%", "20%+"]:
        if bucket not in bucket_groups:
            continue
        group = bucket_groups[bucket]
        gw = sum(1 for r in group if r["outcome"] == "WIN")
        gl = sum(1 for r in group if r["outcome"] == "LOSS")
        gpnl = sum(r["unit_pnl"] for r in group)
        gwr = gw / (gw + gl) if (gw + gl) else 0
        roi = gpnl / ((gw + gl) * UNIT) * 100 if (gw + gl) else 0
        print(f"    edge {bucket:<7} {gw:>3}W-{gl:>3}L  WR={gwr*100:>5.1f}%  "
              f"P&L=${gpnl:>+8.2f}  ROI={roi:>+6.1f}%")

    # Write CSV
    out_dir = REPO_ROOT / "outputs" / "grading"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"plays_{dates[0]}_to_{dates[-1]}.csv"
    if all_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\n  Wrote: {out_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dates", nargs="*", help="Specific dates to grade (default: all)")
    args = ap.parse_args()
    return asyncio.run(main_async(args.dates or None))


if __name__ == "__main__":
    sys.exit(main())
