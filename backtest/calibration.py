"""
Backtest calibration metrics.

Reads outputs/backtest/predictions_*.csv, computes WP-side calibration:
  - Brier score
  - Log-loss
  - Reliability diagram (10 buckets)
  - Expected Calibration Error (ECE)
  - ROC AUC
And total-side accuracy:
  - RMSE on total_proj vs total_actual
  - Brier on over_8_5_hit (using over_wp, currently always 0.5 in v2.5
    until the line-comparison logic moves out of edge_calculator into wp)

No external deps — pure stdlib + math. (matplotlib used optionally in
backtest/report.py for the PNG rendering.)
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _read_predictions(start: str, end: str,
                      out_dir: Path = Path("outputs/backtest")) -> List[Dict]:
    """Read predictions_<date>.csv for each date in [start, end]; concat."""
    from backtest.harness import daterange
    rows: List[Dict] = []
    for date in daterange(start, end):
        f = out_dir / f"predictions_{date}.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def _to_float(v) -> float:
    try:
        return float(v) if v not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _filter_graded_wp(rows: List[Dict]) -> List[Tuple[float, int]]:
    """Return [(home_wp, home_won), ...] for rows with both fields populated."""
    out = []
    for r in rows:
        hw = _to_float(r.get("home_wp"))
        won = r.get("home_won")
        if math.isnan(hw):
            continue
        if won not in ("0", "1"):
            continue
        out.append((hw, int(won)))
    return out


def brier_score(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def log_loss(pairs: List[Tuple[float, int]], eps: float = 1e-9) -> float:
    if not pairs:
        return float("nan")
    total = 0.0
    for p, o in pairs:
        p = max(eps, min(1 - eps, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(pairs)


def reliability_buckets(pairs: List[Tuple[float, int]],
                         n_bins: int = 10) -> List[Dict]:
    """
    Returns one dict per bucket with:
      bucket_lo, bucket_hi, n, mean_predicted, observed_freq, gap.
    """
    width = 1.0 / n_bins
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        # Clamp predicted into [0, 1) so 1.0 lands in last bucket
        idx = min(n_bins - 1, max(0, int(p / width)))
        bins[idx].append((p, o))

    out = []
    for i, bucket in enumerate(bins):
        lo = i * width
        hi = lo + width
        n = len(bucket)
        if n == 0:
            out.append({
                "bucket_lo": round(lo, 2), "bucket_hi": round(hi, 2),
                "n": 0, "mean_predicted": None,
                "observed_freq": None, "gap": None,
            })
            continue
        mp = sum(p for p, _ in bucket) / n
        of = sum(o for _, o in bucket) / n
        out.append({
            "bucket_lo": round(lo, 2), "bucket_hi": round(hi, 2),
            "n": n,
            "mean_predicted": round(mp, 4),
            "observed_freq": round(of, 4),
            "gap": round(of - mp, 4),
        })
    return out


def expected_calibration_error(pairs: List[Tuple[float, int]],
                                n_bins: int = 10) -> float:
    """ECE = Σ (n_i / N) * |observed_i − predicted_i| over non-empty bins."""
    n_total = len(pairs)
    if n_total == 0:
        return float("nan")
    ece = 0.0
    for b in reliability_buckets(pairs, n_bins):
        if b["n"] == 0 or b["mean_predicted"] is None:
            continue
        ece += (b["n"] / n_total) * abs(b["observed_freq"] - b["mean_predicted"])
    return ece


def roc_auc(pairs: List[Tuple[float, int]]) -> float:
    """
    Area under ROC for using home_wp as a classifier of home_won.
    Computed via Mann-Whitney U: the probability that a random positive
    has higher score than a random negative.
    """
    pos = [p for p, o in pairs if o == 1]
    neg = [p for p, o in pairs if o == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def total_rmse(rows: List[Dict]) -> float:
    sq = []
    for r in rows:
        proj = _to_float(r.get("total_proj"))
        actual = _to_float(r.get("total_actual"))
        if math.isnan(proj) or math.isnan(actual):
            continue
        sq.append((proj - actual) ** 2)
    if not sq:
        return float("nan")
    return math.sqrt(sum(sq) / len(sq))


def total_bias(rows: List[Dict]) -> float:
    """Mean (proj - actual). Negative = model underprojects."""
    diffs = []
    for r in rows:
        proj = _to_float(r.get("total_proj"))
        actual = _to_float(r.get("total_actual"))
        if math.isnan(proj) or math.isnan(actual):
            continue
        diffs.append(proj - actual)
    if not diffs:
        return float("nan")
    return sum(diffs) / len(diffs)


def biggest_misses(pairs: List[Tuple[float, int]],
                    rows: List[Dict], n: int = 20) -> List[Dict]:
    """Top n |home_wp − home_won| with game context."""
    by_pk = {r.get("game_pk"): r for r in rows
             if r.get("home_won") in ("0", "1")}
    misses = []
    for r in rows:
        hw = _to_float(r.get("home_wp"))
        won = r.get("home_won")
        if math.isnan(hw) or won not in ("0", "1"):
            continue
        misses.append({
            "date": r.get("date"),
            "matchup": f"{r.get('away')} @ {r.get('home')}",
            "predicted": round(hw, 3),
            "actual": int(won),
            "miss": round(abs(hw - int(won)), 3),
        })
    misses.sort(key=lambda x: x["miss"], reverse=True)
    return misses[:n]


def calibrate(start: str, end: str,
               out_dir: Path = Path("outputs/backtest")) -> Dict:
    rows = _read_predictions(start, end, out_dir)
    pairs = _filter_graded_wp(rows)

    n_total = len(rows)
    n_graded = len(pairs)

    summary = {
        "range": {"start": start, "end": end},
        "n_games_total": n_total,
        "n_games_graded": n_graded,
        "wp_brier_score": round(brier_score(pairs), 4) if pairs else None,
        "wp_log_loss": round(log_loss(pairs), 4) if pairs else None,
        "wp_roc_auc": round(roc_auc(pairs), 4) if pairs else None,
        "wp_ece": round(expected_calibration_error(pairs), 4) if pairs else None,
        "total_rmse": round(total_rmse(rows), 3),
        "total_bias_proj_minus_actual": round(total_bias(rows), 3),
        "home_win_rate_actual": (
            round(sum(o for _, o in pairs) / n_graded, 4)
            if n_graded else None
        ),
        "home_wp_mean_predicted": (
            round(sum(p for p, _ in pairs) / n_graded, 4)
            if n_graded else None
        ),
        "reliability": reliability_buckets(pairs, n_bins=10),
        "biggest_misses": biggest_misses(pairs, rows, n=20),
    }
    return summary


async def run_calibration(start: str, end: str) -> None:
    """CLI entry. Writes JSON summary + reliability CSV + prints metrics."""
    summary = calibrate(start, end)

    reports_dir = Path("outputs/backtest/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"calibration_{start}_to_{end}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    rel_path = reports_dir / f"reliability_{start}_to_{end}.csv"
    with open(rel_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bucket_lo", "bucket_hi", "n",
            "mean_predicted", "observed_freq", "gap",
        ])
        w.writeheader()
        for b in summary["reliability"]:
            w.writerow(b)

    # Console summary
    print()
    print(f"=== Calibration {start} → {end} ===")
    print(f"  games graded: {summary['n_games_graded']}/{summary['n_games_total']}")
    print()
    print(f"  WP Brier:      {summary['wp_brier_score']}  "
          f"(lower is better; 0.25 = random)")
    print(f"  WP log-loss:   {summary['wp_log_loss']}  (lower is better)")
    print(f"  WP ROC AUC:    {summary['wp_roc_auc']}  (0.5 = random)")
    print(f"  WP ECE:        {summary['wp_ece']}  (lower is better)")
    print(f"  Mean home_wp:  {summary['home_wp_mean_predicted']}  "
          f"vs actual home win rate {summary['home_win_rate_actual']}")
    print()
    print(f"  Total RMSE:    {summary['total_rmse']}")
    print(f"  Total bias:    {summary['total_bias_proj_minus_actual']}  "
          f"(neg = model underprojects)")
    print()
    print(f"  Reliability buckets (predicted vs observed):")
    for b in summary["reliability"]:
        if b["n"] == 0:
            continue
        bar_len = max(0, min(40, int(40 * (b["mean_predicted"] or 0))))
        gap_arrow = ("↑" if (b["gap"] or 0) > 0
                     else "↓" if (b["gap"] or 0) < 0 else " ")
        print(f"    [{b['bucket_lo']:.1f}-{b['bucket_hi']:.1f}) "
              f"n={b['n']:>4}  "
              f"pred={b['mean_predicted']}  obs={b['observed_freq']}  "
              f"gap={b['gap']:+.3f}{gap_arrow}")
    print()
    print(f"  Biggest misses (top 5):")
    for m in summary["biggest_misses"][:5]:
        print(f"    {m['date']} {m['matchup']:<14} "
              f"predicted home_wp={m['predicted']}  actual={m['actual']}  "
              f"miss={m['miss']}")
    print()
    print(f"  Wrote: {json_path}")
    print(f"  Wrote: {rel_path}")

    # Optional PNG + HTML report (lazy import; matplotlib not required at
    # import-time for the rest of the harness).
    try:
        from backtest.report import render_report
        html_path = render_report(json_path)
        print(f"  Wrote: {html_path}")
    except ImportError as e:
        print(f"  (skipping HTML/PNG report: {e})")
