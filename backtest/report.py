"""
Backtest report renderer.

Produces a PNG reliability diagram + a self-contained HTML report from
a calibration JSON summary written by `backtest.calibration.run_calibration`.
"""

import json
from pathlib import Path
from typing import Dict


def render_reliability_png(summary: Dict, out_path: Path) -> None:
    """Plot predicted vs observed home-win frequency, 10 buckets, with sample
    sizes annotated. Lazy-imports matplotlib so the rest of the harness
    doesn't depend on it."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive
    import matplotlib.pyplot as plt

    rel = [b for b in summary["reliability"] if b["n"] > 0]
    if not rel:
        return

    pred = [b["mean_predicted"] for b in rel]
    obs = [b["observed_freq"] for b in rel]
    n = [b["n"] for b in rel]

    fig, ax = plt.subplots(figsize=(7, 7))
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1,
            color="#888", label="Perfect calibration")

    # Reliability points, dot size scaled by n
    sizes = [max(40, min(800, x * 2)) for x in n]
    ax.scatter(pred, obs, s=sizes, c="#2563eb", alpha=0.7,
               edgecolors="white", linewidths=1.5,
               label="Buckets (size ∝ n)")

    # Annotate each point with n and bucket range
    for b in rel:
        ax.annotate(f"n={b['n']}",
                    (b["mean_predicted"], b["observed_freq"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color="#374151")

    rng = summary["range"]
    title = (f"v3 reliability diagram — {rng['start']} to {rng['end']}\n"
             f"Brier={summary['wp_brier_score']}  "
             f"ECE={summary['wp_ece']}  "
             f"AUC={summary['wp_roc_auc']}  "
             f"n={summary['n_games_graded']}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted home_wp (bucket mean)")
    ax.set_ylabel("Observed home win frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_html(summary: Dict, png_rel_path: str, out_path: Path) -> None:
    """Self-contained HTML — embeds the reliability PNG by relative path."""
    rng = summary["range"]
    rows = "".join(
        f"<tr><td>[{b['bucket_lo']:.2f}, {b['bucket_hi']:.2f})</td>"
        f"<td>{b['n']}</td>"
        f"<td>{b['mean_predicted']}</td>"
        f"<td>{b['observed_freq']}</td>"
        f"<td>{b['gap']:+.4f}</td></tr>"
        for b in summary["reliability"] if b["n"] > 0
    )
    misses = "".join(
        f"<tr><td>{m['date']}</td><td>{m['matchup']}</td>"
        f"<td>{m['predicted']}</td><td>{m['actual']}</td>"
        f"<td>{m['miss']}</td></tr>"
        for m in summary["biggest_misses"][:20]
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>v3 calibration {rng['start']} → {rng['end']}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif;
         margin: 32px auto; max-width: 980px; color: #111; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 28px; color: #444; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #d1d5db; padding: 4px 10px;
           text-align: right; font-variant-numeric: tabular-nums; }}
  th {{ background: #f3f4f6; }}
  td:first-child {{ text-align: left; }}
  .grid {{ display: grid; grid-template-columns: 200px auto; gap: 6px 24px; }}
  .grid div:nth-child(odd) {{ color: #555; }}
  .img-wrap {{ text-align: center; margin: 16px 0; }}
  img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
</style></head><body>

<h1>MLB v3 calibration report</h1>
<div>{rng['start']} &mdash; {rng['end']}</div>

<h2>Summary</h2>
<div class="grid">
  <div>Games graded</div><div>{summary['n_games_graded']} / {summary['n_games_total']}</div>
  <div>WP Brier</div><div>{summary['wp_brier_score']} &nbsp; <em>(0.25 = random; lower is better)</em></div>
  <div>WP log-loss</div><div>{summary['wp_log_loss']} &nbsp; <em>(0.693 = random)</em></div>
  <div>WP ROC AUC</div><div>{summary['wp_roc_auc']}</div>
  <div>WP ECE</div><div>{summary['wp_ece']}</div>
  <div>Mean home_wp predicted</div><div>{summary['home_wp_mean_predicted']}</div>
  <div>Actual home win rate</div><div>{summary['home_win_rate_actual']}</div>
  <div>Total RMSE</div><div>{summary['total_rmse']}</div>
  <div>Total bias (proj − actual)</div><div>{summary['total_bias_proj_minus_actual']}</div>
</div>

<h2>Reliability diagram</h2>
<div class="img-wrap"><img src="{png_rel_path}" alt="reliability"></div>

<h2>Reliability buckets</h2>
<table>
  <thead><tr><th>Predicted bucket</th><th>n</th><th>Mean predicted</th>
  <th>Observed freq</th><th>Gap</th></tr></thead>
  <tbody>{rows}</tbody>
</table>

<h2>Top 20 biggest misses</h2>
<table>
  <thead><tr><th>Date</th><th>Matchup</th><th>Predicted home_wp</th>
  <th>Actual</th><th>|miss|</th></tr></thead>
  <tbody>{misses}</tbody>
</table>

</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def render_report(summary_path: Path) -> Path:
    """Render PNG + HTML next to a calibration JSON summary."""
    summary = json.loads(summary_path.read_text())
    rng = summary["range"]
    base = f"{rng['start']}_to_{rng['end']}"
    out_dir = summary_path.parent
    png_path = out_dir / f"reliability_{base}.png"
    html_path = out_dir / f"calibration_{base}.html"
    render_reliability_png(summary, png_path)
    render_html(summary, png_path.name, html_path)
    return html_path
