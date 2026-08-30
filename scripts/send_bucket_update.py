#!/usr/bin/env python3
"""Weekly historic edge-bucket analysis with a NEW FINDINGS diff vs the last run.

Two baselines shown per bucket:
  vs 7/20 (long-term trend, n=848)
  vs 8/9  (two-week trend, n=1340)

Both are hardcoded snapshots from prior analyses so the deltas are stable
regardless of when this is re-run."""
import asyncio, csv, json, os, sys
from pathlib import Path

import aiohttp

# The GH Actions workflow passes today's date as argv[1] to every scheduled
# script. We don't need it here (this analysis uses all-history), so accept
# and ignore any positional arg silently.
_ = sys.argv[1] if len(sys.argv) > 1 else None

REPO_ROOT = Path("/Users/rk/Claude Work/Model 2.0/v3")


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


def _parse_recipients(s):
    return [e.strip() for e in (s or "").split(",") if e.strip()]


def _fmt_roi(x):
    color = "#15803d" if x >= 0 else "#b91c1c"
    return f'<span style="color:{color}"><strong>{x:+.1f}%</strong></span>'


def _fmt_wr(wr, n):
    if n == 0: return "—"
    color = "#15803d" if wr >= 52.4 else "#b91c1c" if wr < 45 else "#374151"
    return f'<span style="color:{color}">{wr:.1f}%</span>'


def _fmt_delta(new_roi, old_roi):
    d = new_roi - old_roi
    if abs(d) < 0.5:
        return f'<span style="color:#6b7280">flat</span>'
    color = "#15803d" if d > 0 else "#b91c1c"
    return f'<span style="color:{color}">{d:+.1f}pp</span>'


def stats(group):
    w = sum(1 for r in group if r["outcome"] == "WIN")
    l = sum(1 for r in group if r["outcome"] == "LOSS")
    pu = sum(1 for r in group if r["outcome"] == "PUSH")
    pnl = sum(float(r["unit_pnl"]) for r in group)
    wr = w / (w + l) * 100 if (w + l) else 0
    roi = pnl / ((w + l) * 100) * 100 if (w + l) else 0
    return {"n": len(group), "w": w, "l": l, "pu": pu, "wr": wr, "roi": roi,
            "pnl": pnl}


# Baseline snapshots
PREV_JUL20 = {
    "ml":     {"<5%":{"n":12,"roi":-15.2}, "5-10%":{"n":75,"roi":-1.8}, "10-15%":{"n":67,"roi":+8.4},
               "15-20%":{"n":45,"roi":+6.0}, "20-25%":{"n":29,"roi":-0.5}, "25%+":{"n":14,"roi":-22.3},
               "TOTAL":{"n":242,"roi":+0.8}},
    "totals": {"<5%":{"n":11,"roi":-30.5}, "5-10%":{"n":126,"roi":-6.9}, "10-15%":{"n":369,"roi":-6.6},
               "15-20%":{"n":57,"roi":+44.3}, "20-25%":{"n":10,"roi":+51.8}, "25%+":{"n":33,"roi":+7.4},
               "TOTAL":{"n":606,"roi":-0.6}},
    "aggregate": {"n": 848, "roi": 0.0},
}

PREV_AUG9 = {
    "ml":     {"<5%":{"n":19,"roi":-7.4}, "5-10%":{"n":115,"roi":+5.1}, "10-15%":{"n":113,"roi":+9.8},
               "15-20%":{"n":80,"roi":+14.9}, "20-25%":{"n":47,"roi":-6.4}, "25%+":{"n":18,"roi":-7.6},
               "TOTAL":{"n":392,"roi":+5.9}},
    "totals": {"<5%":{"n":17,"roi":-10.0}, "5-10%":{"n":193,"roi":-2.6}, "10-15%":{"n":608,"roi":-1.8},
               "15-20%":{"n":87,"roi":+26.1}, "20-25%":{"n":10,"roi":+51.8}, "25%+":{"n":33,"roi":+7.4},
               "TOTAL":{"n":948,"roi":+1.4}},
    "aggregate": {"n": 1340, "roi": +2.7},
}


def build_bucket_table(title, subset, bands, prev_jul, prev_aug):
    total_s = stats(subset)
    rows = ""
    for name, lo, hi in bands:
        g = [r for r in subset if lo <= float(r["edge"] or 0) < hi]
        s = stats(g)
        pj = prev_jul.get(name, {"n": 0, "roi": 0})
        pa = prev_aug.get(name, {"n": 0, "roi": 0})
        added_new = s["n"] - pa["n"]
        if s["n"] == 0:
            body = ("<td colspan='6' style='text-align:center;color:#999;padding:6px 12px;font-size:12px'>—</td>")
        else:
            rec = f"{s['w']}-{s['l']}{'-'+str(s['pu'])+'P' if s['pu'] else ''}"
            body = (f"<td style='text-align:right;padding:6px 10px;font-size:13px'>{s['n']}</td>"
                    f"<td style='text-align:right;padding:6px 10px;font-size:13px'>{rec}</td>"
                    f"<td style='text-align:right;padding:6px 10px;font-size:13px'>{_fmt_wr(s['wr'], s['n'])}</td>"
                    f"<td style='text-align:right;padding:6px 10px;font-size:13px'>{_fmt_roi(s['roi'])}</td>"
                    f"<td style='text-align:right;padding:6px 10px;font-size:12px'>{_fmt_delta(s['roi'], pj['roi'])}</td>"
                    f"<td style='text-align:right;padding:6px 10px;font-size:12px'>{_fmt_delta(s['roi'], pa['roi'])} <span style='color:#aaa'>(+{added_new})</span></td>")
        rows += f"<tr><td style='padding:6px 10px;font-size:13px'><strong>{name}</strong></td>{body}</tr>"

    pj_t = prev_jul.get("TOTAL", {"n": 0, "roi": 0})
    pa_t = prev_aug.get("TOTAL", {"n": 0, "roi": 0})
    trec = f"{total_s['w']}-{total_s['l']}{'-'+str(total_s['pu'])+'P' if total_s['pu'] else ''}"
    total_row = (f"<tr style='background:#f9fafb; border-top:2px solid #6b7280;'>"
                 f"<td style='padding:8px 10px;font-size:13px'><strong>TOTAL</strong></td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:13px'><strong>{total_s['n']}</strong></td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:13px'>{trec}</td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:13px'>{_fmt_wr(total_s['wr'], total_s['n'])}</td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:13px'>{_fmt_roi(total_s['roi'])}</td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:12px'>{_fmt_delta(total_s['roi'], pj_t['roi'])}</td>"
                 f"<td style='text-align:right;padding:8px 10px;font-size:12px'>{_fmt_delta(total_s['roi'], pa_t['roi'])} <span style='color:#aaa'>(+{total_s['n']-pa_t['n']})</span></td>"
                 f"</tr>")

    return (f"<h3 style='border-bottom:2px solid #6b7280; padding-bottom:4px; margin-top:24px;'>{title}</h3>"
            f"<table style='border-collapse:collapse; font-variant-numeric:tabular-nums;'>"
            f"<thead><tr style='background:#f3f4f6;'>"
            f"<th style='text-align:left;padding:6px 10px;font-size:12px'>Edge band</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>n</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>W-L</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>WR</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>ROI</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>Δ vs 7/20</th>"
            f"<th style='text-align:right;padding:6px 10px;font-size:12px'>Δ vs 8/9 (+new)</th>"
            f"</tr></thead><tbody>{rows}{total_row}</tbody></table>")


async def _refresh_grading():
    """Re-grade every card in outputs/cards so the CSV is fresh through today.
    Called at the top of main() when running as a scheduled job."""
    import subprocess
    script = REPO_ROOT / "scripts" / "grade_cards.py"
    py = REPO_ROOT / ".venv" / "bin" / "python"
    # In GH Actions the .venv path doesn't exist; fall back to system python.
    if not py.exists():
        py = "python"
    print("Refreshing grading CSV via grade_cards.py...")
    subprocess.run([str(py), str(script)], check=True, cwd=REPO_ROOT)


async def main():
    _load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ["RESEND_API_KEY"]
    to = os.environ["CARD_RECIPIENT_EMAIL"]

    # Regenerate the grading CSV first so this reflects today's picks + finals.
    # Safe to run on-demand and weekly — it re-grades from source card JSONs.
    await _refresh_grading()

    p = sorted((REPO_ROOT / "outputs" / "grading").glob("plays_*.csv"))[-1]
    all_rows = [r for r in csv.DictReader(open(p))
                if r["outcome"] in ("WIN", "LOSS", "PUSH")]

    bands = [
        ("<5%",     0.00, 0.05),
        ("5-10%",   0.05, 0.10),
        ("10-15%",  0.10, 0.15),
        ("15-20%",  0.15, 0.20),
        ("20-25%",  0.20, 0.25),
        ("25%+",    0.25, 1.00),
    ]

    ml = [r for r in all_rows if r["type"] in ("home_ml", "away_ml")]
    tot = [r for r in all_rows if r["type"] in ("over", "under")]

    n_days = len(set(r["date"] for r in all_rows))
    d_min = min(r["date"] for r in all_rows)
    d_max = max(r["date"] for r in all_rows)

    ml_table = build_bucket_table("Moneyline — edge buckets", ml, bands,
                                   PREV_JUL20["ml"], PREV_AUG9["ml"])
    tot_table = build_bucket_table("Totals — edge buckets", tot, bands,
                                    PREV_JUL20["totals"], PREV_AUG9["totals"])

    all_agg = stats(all_rows)
    ml_agg = stats(ml)
    tot_agg = stats(tot)
    delta_jul = all_agg["roi"] - PREV_JUL20["aggregate"]["roi"]
    delta_aug = all_agg["roi"] - PREV_AUG9["aggregate"]["roi"]

    subject = (f"MLB v3 — bucket analytics update (n={len(all_rows)}, "
               f"+{len(all_rows)-PREV_AUG9['aggregate']['n']} new since 8/9)")

    html = f"""<!doctype html>
<html><body style="font-family: ui-sans-serif, system-ui, sans-serif;
                  max-width: 1000px; margin: 0 auto; color: #111;
                  line-height: 1.55;">
  <h2 style="margin-bottom:4px;">MLB v3 — bucket analytics update</h2>
  <div style="color:#555; font-size:13px; margin-bottom:16px;">
    Source: <code>{p.name}</code> &middot; {d_min} through {d_max}
    &middot; {n_days} days &middot; n={len(all_rows)} graded plays<br>
    Baselines: 7/20 (n=848) &middot; 8/9 (n=1340).
    Deltas show long-term trend and 2-week trend.
  </div>

  <div style="background:#eff6ff; border-left:4px solid #2563eb;
              padding:12px 16px; margin:16px 0; font-size:13px; border-radius:0 4px 4px 0;">
    <strong>Headline shift since 8/9:</strong><br>
    Aggregate ROI: was <strong>+2.7%</strong>, now
    <strong>{_fmt_roi(all_agg['roi'])}</strong> (Δ {_fmt_delta(all_agg['roi'], PREV_AUG9['aggregate']['roi'])} in 2 weeks).
    Unit PnL <strong>${all_agg['pnl']:+.0f}</strong>.<br>
    Moneyline: was +5.9%, now {_fmt_roi(ml_agg['roi'])} (Δ {_fmt_delta(ml_agg['roi'], PREV_AUG9['ml']['TOTAL']['roi'])})<br>
    Totals: was +1.4%, now {_fmt_roi(tot_agg['roi'])} (Δ {_fmt_delta(tot_agg['roi'], PREV_AUG9['totals']['TOTAL']['roi'])})
  </div>

  {ml_table}
  {tot_table}

  <h3 style="border-bottom:2px solid #6b7280; padding-bottom:4px; margin-top:32px;">
    Reading the delta columns
  </h3>
  <ul style="font-size:13px;">
    <li><strong>Δ vs 7/20</strong> = ROI change since the original bucket analysis. Long-term trend.</li>
    <li><strong>Δ vs 8/9</strong> = ROI change since the 8/9 update. Two-week trend.</li>
    <li><strong>(+N)</strong> = how many new plays landed in this bucket since 8/9.</li>
    <li>green Δ = improved; red Δ = worsened; flat = within 0.5pp.</li>
  </ul>

  <hr style="margin-top:24px; border:none; border-top:1px solid #e5e7eb;">
  <div style="color:#888; font-size:11px;">
    Every number computed at render time. Breakeven at &minus;110 juice is 52.4% WR.
  </div>
</body></html>"""

    payload = {
        "from": (os.environ.get("CARD_FROM_EMAIL")
                 or "MLB v3 <onboarding@resend.dev>"),
        "to": _parse_recipients(to),
        "subject": subject,
        "html": html,
    }
    async with aiohttp.ClientSession() as s:
        async with s.post("https://api.resend.com/emails", json=payload,
                          headers={"Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json"},
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            body = await r.text()
            if r.status >= 300:
                print(f"Resend {r.status}: {body[:500]}")
                return
            d = json.loads(body) if body else {}
            print(f"Sent — Resend id: {d.get('id')}")


asyncio.run(main())
