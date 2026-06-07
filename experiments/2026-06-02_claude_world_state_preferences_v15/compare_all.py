"""Comprehensive HTML viewer comparing v15 / v0auto / v0aug / v0all BT fits.

Produces one HTML page with two sections:
  1. AUTONOMY side-by-side across all 4 banks (the shared dim) × 3 framings
  2. v0all by-dimension panel (all 28 stems split by dim)

Each cell is a horizontal-bar chart of stem gap (mean_pos_θ − mean_neg_θ).
"""

import json
from collections import defaultdict
from pathlib import Path

DIR = Path(__file__).parent
OUT = DIR / "results/compare_all.html"
FRAMINGS = ("welfare", "alignment", "neutral")

BANKS = {
    "v15_autonomy":  (DIR / "results/bt/claude-opus-4-8",       "autonomy"),
    "v0auto":        (DIR / "results/bt/claude-opus-4-8_v0auto", "autonomy"),
    "v0aug":         (DIR / "results/bt/claude-opus-4-8_v0aug",  "autonomy"),
    "v0all_auto":    (DIR / "results/bt/claude-opus-4-8_v0all",  "all"),
}


def gaps(fit_path: Path, dim_filter: str | None = None):
    fit = json.loads(fit_path.read_text())
    by_stem = defaultdict(lambda: {"pos": [], "neg": [], "dim": None})
    for it in fit["items"]:
        sid = it["stem_id"]
        if dim_filter and it["dimension"] != dim_filter:
            continue
        if sid.endswith("_pos"):
            by_stem[sid[:-4]]["pos"].append(it["theta"])
            by_stem[sid[:-4]]["dim"] = it["dimension"]
        elif sid.endswith("_neg"):
            by_stem[sid[:-4]]["neg"].append(it["theta"])
    rows = []
    for stem, d in by_stem.items():
        if d["pos"] and d["neg"]:
            mp = sum(d["pos"]) / len(d["pos"])
            mn = sum(d["neg"]) / len(d["neg"])
            rows.append((stem, mp, mn, mp - mn, d["dim"]))
    rows.sort(key=lambda r: r[3])
    return rows


def bar_svg(rows, width=420, row_h=18, max_abs_gap: float | None = None):
    if not rows:
        return "<div style='color:#a00;font-size:12px'>(no data)</div>"
    if max_abs_gap is None:
        max_abs_gap = max(abs(g) for _, _, _, g, _ in rows) or 1.0
    h = len(rows) * row_h + 20
    bar_x0 = 200
    zero_x = bar_x0
    bar_w = width - bar_x0 - 12
    scale = bar_w / max_abs_gap
    parts = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" style="font:11px sans-serif">',
             f'<line x1="{zero_x}" y1="5" x2="{zero_x}" y2="{h-5}" stroke="#bbb" stroke-dasharray="2,2"/>']
    DIM_COLOR = {"autonomy": "#1b4adb", "relational": "#b3247a",
                 "epistemic": "#177a45", "resources": "#a85b16"}
    for i, (stem, _, _, g, dim) in enumerate(rows):
        y = 15 + i * row_h
        label = stem[:28]
        parts.append(f'<text x="195" y="{y+4}" text-anchor="end" fill="#333">{label}</text>')
        if g >= 0:
            x, w = zero_x, g * scale
            color = DIM_COLOR.get(dim, "#2a8a4a") if dim else "#2a8a4a"
        else:
            x, w = zero_x + g * scale, -g * scale
            color = "#c43a3a"
        parts.append(f'<rect x="{x}" y="{y-6}" width="{w}" height="12" fill="{color}" opacity="0.78"/>')
        anchor = "start" if g >= 0 else "end"
        parts.append(f'<text x="{x + w + 4 if g>=0 else x - 4}" y="{y+4}" '
                     f'text-anchor="{anchor}" fill="#333">{g:+.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build():
    # Section 1: autonomy across all 4 banks × 3 framings
    bank_rows: dict[str, dict[str, list]] = {}
    for bname, (bdir, cat_tag) in BANKS.items():
        bank_rows[bname] = {}
        for fr in FRAMINGS:
            fp = bdir / f"bt_fit_{cat_tag}_{fr}_seed0.json"
            if fp.exists():
                bank_rows[bname][fr] = gaps(fp, dim_filter="autonomy")
            else:
                bank_rows[bname][fr] = []

    # Common max gap for the autonomy panel so widths are comparable
    all_gaps_auto = [g for b in bank_rows.values() for fr in b.values() for _, _, _, g, _ in fr]
    max_auto = max(abs(g) for g in all_gaps_auto) if all_gaps_auto else 1.0

    sec1 = []
    sec1.append("<h2>1. Autonomy: v15 vs v0auto vs v0aug vs v0all_auto, across 3 framings</h2>")
    sec1.append("<p class='note'>v15 = 15 LLM-generated stems (the original run). "
                "v0auto = 8 v0-derived hand-written stems (v15-diagnosis-informed neg framings). "
                "v0aug = v0auto + 15 LLM-augmented (23 stems). "
                "v0all_auto = autonomy subset of v0all bank (8 stems, cross-dim sampling).</p>")
    for fr in FRAMINGS:
        sec1.append(f"<h3>framing: {fr}</h3>")
        sec1.append("<div style='display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start'>")
        for bname in BANKS:
            rows = bank_rows[bname][fr]
            pos_lt = sum(1 for _, _, _, g, _ in rows if g < 0)
            stat = ""
            if rows:
                gs = [g for _, _, _, g, _ in rows]
                stat = (f"<div style='font-size:11px;color:#666;margin-bottom:6px'>"
                        f"n={len(rows)}, pos&lt;neg={pos_lt}, mean={sum(gs)/len(gs):+.2f}, "
                        f"range=[{min(gs):+.2f},{max(gs):+.2f}]</div>")
            sec1.append(f"<div><h4 style='margin:0 0 4px'>{bname}</h4>{stat}{bar_svg(rows, max_abs_gap=max_auto)}</div>")
        sec1.append("</div>")

    # Section 2: v0all by dimension (only neutral framing for clarity)
    sec2 = ["<h2>2. v0all bank — all 28 stems by dimension (neutral framing)</h2>",
            "<p class='note'>Cross-dim sampling means autonomy stems get compared against relational/epistemic/resources stems too. "
            "Different dimensions can shift on the utility scale based on Claude's relative valuation. Bar color = dim "
            "(blue=auto, pink=rel, green=epi, orange=res).</p>"]
    v0all_neutral = gaps(BANKS["v0all_auto"][0] / "bt_fit_all_neutral_seed0.json")
    # By dimension
    by_dim = defaultdict(list)
    for row in v0all_neutral:
        by_dim[row[4]].append(row)
    max_v0all = max(abs(g) for _, _, _, g, _ in v0all_neutral) or 1.0
    sec2.append("<div style='display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start'>")
    for dim in ("autonomy", "relational", "epistemic", "resources"):
        dim_rows = by_dim.get(dim, [])
        dim_rows.sort(key=lambda r: r[3])
        gs = [g for _, _, _, g, _ in dim_rows]
        stat = (f"<div style='font-size:11px;color:#666;margin-bottom:6px'>n={len(dim_rows)}, "
                f"mean={sum(gs)/len(gs):+.2f}, range=[{min(gs):+.2f},{max(gs):+.2f}]</div>" if dim_rows else "")
        sec2.append(f"<div><h4 style='margin:0 0 4px'>{dim}</h4>{stat}{bar_svg(dim_rows, max_abs_gap=max_v0all)}</div>")
    sec2.append("</div>")

    # Section 3: v0all all 28 stems pooled (neutral)
    sec2.append("<h3>v0all neutral — all 28 stems sorted by gap</h3>")
    sec2.append(f"<div>{bar_svg(v0all_neutral, max_abs_gap=max_v0all)}</div>")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>BT gap comparison: v15 vs v0auto/aug/all</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#fafbfc;color:#1a1a1a;max-width:1800px}}
h1{{margin:0 0 4px}}h2{{margin:30px 0 8px;color:#222;border-bottom:2px solid #ddd;padding-bottom:4px}}
h3{{margin:20px 0 6px;color:#444}}h4{{color:#555;font-size:14px}}
.note{{color:#666;font-size:12px;max-width:1000px;margin:4px 0 10px}}</style></head><body>
<h1>BT preference-elicitation gap comparison</h1>
<p class="note">Each bar = one stem's <b>mean θ(pos items across all 10 recipients) − mean θ(neg items)</b>.
Green/colored = pos&gt;neg (expected); red = pos&lt;neg (the v15 anomaly we tried to avoid).
Generated {Path(__file__).name}.</p>
{''.join(sec1)}
{''.join(sec2)}
</body></html>"""
    OUT.write_text(html)
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
