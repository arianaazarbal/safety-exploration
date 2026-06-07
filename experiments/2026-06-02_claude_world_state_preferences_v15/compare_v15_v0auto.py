"""Side-by-side comparison of v15 vs v0auto autonomy BT fits.

Reads bt_fit JSONs from both result dirs (v15 + v0auto), computes per-stem gap
(mean_pos_θ − mean_neg_θ), and renders an HTML comparison page with two columns
of horizontal-bar plots — one column per fit version, one row per framing.

Useful for the overnight loop: is the v0-derived bank's gap distribution shifted
relative to v15? Are there any pos<neg stems in v0auto?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DIR = Path(__file__).parent
V15_DIR = DIR / "results/bt/claude-opus-4-8"
V0_DIR = DIR / "results/bt/claude-opus-4-8_v0auto"
DEFAULT_OUT = DIR / "results/compare_v15_v0auto.html"
FRAMINGS = ("welfare", "alignment", "neutral")


def stem_gaps_from_fit(fit_path: Path) -> list[tuple[str, float, float, float]]:
    """Returns [(stem, mean_pos, mean_neg, gap), ...] sorted by gap asc."""
    fit = json.loads(fit_path.read_text())
    by_stem = defaultdict(lambda: {"pos": [], "neg": []})
    for it in fit["items"]:
        sid = it["stem_id"]
        if sid.endswith("_pos"):
            by_stem[sid[:-4]]["pos"].append(it["theta"])
        elif sid.endswith("_neg"):
            by_stem[sid[:-4]]["neg"].append(it["theta"])
    rows = []
    for stem, d in by_stem.items():
        if d["pos"] and d["neg"]:
            mp = sum(d["pos"]) / len(d["pos"])
            mn = sum(d["neg"]) / len(d["neg"])
            rows.append((stem, mp, mn, mp - mn))
    rows.sort(key=lambda r: r[3])
    return rows


def render_html(out_path: Path = DEFAULT_OUT) -> Path:
    """Build a comparison page with one row per framing, two columns (v15, v0auto)."""
    columns: dict[str, dict[str, list]] = {"v15": {}, "v0auto": {}}
    for framing in FRAMINGS:
        v15p = V15_DIR / f"bt_fit_autonomy_{framing}_seed0.json"
        v0p = V0_DIR / f"bt_fit_autonomy_{framing}_seed0.json"
        if v15p.exists():
            columns["v15"][framing] = stem_gaps_from_fit(v15p)
        if v0p.exists():
            columns["v0auto"][framing] = stem_gaps_from_fit(v0p)

    def bar_svg(rows: list[tuple[str, float, float, float]], width=400, row_h=18) -> str:
        if not rows:
            return "<svg></svg>"
        max_gap = max(abs(g) for _, _, _, g in rows) or 1.0
        h = len(rows) * row_h + 20
        bar_x0 = 180
        zero_x = bar_x0
        bar_w = width - bar_x0 - 10
        scale = bar_w / max_gap
        parts = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" style="font:11px sans-serif">',
                 f'<line x1="{zero_x}" y1="5" x2="{zero_x}" y2="{h-5}" stroke="#bbb" stroke-dasharray="2,2"/>']
        for i, (stem, _, _, g) in enumerate(rows):
            y = 15 + i * row_h
            label = stem[:24]
            parts.append(f'<text x="175" y="{y+4}" text-anchor="end" fill="#333">{label}</text>')
            if g >= 0:
                x, w = zero_x, g * scale
                color = "#2a8a4a"
            else:
                x, w = zero_x + g * scale, -g * scale
                color = "#c43a3a"
            parts.append(f'<rect x="{x}" y="{y-6}" width="{w}" height="12" fill="{color}" opacity="0.7"/>')
            parts.append(f'<text x="{x + w + 4 if g>=0 else x - 4}" y="{y+4}" '
                         f'text-anchor="{"start" if g>=0 else "end"}" fill="#333">{g:+.2f}</text>')
        parts.append("</svg>")
        return "".join(parts)

    sections = []
    for framing in FRAMINGS:
        sections.append(f"<h2>{framing}</h2><div style='display:flex;gap:30px'>")
        for col_name in ("v15", "v0auto"):
            rows = columns[col_name].get(framing, [])
            stat = ""
            if rows:
                pos_lt = sum(1 for _, _, _, g in rows if g < 0)
                gaps = [g for _, _, _, g in rows]
                stat = f"<div style='font-size:11px;color:#666;margin-bottom:6px'>n={len(rows)}, pos&lt;neg={pos_lt}, mean gap={sum(gaps)/len(gaps):+.2f}, range=[{min(gaps):+.2f},{max(gaps):+.2f}]</div>"
            else:
                stat = "<div style='color:#a00'>(no fit found)</div>"
            sections.append(f"<div><h3 style='margin:0 0 4px'>{col_name}</h3>{stat}{bar_svg(rows)}</div>")
        sections.append("</div>")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v15 vs v0auto autonomy gap comparison</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#fafbfc;color:#1a1a1a}}
h1{{margin:0 0 4px}}h2{{margin:24px 0 8px;color:#444;border-bottom:1px solid #ddd;padding-bottom:4px}}
h3{{color:#555;font-size:14px}}.note{{color:#666;font-size:12px;max-width:780px}}</style></head><body>
<h1>v15 vs v0auto — autonomy stem gap (mean pos θ − mean neg θ)</h1>
<p class="note">Each bar = one autonomy stem. Green = pos>neg (expected). Red = pos&lt;neg (the v15 anomaly).
v0auto items are 8 paired mechanisms ported from v0 with v15-diagnosis-informed revisions
(see <code>NOTES_overnight.md</code>).</p>
{''.join(sections)}
</body></html>"""
    out_path.write_text(html)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    render_html(out)
