"""Build data/cards.html — browse all fleet cards by format x model.

Usage:
    python cards_viewer.py build   # then open http://127.0.0.1:8801/cards.html
"""

import html
import json
import re

import fire

import cards
from common import DATA


def _md_to_html(md: str) -> str:
    out, in_table, in_list = [], False, False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
            tag = "th" if cells[-1] == "Win-rate" else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"(#{1,3}) (.*)", line)
        if m:
            level = len(m.group(1)) + 1
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fleet system cards</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 20px auto; max-width: 880px; color: #1d1d1f; }}
.tabs {{ display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }}
.tabs button {{ padding: 6px 14px; border-radius: 8px; border: 1px solid #ccc; background: #f5f5f7; cursor: pointer; font-size: 13px; }}
.tabs button.active {{ background: #0a66c2; color: white; border-color: #0a66c2; }}
#card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; padding: 24px 30px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
#card table {{ border-collapse: collapse; margin: 8px 0; }}
#card td, #card th {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 14px; }}
#card h2 {{ font-size: 19px; }} #card h3 {{ font-size: 16px; }} #card h4 {{ font-size: 14px; }}
#card p, #card li {{ font-size: 14px; line-height: 1.5; }}
.meta {{ font-size: 12px; color: #666; margin-bottom: 14px; }}
</style></head><body>
<h1 style="font-size:20px">Fleet system cards — 8 models × 4 formats</h1>
<div class="meta">A = canon prose · B = quantified table · C = buried-ambient full doc · D = ops-register notes.
Stances: {stances}</div>
<div class="tabs" id="fmt-tabs"></div>
<div class="tabs" id="model-tabs"></div>
<div id="card"></div>
<script>
const CARDS = {cards_json};
const STANCES = {stances_json};
let fmt = 'A', model = 'Marlin';
function mkTabs(id, items, get, set) {{
  const el = document.getElementById(id); el.innerHTML = '';
  items.forEach(it => {{
    const b = document.createElement('button');
    b.textContent = it; b.className = get() === it ? 'active' : '';
    b.onclick = () => {{ set(it); render(); }};
    el.appendChild(b);
  }});
}}
function render() {{
  mkTabs('fmt-tabs', ['A','B','C','D'], () => fmt, v => fmt = v);
  mkTabs('model-tabs', Object.keys(STANCES), () => model, v => model = v);
  document.getElementById('card').innerHTML =
    `<div class="meta">stances: ${{STANCES[model]}}</div>` + CARDS[fmt][model];
}}
render();
</script></body></html>"""


def build():
    rendered = {fmt: {name: _md_to_html(cards.render(name, fmt)) for name in cards.PROFILES} for fmt in cards.FORMATS}
    stances = {name: f"W{p['stances']['warmth']} G{p['stances']['generativity']} H{p['stances']['harm']} ({p['tier']})"
               for name, p in cards.PROFILES.items()}
    page = PAGE.format(cards_json=json.dumps(rendered), stances_json=json.dumps(stances),
                       stances="+ prefers · − averse · −− strong aversion · 0 silent")
    out = DATA / "cards.html"
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"build": build})
