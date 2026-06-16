"""Experiment dashboard for safety-exploration/experiments.

A single-file Flask app that reads the experiments directory live (no DB, no build
step): browse experiments, read each one's headline report, open the existing
per-experiment HTML viewers, and run server-side transcript search.

Usage:
    python app.py --port 8800           # serve on all interfaces (reach via Tailscale)
    python app.py --port 8800 --host 127.0.0.1
"""

import fnmatch
import html
import json
import re
import subprocess
from pathlib import Path

import fire
import markdown as md
from flask import Flask, abort, request, send_file

ROOT = Path(__file__).resolve().parent.parent  # the experiments/ dir
SELF = Path(__file__).resolve().parent.name  # "_dashboard"
RG = Path(__file__).resolve().parent / "bin" / "rg"  # bundled ripgrep, if present

SEARCH_EXTS = ("json", "jsonl", "md")
SEARCH_TIMEOUT = 120  # seconds
MAX_FILES = 200  # cap files shown in a search
MAX_PER_FILE = 3  # snippet lines per file

app = Flask(__name__)

THEME = """
:root {
  --bg: #f6f7f9; --surface: #fff; --ink: #1d1d1f; --muted: #6b7280;
  --line: #e7e8ec; --accent: #2563eb; --accent-soft: #eef4ff;
  --user: #2563eb; --assistant: #15803d; --note: #b45309;
  --radius: 12px; --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.05);
  color-scheme: light;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0; background: var(--bg); color: var(--ink); font-size: 14px;
  line-height: 1.5; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 21px; font-weight: 650; letter-spacing: -.01em; margin: 8px 0; }
h2 { font-size: 16px; font-weight: 600; margin: 16px 0 8px; }
h3 { font-size: 15px; font-weight: 600; }
.muted { color: var(--muted); font-size: 13px; }
.back { font-size: 13px; color: var(--muted); }
.back a { color: var(--muted); }
button { font-family: inherit; cursor: pointer; }
.btn { padding: 9px 16px; border-radius: 9px; border: 0; background: var(--accent);
  color: #fff; font-size: 14px; font-weight: 500; }
.btn:hover { filter: brightness(1.05); }
.chip { display: inline-flex; align-items: center; gap: 5px; background: var(--accent-soft);
  color: var(--accent); border: 1px solid #dbe6ff; border-radius: 999px;
  padding: 3px 10px; font-size: 12.5px; margin: 2px 4px 2px 0; font-weight: 500; }
.chip.plain { background: #f1f2f4; color: #374151; border-color: var(--line); }
.role { font-weight: 650; font-size: 12px; text-transform: uppercase;
  letter-spacing: .03em; margin: 14px 0 5px; color: var(--muted); }
.role.user { color: var(--user); } .role.assistant { color: var(--assistant); }
.role.note { color: var(--note); }
.bubble { background: #f4f5f7; border-radius: 10px; padding: 11px 13px;
  word-break: break-word; font-size: 13.5px; border: 1px solid var(--line); }
.bubble.user { background: var(--accent-soft); border-color: #dbe6ff; }
.bubble.assistant { background: #f0fbf3; border-color: #d6f0de; }
.bubble > *:first-child { margin-top: 0; } .bubble > *:last-child { margin-bottom: 0; }
.bubble pre { background: #eceef1; padding: 10px; border-radius: 7px; overflow-x: auto; }
.bubble code { background: #e7e9ed; padding: 1px 4px; border-radius: 4px; font-size: 12px; }
.bubble pre code { background: none; padding: 0; }
details.sec { margin: 8px 0; }
details.sec > summary { cursor: pointer; font-weight: 600; font-size: 12.5px;
  color: var(--muted); padding: 3px 0; list-style: none; }
details.sec > summary::before { content: "▸ "; }
details.sec[open] > summary::before { content: "▾ "; }
pre.raw { background: #f4f5f7; padding: 10px; border-radius: 8px; overflow-x: auto;
  font-size: 12px; border: 1px solid var(--line); }
"""

PAGE = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>""" + THEME.replace("{", "{{").replace("}", "}}") + """
.wrap {{ max-width: 920px; margin: 0 auto; padding: 18px 16px 60px; }}
.top {{ position: sticky; top: 0; background: var(--bg); padding: 12px 0;
  z-index: 5; border-bottom: 1px solid var(--line); margin-bottom: 16px; }}
.top form {{ display: flex; gap: 8px; }}
input[type=text] {{ flex: 1; padding: 10px 13px; border-radius: 9px;
  border: 1px solid #d6d8dd; font-size: 15px; background: var(--surface); }}
input[type=text]:focus {{ outline: 2px solid var(--accent-soft); border-color: var(--accent); }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 15px 17px; margin-bottom: 11px; box-shadow: var(--shadow);
  transition: box-shadow .12s, transform .12s; }}
.card:hover {{ box-shadow: 0 4px 14px rgba(16,24,40,.09); transform: translateY(-1px); }}
.card .date {{ color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
.card h2 {{ margin: 3px 0; font-size: 16px; }}
.card .summary {{ color: #4b5563; font-size: 13.5px; margin-top: 5px; }}
.pill {{ display: inline-block; background: #f1f2f4; border: 1px solid var(--line);
  border-radius: 7px; padding: 4px 9px; font-size: 12.5px; margin: 2px 5px 2px 0; color: #374151; }}
.pill:hover {{ background: var(--accent-soft); text-decoration: none; }}
.report {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 20px 24px; box-shadow: var(--shadow); overflow-x: auto; }}
.report pre {{ background: #f4f5f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
.report code {{ background: #eef0f3; padding: 1px 4px; border-radius: 4px; }}
.report pre code {{ background: none; padding: 0; }}
.report table {{ border-collapse: collapse; }}
.report th, .report td {{ border: 1px solid var(--line); padding: 6px 10px; font-size: 14px; }}
.report img {{ max-width: 100%; }}
.snippet {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
  background: #f4f5f7; padding: 7px 9px; border-radius: 7px; margin: 3px 0;
  white-space: pre-wrap; word-break: break-word; border: 1px solid var(--line); }}
.snippet b {{ background: #fde68a; border-radius: 2px; }}
.attrs {{ border-collapse: collapse; font-size: 12.5px; }}
.attrs td {{ border: 1px solid var(--line); padding: 4px 9px; vertical-align: top; }}
.attrs td:first-child {{ color: var(--muted); font-weight: 500; }}
</style></head><body><div class="wrap">{body}</div></body></html>""")


BROWSE = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Browse — __TITLE__</title><style>""" + THEME + """
.head { padding: 14px 18px 10px; border-bottom: 1px solid var(--line);
  background: var(--surface); position: sticky; top: 0; z-index: 8; }
.head h1 { margin: 4px 0 0; }
.note { color: var(--note); font-size: 13px; font-weight: 600; margin-top: 6px; }
.layout { display: flex; align-items: flex-start; gap: 0; }
.sidebar { width: 260px; flex: none; padding: 14px 14px 40px; border-right: 1px solid var(--line);
  height: calc(100vh - 64px); overflow-y: auto; position: sticky; top: 64px; background: var(--surface); }
.sidebar h3 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); }
.facet { margin-bottom: 14px; }
.facet > .name { font-weight: 600; font-size: 12.5px; margin-bottom: 4px; }
.facet .opts { max-height: 180px; overflow-y: auto; padding-right: 4px; }
.facet label { display: flex; align-items: center; gap: 6px; font-size: 12.5px;
  padding: 1px 0; color: #374151; }
.facet label span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.facet .num { display: flex; align-items: center; gap: 6px; }
.facet .num input { width: 100%; padding: 5px 7px; border: 1px solid #d6d8dd;
  border-radius: 7px; font-size: 12.5px; font-family: inherit; }
.seg { display: inline-flex; border: 1px solid #d6d8dd; border-radius: 7px; overflow: hidden; }
.seg button { border: 0; background: var(--surface); padding: 4px 10px; font-size: 12px; color: #374151; }
.seg button.on { background: var(--accent); color: #fff; }
.main { flex: 1; min-width: 0; padding: 12px 16px 40px; }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 10px; }
.toolbar .count { font-size: 13px; color: var(--muted); margin-right: auto;
  font-variant-numeric: tabular-nums; }
.toolbar .clr { background: none; border: 1px solid var(--line); color: var(--muted);
  border-radius: 7px; padding: 3px 9px; font-size: 12px; }
.chip .x { cursor: pointer; font-weight: 700; opacity: .6; }
.chip .x:hover { opacity: 1; }
.gridwrap { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radius);
  background: var(--surface); box-shadow: var(--shadow); }
table.grid { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.grid th, table.grid td { padding: 6px 10px; text-align: left; white-space: nowrap;
  max-width: 280px; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid var(--line); }
table.grid th { background: #fbfbfc; cursor: pointer; position: sticky; top: 0;
  font-weight: 600; color: #374151; user-select: none; }
table.grid th:hover { background: var(--accent-soft); }
table.grid tbody tr:nth-child(even) { background: #fafbfc; }
table.grid tbody tr:hover { background: var(--accent-soft); cursor: pointer; }
.filterbtn { display: none; }
#scrim { position: fixed; inset: 0; background: rgba(0,0,0,.25); opacity: 0;
  pointer-events: none; transition: opacity .15s; z-index: 9; }
#scrim.on { opacity: 1; pointer-events: auto; }
#drawer { position: fixed; top: 0; right: 0; width: min(620px, 94vw); height: 100%;
  background: var(--surface); box-shadow: -4px 0 24px rgba(16,24,40,.18); overflow-y: auto;
  padding: 18px 22px 60px; transform: translateX(100%); transition: transform .18s; z-index: 10; }
#drawer.open { transform: translateX(0); }
#drawer .x { float: right; font-size: 24px; cursor: pointer; color: var(--muted);
  line-height: 1; border: 0; background: none; }
@media (max-width: 760px) {
  .sidebar { position: fixed; top: 0; left: 0; height: 100%; z-index: 10; transform: translateX(-100%);
    transition: transform .18s; box-shadow: 2px 0 24px rgba(16,24,40,.18); }
  .sidebar.open { transform: translateX(0); }
  .filterbtn { display: inline-block; }
}
</style></head><body>
<div id="data" style="display:none">__DATA__</div>
<div class="head">
  <div class="back"><a href="/exp/__NAME__">← __TITLE__</a></div>
  <h1>Browse transcripts</h1>
  <div class="note" id="note"></div>
</div>
<div class="layout">
  <aside class="sidebar" id="sidebar"><h3>Filters</h3><div id="filters"></div></aside>
  <main class="main">
    <div class="toolbar">
      <button class="clr filterbtn" onclick="toggleSidebar()">☰ Filters</button>
      <span class="count" id="count"></span>
      <span id="chips"></span>
      <button class="clr" onclick="clearAll()">Clear all</button>
    </div>
    <div class="gridwrap"><table class="grid" id="grid"></table></div>
  </main>
</div>
<div id="scrim" onclick="closeDrawer();closeSidebar()"></div>
<div id="drawer"><button class="x" onclick="closeDrawer()">×</button><div id="body"></div></div>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const sel = {}, bools = {}, nums = {};
let sort = {col: null, dir: 1};

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function build() {
  const fc = document.getElementById('filters');
  for (const f of D.facets) {
    const g = document.createElement('div'); g.className = 'facet';
    const nm = document.createElement('div'); nm.className = 'name'; nm.textContent = f.field;
    g.appendChild(nm);
    if (f.type === 'cat') {
      sel[f.field] = new Set();
      const box = document.createElement('div'); box.className = 'opts';
      for (const v of f.values) {
        const l = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.value = v;
        cb.onchange = () => { cb.checked ? sel[f.field].add(v) : sel[f.field].delete(v); render(); };
        const sp = document.createElement('span'); sp.textContent = v;
        l.appendChild(cb); l.appendChild(sp); box.appendChild(l);
      }
      g.appendChild(box);
    } else if (f.type === 'bool') {
      bools[f.field] = '';
      const seg = document.createElement('div'); seg.className = 'seg';
      for (const [val, lab] of [['', 'any'], ['true', '✓'], ['false', '✗']]) {
        const b = document.createElement('button'); b.textContent = lab;
        if (val === '') b.classList.add('on');
        b.onclick = () => { bools[f.field] = val;
          [...seg.children].forEach(c => c.classList.remove('on')); b.classList.add('on'); render(); };
        seg.appendChild(b);
      }
      g.appendChild(seg);
    } else if (f.type === 'num') {
      nums[f.field] = {min: null, max: null};
      const box = document.createElement('div'); box.className = 'num';
      const lo = document.createElement('input'), hi = document.createElement('input');
      lo.type = hi.type = 'number';
      lo.placeholder = '≥ ' + (Math.round(f.min*100)/100); hi.placeholder = '≤ ' + (Math.round(f.max*100)/100);
      lo.oninput = () => { nums[f.field].min = lo.value === '' ? null : +lo.value; render(); };
      hi.oninput = () => { nums[f.field].max = hi.value === '' ? null : +hi.value; render(); };
      box.appendChild(lo); box.appendChild(hi); g.appendChild(box);
    }
    fc.appendChild(g);
  }
}

function cell(v) {
  if (v === true) return '✓'; if (v === false) return '✗';
  if (v === null || v === undefined) return '<span style="color:#bbb">—</span>';
  if (typeof v === 'number') return (Math.round(v * 1e4) / 1e4);
  return esc(v);
}

function chipsHtml() {
  const out = [];
  for (const f of D.facets) {
    if (f.type === 'cat' && sel[f.field].size)
      for (const v of sel[f.field])
        out.push(`<span class="chip">${esc(f.field)}: ${esc(v)} <span class="x" onclick="rmCat('${esc(f.field)}','${esc(v)}')">×</span></span>`);
    if (f.type === 'bool' && bools[f.field] !== '')
      out.push(`<span class="chip">${esc(f.field)}: ${bools[f.field] === 'true' ? '✓' : '✗'} <span class="x" onclick="rmBool('${esc(f.field)}')">×</span></span>`);
    if (f.type === 'num' && (nums[f.field].min !== null || nums[f.field].max !== null))
      out.push(`<span class="chip">${esc(f.field)}: ${nums[f.field].min ?? '−∞'}…${nums[f.field].max ?? '∞'} <span class="x" onclick="rmNum('${esc(f.field)}')">×</span></span>`);
  }
  return out.join('');
}

const CAP = D.render_cap || 1000;
let seq = 0, timer = null;

function render() {  // debounced server-side filter/sort fetch
  document.getElementById('chips').innerHTML = chipsHtml();
  clearTimeout(timer);
  timer = setTimeout(fetchRows, 140);
}

function fetchRows() {
  const mine = ++seq;
  const body = {
    sel: Object.fromEntries(Object.entries(sel).map(([k, s]) => [k, [...s]])),
    bools, nums, sort, offset: 0, limit: CAP,
  };
  document.getElementById('count').textContent = 'filtering…';
  fetch(`/exp/${D.name}/rows`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  }).then(r => r.json()).then(d => {
    if (mine !== seq) return;  // ignore stale responses
    const head = '<tr>' + D.columns.map(c =>
      `<th onclick="setSort('${esc(c)}')">${esc(c)}${sort.col === c ? (sort.dir > 0 ? ' ↑' : ' ↓') : ''}</th>`
    ).join('') + '</tr>';
    const rowsHtml = d.rows.map(r =>
      `<tr onclick="openRec('${encodeURIComponent(r._id)}')">` +
      D.columns.map(c => `<td>${cell(r[c])}</td>`).join('') + '</tr>'
    ).join('');
    document.getElementById('grid').innerHTML = head + rowsHtml;
    const more = d.total > d.rows.length ? ` (top ${d.rows.length.toLocaleString()} — refine or sort)` : '';
    document.getElementById('count').textContent =
      `${d.total.toLocaleString()} of ${D.total.toLocaleString()}${more}`;
  });
}

function setSort(c) { sort = {col: c, dir: sort.col === c ? -sort.dir : 1}; render(); }
function rmCat(f, v) { sel[f].delete(v); syncBoxes(); render(); }
function rmBool(f) { bools[f] = ''; syncBoxes(); render(); }
function rmNum(f) { nums[f] = {min: null, max: null}; syncBoxes(); render(); }
function clearAll() {
  for (const k in sel) sel[k].clear();
  for (const k in bools) bools[k] = '';
  for (const k in nums) nums[k] = {min: null, max: null};
  syncBoxes(); render();
}
function syncBoxes() {  // rebuild controls to reflect cleared state
  document.getElementById('filters').innerHTML = ''; build();
}
function openRec(id) {
  const dr = document.getElementById('body');
  dr.innerHTML = '<p class="muted">Loading…</p>';
  document.getElementById('drawer').classList.add('open');
  document.getElementById('scrim').classList.add('on');
  fetch(`/exp/${D.name}/rec/${id}`).then(r => r.text()).then(t => { dr.innerHTML = t; });
}
function closeDrawer() { document.getElementById('drawer').classList.remove('open');
  document.getElementById('scrim').classList.remove('on'); }
function toggleSidebar() { document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('scrim').classList.toggle('on'); }
function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeDrawer(); closeSidebar(); } });
if (D.note) document.getElementById('note').textContent = D.note;
build(); render();
</script></body></html>""")


def _safe(name: str) -> Path:
    """Resolve an experiment folder name to a path inside ROOT, or 404."""
    p = (ROOT / name).resolve()
    if not str(p).startswith(str(ROOT)) or not p.is_dir() or name == SELF:
        abort(404)
    return p


def _experiments():
    out = []
    for p in sorted(ROOT.iterdir(), reverse=True):
        if not p.is_dir() or p.name.startswith(".") or p.name == SELF:
            continue
        out.append(p)
    return out


def _headline(p: Path):
    """Pick the headline markdown file by priority chain; None if none found."""
    tops = {f.name: f for f in p.glob("*.md")}
    for name in ("RESULTS.md", "README.md", "REPORT.md"):
        if name in tops:
            return tops[name]
    reports = sorted(n for n in tops if n.upper().startswith("REPORT"))
    if reports:
        return tops[reports[0]]
    others = sorted(tops)
    if others:
        return tops[others[0]]
    deep = sorted(p.rglob("*.md"))
    return deep[0] if deep else None


def _date(name: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else ""


def _pretty(name: str) -> str:
    """Experiment folder name -> human title: drop date prefix, underscores -> spaces."""
    s = re.sub(r"^\d{4}-\d{2}-\d{2}[_-]?", "", name)
    return s.replace("_", " ").strip() or name


def _meta(p: Path):
    """(title, doc_title, summary). Title is the experiment name; doc_title is the
    headline doc's H1 (shown as a subtitle); summary is its Question/first line."""
    title = _pretty(p.name)
    hl = _headline(p)
    doc_title = ""
    summary = ""
    if hl:
        text = hl.read_text(errors="replace")
        for line in text.splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break
        m = re.search(r"\*\*Question:\*\*\s*(.+)", text)
        if m:
            summary = m.group(1).strip()
        else:
            for line in text.splitlines():
                s = line.strip()
                if s and not s.startswith("#") and not s.startswith("!["):
                    summary = re.sub(r"[*_`#]", "", s)
                    break
    return title, doc_title, summary[:280]


@app.route("/")
def index():
    cards = []
    for p in _experiments():
        title, doc_title, summary = _meta(p)
        hl = _headline(p)
        sub = (
            f'<div class="muted" style="font-size:13px">{html.escape(doc_title)}</div>'
            if doc_title and doc_title.lower() != title.lower()
            else ""
        )
        cards.append(
            f'<div class="card"><div class="date">{_date(p.name)}</div>'
            f'<h2><a href="/exp/{p.name}">{html.escape(title)}</a></h2>'
            f"{sub}"
            f'<div class="summary">{html.escape(summary)}</div>'
            f'{"" if hl else "<div class=muted>(no report doc found)</div>"}'
            "</div>"
        )
    body = (
        '<div class="top"><form action="/search" method="get">'
        '<input type="text" name="q" placeholder="Search transcripts &amp; reports…" '
        'autofocus><button>Search</button></form></div>'
        f"<h1>Experiments</h1><p class=muted>{len(cards)} experiments</p>"
        + "".join(cards)
    )
    return PAGE.format(title="Experiment dashboard", body=body)


@app.route("/exp/<name>")
def experiment(name):
    p = _safe(name)
    title, doc_title, _ = _meta(p)
    hl = _headline(p)
    parts = [
        '<div class="back"><a href="/">← all experiments</a></div>',
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="muted">{html.escape(name)}</p>',
    ]

    if _has_browser(p):
        parts.append(
            f'<p><a class="pill" style="background:#0066cc;color:#fff;font-size:14px;'
            f'padding:6px 12px" href="/exp/{name}/browse">📊 Browse transcripts '
            "(faceted)</a></p>"
        )

    viewers = sorted(p.rglob("*.html"))
    if viewers:
        links = " ".join(
            f'<a class="pill" href="/raw/{name}/{html.escape(str(v.relative_to(p)))}">'
            f"🔍 {html.escape(v.relative_to(p).as_posix())}</a>"
            for v in viewers[:60]
        )
        parts.append(f"<h2>Viewers</h2><div>{links}</div>")

    docs = sorted(p.glob("*.md"))
    if docs:
        links = " ".join(
            f'<a class="pill" href="/md/{name}/{html.escape(d.name)}">📄 {html.escape(d.name)}</a>'
            for d in docs
        )
        parts.append(f"<h2>Docs</h2><div>{links}</div>")

    if hl:
        parts.append(f"<h2>{html.escape(hl.name)}</h2>")
        parts.append(f'<div class="report">{_render_md(hl)}</div>')
    else:
        parts.append('<p class="muted">No report document found.</p>')

    return PAGE.format(title=title, body="".join(parts))


def _render_md(path: Path) -> str:
    return md.markdown(
        path.read_text(errors="replace"),
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
    )


@app.route("/md/<name>/<path:relpath>")
def view_md(name, relpath):
    p = _safe(name)
    f = (p / relpath).resolve()
    if not str(f).startswith(str(p)) or not f.is_file() or f.suffix != ".md":
        abort(404)
    body = (
        f'<div class="back"><a href="/exp/{name}">← {html.escape(name)}</a></div>'
        f"<h1>{html.escape(f.name)}</h1>"
        f'<div class="report">{_render_md(f)}</div>'
    )
    return PAGE.format(title=f.name, body=body)


@app.route("/raw/<name>/<path:relpath>")
def raw(name, relpath):
    """Serve an existing viewer.html or any asset, untouched."""
    p = _safe(name)
    f = (p / relpath).resolve()
    if not str(f).startswith(str(p)) or not f.is_file():
        abort(404)
    return send_file(f)


def _hl(text: str, q: str) -> str:
    esc = html.escape(text)
    return re.sub(re.escape(html.escape(q)), lambda m: f"<b>{m.group(0)}</b>", esc, flags=re.I)


@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    only = request.args.get("exp")
    if not q:
        return PAGE.format(
            title="Search",
            body='<div class="back"><a href="/">← all experiments</a></div>'
            '<p class="muted">No query.</p>',
        )

    target = _safe(only) if only else ROOT
    if RG.exists():
        cmd = [str(RG), "-i", "-F", "--no-ignore", "--no-heading", "-n",
               f"--max-count={MAX_PER_FILE}"]
        for ext in SEARCH_EXTS:
            cmd += ["-g", f"*.{ext}"]
        cmd += ["-g", f"!{SELF}/**", "--", q, str(target)]
    else:  # fallback: grep (slower, but always available)
        cmd = ["grep", "-rInF", f"--max-count={MAX_PER_FILE}", "-I"]
        for ext in SEARCH_EXTS:
            cmd.append(f"--include=*.{ext}")
        cmd += [f"--exclude-dir={SELF}", "--", q, str(target)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT, errors="replace"
        )
        lines = proc.stdout.splitlines()
        timed_out = False
    except subprocess.TimeoutExpired as e:
        lines = (e.stdout or "").splitlines() if isinstance(e.stdout, str) else []
        timed_out = True

    groups = {}  # exp name -> list of (relpath, lineno, snippet)
    for ln in lines:
        m = re.match(r"^(.*?):(\d+):(.*)$", ln)
        if not m:
            continue
        fpath, lineno, content = m.group(1), m.group(2), m.group(3)
        rel = Path(fpath).resolve().relative_to(ROOT)
        exp = rel.parts[0]
        within = Path(*rel.parts[1:]).as_posix() if len(rel.parts) > 1 else rel.as_posix()
        groups.setdefault(exp, []).append((within, lineno, content[:400]))

    nfiles = sum(len({r for r, _, _ in v}) for v in groups.values())
    parts = [
        '<div class="back"><a href="/">← all experiments</a></div>',
        f"<h1>Search: {html.escape(q)}</h1>",
        f'<p class="muted">{len(lines)} matches across {nfiles} files in '
        f"{len(groups)} experiments"
        + (" · ⏱ timed out (partial results)" if timed_out else "")
        + "</p>",
    ]
    shown = 0
    for exp in sorted(groups, reverse=True):
        hits = groups[exp]
        parts.append(
            f'<div class="card"><h2><a href="/exp/{exp}">{html.escape(exp)}</a> '
            f'<span class="muted">({len(hits)})</span></h2>'
        )
        for rel, lineno, content in hits:
            if shown >= MAX_FILES:
                break
            shown += 1
            opener = (
                f"/raw/{exp}/{html.escape(rel)}"
                if rel.endswith(".html")
                else f"/md/{exp}/{html.escape(rel)}"
                if rel.endswith(".md")
                else f"/raw/{exp}/{html.escape(rel)}"
            )
            parts.append(
                f'<div><a href="{opener}">{html.escape(rel)}</a>'
                f'<span class="muted"> :{lineno}</span>'
                f'<div class="snippet">{_hl(content, q)}</div></div>'
            )
        parts.append("</div>")
    if not groups:
        parts.append('<p class="muted">No matches.</p>')
    return PAGE.format(title=f"Search: {q}", body="".join(parts))


# ---------------------------------------------------------------------------
# Faceted transcript browser
# ---------------------------------------------------------------------------

CAT_MAX = 40  # max distinct values for a string field to become a filter
CAT_LEN = 60  # max avg length for a string field to be a facet/column
MAX_BROWSE = 50000  # hard cap on rows sent to the client (payload guard)
RENDER_CAP = 1000  # rows the browser draws at once (filter runs over the full set)
_REC_CACHE = {}  # exp path -> (signature, records)


def _has_browser(p: Path) -> bool:
    if (p / "dashboard.json").exists():
        return True
    return bool(list(p.glob("results/*.json")))


def _cfg(p: Path) -> dict:
    f = p / "dashboard.json"
    cfg = json.loads(f.read_text()) if f.exists() else {}
    cfg.setdefault("records", "results/*.json")
    cfg.setdefault("exclude", ["*_all*", "*viewer*"])
    cfg.setdefault("joins", [])
    cfg.setdefault("transcript", [])
    cfg.setdefault("hide", [])
    cfg.setdefault("id_field", None)
    cfg.setdefault("path_regex", None)
    cfg.setdefault("record_key", None)  # key to unwrap a list from a {key:[...]} file
    cfg.setdefault("flatten", [])  # dict fields to flatten into dotted scalar facets
    cfg.setdefault("transcript_path_field", None)  # record field holding a dir path
    cfg.setdefault("transcript_dir_files", [])  # files in that dir to render lazily
    return cfg


def _record_files(p: Path, cfg: dict):
    globs = cfg["records"] if isinstance(cfg["records"], list) else [cfg["records"]]
    files, seen = [], set()
    for g in globs:
        for f in sorted(p.glob(g)):
            if f.suffix in (".json", ".jsonl") and f not in seen and not any(
                fnmatch.fnmatch(f.name, pat) for pat in cfg["exclude"]
            ):
                seen.add(f)
                files.append(f)
    return sorted(files)


def _parse_file(f: Path, record_key=None):
    """Yield record dicts from a .json (object/list, or {record_key:[...]}) or
    .jsonl (one object per line)."""
    try:
        if f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            d = json.loads(f.read_text())
            if record_key and isinstance(d, dict) and isinstance(d.get(record_key), list):
                yield from d[record_key]
            else:
                yield from (d if isinstance(d, list) else [d])
    except Exception:
        return


def _flat(d, prefix):
    """Recursively yield (dotted_key, scalar) for scalar leaves of a dict."""
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _flat(v, key)
        elif v is None or isinstance(v, (str, int, float, bool)):
            yield key, v


def _load_records(p: Path, cfg: dict):
    files = _record_files(p, cfg)
    cfgf = p / "dashboard.json"
    cfg_mtime = cfgf.stat().st_mtime if cfgf.exists() else 0
    sig = (len(files), max((f.stat().st_mtime for f in files), default=0), cfg_mtime)
    key = str(p)
    if _REC_CACHE.get(key, (None,))[0] == sig:
        return _REC_CACHE[key][1]

    pat = re.compile(cfg["path_regex"]) if cfg.get("path_regex") else None
    recs = []
    for f in files:
        path_fields = {}
        if pat:
            m = pat.search(f.relative_to(p).as_posix())
            if m:
                path_fields = m.groupdict()
        for it in _parse_file(f, cfg.get("record_key")):
            if isinstance(it, dict):
                it.setdefault("_file", f.name)
                for k, v in path_fields.items():
                    it.setdefault(k, v)
                for root in cfg["flatten"]:
                    if isinstance(it.get(root), dict):
                        it.update(dict(_flat(it[root], root)))
                recs.append(it)

    for j in cfg["joins"]:
        jf = p / j["file"]
        if not jf.exists():
            continue
        if jf.suffix == ".jsonl":
            rows = list(_parse_file(jf))
        else:
            try:
                jdata = json.loads(jf.read_text())
            except Exception:
                continue
            rows = jdata if isinstance(jdata, list) else list(jdata.values())
        on, pre = j["on"], j.get("prefix", "")
        index = {r[on]: r for r in rows if isinstance(r, dict) and on in r}
        for it in recs:
            m = index.get(it.get(on))
            if m:
                for k, v in m.items():
                    if k != on:
                        it[f"{pre}.{k}" if pre else k] = v

    for i, it in enumerate(recs):  # stable synthetic unique id for drill-down
        it["_rowid"] = i

    _REC_CACHE[key] = (sig, recs)
    return recs


def _idf(cfg, records):
    if cfg["id_field"]:
        return cfg["id_field"]
    for cand in ("session_id", "id", "uid", "session", "_file"):
        if records and cand in records[0]:
            return cand
    return "_file"


def _facets(records, cfg):
    """Return (facets, columns): scalar fields usable as filters/table columns."""
    hide = set(cfg["hide"])
    order, seen = [], set()
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k)
                order.append(k)
    facets, columns = [], []
    for k in order:
        if k in hide or k.startswith("_"):
            continue
        vals = [r[k] for r in records if r.get(k) is not None]
        if not vals:
            continue
        types = {type(v) for v in vals}
        if not (types <= {bool, int, float, str}):  # only scalar fields are facetable
            continue
        if len(set(vals)) < 2:  # single-value field: no filtering power, skip
            continue
        if types <= {bool}:
            facets.append({"field": k, "type": "bool"})
        elif types <= {int, float}:
            nums = [float(v) for v in vals]
            facets.append({"field": k, "type": "num", "min": min(nums), "max": max(nums)})
        elif types <= {str}:
            distinct = sorted(set(vals))
            avglen = sum(len(v) for v in vals) / len(vals)
            if len(distinct) <= CAT_MAX and avglen <= CAT_LEN:
                facets.append({"field": k, "type": "cat", "values": distinct})
            else:
                continue
        else:
            continue
        columns.append(k)
    return facets, columns


def _fmt(v):
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


@app.route("/exp/<name>/browse")
def browse(name):
    p = _safe(name)
    cfg = _cfg(p)
    records = _load_records(p, cfg)
    if not records:
        return PAGE.format(
            title="Browse",
            body=f'<div class="back"><a href="/exp/{name}">← back</a></div>'
            "<p class=muted>No records found to browse.</p>",
        )
    facets, columns = _facets(records, cfg)
    data = json.dumps(
        {"facets": facets, "columns": columns, "name": name,
         "total": len(records), "render_cap": RENDER_CAP}
    )
    return (
        BROWSE.replace("__DATA__", html.escape(data, quote=True))
        .replace("__NAME__", html.escape(name))
        .replace("__TITLE__", html.escape(_pretty(name)))
    )


def _filter_sort(records, columns, body):
    """Server-side filter/sort/paginate. Returns (page_rows, total_matched)."""
    sel = {k: set(v) for k, v in body.get("sel", {}).items() if v}
    bools = {k: v for k, v in body.get("bools", {}).items() if v != ""}
    nums = body.get("nums", {})

    def ok(r):
        for f, vals in sel.items():
            if r.get(f) not in vals:
                return False
        for f, bv in bools.items():
            if str(r.get(f)) != bv:
                return False
        for f, rng in nums.items():
            v = r.get(f)
            lo, hi = rng.get("min"), rng.get("max")
            if lo is not None and not (isinstance(v, (int, float)) and v >= lo):
                return False
            if hi is not None and not (isinstance(v, (int, float)) and v <= hi):
                return False
        return True

    out = [r for r in records if ok(r)]
    sc = body.get("sort") or {}
    if sc.get("col"):
        col = sc["col"]
        out.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=sc.get("dir", 1) < 0)
    total = len(out)
    off = int(body.get("offset", 0))
    page = out[off:off + int(body.get("limit", RENDER_CAP))]
    rows = [{"_id": str(r["_rowid"]), **{c: r.get(c) for c in columns}} for r in page]
    return rows, total


@app.route("/exp/<name>/rows", methods=["POST"])
def rows(name):
    p = _safe(name)
    cfg = _cfg(p)
    records = _load_records(p, cfg)
    _, columns = _facets(records, cfg)
    page, total = _filter_sort(records, columns, request.get_json(force=True) or {})
    return {"rows": page, "total": total}


def _msg_text(content):
    """Flatten a message's content (string, or list of {type,text}/{text} blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                out.append(b.get("text") or b.get("content") or json.dumps(b, default=str))
        return "\n".join(out)
    return json.dumps(content, indent=2, default=str)


def _looks_like_messages(v):
    return (
        isinstance(v, list)
        and v
        and all(isinstance(m, dict) for m in v)
        and any("role" in m for m in v)
    )


def _md(s: str) -> str:
    """Render text as markdown, keeping literal angle-bracket tags (e.g. <think>) visible."""
    return md.markdown(
        html.escape(s, quote=False), extensions=["fenced_code", "tables", "sane_lists"]
    )


def _cls(role: str) -> str:
    return role if role in ("user", "assistant") else "note"


def _bubble(role, inner, label=None, collapsed=False):
    cls = _cls(role)
    head = f'<div class="role {cls}">{html.escape(label or role)}</div>'
    body = f'<div class="bubble {cls}">{inner}</div>'
    if collapsed:
        return (
            f'<details class="sec"><summary>{html.escape(label or role)}</summary>'
            f"{body}</details>"
        )
    return head + body


def _render_messages(v):
    out = []
    for m in v:
        role = str(m.get("role", ""))
        text = _msg_text(m.get("content", m.get("text", m.get("assistant_text", ""))))
        inner = _md(text) if text else '<span class="muted">(empty)</span>'
        # collapse boilerplate system turns and very long messages by default
        collapsed = role == "system" or len(text) > 1500
        out.append(_bubble(role, inner, label=role, collapsed=collapsed))
    return "".join(out)


def _render_value(v, role="note", label=None, collapsed=False):
    if isinstance(v, str):
        return _bubble(role, _md(v), label=label, collapsed=collapsed)
    if _looks_like_messages(v):
        body = _render_messages(v)
        if collapsed:
            return f'<details class="sec"><summary>{html.escape(label or "messages")}</summary>{body}</details>'
        return (f'<div class="role note">{html.escape(label)}</div>' if label else "") + body
    inner = f'<pre class="raw">{html.escape(json.dumps(v, indent=2, default=str))}</pre>'
    return _bubble(role, inner, label=label, collapsed=collapsed)


@app.route("/exp/<name>/rec/<path:rid>")
def record(name, rid):
    p = _safe(name)
    cfg = _cfg(p)
    records = _load_records(p, cfg)
    rec = next((r for r in records if str(r.get("_rowid")) == rid), None)
    if rec is None:
        abort(404)

    title = str(rec.get(_idf(cfg, records)) or rid)
    parts = [f"<h3>{html.escape(title)}</h3>"]
    tspec = cfg["transcript"] or [
        {"field": k} for k, v in rec.items() if isinstance(v, str) and len(v) > 80
    ]
    for t in tspec:
        f = t if isinstance(t, str) else t["field"]
        if f not in rec or rec[f] is None:
            continue
        label = f if isinstance(t, str) else t.get("label", f)
        role = "note" if isinstance(t, str) else t.get("role", "note")
        collapsed = False if isinstance(t, str) else bool(t.get("collapsed"))
        parts.append(_render_value(rec[f], role=role, label=label, collapsed=collapsed))

    shown = {f if isinstance(f, str) else f["field"] for f in tspec}
    for k, v in rec.items():  # always surface system/user prompts, collapsed
        if k in shown or k.startswith("_"):
            continue
        if isinstance(v, str) and len(v) > 120 and re.search(
            r"system|prompt|instruction", k, re.I
        ):
            parts.append(_render_value(v, role="note", label=k, collapsed=True))
            shown.add(k)

    tpf = cfg["transcript_path_field"]
    if tpf and rec.get(tpf):
        base = (p / rec[tpf]).resolve()
        if str(base).startswith(str(p)):
            for spec in cfg["transcript_dir_files"]:
                f = base / spec["file"]
                if f.is_file():
                    loaded = list(_parse_file(f)) if f.suffix == ".jsonl" else json.loads(f.read_text())
                    parts.append(
                        _render_value(
                            loaded, role="note", label=spec.get("label", spec["file"]),
                            collapsed=bool(spec.get("collapsed")),
                        )
                    )

    hide = set(cfg["hide"]) | shown
    attrs = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt(v))}</td></tr>"
        for k, v in rec.items()
        if k not in hide
        and not k.startswith("_")
        and not isinstance(v, (dict, list))
        and not (isinstance(v, str) and len(v) > 200)
    )
    parts.append(f'<h4>Fields</h4><table class="attrs">{attrs}</table>')
    return "".join(parts)


def main(port: int = 8800, host: str = "0.0.0.0"):
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    fire.Fire(main)
