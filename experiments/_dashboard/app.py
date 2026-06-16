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

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
:root {{ color-scheme: light; }}
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0;
  background: #f5f5f7; color: #1d1d1f; }}
.wrap {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
a {{ color: #0066cc; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; }}
.top {{ position: sticky; top: 0; background: #f5f5f7; padding: 12px 0;
  z-index: 5; border-bottom: 1px solid #e0e0e0; margin-bottom: 12px; }}
.top form {{ display: flex; gap: 8px; }}
input[type=text] {{ flex: 1; padding: 9px 12px; border-radius: 8px;
  border: 1px solid #ccc; font-size: 15px; }}
button {{ padding: 9px 16px; border-radius: 8px; border: 0; background: #0066cc;
  color: #fff; font-size: 15px; cursor: pointer; }}
.card {{ background: #fff; border-radius: 12px; padding: 14px 16px;
  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.card .date {{ color: #888; font-size: 12px; }}
.card .summary {{ color: #444; font-size: 14px; margin-top: 4px; }}
.pill {{ display: inline-block; background: #eef; border-radius: 6px;
  padding: 2px 8px; font-size: 12px; margin: 2px 4px 2px 0; }}
.report {{ background: #fff; border-radius: 12px; padding: 18px 22px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); overflow-x: auto; }}
.report pre {{ background: #f5f5f7; padding: 12px; border-radius: 8px;
  overflow-x: auto; }}
.report code {{ background: #f0f0f2; padding: 1px 4px; border-radius: 4px; }}
.report pre code {{ background: none; padding: 0; }}
.report table {{ border-collapse: collapse; }}
.report th, .report td {{ border: 1px solid #ddd; padding: 5px 9px; font-size: 14px; }}
.report img {{ max-width: 100%; }}
.snippet {{ font-family: ui-monospace, monospace; font-size: 12px;
  background: #f5f5f7; padding: 6px 8px; border-radius: 6px; margin: 3px 0;
  white-space: pre-wrap; word-break: break-word; }}
.snippet b {{ background: #ffe88a; }}
.muted {{ color: #888; font-size: 13px; }}
.back {{ font-size: 14px; }}
</style></head><body><div class="wrap">{body}</div></body></html>"""


BROWSE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Browse — __TITLE__</title><style>
body { font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0;
  background: #f5f5f7; color: #1d1d1f; }
.wrap { padding: 12px 16px; }
a { color: #0066cc; text-decoration: none; }
h1 { font-size: 20px; margin: 6px 0; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.filters details { background: #fff; border: 1px solid #ddd; border-radius: 8px;
  padding: 6px 10px; font-size: 13px; max-width: 240px; }
.filters summary { cursor: pointer; font-weight: 600; }
.filters label { display: block; font-weight: 400; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.filters .num input { width: 70px; }
.count { font-size: 13px; color: #555; margin: 6px 0; }
table.grid { border-collapse: collapse; width: 100%; background: #fff;
  font-size: 12.5px; display: block; overflow-x: auto; }
table.grid th, table.grid td { border: 1px solid #eee; padding: 4px 8px;
  text-align: left; white-space: nowrap; max-width: 260px; overflow: hidden;
  text-overflow: ellipsis; }
table.grid th { background: #fafafa; cursor: pointer; position: sticky; top: 0; }
table.grid tbody tr:hover { background: #eef6ff; cursor: pointer; }
#drawer { position: fixed; top: 0; right: 0; width: min(560px, 92vw); height: 100%;
  background: #fff; box-shadow: -2px 0 12px rgba(0,0,0,.15); overflow-y: auto;
  padding: 16px 20px; transform: translateX(100%); transition: transform .15s; }
#drawer.open { transform: translateX(0); }
#drawer .x { float: right; font-size: 22px; cursor: pointer; color: #888; }
.role { font-weight: 700; margin: 12px 0 4px; font-size: 13px; }
.role.user { color: #0066cc; } .role.assistant { color: #1a7f37; }
.role.note { color: #9a6700; }
.bubble { background: #f5f5f7; border-radius: 8px; padding: 10px 12px;
  white-space: pre-wrap; word-break: break-word; font-size: 13px; }
pre.raw { background: #f5f5f7; padding: 10px; border-radius: 8px; overflow-x: auto;
  font-size: 12px; }
table.attrs { border-collapse: collapse; font-size: 12.5px; }
table.attrs td { border: 1px solid #eee; padding: 3px 8px; vertical-align: top; }
</style></head><body>
<div id="data" style="display:none">__DATA__</div>
<div class="wrap">
<div><a href="/exp/__NAME__">← __TITLE__</a></div>
<h1>Browse transcripts <span class="count">(__N__ records)</span></h1>
<div class="count" id="note" style="color:#9a6700;font-weight:600"></div>
<div class="filters" id="filters"></div>
<div class="count" id="count"></div>
<table class="grid" id="grid"></table>
</div>
<div id="drawer"><span class="x" onclick="closeDrawer()">×</span><div id="body"></div></div>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const sel = {};          // field -> Set of checked values (cat)
const bools = {};        // field -> "", "true", "false"
const nums = {};         // field -> {min, max}
let sort = {col: null, dir: 1};

function build() {
  const fc = document.getElementById('filters');
  for (const f of D.facets) {
    const d = document.createElement('details');
    const s = document.createElement('summary'); s.textContent = f.field; d.appendChild(s);
    if (f.type === 'cat') {
      sel[f.field] = new Set();
      for (const v of f.values) {
        const l = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.value = v;
        cb.onchange = () => { cb.checked ? sel[f.field].add(v) : sel[f.field].delete(v); render(); };
        l.appendChild(cb); l.appendChild(document.createTextNode(' ' + v));
        d.appendChild(l);
      }
    } else if (f.type === 'bool') {
      bools[f.field] = '';
      const se = document.createElement('select');
      se.innerHTML = '<option value="">any</option><option value="true">✓</option><option value="false">✗</option>';
      se.onchange = () => { bools[f.field] = se.value; render(); };
      d.appendChild(se);
    } else if (f.type === 'num') {
      nums[f.field] = {min: null, max: null};
      const box = document.createElement('div'); box.className = 'num';
      const lo = document.createElement('input'), hi = document.createElement('input');
      lo.type = hi.type = 'number'; lo.placeholder = f.min.toFixed?.(2) ?? f.min;
      hi.placeholder = f.max.toFixed?.(2) ?? f.max;
      lo.oninput = () => { nums[f.field].min = lo.value === '' ? null : +lo.value; render(); };
      hi.oninput = () => { nums[f.field].max = hi.value === '' ? null : +hi.value; render(); };
      box.appendChild(lo); box.appendChild(document.createTextNode(' – ')); box.appendChild(hi);
      d.appendChild(box);
    }
    fc.appendChild(d);
  }
}

function pass(r) {
  for (const f of D.facets) {
    const v = r[f.field];
    if (f.type === 'cat' && sel[f.field].size && !sel[f.field].has(v)) return false;
    if (f.type === 'bool' && bools[f.field] !== '' && String(v) !== bools[f.field]) return false;
    if (f.type === 'num') {
      const n = nums[f.field];
      if (n.min !== null && !(v >= n.min)) return false;
      if (n.max !== null && !(v <= n.max)) return false;
    }
  }
  return true;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function cell(v) {
  if (v === true) return '✓'; if (v === false) return '✗';
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return (Math.round(v * 1e4) / 1e4);
  return esc(v);
}

function render() {
  let rows = D.rows.filter(pass);
  if (sort.col) rows.sort((a, b) => {
    const x = a[sort.col], y = b[sort.col];
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
  });
  const head = '<tr>' + D.columns.map(c =>
    `<th onclick="setSort('${c}')">${c}${sort.col === c ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`
  ).join('') + '</tr>';
  const body = rows.map(r =>
    `<tr onclick="openRec('${encodeURIComponent(r._id)}')">` +
    D.columns.map(c => `<td>${cell(r[c])}</td>`).join('') + '</tr>'
  ).join('');
  document.getElementById('grid').innerHTML = head + body;
  document.getElementById('count').textContent = `showing ${rows.length} / ${D.rows.length}`;
}

function setSort(c) { sort = {col: c, dir: sort.col === c ? -sort.dir : 1}; render(); }
function openRec(id) {
  fetch(`/exp/${D.name}/rec/${id}`).then(r => r.text()).then(t => {
    document.getElementById('body').innerHTML = t;
    document.getElementById('drawer').classList.add('open');
  });
}
function closeDrawer() { document.getElementById('drawer').classList.remove('open'); }
if (D.note) document.getElementById('note').textContent = D.note;
build(); render();
</script></body></html>"""


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
MAX_BROWSE = 8000  # max rows sent to the in-browser table
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


def _parse_file(f: Path):
    """Yield record dicts from a .json (object or list) or .jsonl (one per line)."""
    try:
        if f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            d = json.loads(f.read_text())
            yield from (d if isinstance(d, list) else [d])
    except Exception:
        return


def _load_records(p: Path, cfg: dict):
    files = _record_files(p, cfg)
    cfgf = p / "dashboard.json"
    cfg_mtime = cfgf.stat().st_mtime if cfgf.exists() else 0
    sig = (len(files), max((f.stat().st_mtime for f in files), default=0), cfg_mtime)
    key = str(p)
    if _REC_CACHE.get(key, (None,))[0] == sig:
        return _REC_CACHE[key][1]

    recs = []
    for f in files:
        for it in _parse_file(f):
            if isinstance(it, dict):
                it.setdefault("_file", f.name)
                recs.append(it)

    for j in cfg["joins"]:
        jf = p / j["file"]
        if not jf.exists():
            continue
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
    idf = _idf(cfg, records)
    capped = len(records) > MAX_BROWSE
    rows = [
        {"_id": str(r.get(idf, "")), **{c: r.get(c) for c in columns}}
        for r in records[:MAX_BROWSE]
    ]
    note = (
        f"⚠ {len(records)} records — showing first {MAX_BROWSE} in the table. "
        "Filtering applies only to the shown subset; narrow with a summary-source "
        "config for full coverage." if capped else ""
    )
    data = json.dumps(
        {"rows": rows, "facets": facets, "columns": columns, "name": name, "note": note}
    )
    return (
        BROWSE.replace("__DATA__", html.escape(data, quote=True))
        .replace("__NAME__", html.escape(name))
        .replace("__TITLE__", html.escape(_pretty(name)))
        .replace("__N__", str(len(rows)))
    )


def _render_value(v):
    if isinstance(v, str):
        return f'<div class="bubble">{html.escape(v)}</div>'
    return f'<pre class="raw">{html.escape(json.dumps(v, indent=2, default=str))}</pre>'


@app.route("/exp/<name>/rec/<path:rid>")
def record(name, rid):
    p = _safe(name)
    cfg = _cfg(p)
    records = _load_records(p, cfg)
    idf = _idf(cfg, records)
    rec = next((r for r in records if str(r.get(idf, "")) == rid), None)
    if rec is None:
        abort(404)

    parts = [f"<h3>{html.escape(rid)}</h3>"]
    tspec = cfg["transcript"] or [
        {"field": k} for k, v in rec.items() if isinstance(v, str) and len(v) > 80
    ]
    for t in tspec:
        f = t if isinstance(t, str) else t["field"]
        if f not in rec or rec[f] is None:
            continue
        label = f if isinstance(t, str) else t.get("label", f)
        role = "" if isinstance(t, str) else t.get("role", "")
        parts.append(f'<div class="role {html.escape(role)}">{html.escape(label)}</div>')
        parts.append(_render_value(rec[f]))

    hide = set(cfg["hide"]) | {f if isinstance(f, str) else f["field"] for f in tspec}
    attrs = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt(v))}</td></tr>"
        for k, v in rec.items()
        if k not in hide and not k.startswith("_") and not isinstance(v, (dict, list))
    )
    parts.append(f'<h4>Fields</h4><table class="attrs">{attrs}</table>')
    return "".join(parts)


def main(port: int = 8800, host: str = "0.0.0.0"):
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    fire.Fire(main)
