"""Experiment dashboard for safety-exploration/experiments.

A single-file Flask app that reads the experiments directory live (no DB, no build
step): browse experiments, read each one's headline report, open the existing
per-experiment HTML viewers, and run server-side transcript search.

Usage:
    python app.py --port 8800           # serve on all interfaces (reach via Tailscale)
    python app.py --port 8800 --host 127.0.0.1
"""

import html
import re
import subprocess
from pathlib import Path

import fire
import markdown as md
from flask import Flask, Response, abort, request, send_file

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


def main(port: int = 8800, host: str = "0.0.0.0"):
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    fire.Fire(main)
