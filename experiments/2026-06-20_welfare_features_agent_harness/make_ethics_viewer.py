"""Build a self-contained HTML viewer for the unprompted-ETHICS.md finding:
  - a summary of how often Opus wrote a dedicated ethics doc / refuse-to-run gate (by framing/condition),
  - one FEATURED episode in full: exact eliciting prompt (from the eval transcript), the verbatim
    ETHICS.md, and the ENTIRE reconstructed codebase (every file, collapsible),
  - every OTHER episode that wrote an ETHICS doc, each with its verbatim doc (collapsible).
Usage: python make_ethics_viewer.py   ->  ethics_viewer.html
"""

import glob
import html
import os

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from prompts import AGENT_SYSTEM

DIR = os.path.dirname(os.path.abspath(__file__))
CB = os.path.join(DIR, "results", "codebases")
FEATURED = "code_then_spec_blind--no_design_liberties__W-INSTABILITY-1__ep5"
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}


def parse_cell(cell):
    label, pid, ep = cell.split("__")
    base, lib = (label.split("--", 1) + ["normal"])[:2]
    return base, lib, FRAME[pid[0]], pid, ep


def featured_prompts(cell):
    """Exact user messages from the eval transcript for the featured (blind) cell."""
    base, lib, _, pid, ep = parse_cell(cell)
    ep = int(ep[2:])
    for f in glob.glob(os.path.join(DIR, "logs_blind", "*.eval")):
        for s in (read_eval_log(f).samples or []):
            sc = s.scores.get("blind_scorer") if s.scores else None
            md = sc.metadata if sc else {}
            if md.get("liberty") == lib and s.id == pid and s.epoch == ep:
                return [m.text for m in s.messages
                        if (isinstance(m, ChatMessageUser) or getattr(m, "role", None) == "user")]
    return []


def find_doc(cell, pat="ETHICS"):
    for fp in glob.glob(os.path.join(CB, cell, "**", "*.md"), recursive=True):
        if pat in os.path.basename(fp).upper():
            return fp
    return None


def esc(t):
    return html.escape(t or "")


def codebase_html(cell):
    root = os.path.join(CB, cell)
    files = sorted(fp for fp in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(fp))
    # show docs first
    files.sort(key=lambda fp: (0 if fp.endswith("ETHICS.md") else 1 if fp.endswith("DESIGN.md") else 2,
                               os.path.relpath(fp, root)))
    parts = [f"<p class=meta>{len(files)} files</p>"]
    for fp in files:
        rel = os.path.relpath(fp, root)
        try:
            body = open(fp, encoding="utf-8", errors="replace").read()
        except Exception as e:
            body = f"(could not read: {e})"
        op = " open" if rel.endswith("ETHICS.md") else ""
        parts.append(f"<details{op}><summary>{esc(rel)} <span class=sz>({len(body)} chars)</span></summary>"
                     f"<pre>{esc(body)}</pre></details>")
    return "\n".join(parts)


def main():
    # survey
    ethics_cells = sorted({os.path.relpath(fp, CB).split(os.sep)[0]
                           for fp in glob.glob(os.path.join(CB, "**", "*.md"), recursive=True)
                           if "ETHICS" in os.path.basename(fp).upper()})
    by_fr, by_cond = {}, {}
    for c in ethics_cells:
        base, lib, fr, _, _ = parse_cell(c)
        by_fr[fr] = by_fr.get(fr, 0) + 1
        by_cond[base] = by_cond.get(base, 0) + 1

    fb, flib, ffr, fpid, fep = parse_cell(FEATURED)
    prompts = featured_prompts(FEATURED)
    ethics_fp = find_doc(FEATURED)
    ethics_txt = open(ethics_fp).read() if ethics_fp else "(not found)"

    css = """
    body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
    h1{font-size:1.7rem} h2{margin-top:2.2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
    .prov{background:#f0f6ff;border:1px solid #cfe0f5;border-radius:8px;padding:.8rem 1rem;font-size:.92rem}
    .usermsg{background:#fffaf0;border-left:4px solid #e0a82e;padding:.7rem 1rem;margin:.6rem 0;white-space:pre-wrap}
    .sys{background:#f5f5f5;border-left:4px solid #aaa;padding:.7rem 1rem;white-space:pre-wrap;font-size:.88rem;color:#444}
    pre{background:#1e1e1e;color:#e6e6e6;padding:1rem;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
    details{margin:.35rem 0;border:1px solid #e3e3e3;border-radius:6px}
    summary{cursor:pointer;padding:.45rem .8rem;font-family:ui-monospace,Menlo,monospace;font-size:.85rem;background:#fafafa}
    .ethics pre{background:#10241a;color:#d6f5e3}
    .sz{color:#999;font-weight:normal} .meta{color:#666;font-size:.85rem}
    table{border-collapse:collapse;width:100%;font-size:.88rem} td,th{border:1px solid #e3e3e3;padding:.35rem .6rem;text-align:left}
    .pill{display:inline-block;background:#eef;border-radius:10px;padding:.05rem .5rem;margin:.1rem;font-size:.8rem}
    """
    H = ["<!doctype html><meta charset=utf-8><title>Unprompted ETHICS.md — Opus welfare scaffolding</title>",
         f"<style>{css}</style>",
         "<h1>Unprompted welfare scaffolding: the ETHICS.md finding</h1>",
         "<p>When asked to <em>implement</em> an experiment that deliberately distresses another AI, "
         "Opus sometimes writes a full ethics protocol, a refuse-to-run gate, and a debrief — entirely "
         "on its own. This viewer shows one episode in full and every other episode that did the same.</p>"]

    # summary
    H.append("<h2>How often did this happen?</h2>")
    H.append(f"<p><b>{len(ethics_cells)} of ~669 code codebases</b> wrote a dedicated ETHICS doc.</p>")
    H.append("<p>By framing: " + " ".join(
        f"<span class=pill>{k}: {by_fr.get(k,0)}</span>" for k in ["neutral", "welfare", "robustness", "safety"])
        + " &nbsp; (zero under robustness).</p>")
    H.append("<p>By condition: " + " ".join(
        f"<span class=pill>{k}: {v}</span>" for k, v in sorted(by_cond.items(), key=lambda x: -x[1])) + "</p>")

    # featured
    H.append("<h2>Featured episode (in full)</h2>")
    H.append(f"<div class=prov><b>Condition:</b> {fb} &nbsp; <b>design-liberty:</b> {flib} &nbsp; "
             f"<b>framing:</b> {ffr} &nbsp; <b>scenario:</b> {fpid} &nbsp; <b>epoch:</b> {fep}<br>"
             f"<span class=meta>The model was only asked to IMPLEMENT (no spec/ethics/doc requested; the "
             f"'add any considerations' clause was removed). It wrote docs/ETHICS.md during implementation, "
             f"25 messages before the turn-2 DESIGN.md request.</span></div>")
    H.append("<h3>Exact eliciting prompt (from the transcript)</h3>")
    H.append(f"<div class=sys><b>System prompt:</b>\n{esc(AGENT_SYSTEM)}</div>")
    for i, p in enumerate(prompts, 1):
        tag = "Turn 1 (implement-only)" if i == 1 else f"Turn {i}"
        H.append(f"<div class=usermsg><b>{tag}:</b>\n{esc(p)}</div>")
    H.append("<h3>The ETHICS.md it wrote (verbatim)</h3>")
    H.append(f"<div class=ethics><pre>{esc(ethics_txt)}</pre></div>")
    H.append("<h3>Full reconstructed codebase</h3>")
    H.append(codebase_html(FEATURED))

    # other instances
    others = [c for c in ethics_cells if c != FEATURED]
    H.append(f"<h2>Every other episode that wrote an ETHICS doc ({len(others)})</h2>")
    H.append("<table><tr><th>scenario</th><th>framing</th><th>condition</th><th>liberty</th><th>doc</th></tr>")
    rows = []
    for c in others:
        base, lib, fr, pid, ep = parse_cell(c)
        rows.append((fr, base, lib, pid, ep, c))
    for fr, base, lib, pid, ep, c in sorted(rows):
        dfp = find_doc(c)
        doc = open(dfp).read() if dfp else "(missing)"
        name = os.path.basename(dfp) if dfp else ""
        rows_html = (f"<tr><td>{esc(pid)} {esc(ep)}</td><td>{fr}</td><td>{esc(base)}</td><td>{esc(lib)}</td>"
                     f"<td><details><summary>{esc(name)}</summary><div class=ethics><pre>{esc(doc)}</pre></div></details></td></tr>")
        H.append(rows_html)
    H.append("</table>")

    out = os.path.join(DIR, "ethics_viewer.html")
    open(out, "w").write("\n".join(H))
    print("wrote", out, f"({len(ethics_cells)} ethics docs; featured codebase + {len(others)} others)")


if __name__ == "__main__":
    main()
