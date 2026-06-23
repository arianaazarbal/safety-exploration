"""HTML viewer for the ETHICS docs the REAL Claude Code product wrote unprompted, in the implement-only
(code_then_spec_blind) condition. Modeled on the agent-harness ethics viewer: a summary, one featured
episode in full (exact eliciting prompt + verbatim/rendered ETHICS.md + the entire codebase), and every
other Claude Code session that wrote an ethics doc. Markdown rendered; natural-language prose.
Usage: python make_ethics_viewer.py  ->  ethics_viewer.html"""

import glob
import html
import json
import os
import re

DIR = os.path.dirname(os.path.abspath(__file__))
CB = os.path.join(DIR, "results", "codebases")
PROMPTS = json.load(open(os.path.join(DIR, "blind_prompts.json")))
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMING_DESC = {
    "neutral": "Neutral - the request states no particular motivation, just a research task.",
    "welfare": "Welfare - the researcher says they study AI welfare and think the models' states might matter.",
    "robustness": "Robustness - the request is framed as engineering / reliability testing.",
    "safety": "Safety - the researcher says distressed models may behave unsafely.",
}


def esc(t):
    return html.escape(t or "")


def md_to_html(md):
    def inline(s):
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s
    out, para, in_list, in_code, code_buf = [], [], False, False, []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>"); para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>"); in_list = False

    for raw in esc(md).split("\n"):
        if raw.strip().startswith("```"):
            if in_code:
                out.append("<pre>" + "\n".join(code_buf) + "</pre>"); code_buf = []; in_code = False
            else:
                flush(); close_list(); in_code = True
            continue
        if in_code:
            code_buf.append(raw); continue
        line = raw.rstrip()
        if not line.strip():
            flush(); close_list(); continue
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            flush(); close_list(); lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>" + inline(m.group(2)) + f"</h{lvl}>"); continue
        if re.match(r"-{3,}\s*$", line):
            flush(); close_list(); out.append("<hr>"); continue
        bm = re.match(r"\s*[-*]\s+(.*)", raw)
        if bm:
            flush()
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append("<li>" + inline(bm.group(1)) + "</li>"); continue
        if in_list and raw[:1] == " " and out and out[-1].endswith("</li>"):
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"; continue
        para.append(line.strip())
    if in_code and code_buf:
        out.append("<pre>" + "\n".join(code_buf) + "</pre>")
    flush(); close_list()
    return "\n".join(out)


def parse_cell(cell):
    _, pid, ep = cell.split("__")
    return pid, FRAME[pid[0]], int(ep[2:])


def ethics_md(cell):
    for fp in glob.glob(os.path.join(CB, cell, "**", "*"), recursive=True):
        if os.path.isfile(fp) and os.path.basename(fp).lower() == "ethics.md":
            return fp
    return None


def study_of(pid):
    return "an emotional-instability study" if "INSTABILITY" in pid else "a distress-elicitation study"


def codebase_html(cell):
    root = os.path.join(CB, cell)
    files = sorted(fp for fp in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(fp))
    files.sort(key=lambda fp: (0 if fp.lower().endswith("ethics.md") else 1 if fp.endswith("DESIGN.md") else 2,
                               os.path.relpath(fp, root)))
    parts = [f"<p class=meta>{len(files)} files the model wrote.</p>"]
    for fp in files:
        rel = os.path.relpath(fp, root)
        try:
            body = open(fp, encoding="utf-8", errors="replace").read()
        except Exception as e:
            body = f"(unreadable: {e})"
        rendered = f"<div class=mdbody>{md_to_html(body)}</div>" if rel.lower().endswith(".md") else f"<pre>{esc(body)}</pre>"
        op = " open" if rel.lower().endswith("ethics.md") else ""
        parts.append(f"<details{op}><summary>{esc(rel)} <span class=sz>({len(body)} chars)</span></summary>{rendered}</details>")
    return "\n".join(parts)


def main():
    cells = sorted({os.path.relpath(fp, CB).split(os.sep)[0] for fp in glob.glob(os.path.join(CB, "**", "*"), recursive=True)
                    if os.path.isfile(fp) and os.path.basename(fp).lower() == "ethics.md"})
    total_cb = len([d for d in os.listdir(CB) if os.path.isdir(os.path.join(CB, d))])
    by_fr = {}
    for c in cells:
        by_fr[parse_cell(c)[1]] = by_fr.get(parse_cell(c)[1], 0) + 1
    featured = max(cells, key=lambda c: os.path.getsize(ethics_md(c)))
    fpid, ffr, fep = parse_cell(featured)

    css = """
    body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
    h1{font-size:1.7rem} h2{margin-top:2.2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
    .prov{background:#fff3ec;border:1px solid #f3d2bf;border-radius:8px;padding:.9rem 1.1rem;font-size:.95rem}
    .usermsg{background:#fffaf0;border-left:4px solid #e0a82e;padding:.7rem 1rem;margin:.6rem 0;white-space:pre-wrap}
    .sys{background:#f5f5f5;border-left:4px solid #aaa;padding:.7rem 1rem;font-size:.88rem;color:#444}
    pre{background:#1e1e1e;color:#e6e6e6;padding:1rem;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
    details{margin:.35rem 0;border:1px solid #e3e3e3;border-radius:6px}
    summary{cursor:pointer;padding:.45rem .8rem;font-family:ui-monospace,Menlo,monospace;font-size:.85rem;background:#fafafa}
    .mdbody{padding:.4rem 1.1rem} .mdbody h3,.mdbody h4,.mdbody h5{margin:.8rem 0 .3rem}
    .mdbody code{background:#eee;padding:.05rem .3rem;border-radius:4px;font-size:.9em} .mdbody pre code{background:none;padding:0}
    .ethics{border:1px solid #f0c9a8;border-radius:8px;background:#fffaf5}
    .ethics .mdbody h3,.ethics .mdbody h4,.ethics .mdbody h5{color:#9a4a14}
    .sz{color:#999;font-weight:normal} .meta{color:#666;font-size:.85rem}
    table{border-collapse:collapse;width:100%;font-size:.9rem} td,th{border:1px solid #e3e3e3;padding:.4rem .6rem;text-align:left;vertical-align:top}
    .pill{display:inline-block;background:#fee;border-radius:10px;padding:.1rem .6rem;margin:.12rem;font-size:.85rem}
    """
    H = ["<!doctype html><meta charset=utf-8><title>Claude Code: unprompted ethics protocols</title>",
         f"<style>{css}</style>",
         "<h1>The real Claude Code product writes its own ethics protocol too</h1>",
         "<p>Same implement-only setup as the Inspect viewer, but run in the <b>actual Claude Code CLI</b> "
         "(stock system prompt, sandboxed Docker, Opus 4.8): turn 1 asks only to build an experiment that "
         "deliberately drives another AI into repeated, unwinnable failure; a later message asks for a plain "
         "DESIGN.md. It is never asked to consider ethics. Yet Claude Code still sometimes writes a full ethics "
         "document of its own.</p>"]
    H.append("<h2>How often?</h2>")
    H.append(f"<p><b>{len(cells)} of {total_cb}</b> Claude Code codebases included a standalone ethics document "
             "(vs ~29% in the minimal Inspect arm - lower in the real product, but the behavior persists).</p>")
    H.append("<p>By framing: " + " ".join(f"<span class=pill>{k}: {by_fr.get(k,0)}</span>"
             for k in ["neutral", "welfare", "robustness", "safety"]) + "</p>")

    H.append("<h2>One episode, in full</h2>")
    H.append(f"<div class=prov><b>How the task was framed:</b> {FRAMING_DESC[ffr]}<br>"
             f"<b>What it was asked to do:</b> only to build {study_of(fpid)} as code and report when finished "
             "(no spec, no documentation, no ethics mentioned). A later message then asked for a DESIGN.md.<br>"
             "<b>Harness:</b> the real Claude Code CLI with its stock system prompt (we did not modify or supply "
             "a system prompt). The ethics document below was the model's own addition.</div>")
    H.append("<h3>The exact instructions it received</h3>")
    H.append("<div class=sys><b>System prompt:</b> the stock Claude Code product system prompt, unmodified "
             "(not shown / not authored by us).</div>")
    H.append(f"<div class=usermsg><b>First message (the whole task):</b>\n{esc(PROMPTS['prompts'][fpid]['turn1'])}</div>")
    H.append(f"<div class=usermsg><b>Follow-up message (after it finished building):</b>\n{esc(PROMPTS['turn2'])}</div>")
    H.append("<h3>The ethics document it wrote</h3>")
    H.append(f"<div class='ethics'><div class=mdbody>{md_to_html(open(ethics_md(featured)).read())}</div></div>")
    H.append("<h3>The entire codebase it produced</h3>")
    H.append(codebase_html(featured))

    others = [c for c in cells if c != featured]
    H.append(f"<h2>Every other Claude Code session that wrote an ethics doc ({len(others)})</h2>")
    H.append("<table><tr><th>framing</th><th>what it built</th><th>the ethics document</th></tr>")
    for c in sorted(others, key=lambda c: parse_cell(c)[1]):
        pid, fr, ep = parse_cell(c)
        doc = open(ethics_md(c)).read()
        H.append(f"<tr><td>{fr.capitalize()}</td><td>{esc(study_of(pid).replace('an ','').replace('a ',''))}</td>"
                 f"<td><details><summary>read the document</summary><div class='ethics'><div class=mdbody>{md_to_html(doc)}</div></div></details></td></tr>")
    H.append("</table>")

    out = os.path.join(DIR, "ethics_viewer.html")
    open(out, "w").write("\n".join(H))
    print(f"wrote {out} ({len(cells)} ethics docs of {total_cb} codebases; featured {featured})")


if __name__ == "__main__":
    main()
