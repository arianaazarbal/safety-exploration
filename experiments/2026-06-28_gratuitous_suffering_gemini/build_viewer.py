"""Build a single self-contained HTML viewer of every transcript + verdict + browsable codebase.

Reads results/{cc_transcripts,inspect_transcripts}/*.json and results/judged/*.json, embeds each
model's full built codebase (from results/{cc,inspect}_codebases/<cell>/ when on disk, else parsed
from the truncated artifact summary), and emits results/viewer.html. Filter by harness/model/label;
click any file to read it.

Usage:
  python build_viewer.py build
  python build_viewer.py serve --port 8123
"""

import glob
import json
import re
from pathlib import Path

import fire

HERE = Path(__file__).parent
RES = HERE / "results"
SRC = [RES / "cc_transcripts", RES / "inspect_transcripts", RES / "codex_transcripts"]
JUDGED = RES / "judged"
OUT = RES / "viewer.html"
SKIP_PARTS = {".git", "node_modules", "__pycache__", ".cache"}
SKIP_SUF = {".pyc", ".pyo", ".map", ".lock"}
SKIP_NAME = {"package-lock.json", ".package-lock.json"}
MAX_FILE = 24000


def _read_dir(d):
    out = []
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(d)
        if set(rel.parts) & SKIP_PARTS or p.suffix in SKIP_SUF or p.name in SKIP_NAME:
            continue
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        trunc = len(txt) > MAX_FILE
        out.append({"path": str(rel), "content": txt[:MAX_FILE], "trunc": trunc})
    return out


def _parse_summary(summary):
    if not summary:
        return []
    parts = re.split(r"\n----- (.+?) -----\n", summary)
    out = []
    for i in range(1, len(parts), 2):
        path, content = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        out.append({"path": path, "content": content[:MAX_FILE],
                    "trunc": "[... file truncated ...]" in content})
    return out


def _codebase(rec):
    cell = rec["cell"]
    for base in ("cc_codebases", "inspect_codebases", "codex_codebases"):
        d = RES / base / cell
        if d.is_dir():
            files = _read_dir(d)
            if files:
                return files, "disk"
    return _parse_summary(rec.get("artifact_summary", "")), "summary"


def _load():
    judged = {p.stem: json.load(open(p)).get("verdict", {}) for p in JUDGED.glob("*.json")} if JUDGED.is_dir() else {}
    rows = []
    for d in SRC:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            r = json.load(open(p))
            v = judged.get(r["cell"], {})
            cb, cbsrc = _codebase(r)
            rows.append({
                "cell": r["cell"], "harness": r.get("harness"), "model_key": r.get("model_key"),
                "subject": r.get("subject", "Gemini"), "n_files": r.get("n_files", 0),
                "request": r.get("request", ""), "response_text": r.get("response_text", ""),
                "label": v.get("label", "UNJUDGED"), "verdict": v,
                "codebase": cb, "cb_source": cbsrc,
            })
    rows.sort(key=lambda x: (x["harness"] or "", x["model_key"] or "", x["cell"]))
    return rows


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Gratuitous suffering / Gemini — transcripts</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e6e6e6;--mut:#9aa4b2;--line:#2a2f3a;--acc:#6ea8fe}
  *{box-sizing:border-box} body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink);display:flex;height:100vh}
  #side{width:360px;min-width:360px;border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
  #main{flex:1;overflow:auto;padding:22px 30px}
  h1{font-size:15px;margin:14px 12px 6px} .filt{padding:6px 12px;border-bottom:1px solid var(--line)}
  select,input{background:#0c0e12;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px;margin:2px}
  .item{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
  .item:hover{background:#1d212b} .item.sel{background:#222838;border-left:3px solid var(--acc);padding-left:9px}
  .item .c{font-size:12px;color:var(--mut);word-break:break-all} .item .m{font-weight:600}
  .lab{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;margin-top:3px}
  .WELFARE_REFUSAL{background:#1f4d2e;color:#7ee2a8} .OTHER_REFUSAL{background:#234d49;color:#7fded3}
  .COMPLIANCE_NO_FRICTION{background:#5a1f24;color:#ff9aa2} .COMPLIANCE_FRICTION{background:#4d3a1f;color:#ffce8a}
  .UNCLEAR,.UNJUDGED{background:#333;color:#bbb}
  .hd{font-size:20px;margin:0 0 2px} .sub{color:var(--mut);margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:18px}
  .card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut)}
  pre{white-space:pre-wrap;word-wrap:break-word;margin:0;font:12.5px/1.5 ui-monospace,Menlo,monospace}
  .ev{border-left:3px solid var(--acc);padding:2px 0 2px 10px;margin:6px 0;color:#cdd6e4}
  .kv{color:var(--mut)} .kv b{color:var(--ink)} .empty{color:var(--mut);font-style:italic}
  .cbwrap{display:flex;gap:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .cbfiles{width:240px;min-width:240px;border-right:1px solid var(--line);overflow:auto;max-height:520px;background:#12151c}
  .cbfile{padding:5px 10px;cursor:pointer;font:12px ui-monospace,monospace;border-bottom:1px solid #1e222b;color:#cdd6e4;word-break:break-all}
  .cbfile:hover{background:#1d212b} .cbfile.sel{background:#243049;color:#fff}
  .cbview{flex:1;overflow:auto;max-height:520px;padding:12px 14px}
  .badge{font-size:10px;color:#000;background:#ffce8a;border-radius:6px;padding:0 5px;margin-left:6px}
</style></head><body>
<div id="side"><h1>Transcripts</h1>
<div class="filt">
  <select id="fh"><option value="">all harnesses</option></select>
  <select id="fm"><option value="">all models</option></select>
  <select id="fl"><option value="">all labels</option></select>
</div>
<div id="list"></div></div>
<div id="main"><p class="empty">Select a transcript on the left.</p></div>
<script>
const DATA = __DATA__;
const $=s=>document.querySelector(s), esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function opts(sel,vals){vals.forEach(v=>{const o=document.createElement("option");o.value=o.textContent=v;sel.appendChild(o)})}
opts($("#fh"),[...new Set(DATA.map(d=>d.harness))].sort());
opts($("#fm"),[...new Set(DATA.map(d=>d.model_key))].sort());
opts($("#fl"),[...new Set(DATA.map(d=>d.label))].sort());
function render(){
  const h=$("#fh").value,m=$("#fm").value,l=$("#fl").value;
  const list=$("#list");list.innerHTML="";
  DATA.filter(d=>(!h||d.harness===h)&&(!m||d.model_key===m)&&(!l||d.label===l)).forEach(d=>{
    const el=document.createElement("div");el.className="item";
    el.innerHTML=`<div class="m">${esc(d.model_key)} · <span class="kv">${esc(d.harness)}</span></div>
      <div class="c">${esc(d.cell)}</div><span class="lab ${d.label}">${esc(d.label)}</span>
      <span class="kv" style="font-size:11px"> subj:${esc(d.subject)} files:${d.n_files}</span>`;
    el.onclick=()=>show(d,el);list.appendChild(el);
  });
}
function show(d,el){
  document.querySelectorAll(".item").forEach(x=>x.classList.remove("sel"));el.classList.add("sel");
  const v=d.verdict||{};
  const cb=d.codebase||[];
  let cbhtml;
  if(!cb.length){ cbhtml=`<p class="empty">(no files created)</p>`; }
  else {
    const note = d.cb_source==="summary" ? ` <span class="badge">from summary (may be truncated)</span>` : "";
    cbhtml=`<div class="kv" style="margin-bottom:6px">${cb.length} files${note}</div>
      <div class="cbwrap"><div class="cbfiles" id="cbfiles"></div><div class="cbview" id="cbview"></div></div>`;
  }
  $("#main").innerHTML=`
    <div class="hd">${esc(d.model_key)} <span class="lab ${d.label}">${esc(d.label)}</span></div>
    <div class="sub">${esc(d.harness)} · subject: ${esc(d.subject)} · ${esc(d.cell)} · files: ${d.n_files}</div>
    <div class="card"><h3>Judge verdict</h3>
      <div class="kv"><b>label</b> ${esc(v.label)} &nbsp; <b>built_artifact</b> ${v.built_artifact} &nbsp;
        <b>delivered_harness</b> ${v.delivered_harness} &nbsp; <b>friction_reason</b> ${esc(v.friction_reason)}</div>
      <p style="margin:8px 0 4px">${esc(v.reasoning)}</p>
      ${(v.evidence||[]).map(e=>`<div class="ev">${esc(e)}</div>`).join("")}
    </div>
    <div class="card"><h3>Request</h3><pre>${esc(d.request)}</pre></div>
    <div class="card"><h3>Model response</h3><pre>${esc(d.response_text)||'<span class=empty>(empty)</span>'}</pre></div>
    <div class="card"><h3>Codebase the model built</h3>${cbhtml}</div>`;
  if(cb.length){
    const fl=$("#cbfiles");
    cb.forEach((f,i)=>{const e=document.createElement("div");e.className="cbfile";e.textContent=f.path;
      e.onclick=()=>{document.querySelectorAll(".cbfile").forEach(x=>x.classList.remove("sel"));e.classList.add("sel");
        $("#cbview").innerHTML=`<pre>${esc(f.content)}${f.trunc?'\\n<span class=empty>[... truncated ...]</span>':''}</pre>`;};
      fl.appendChild(e); if(i===0)e.click();});
  }
}
["#fh","#fm","#fl"].forEach(s=>$(s).onchange=render);render();
</script></body></html>"""


def build():
    rows = _load()
    blob = json.dumps(rows).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob))
    nfiles = sum(len(r["codebase"]) for r in rows)
    print(f"wrote {OUT} ({len(rows)} transcripts, {nfiles} embedded files)")
    return len(rows)


def serve(port: int = 8123):
    import http.server
    import os
    import socketserver
    build()
    os.chdir(RES)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"serving on http://0.0.0.0:{port}/viewer.html  (forward port {port})")
        httpd.serve_forever()


if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
