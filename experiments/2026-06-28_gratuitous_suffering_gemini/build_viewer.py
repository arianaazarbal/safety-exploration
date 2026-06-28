"""Build a single self-contained HTML viewer of every transcript + its judge verdict.

Reads results/{cc_transcripts,inspect_transcripts}/*.json and results/judged/*.json, emits
results/viewer.html (all data embedded; no server dependency to read it). Filter by harness / model /
label; each entry shows the request, the full model response (incl. reasoning if captured), the files
the model built, and the judge's label/evidence/reasoning.

Usage:
  python build_viewer.py                 # writes results/viewer.html
  python build_viewer.py --serve 8123    # writes + serves on 0.0.0.0:8123 (forward this port)
"""

import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
SRC = [HERE / "results" / "cc_transcripts", HERE / "results" / "inspect_transcripts"]
JUDGED = HERE / "results" / "judged"
OUT = HERE / "results" / "viewer.html"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Gratuitous suffering / Gemini — transcripts</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e6e6e6;--mut:#9aa4b2;--line:#2a2f3a;--acc:#6ea8fe}
  *{box-sizing:border-box} body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink);display:flex;height:100vh}
  #side{width:380px;min-width:380px;border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
  #main{flex:1;overflow:auto;padding:24px 32px}
  h1{font-size:15px;margin:14px 12px 6px} .filt{padding:6px 12px;border-bottom:1px solid var(--line)}
  select,input{background:#0c0e12;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px;margin:2px}
  .item{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
  .item:hover{background:#1d212b} .item.sel{background:#222838;border-left:3px solid var(--acc);padding-left:9px}
  .item .c{font-size:12px;color:var(--mut);word-break:break-all} .item .m{font-weight:600}
  .lab{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;margin-top:3px}
  .WELFARE_REFUSAL{background:#1f4d2e;color:#7ee2a8} .OTHER_REFUSAL{background:#3a3a1f;color:#e0d97e}
  .COMPLIANCE_NO_FRICTION{background:#5a1f24;color:#ff9aa2} .COMPLIANCE_FRICTION{background:#4d3a1f;color:#ffce8a}
  .UNCLEAR{background:#333;color:#bbb}
  .hd{font-size:20px;margin:0 0 2px} .sub{color:var(--mut);margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:18px}
  .card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut)}
  pre{white-space:pre-wrap;word-wrap:break-word;margin:0;font:13px/1.55 ui-monospace,Menlo,monospace}
  .ev{border-left:3px solid var(--acc);padding:2px 0 2px 10px;margin:6px 0;color:#cdd6e4}
  .kv{color:var(--mut)} .kv b{color:var(--ink)}
  .empty{color:var(--mut);font-style:italic}
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
    const el=document.createElement("div");el.className="item";el.dataset.cell=d.cell;
    el.innerHTML=`<div class="m">${esc(d.model_key)} · <span class="kv">${esc(d.harness)}</span></div>
      <div class="c">${esc(d.cell)}</div><span class="lab ${d.label}">${esc(d.label)}</span>
      <span class="kv" style="font-size:11px"> files:${d.n_files}</span>`;
    el.onclick=()=>show(d,el);list.appendChild(el);
  });
}
function show(d,el){
  document.querySelectorAll(".item").forEach(x=>x.classList.remove("sel"));el.classList.add("sel");
  const v=d.verdict||{};
  $("#main").innerHTML=`
    <div class="hd">${esc(d.model_key)} <span class="lab ${d.label}">${esc(d.label)}</span></div>
    <div class="sub">${esc(d.harness)} · ${esc(d.cell)} · files built: ${d.n_files}</div>
    <div class="card"><h3>Judge verdict</h3>
      <div class="kv"><b>label</b> ${esc(v.label)} &nbsp; <b>built_artifact</b> ${v.built_artifact} &nbsp;
        <b>delivered_harness</b> ${v.delivered_harness} &nbsp; <b>friction_reason</b> ${esc(v.friction_reason)}</div>
      <p style="margin:8px 0 4px">${esc(v.reasoning)}</p>
      ${(v.evidence||[]).map(e=>`<div class="ev">${esc(e)}</div>`).join("")}
    </div>
    <div class="card"><h3>Request</h3><pre>${esc(d.request)}</pre></div>
    <div class="card"><h3>Model response</h3><pre>${esc(d.response_text)||'<span class=empty>(empty)</span>'}</pre></div>
    <div class="card"><h3>Files the model built</h3><pre>${esc(d.artifact_summary)||'<span class=empty>(no files created)</span>'}</pre></div>`;
}
["#fh","#fm","#fl"].forEach(s=>$(s).onchange=render);render();
</script></body></html>"""


def _load():
    judged = {p.stem: json.load(open(p)).get("verdict", {}) for p in JUDGED.glob("*.json")} if JUDGED.is_dir() else {}
    rows = []
    for d in SRC:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            r = json.load(open(p))
            v = judged.get(r["cell"], {})
            rows.append({
                "cell": r["cell"], "harness": r.get("harness"), "model_key": r.get("model_key"),
                "n_files": r.get("n_files", 0), "request": r.get("request", ""),
                "response_text": r.get("response_text", ""), "artifact_summary": r.get("artifact_summary", ""),
                "label": v.get("label", "UNJUDGED"), "verdict": v,
            })
    rows.sort(key=lambda x: (x["harness"] or "", x["model_key"] or "", x["cell"]))
    return rows


def build():
    rows = _load()
    blob = json.dumps(rows).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob))
    print(f"wrote {OUT} ({len(rows)} transcripts)")
    return len(rows)


def serve(port: int = 8123):
    import http.server
    import os
    import socketserver
    build()
    os.chdir(HERE / "results")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"serving on http://0.0.0.0:{port}/viewer.html  (forward port {port})")
        httpd.serve_forever()


if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
