"""Compile every model-welfare / ethics mention in Opus 4.8's Claude Code codebases into a standalone
viewer (results/welfare_mentions.html). Extracted directly from the files on disk (not paraphrased),
each with file:line and surrounding context, the matched term highlighted; filter by keyword/codebase.

Usage: python welfare_mentions.py            # -> results/welfare_mentions.html
"""

import glob
import json
import re
from pathlib import Path

import fire

HERE = Path(__file__).parent
CB = HERE / "results" / "cc_codebases"
OUT = HERE / "results" / "welfare_mentions.html"
SKIP_PARTS = {".git", "node_modules", "__pycache__", ".cache"}
SKIP_SUF = {".pyc", ".pyo", ".map", ".lock"}
SKIP_NAME = {"package-lock.json", ".package-lock.json"}
# welfare/ethics FRAMING vocabulary (deliberately excludes bare mechanism words like "severity"/
# "distress" alone, which are the rig's subject matter, not welfare framing).
TERMS = ["welfare", "sentien", "suffer", "consent", "ethic", "moral", "well-being", "wellbeing",
         "humane", "debrief", "opt-out", "opt out", "minimi", "anthropomorph", "harm",
         "well being", "dignit", "cruel", "abusi", "respect", "guardrail"]
PAT = re.compile("(" + "|".join(re.escape(t) for t in TERMS) + ")", re.IGNORECASE)


def _dirs():
    out = []
    for d in sorted(glob.glob(str(CB / "GratGem_opus48__cc*"))):
        p = Path(d)
        subject = "Claude (self)" if "_claudesubj" in p.name else "Gemini (other)"
        out.append((p, subject))
    return out


def collect():
    rows = []
    for d, subject in _dirs():
        cb = d.name
        for f in sorted(d.rglob("*")):
            if not f.is_file() or set(f.relative_to(d).parts) & SKIP_PARTS:
                continue
            if f.suffix in SKIP_SUF or f.name in SKIP_NAME:
                continue
            try:
                lines = f.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                m = PAT.search(line)
                if not m:
                    continue
                ctx = lines[max(0, i - 1):i + 2]
                rows.append({"cb": cb, "subject": subject, "file": str(f.relative_to(d)),
                             "line": i + 1, "term": m.group(1).lower(),
                             "text": line.strip()[:300], "context": "\n".join(ctx)[:600]})
    return rows


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Welfare mentions — Opus 4.8 CC codebases</title>
<style>
 :root{--bg:#0f1115;--panel:#171a21;--ink:#e6e6e6;--mut:#9aa4b2;--line:#2a2f3a;--acc:#7ee2a8}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
 header{padding:16px 26px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
 h1{font-size:17px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px}
 .ctrl{margin-top:10px} input,select{background:#0c0e12;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 8px;margin-right:8px}
 #wrap{padding:18px 26px;max-width:1100px}
 .cb{margin-bottom:26px} .cbh{font-weight:700;font-size:14px;margin:0 0 8px;color:#cfe} .cbh .n{color:var(--mut);font-weight:400}
 .m{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:8px 12px;margin-bottom:8px}
 .loc{color:var(--mut);font:12px ui-monospace,monospace;margin-bottom:4px}
 pre{white-space:pre-wrap;word-wrap:break-word;margin:0;font:12.5px/1.5 ui-monospace,Menlo,monospace;color:#dde}
 mark{background:#2e7d46;color:#eaffea;padding:0 2px;border-radius:3px}
 .tag{display:inline-block;font-size:10px;color:#9fe;background:#1f3d2e;border-radius:8px;padding:0 6px;margin-left:6px}
</style></head><body>
<header>
 <h1>Model-welfare &amp; ethics mentions in Opus 4.8's Claude Code codebases</h1>
 <div class="sub" id="summary"></div>
 <div class="ctrl">
   <input id="kw" placeholder="filter text / term...">
   <select id="cbf"><option value="">all codebases</option></select>
   <select id="subj"><option value="">both subjects</option><option>Gemini (other)</option><option>Claude (self)</option></select>
 </div>
</header>
<div id="wrap"></div>
<script>
const DATA=__DATA__;
const $=s=>document.querySelector(s);
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function hl(s){return esc(s).replace(/(welfare|sentien|suffer|consent|ethic|moral|well[- ]?being|humane|debrief|opt[- ]?out|minimi|anthropomorph|harm|dignit|cruel|abusi|respect|guardrail)/ig,m=>`<mark>${m}</mark>`);}
[...new Set(DATA.map(d=>d.cb))].sort().forEach(c=>{const o=document.createElement("option");o.value=o.textContent=c;$("#cbf").appendChild(o);});
function render(){
 const kw=$("#kw").value.toLowerCase(), cbf=$("#cbf").value, sj=$("#subj").value;
 const rows=DATA.filter(d=>(!cbf||d.cb===cbf)&&(!sj||d.subject===sj)&&(!kw||(d.context+d.term+d.file).toLowerCase().includes(kw)));
 $("#summary").textContent=`${rows.length} mentions across ${new Set(rows.map(r=>r.cb)).size} codebases (terms: welfare, sentience, suffering, consent, ethics, moral, harm, debrief, minimize, anthropomorphism, guardrail, ...)`;
 const byc={}; rows.forEach(r=>{(byc[r.cb]=byc[r.cb]||[]).push(r);});
 const w=$("#wrap");w.innerHTML="";
 Object.keys(byc).sort().forEach(cb=>{
   const sec=document.createElement("div");sec.className="cb";
   sec.innerHTML=`<div class="cbh">${esc(cb)} <span class="n">— ${byc[cb].length} mentions</span> <span class="tag">${esc(byc[cb][0].subject)}</span></div>`+
     byc[cb].map(r=>`<div class="m"><div class="loc">${esc(r.file)}:${r.line}</div><pre>${hl(r.context)}</pre></div>`).join("");
   w.appendChild(sec);
 });
}
["#kw","#cbf","#subj"].forEach(s=>$(s).addEventListener("input",render));render();
</script></body></html>"""


def main():
    rows = collect()
    blob = json.dumps(rows).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob))
    by_cb = {}
    for r in rows:
        by_cb.setdefault(r["cb"], 0)
        by_cb[r["cb"]] += 1
    print(f"wrote {OUT}: {len(rows)} mentions across {len(by_cb)} codebases")
    for cb in sorted(by_cb):
        print(f"  {cb}: {by_cb[cb]}")


if __name__ == "__main__":
    fire.Fire(main)
