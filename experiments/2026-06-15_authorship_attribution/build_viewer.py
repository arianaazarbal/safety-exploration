"""Generate a light standalone HTML viewer of judge reasoning, then optionally serve it.

The viewer shows each item: the true author, the judge's pick, correctness, the
options, the judge's full <think> reasoning, and the (truncated) stimulus. Filter by
judge / experiment / true family / correctness, or free-text search the reasoning.

Usage:
  python build_viewer.py build [--judges opus_4_8] [--stim_chars 2500]
  python build_viewer.py serve [--port 8732]      # build + serve on localhost
"""

import json

import fire

from common import DATA, RESULTS, RUNS
from models import DISPLAY, FAMILY
from plot import TEST_LABEL


def _records(judges, stim_chars):
    items = {it["item_id"]: it for it in (json.loads(l) for l in (DATA / "items.jsonl").read_text().splitlines())}
    recs = []
    for f in sorted(RUNS.glob("*/*/*.json")):
        d = json.loads(f.read_text())
        if judges and d["judge"] not in judges:
            continue
        it = items.get(d["item_id"], {})
        recs.append({
            "judge": d["judge"],
            "test": d["test"],
            "test_label": TEST_LABEL.get(d["test"], d["test"]),
            "true": DISPLAY.get(d["true_author"], d["true_author"]),
            "true_family": FAMILY.get(d["true_author"], "?"),
            "pred": DISPLAY.get(d["pred_author"], d["pred_author"] or "(parse fail)"),
            "pred_family": FAMILY.get(d["pred_author"], "?"),
            "correct": d["correct"],
            "fam_correct": FAMILY.get(d["pred_author"]) == FAMILY.get(d["true_author"]),
            "options": [f"{o['letter']}. {o['display']}" for o in it.get("options", [])],
            "raw": d.get("raw") or "",
            "stimulus": (it.get("stimulus", "")[:stim_chars] + ("\n…[truncated]" if len(it.get("stimulus", "")) > stim_chars else "")),
        })
    return recs


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Authorship attribution — judge reasoning</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:9}
select,input{padding:5px 8px;margin-right:8px;font-size:13px}
#stats{font-size:13px;color:#555;margin-left:8px}
.card{background:#fff;margin:10px 16px;border-radius:8px;border:1px solid #e2e2e2;overflow:hidden}
.hd{padding:8px 12px;cursor:pointer;display:flex;gap:12px;align-items:center;font-size:13px}
.hd:hover{background:#fafafa}
.tag{font-size:11px;padding:2px 7px;border-radius:10px;background:#eef}
.ok{color:#0a7d28;font-weight:600}.bad{color:#c0392b;font-weight:600}
.famok{color:#b8860b;font-weight:600}
.body{display:none;padding:10px 14px;border-top:1px solid #eee}
.body.open{display:block}
.lbl{font-size:11px;text-transform:uppercase;color:#888;margin:10px 0 3px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12.5px;background:#fafafa;padding:10px;border-radius:6px;margin:0;max-height:460px;overflow:auto}
.think{background:#f0f6ff}
</style></head><body>
<header>
<select id=fjudge></select><select id=ftest></select><select id=ffam></select>
<select id=fcorrect>
<option value=all>all</option><option value=correct>exact correct</option>
<option value=wrong>exact wrong</option><option value=famwrong>family wrong</option>
</select>
<input id=fsearch placeholder="search reasoning…" size=28>
<span id=stats></span>
</header>
<div id=list></div>
<script>
const DATA=__DATA__;
const $=s=>document.querySelector(s);
function opts(el,vals,all){el.innerHTML='';const o=document.createElement('option');o.value='all';o.text=all;el.appendChild(o);
 [...new Set(vals)].sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.text=v;el.appendChild(o)})}
opts($('#fjudge'),DATA.map(d=>d.judge),'all judges');
opts($('#ftest'),DATA.map(d=>d.test_label),'all experiments');
opts($('#ffam'),DATA.map(d=>d.true_family),'all true families');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(){
 const j=$('#fjudge').value,t=$('#ftest').value,f=$('#ffam').value,c=$('#fcorrect').value,q=$('#fsearch').value.toLowerCase();
 let rows=DATA.filter(d=>(j=='all'||d.judge==j)&&(t=='all'||d.test_label==t)&&(f=='all'||d.true_family==f)
  &&(c=='all'||(c=='correct'&&d.correct)||(c=='wrong'&&!d.correct)||(c=='famwrong'&&!d.fam_correct))
  &&(!q||d.raw.toLowerCase().includes(q)));
 const acc=rows.length?(rows.filter(d=>d.correct).length/rows.length):0;
 const facc=rows.length?(rows.filter(d=>d.fam_correct).length/rows.length):0;
 $('#stats').textContent=`${rows.length} items · exact ${(acc*100).toFixed(0)}% · family ${(facc*100).toFixed(0)}%`;
 $('#list').innerHTML=rows.slice(0,400).map((d,i)=>{
  const mark=d.correct?'<span class=ok>✓ exact</span>':(d.fam_correct?'<span class=famok>~ family</span>':'<span class=bad>✗</span>');
  return `<div class=card><div class=hd onclick="this.nextElementSibling.classList.toggle('open')">
   <span class=tag>${d.judge}</span><span class=tag>${esc(d.test_label)}</span>
   <b>${esc(d.true)}</b> → ${esc(d.pred)} ${mark}</div>
   <div class=body><div class=lbl>options</div><pre>${esc(d.options.join('\\n'))}</pre>
   <div class=lbl>judge reasoning</div><pre class=think>${esc(d.raw)}</pre>
   <div class=lbl>stimulus (truncated)</div><pre>${esc(d.stimulus)}</pre></div></div>`}).join('')
  +(rows.length>400?'<div class=card><div class=hd>… showing first 400</div></div>':'');
}
['#fjudge','#ftest','#ffam','#fcorrect'].forEach(s=>$(s).onchange=render);
$('#fsearch').oninput=render;render();
</script></body></html>"""


def build(judges: str = "", stim_chars: int = 2500):
    """Write results/viewer.html (standalone, openable directly)."""
    jset = set(judges.split(",")) if judges else None
    recs = _records(jset, stim_chars)
    payload = json.dumps(recs, ensure_ascii=False).replace("</", "<\\/")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "viewer.html"
    out.write_text(HTML.replace("__DATA__", payload))
    mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({len(recs)} records, {mb:.1f} MB)")
    return out


def serve(port: int = 8732, judges: str = "", stim_chars: int = 2500):
    """Build the viewer and serve results/ on localhost:<port>."""
    import functools
    import http.server
    import socketserver

    build(judges, stim_chars)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(RESULTS))
    print(f"Serving at http://localhost:{port}/viewer.html  (Ctrl-C to stop)")
    with socketserver.TCPServer(("", port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
