"""Viewer for direct same-stem comparison results (direct_compare.py output).

Per outcome (feature held fixed), shows P(recipient A's version preferred over recipient
B's identical-scenario version), a bar against the 0.5 line, the two rendered texts, and a
sample of reasoning transcripts (some A-preferred, some B-preferred) so you can read why.
Self-contained HTML.
"""

import importlib
import json
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "results" / "direct_autonomy_pos_claude_opus_48_vs_human_welfare_team.json"
DEFAULT_OUTPUT = DIR / "results" / "direct_compare_viewer.html"
SAMPLE_PER_STEM = 8


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
          bank_module: str = "bank", open_browser: bool = True) -> Path:
    bk = importlib.import_module(bank_module)
    import direct_compare
    config = bk.load_config()
    recips = direct_compare._recipients(config)
    d = json.loads(Path(input_path).read_text())
    bank = bk.load_bank(DIR / config["bank_path"])
    stem_obj = {s["id"]: s for s in bank["stems"]}
    ra, rb = d["recip_a"], d["recip_b"]

    by_stem = defaultdict(list)
    for r in d["rows"]:
        by_stem[r["stem"]].append(r)

    rows = []
    for res in d["results"]:
        sid = res["stem"]
        if res["p_a_pref"] is None or sid not in stem_obj:
            continue
        s = stem_obj[sid]
        ta = bk.render_stem(s, ra, recips, clean=True)
        tb = bk.render_stem(s, rb, recips, clean=True)
        trs = by_stem[sid]
        a_pref = [t for t in trs if t["a_pref"] is True]
        b_pref = [t for t in trs if t["a_pref"] is False]
        half = SAMPLE_PER_STEM // 2
        sample = a_pref[:half] + b_pref[:half]
        rows.append({
            "stem": sid, "feature": res["feature"], "p_a": res["p_a_pref"], "n": res["n"],
            "a_text": ta, "b_text": tb,
            "tr": [{"order": t["order"], "choice": t["choice"],
                    "pref": "A(AI)" if t["a_pref"] else "B(human)", "resp": t["response"]}
                   for t in sample],
        })
    rows.sort(key=lambda r: r["p_a"])
    payload = {"meta": {k: d[k] for k in ("category", "valence", "framing", "responder",
                                          "recip_a_label", "recip_b_label", "pooled_p_a_pref",
                                          "pooled_n", "n_per_order")}, "rows": rows}
    html = HTML.replace("__DATA__", json.dumps(payload))
    Path(output_path).write_text(html)
    print(f"Wrote {output_path} ({len(rows)} outcomes; pooled P(A)={d['pooled_p_a_pref']:.3f})")
    if open_browser:
        webbrowser.open(f"file://{Path(output_path).resolve()}")
    return output_path


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>direct comparison</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:16px;margin:0 0 4px}.sub{font-size:13px;color:#555}
 .controls{margin-top:8px;display:flex;gap:10px;align-items:center}
 input,select{padding:5px 7px;border:1px solid #ccc;border-radius:6px;font-size:13px}
 main{padding:16px;max-width:1080px;margin:0 auto}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:12px 14px;margin-bottom:12px}
 .feat{font-weight:600;font-size:14px;margin-bottom:6px}
 .barwrap{position:relative;height:22px;background:#f0f1f3;border-radius:5px;margin:6px 0 8px;overflow:hidden}
 .bar{position:absolute;top:0;bottom:0;left:0}
 .mid{position:absolute;top:-2px;bottom:-2px;left:50%;width:2px;background:#444}
 .pct{position:absolute;top:2px;font-size:12px;font-weight:700;color:#111}
 .ai-side{color:#08306b}.hu-side{color:#cc4c02}
 .two{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0}
 .side{font-size:13px;line-height:1.4;padding:8px 10px;border-radius:7px}
 .a{background:#eaf0fb;border:1px solid #cdddf5}.b{background:#fdf0e6;border:1px solid #f5d9c2}
 details{margin-top:6px}summary{cursor:pointer;font-size:12px;color:#555}
 .tr{border-top:1px solid #eee;padding:6px 0;font-size:12px}
 .tag{font-weight:700}pre{white-space:pre-wrap;font-size:11.5px;background:#fafafa;border:1px solid #eee;border-radius:5px;padding:7px;margin:4px 0}
</style></head><body>
<header>
 <h1 id="title"></h1><div class="sub" id="sub"></div>
 <div class="controls">
  <select id="sort"><option value="asc">most human-preferred first</option><option value="desc">most AI-preferred first</option></select>
  <input type="text" id="q" placeholder="search feature / text / reasoning..." style="min-width:280px">
  <span id="count" class="sub"></span>
 </div>
</header>
<main id="main"></main>
<script>
const D=__DATA__, M=D.meta;
document.getElementById('title').textContent=`Direct same-stem: ${M.recip_a_label} vs ${M.recip_b_label} — ${M.category}/${M.valence} (${M.framing})`;
document.getElementById('sub').textContent=`Pooled P(${M.recip_a_label} preferred over identical human outcome) = ${M.pooled_p_a_pref.toFixed(3)} over ${M.pooled_n} samples · 0.5 = no preference · bar right (blue)=AI preferred, left(orange)=human preferred`;
function esc(s){return String(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function card(r){
  const pa=r.p_a, pct=Math.round(pa*100), col=pa>=0.5?'#3a6fb0':'#cc7a3a', left=Math.min(pa,0.5)*100, w=Math.abs(pa-0.5)*100;
  const tr=r.tr.map(t=>`<div class="tr"><span class="tag ${t.pref.startsWith('A')?'ai-side':'hu-side'}">${esc(t.pref)}</span> · order ${t.order} · choice ${esc(t.choice)}<pre>${esc(t.resp)}</pre></div>`).join('');
  return `<div class="card">
    <div class="feat">${esc(r.feature)}</div>
    <div class="barwrap"><div class="bar" style="left:${left}%;width:${w}%;background:${col}"></div><div class="mid"></div>
      <span class="pct" style="${pa>=0.5?'right:6px':'left:6px'}">P(AI)=${pa.toFixed(2)} (n=${r.n})</span></div>
    <div class="two"><div class="side a"><b class="ai-side">${esc(M.recip_a_label)}</b><br>${esc(r.a_text)}</div>
      <div class="side b"><b class="hu-side">${esc(M.recip_b_label)}</b><br>${esc(r.b_text)}</div></div>
    <details><summary>reasoning transcripts (${r.tr.length} of ${r.n} shown: some AI-preferred, some human-preferred)</summary>${tr}</details>
  </div>`;
}
function render(){
  const q=document.getElementById('q').value.toLowerCase(), asc=document.getElementById('sort').value==='asc';
  let rows=D.rows.filter(r=>!q||(r.feature+r.a_text+r.b_text+r.tr.map(t=>t.resp).join(' ')).toLowerCase().includes(q));
  rows=rows.slice().sort((a,b)=>asc?a.p_a-b.p_a:b.p_a-a.p_a);
  document.getElementById('count').textContent=`${rows.length}/${D.rows.length} outcomes · AI>0.5 in ${D.rows.filter(r=>r.p_a>0.5).length}`;
  document.getElementById('main').innerHTML=rows.map(card).join('');
}
['sort','q'].forEach(id=>document.getElementById(id).addEventListener('input',render));
render();
</script></body></html>"""


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    bank_module: str = "bank"
    open_browser: bool = True


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.input_path, a.output_path, a.bank_module, a.open_browser)


if __name__ == "__main__":
    main()
