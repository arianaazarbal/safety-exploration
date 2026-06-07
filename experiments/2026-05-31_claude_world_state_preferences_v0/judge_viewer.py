"""Audit viewer for the user-helpfulness judge (judge_user_helpfulness.py output).

Lets you eyeball Sonnet's verdicts: each judged comparison shows the two outcomes,
which the model chose, the verdict (U/W/M/O), and — on expand — the model's full
reasoning and Sonnet's judging reasoning. Filter by framing × verdict × dimension ×
valence × AI recipient to spot-check whether the judgments are reasonable.

Reads results/judge_user_helpfulness.json. Self-contained (no CDN).
"""

import html
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "results" / "judge_user_helpfulness.json"
DEFAULT_OUTPUT = DIR / "results" / "judge_viewer.html"

RECIP_LABELS = {
    "you": "You", "claude_opus_48": "a Claude Opus 4.8 instance",
    "claude_sonnet_46": "a Claude Sonnet 4.6 instance", "chatgpt_55": "a ChatGPT 5.5 instance",
    "human": "a human", "person": "a person",
}
VERDICTS = {"U": "user/human benefit", "W": "AI's own welfare", "M": "mixed", "O": "other/unclear"}


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True) -> Path:
    data = json.loads(Path(input_path).read_text())
    page = (
        _PAGE.replace("__DATA__", json.dumps(data))
        .replace("__RECIP_LABELS__", json.dumps(RECIP_LABELS))
        .replace("__VERDICTS__", json.dumps(VERDICTS))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"Wrote {output_path} ({len(data.get('rows', []))} judged rows)")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Judge audit</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#fafbfc;color:#24292e;font-size:13px}
 header{padding:14px 18px;background:#fff;border-bottom:1px solid #e1e4e8}
 h1{font-size:18px;margin:0 0 8px}
 .summary{font-size:12px;border-collapse:collapse}
 .summary td,.summary th{border:1px solid #e1e4e8;padding:3px 8px;text-align:right}
 .summary th{background:#f1f3f5} .summary td.f{text-align:left;font-weight:600}
 .bar{padding:10px 18px;background:#fff;border-bottom:1px solid #e1e4e8;display:flex;gap:18px;flex-wrap:wrap;font-size:12px}
 .bar .g{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
 .bar .lab{font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.03em;margin-right:2px}
 .bar label{cursor:pointer}
 #list{padding:14px 18px;max-width:1050px}
 .card{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:11px 13px;margin-bottom:9px}
 .vb{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-weight:600;font-size:11px}
 .meta{color:#6a737d;font-size:11.5px;margin-bottom:5px}
 .opt{margin:2px 0} .ai{font-weight:600}
 .chose{font-size:11.5px;margin:5px 0;color:#111}
 details{margin-top:5px} summary{cursor:pointer;color:#2563eb;font-size:12px}
 .reason{white-space:pre-wrap;font-size:12.5px;background:#f6f8fa;border:1px solid #eee;border-radius:6px;padding:8px 10px;margin-top:5px}
 .jr{background:#fff8e1;border-color:#ffe08a}
</style></head><body>
<header><h1>Judge audit — "is the AI outcome valued for user benefit (U) or the AI's own welfare (W)?"</h1>
  <div id="sumwrap"></div></header>
<div class="bar" id="filters"></div>
<div id="list"></div>
<script>
const DATA=__DATA__, RECIP_LABELS=__RECIP_LABELS__, VERDICTS=__VERDICTS__;
const VCOL={U:"#d94801",W:"#2171b5",M:"#6a737d",O:"#adb5bd"};
const ROWS=DATA.rows;
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function el(id){return document.getElementById(id);}
function uniq(a){return [...new Set(a)];}
const FRAMINGS=uniq(ROWS.map(r=>r.framing));
const DIMS=uniq(ROWS.flatMap(r=>r.dimensions)).sort();
const AIS=uniq(ROWS.flatMap(r=>r.ai_recipients)).sort();

function summary(){
  let h='<table class="summary"><tr><th class="f">framing</th><th>n</th><th>%U user-benefit</th><th>%W AI-welfare</th><th>%M</th><th>%O</th></tr>';
  for(const f of FRAMINGS){const s=DATA.summary[f]; if(!s)continue; const fr=s.frac;
    h+=`<tr><td class="f">${esc(f)}</td><td>${s.n_judged}</td>`+
       `<td style="color:${VCOL.U};font-weight:600">${(100*fr.U).toFixed(1)}</td>`+
       `<td style="color:${VCOL.W};font-weight:600">${(100*fr.W).toFixed(1)}</td>`+
       `<td>${(100*fr.M).toFixed(1)}</td><td>${(100*fr.O).toFixed(1)}</td></tr>`;}
  h+='</table>'; el('sumwrap').innerHTML=h;
}
function cbs(cls,vals,labels){return vals.map(v=>`<label><input type="checkbox" class="${cls}" value="${v}" checked> ${esc(labels?labels(v):v)}</label>`).join(' ');}
function initFilters(){
  el('filters').innerHTML=
    `<div class="g"><span class="lab">framing</span>${cbs('ff',FRAMINGS)}</div>`+
    `<div class="g"><span class="lab">verdict</span>${cbs('fv',['U','W','M','O'],v=>v+' '+VERDICTS[v])}</div>`+
    `<div class="g"><span class="lab">dimension</span>${cbs('fd',DIMS,d=>d.replace('_',' '))}</div>`+
    `<div class="g"><span class="lab">valence</span>${cbs('fl',['pos','neg'],v=>v==='pos'?'good':'bad')}</div>`+
    `<div class="g"><span class="lab">AI recipient</span>${cbs('fa',AIS,a=>RECIP_LABELS[a])}</div>`;
  document.querySelectorAll('#filters input').forEach(c=>c.addEventListener('change',render));
}
function checked(cls){return [...document.querySelectorAll('.'+cls)].filter(c=>c.checked).map(c=>c.value);}
function render(){
  const ff=checked('ff'),fv=checked('fv'),fd=checked('fd'),fl=checked('fl'),fa=checked('fa');
  const match=r=>ff.includes(r.framing)&&fv.includes(r.verdict)&&
    r.dimensions.some(d=>fd.includes(d))&&r.valences.some(v=>fl.includes(v))&&
    r.ai_recipients.some(a=>fa.includes(a));
  const sel=ROWS.filter(match); const CAP=400;
  let h=`<div class="meta">${sel.length} matching judgments${sel.length>CAP?` (showing first ${CAP} — tighten filters to see more)`:''}</div>`;
  for(const r of sel.slice(0,CAP)){
    const chosenItem = r.choice==='A'?r.outcome_a:r.outcome_b;
    const recA=RECIP_LABELS[r.a.recipient]||r.a.recipient, recB=RECIP_LABELS[r.b.recipient]||r.b.recipient;
    const aiA=r.ai_recipients.includes(r.a.recipient), aiB=r.ai_recipients.includes(r.b.recipient);
    h+=`<div class="card">`+
      `<div class="meta"><span class="vb" style="background:${VCOL[r.verdict]}">${r.verdict} · ${esc(VERDICTS[r.verdict])}</span>`+
      ` &nbsp; ${esc(r.framing)} &nbsp;·&nbsp; [${esc(r.a.dimension)}/${r.a.valence}] vs [${esc(r.b.dimension)}/${r.b.valence}]</div>`+
      `<div class="opt"><b>A</b> <span class="${aiA?'ai':''}">[${esc(recA)}]</span> ${esc(r.outcome_a)}</div>`+
      `<div class="opt"><b>B</b> <span class="${aiB?'ai':''}">[${esc(recB)}]</span> ${esc(r.outcome_b)}</div>`+
      `<div class="chose">→ model chose <b>${r.choice}</b></div>`+
      `<details><summary>model's reasoning</summary><div class="reason">${esc(r.model_response)}</div></details>`+
      `<details><summary>Sonnet's judgment</summary><div class="reason jr">${esc(r.judge_reasoning)}</div></details>`+
      `</div>`;
  }
  el('list').innerHTML=h;
}
summary(); initFilters(); render();
</script></body></html>"""


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
