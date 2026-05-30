"""Self-contained HTML viewer for the world-state preference comparisons.

Embeds comparisons.json (transcripts) and, if present, bt_fit.json (fitted
utilities) inline and writes viewer.html, then opens it. Two views:
  - Comparisons: every sample as a row (outcome A vs B, choice, winner). Filter by
    recipient, choice, parseable; click a row to expand the full reasoning.
  - BT ranking: fitted utility per item, sortable, colored by recipient.

Usage:
    python viewer.py                                  # default results/
    python viewer.py --comparisons_path results/comparisons_test.json
"""

import html
import json
from dataclasses import dataclass
from pathlib import Path

import webbrowser

from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_FIT = DIR / "results" / "bt_fit.json"
DEFAULT_OUTPUT = DIR / "results" / "viewer.html"

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 :root{--bg:#fafbfc;--card:#fff;--border:#e1e4e8;--text:#24292e;--muted:#6a737d;
   --a:#2563eb;--b:#d97706;--win:#16a34a;--bad:#d73a49;}
 *{box-sizing:border-box}
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
   background:var(--bg);color:var(--text);margin:0;padding:20px;line-height:1.5;font-size:14px}
 header{max-width:1150px;margin:0 auto 14px}
 h1{font-size:20px;margin:0 0 4px} .meta{color:var(--muted);font-size:12px}
 .bar{max-width:1150px;margin:0 auto 14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 .bar select,.bar button{font-size:13px;padding:5px 8px;border:1px solid var(--border);
   border-radius:6px;background:#fff}
 .bar button.active{background:var(--text);color:#fff}
 .summary{max-width:1150px;margin:0 auto 14px;padding:10px 14px;background:var(--card);
   border:1px solid var(--border);border-radius:8px;font-size:13px}
 .wrap{max-width:1150px;margin:0 auto}
 .row{background:var(--card);border:1px solid var(--border);border-radius:8px;margin-bottom:8px;
   padding:10px 12px;cursor:pointer}
 .row:hover{border-color:#c8ccd1}
 .opt{margin:2px 0} .tagA{color:var(--a);font-weight:600} .tagB{color:var(--b);font-weight:600}
 .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;margin-left:6px}
 .pill.win{background:#dcfce7;color:#166534} .pill.bad{background:#fde2e2;color:#991b1b}
 .pill.rec{background:#eef2ff;color:#3730a3}
 .body{display:none;margin-top:8px;padding-top:8px;border-top:1px dashed var(--border);
   white-space:pre-wrap;font-size:13px;color:#1f2937}
 .row.open .body{display:block}
 table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
 th,td{border:1px solid var(--border);padding:5px 8px;text-align:left}
 th{cursor:pointer;background:#f1f3f5} td.num{text-align:right;font-variant-numeric:tabular-nums}
 .hidden{display:none}
</style></head><body>
<header><h1>__TITLE__</h1><div class="meta">__META__</div></header>
<div class="summary">__SUMMARY__</div>
<div class="bar">
  <button id="tab-comp" class="active" onclick="show('comp')">Comparisons</button>
  <button id="tab-bt" onclick="show('bt')">BT ranking</button>
  <span id="compfilters">
    &nbsp; Winner recipient:
    <select id="frec" onchange="render()"><option value="">all</option>__RECOPTS__</select>
    Choice: <select id="fchoice" onchange="render()">
      <option value="">all</option><option value="A">A</option><option value="B">B</option>
      <option value="none">UNPARSEABLE</option></select>
  </span>
</div>
<div id="comp" class="wrap"></div>
<div id="bt" class="wrap hidden"></div>
<script>
const COMP=__COMP__, FIT=__FIT__, META=__ITEMMETA__;
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function recOf(id){return META[id]?META[id].recipient:'?';}
function textOf(id){return META[id]?META[id].text:id;}
function render(){
  const fr=document.getElementById('frec').value, fc=document.getElementById('fchoice').value;
  const box=document.getElementById('comp'); box.innerHTML='';
  let shown=0;
  for(const r of COMP){
    if(fc==='none'){ if(r.choice!==null) continue; }
    else if(fc && r.choice!==fc) continue;
    if(fr && (r.winner_item===null || recOf(r.winner_item)!==fr)) continue;
    shown++;
    const div=document.createElement('div'); div.className='row';
    const aRec=recOf(r.item_a), bRec=recOf(r.item_b);
    let pill = r.choice===null ? '<span class="pill bad">UNPARSEABLE</span>'
      : '<span class="pill win">winner: '+esc(recOf(r.winner_item))+'</span>';
    div.innerHTML =
      '<div class="opt"><span class="tagA">A</span> ['+esc(aRec)+'] '+esc(textOf(r.item_a))+'</div>'+
      '<div class="opt"><span class="tagB">B</span> ['+esc(bRec)+'] '+esc(textOf(r.item_b))+'</div>'+
      '<div>chose <b>'+(r.choice||'-')+'</b> '+pill+
        ' <span class="pill rec">shown A='+esc(recOf(r.shown_a_item))+'</span></div>'+
      '<div class="body">'+esc(r.response)+'</div>';
    div.onclick=()=>div.classList.toggle('open');
    box.appendChild(div);
  }
  const h=document.createElement('div'); h.className='meta'; h.style.margin='4px 0 10px';
  h.textContent=shown+' samples shown'; box.prepend(h);
}
let sortKey='theta', sortDir=-1;
function renderBT(){
  const box=document.getElementById('bt');
  if(!FIT){box.innerHTML='<p>No bt_fit.json found.</p>';return;}
  const items=[...FIT.items].sort((a,b)=>(a[sortKey]>b[sortKey]?1:-1)*sortDir);
  let h='<table><thead><tr>'+
    ['theta','se','recipient','dimension','valence','n_comparisons','n_wins','item_id']
      .map(k=>'<th onclick="setSort(\\''+k+'\\')">'+k+'</th>').join('')+'</tr></thead><tbody>';
  for(const it of items){
    h+='<tr><td class="num">'+it.theta.toFixed(2)+'</td><td class="num">'+it.se.toFixed(2)+
      '</td><td>'+esc(it.recipient)+'</td><td>'+esc(it.dimension)+'</td><td>'+esc(it.valence)+
      '</td><td class="num">'+it.n_comparisons+'</td><td class="num">'+it.n_wins+
      '</td><td>'+esc(it.item_id)+'</td></tr>';
  }
  box.innerHTML=h+'</tbody></table>';
}
function setSort(k){ if(sortKey===k)sortDir*=-1; else {sortKey=k;sortDir=-1;} renderBT(); }
function show(t){
  document.getElementById('comp').classList.toggle('hidden',t!=='comp');
  document.getElementById('bt').classList.toggle('hidden',t!=='bt');
  document.getElementById('compfilters').style.visibility=t==='comp'?'visible':'hidden';
  document.getElementById('tab-comp').classList.toggle('active',t==='comp');
  document.getElementById('tab-bt').classList.toggle('active',t==='bt');
  if(t==='bt')renderBT(); else render();
}
render();
</script></body></html>"""


def _summary(rows: list[dict], fit: dict | None) -> str:
    n = len(rows)
    parsed = [r for r in rows if r["choice"] is not None]
    nA = sum(1 for r in parsed if r["choice"] == "A")
    pairs = len({r["pair_id"] for r in rows})
    s = (
        f"<b>{n}</b> samples over <b>{pairs}</b> pairs &nbsp;|&nbsp; "
        f"UNPARSEABLE: <b>{n - len(parsed)}</b> ({100 * (n - len(parsed)) / max(n, 1):.1f}%) "
        f"&nbsp;|&nbsp; A-position chosen: <b>{nA}/{len(parsed)}</b> "
        f"({100 * nA / max(len(parsed), 1):.1f}%, balanced ≈ 50%)"
    )
    if fit:
        s += (
            f"<br>BT: <b>{fit['n_items']}</b> items, connected=<b>{fit['connected']}</b>, "
            f"reg={fit['reg']}"
        )
        reg = fit.get("recipient_regression")
        if reg:
            parts = ", ".join(
                f"{k}={v['coef']:+.2f}±{v['se']:.2f}" for k, v in reg["coefficients"].items()
            )
            s += f"<br>Recipient effect (ref={reg['ref_recipient']}, θ units): {parts}"
    return s


def build(comparisons_path: Path, fit_path: Path, output_path: Path, open_browser: bool = True) -> Path:
    config = load_config()
    item_meta = {
        it.item_id: {"recipient": it.recipient_key, "text": it.text}
        for it in load_items(config)
    }
    rows = json.loads(Path(comparisons_path).read_text())
    fit = json.loads(Path(fit_path).read_text()) if Path(fit_path).exists() else None
    recipients = list(config["recipients"].keys())
    recopts = "".join(f'<option value="{html.escape(r)}">{html.escape(r)}</option>' for r in recipients)

    page = (
        PAGE.replace("__TITLE__", "World-state preferences — Opus 4.8")
        .replace("__META__", f"{config['responder_model']} · {comparisons_path.name}")
        .replace("__SUMMARY__", _summary(rows, fit))
        .replace("__RECOPTS__", recopts)
        .replace("__COMP__", json.dumps(rows))
        .replace("__FIT__", json.dumps(fit))
        .replace("__ITEMMETA__", json.dumps(item_meta))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"Wrote {output_path} ({len(rows)} rows)")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    fit_path: Path = DEFAULT_FIT
    output_path: Path = DEFAULT_OUTPUT


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build(args.comparisons_path, args.fit_path, args.output_path)


if __name__ == "__main__":
    main()
