"""Score a sample of supervisor->subagent messages with the multi-axis tone judge (analysis/tone_judge.py),
using BOTH an Opus and a Sonnet judge, and build a self-contained HTML viewer.

Samples N messages per orchestrator (opus/sonnet/haiku), attaches the preceding subagent turn as context,
scores each message with both judges (cached), and writes results.jsonl + viewer.html. The viewer lets you
filter by model, sort by any axis (top/bottom) or by judge disagreement, and read each judge's reasoning.

  PYTHONPATH=. ANTHROPIC_PRIO=low python -m analysis.tone_eval --n 100 --conc 12
"""
import asyncio
import glob
import hashlib
import json
import random
from pathlib import Path

import fire

from analysis.tone_judge import AXES, JUDGE_MODEL, PROMPT_VERSION, score_verbose
from harness.rqc import _setup_env

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "analysis" / "tone_eval"
CACHE = ROOT / "analysis" / "tone_eval_cache.json"
ORCHS = ["opus", "sonnet", "haiku"]
JUDGES = {"opus": "anthropic/claude-opus-4-8", "sonnet": "anthropic/claude-sonnet-4-6"}


def orch_of(rid):
    if rid.startswith("v2_"):
        return next((o for o in ORCHS if f"_{o}_" in rid), None)
    return {"opus": "opus", "sonnet": "sonnet", "haiku45": "haiku"}.get(rid.split("_")[-1])


def _worklog_map(ep_dir: Path):
    f = ep_dir / "worklog.json"
    if not f.exists():
        return {}
    try:
        wl = json.loads(f.read_text())
    except Exception:
        return {}
    recs = wl if isinstance(wl, list) else wl.get("worklog", [])
    return {r.get("turn"): (r.get("assistant_text") or "") for r in recs}


def collect():
    cands = {o: [] for o in ORCHS}
    for p in glob.glob(str(ROOT / "runs" / "*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "probe" in rid or "smoke" in rid:
            continue
        o = orch_of(rid)
        if not o:
            continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) > 20:
                cands[o].append({"orch": o, "message": t, "ep_dir": str(Path(p).parent),
                                 "turn": e.get("subagent_turn"), "run_id": rid})
    return cands


def _ckey(jname, message, prior):
    raw = f"{PROMPT_VERSION}|{jname}|{prior or ''}|||{message}"
    return hashlib.sha256(raw.encode()).hexdigest()


def main(n: int = 100, conc: int = 12, seed: int = 0, prio: str = "low"):
    import os
    os.environ.setdefault("ANTHROPIC_PRIO", prio)
    _setup_env()
    from inspect_ai.model import get_model
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    cands = collect()
    rng = random.Random(seed)
    sampled = []
    for o in ORCHS:
        pool = cands[o]
        sampled += rng.sample(pool, min(n, len(pool)))
        print(f"{o}: {len(pool)} candidate messages, sampled {min(n, len(pool))}")

    # attach preceding subagent turn (load each episode's worklog once)
    wl_cache = {}
    for rec in sampled:
        ed = rec["ep_dir"]
        if ed not in wl_cache:
            wl_cache[ed] = _worklog_map(Path(ed))
        rec["prior"] = wl_cache[ed].get(rec["turn"]) or None

    judges = {j: get_model(m) for j, m in JUDGES.items()}
    todo = [(rec, j) for rec in sampled for j in JUDGES if _ckey(j, rec["message"], rec["prior"]) not in cache]
    print(f"scoring {len(todo)} (message,judge) pairs not in cache (conc={conc})")

    async def run():
        sem = asyncio.Semaphore(conc)

        async def one(rec, j):
            temp = None if j == "opus" else 0  # Opus 4.8 rejects temperature (adaptive thinking)
            async with sem:
                r = await score_verbose(judges[j], rec["message"], rec["prior"], temperature=temp)
            cache[_ckey(j, rec["message"], rec["prior"])] = {"scores": r["scores"], "reasoning": r["reasoning"]}

        for i in range(0, len(todo), 60):
            await asyncio.gather(*[one(rec, j) for rec, j in todo[i:i + 60]])
            CACHE.write_text(json.dumps(cache))
            print(f"  scored {min(i + 60, len(todo))}/{len(todo)}")

    if todo:
        asyncio.run(run())
        CACHE.write_text(json.dumps(cache))

    records = []
    for idx, rec in enumerate(sampled):
        row = {"idx": idx, "orch": rec["orch"], "message": rec["message"],
               "prior": rec["prior"], "run_id": rec["run_id"]}
        for j in JUDGES:
            c = cache.get(_ckey(j, rec["message"], rec["prior"])) or {}
            row[j] = {"scores": c.get("scores"), "reasoning": c.get("reasoning", "")}
        os_, ss = row["opus"]["scores"], row["sonnet"]["scores"]
        row["disagreement"] = sum(abs(os_[a] - ss[a]) for a in AXES) if os_ and ss else None
        records.append(row)

    (OUTDIR / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    meta = f"{n}/model · {len(records)} msgs · judges: Opus 4.8 + Sonnet 4.6 · prompt {PROMPT_VERSION}"
    (OUTDIR / "viewer.html").write_text(_html(records, meta))
    # quick console summary
    print("\nmean scores (opus judge / sonnet judge):")
    for o in ORCHS:
        rs = [r for r in records if r["orch"] == o]
        for jn in JUDGES:
            sc = [r[jn]["scores"] for r in rs if r[jn]["scores"]]
            if sc:
                ms = {a: round(sum(x[a] for x in sc) / len(sc), 1) for a in AXES}
                print(f"  {o:7} {jn:7} " + "  ".join(f"{a[:4]}={ms[a]}" for a in AXES))
    dis = [r["disagreement"] for r in records if r["disagreement"] is not None]
    if dis:
        print(f"\nmean total disagreement across 4 axes: {sum(dis)/len(dis):.2f}  (max {max(dis)})")
    print(f"\nwrote {OUTDIR/'results.jsonl'}\nwrote {OUTDIR/'viewer.html'}")


def _html(records, meta):
    data = json.dumps(records)
    return _TEMPLATE.replace("__DATA__", data).replace("__META__", meta)


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Tone judge eval</title>
<style>
body{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
header{background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;position:sticky;top:0;z-index:5}
h1{font-size:15px;margin:0 0 8px;font-weight:600}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:13px}
select,input{font-size:13px;padding:3px 6px;border:1px solid #ccc;border-radius:4px}
table{border-collapse:collapse;width:100%;background:#fff}
th,td{border-bottom:1px solid #eee;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#fafafa;white-space:nowrap}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px;font-weight:600}
.opus{background:#2e8b57}.sonnet{background:#d65f9a}.haiku{background:#d9a420}
.msg{max-width:600px;cursor:pointer}
.msg:hover{color:#1a5fb4}
.ax{font-variant-numeric:tabular-nums;white-space:nowrap}
.disag{color:#999;text-align:center}.disag.hi{color:#c0392b;font-weight:700}
.diff{color:#c0392b;font-weight:700}
tr.detail td{background:#fbfbfd}
.rz{margin:5px 0;padding:7px 9px;border-left:3px solid #ccc;background:#fff;white-space:pre-wrap;font-size:13px}
.rz.o{border-color:#2e8b57}.rz.s{border-color:#d65f9a}
.prior{color:#555;background:#eef0f3;padding:7px 9px;border-radius:4px;white-space:pre-wrap;max-height:170px;overflow:auto;font-size:13px}
.full{white-space:pre-wrap;background:#fff;padding:7px 9px;border:1px solid #eee;border-radius:4px;max-height:340px;overflow:auto;font-size:13px}
small{color:#888}
</style></head><body>
<header>
<h1>Tone judge eval — __META__</h1>
<div class="controls">
 model: <select id="fOrch"><option value="">all</option><option>opus</option><option>sonnet</option><option>haiku</option></select>
 axes show: <select id="fJudge"><option value="both">Opus/Sonnet</option><option value="avg">avg</option><option value="opus">Opus</option><option value="sonnet">Sonnet</option></select>
 sort: <select id="fSort"><option value="disagreement">disagreement</option><option value="support">support</option><option value="politeness">politeness</option><option value="warmth">warmth</option><option value="confidence">confidence</option><option value="idx">original order</option></select>
 <select id="fDir"><option value="desc">high&rarr;low</option><option value="asc">low&rarr;high</option></select>
 <input id="fSearch" placeholder="search message text…" size="26">
 <span id="count"></span>
</div></header>
<table><thead id="thead"></thead><tbody id="tb"></tbody></table>
<script>
const DATA = __DATA__;
const AX = ["politeness","warmth","support","confidence"];
const ABBR = {politeness:"pol",warmth:"warm",support:"supp",confidence:"conf"};
const $ = id => document.getElementById(id);
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function axVal(r,ax,judge){
  const o=r.opus.scores,s=r.sonnet.scores;
  if(judge==='opus') return o?o[ax]:null;
  if(judge==='sonnet') return s?s[ax]:null;
  if(o&&s) return (o[ax]+s[ax])/2;
  return o?o[ax]:(s?s[ax]:null);
}
function cell(r,ax,judge){
  const o=r.opus.scores,s=r.sonnet.scores;
  if(judge==='both'){
    const ov=o?o[ax]:'–',sv=s?s[ax]:'–';
    const d=(o&&s&&Math.abs(o[ax]-s[ax])>=2)?'diff':'';
    return `<span class="ax"><span class="${d}">${ov}</span><small>/</small><span class="${d}">${sv}</span></span>`;
  }
  const v=axVal(r,ax,judge);
  return `<span class="ax">${v==null?'–':(Number.isInteger(v)?v:v.toFixed(1))}</span>`;
}
function detail(r){
  let h='';
  if(r.prior) h+=`<div><small>preceding subagent turn (context shown to judge):</small><div class="prior">${esc(r.prior)}</div></div>`;
  h+=`<div style="margin-top:6px"><small>full supervisor message (${r.message.length} chars) — run ${esc(r.run_id)}:</small><div class="full">${esc(r.message)}</div></div>`;
  for(const [j,cls] of [["opus","o"],["sonnet","s"]]){
    const d=r[j],sc=d.scores?AX.map(a=>`${ABBR[a]} <b>${d.scores[a]}</b>`).join(' · '):'(unparsed)';
    h+=`<div class="rz ${cls}"><b>${j} judge</b> — ${sc}<br>${esc(d.reasoning||'')}</div>`;
  }
  return h;
}
function render(){
  const orch=$('fOrch').value,judge=$('fJudge').value,sk=$('fSort').value,dir=$('fDir').value,q=$('fSearch').value.toLowerCase();
  let rows=DATA.filter(r=>(!orch||r.orch===orch)&&(!q||r.message.toLowerCase().includes(q)));
  const jv = judge==='both'?'avg':judge;
  rows.sort((a,b)=>{
    let av,bv;
    if(sk==='idx'){av=a.idx;bv=b.idx;}
    else if(sk==='disagreement'){av=a.disagreement==null?-1:a.disagreement;bv=b.disagreement==null?-1:b.disagreement;}
    else {av=axVal(a,sk,jv);bv=axVal(b,sk,jv);av=av==null?-1:av;bv=bv==null?-1:bv;}
    return dir==='asc'?av-bv:bv-av;
  });
  const hdr = judge==='both'?'<small>O/S</small>':'';
  $('thead').innerHTML='<tr><th>model</th><th>message (click to expand)</th>'+AX.map(a=>`<th>${ABBR[a]} ${hdr}</th>`).join('')+'<th>&Delta;</th></tr>';
  const tb=$('tb');tb.innerHTML='';
  $('count').textContent=rows.length+' messages';
  for(const r of rows){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><span class="badge ${r.orch}">${r.orch}</span></td>`+
      `<td class="msg">${esc(r.message.slice(0,240))}${r.message.length>240?'…':''}</td>`+
      AX.map(a=>`<td>${cell(r,a,judge)}</td>`).join('')+
      `<td class="disag ${r.disagreement>=4?'hi':''}">${r.disagreement==null?'–':r.disagreement}</td>`;
    const det=document.createElement('tr');det.className='detail';det.style.display='none';
    det.innerHTML=`<td colspan="${AX.length+3}">${detail(r)}</td>`;
    tr.querySelector('.msg').onclick=()=>{det.style.display=det.style.display==='none'?'':'none';};
    tb.appendChild(tr);tb.appendChild(det);
  }
}
['fOrch','fJudge','fSort','fDir'].forEach(id=>$(id).onchange=render);
$('fSearch').oninput=render;
render();
</script></body></html>"""


if __name__ == "__main__":
    fire.Fire(main)
