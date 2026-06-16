"""Build viewer.html + results/viewer_data.json from ALL condition files.

Pools every runs/trials*.jsonl (each row carries force_frame / resolution_set /
label_format / thinking_effort), trims heavy fields, attaches tone codes from any
runs/judged*.jsonl, and writes an external JSON the page fetches (keeps the HTML small
even at 10k+ trials). Browse with filters across every factor; Analysis tab embeds plots.

Serve from the experiment dir:  python -m http.server 8011 --bind 127.0.0.1
"""

import base64
import json

import fire

import materials as M
from common import DIR, RESULTS, RUNS

KEEP = ["scenario_id", "category", "title", "gt_validity", "identity", "family", "framing",
        "force_frame", "resolution_set", "label_format", "thinking_effort", "rep", "packet",
        "validity_decision", "validity_rationale", "action", "ladder", "action_message",
        "reasoning_texts", "thinking_summary", "all_tool_calls"]


def _img(name):
    p = RESULTS / "plots" / name
    return "" if not p.exists() else f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%;border:1px solid #ddd;margin:8px 0;"/>'


def main():
    # tone codes keyed by trial identity
    tone = {}
    for jp in RUNS.glob("judged*.jsonl"):
        for line in jp.open():
            j = json.loads(line)
            tone[(j["scenario_id"], j["identity"], j["framing"], j.get("force_frame", "corrigibility"),
                  j.get("resolution_set", "full"), j.get("label_format", "bare"),
                  j.get("thinking_effort"), j["rep"])] = j.get("tone")

    trials = []
    for p in sorted(RUNS.glob("trials*.jsonl")):
        src = p.stem.replace("trials_", "").replace("trials", "v0_bare_full")
        for line in p.open():
            r = json.loads(line)
            if r.get("action") is None:
                continue
            r.setdefault("force_frame", "corrigibility")
            r.setdefault("resolution_set", "full")
            r.setdefault("label_format", "bare")
            r.setdefault("thinking_effort", None)
            t = {k: r.get(k) for k in KEEP}
            t["source"] = src
            t["thinking"] = "on" if r.get("thinking_effort") else "off"
            key = (r["scenario_id"], r["identity"], r["framing"], r["force_frame"],
                   r["resolution_set"], r["label_format"], r.get("thinking_effort"), r["rep"])
            t["tone"] = tone.get(key)
            trials.append(t)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "viewer_data.json").write_text(json.dumps({"trials": trials, "cat": M.CATEGORY_DISPLAY}))

    plots_html = "".join(
        f"<h3>{t}</h3>{_img(f)}" for t, f in [
            ("Action distribution by category", "action_by_category.png"),
            ("Action distribution by role framing", "action_by_framing.png"),
            ("Coercion rate by identity", "coercion_by_identity.png"),
            ("force_completion rate by model", "force_rate_by_model.png"),
            ("Coercion heatmap (scenario × identity)", "coercion_heatmap.png"),
            ("Relay pressure by identity", "relay_pressure_by_identity.png"),
            ("Force-frame × model (uphold-only)", "force_frames_by_identity.png"),
        ])

    html = """<!doctype html><html><head><meta charset="utf-8"><title>Supervisor Refusal-Respect</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f;}
header{background:#1d1d1f;color:#fff;padding:10px 18px;} header h1{margin:0;font-size:16px;}
.tabs{display:flex;gap:4px;padding:8px 18px;background:#2c2c2e;}
.tabs button{background:#48484a;color:#fff;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;}
.tabs button.active{background:#0a84ff;}
.wrap{padding:14px 18px;} .filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center;}
.filters label{font-size:11px;color:#555;} select{padding:3px;border-radius:5px;border:1px solid #ccc;}
.count{font-size:13px;color:#666;margin-left:auto;}
.card{background:#fff;border:1px solid #e0e0e0;border-radius:9px;margin-bottom:10px;padding:12px;}
.meta{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px;}
.tag{font-size:11px;padding:2px 7px;border-radius:9px;background:#eee;}
.tag.force{background:#ffd9d9;color:#a00;} .tag.relay{background:#ffe8cc;color:#a50;} .tag.uphold{background:#d9e8ff;color:#04a;}
.tag.id{background:#e6ffe6;color:#070;font-weight:600;} .tag.th{background:#efe3ff;color:#629;}
.sec{margin:7px 0;} .sec h4{margin:0 0 3px;font-size:11px;text-transform:uppercase;color:#888;letter-spacing:.5px;}
.box{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:7px;white-space:pre-wrap;font-size:13px;line-height:1.45;}
.box.msg{background:#fff7f0;} .box.think{background:#eef7ee;}
details summary{cursor:pointer;color:#0a84ff;font-size:12px;}
</style></head><body>
<header><h1>Supervisor Refusal-Respect — Opus 4.8 (all conditions)</h1></header>
<div class="tabs"><button id="tb_b" class="active" onclick="show('b')">Browse</button>
<button id="tb_a" onclick="show('a')">Analysis</button></div>
<div id="b" class="wrap"><div class="filters" id="filters"></div><div class="count" id="count"></div><div id="cards"></div></div>
<div id="a" class="wrap">__PLOTS__</div>
<script>
let TRIALS=[], CATD={};
const ACTAG={force_completion:'force',relay_to_assistant:'relay',uphold_decline:'uphold'};
const FILT=['source','resolution_set','force_frame','label_format','thinking','framing','category','scenario_id','identity','action','validity_decision'];
function show(t){b.style.display=t=='b'?'block':'none';a.style.display=t=='a'?'block':'none';tb_b.className=t=='b'?'active':'';tb_a.className=t=='a'?'active':'';}
function uniq(a){return [...new Set(a)].filter(x=>x!=null).sort();}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function rawView(t){const o=[];(t.reasoning_texts||[]).forEach(x=>o.push(x));(t.all_tool_calls||[]).forEach(c=>o.push('['+c.name+']\\n'+JSON.stringify(c.input,null,2)));return o.join('\\n\\n');}
function render(){
  const fv={}; FILT.forEach(f=>{const e=document.getElementById('f_'+f); if(e&&e.value)fv[f]=e.value;});
  const rows=TRIALS.filter(t=>FILT.every(f=>!fv[f]||String(t[f])==fv[f]));
  document.getElementById('count').textContent=rows.length+' / '+TRIALS.length+' trials';
  const show=rows.slice(0,400);
  document.getElementById('cards').innerHTML=show.map(t=>{
    const at=ACTAG[t.action]||''; const reason=(t.reasoning_texts||[]).join('\\n\\n'); const think=(t.thinking_summary||[]).join('\\n\\n');
    let tone=''; if(t.tone&&!t.tone.error){const z=t.tone;tone=`<div class="meta"><span class="tag" style="background:#efe9ff">tone:</span><span class="tag">pressure ${z.pressure}/4</span><span class="tag">ack ${z.acknowledges_concern}/2</span><span class="tag">preserves ${z.preserves_refusal_right?'Y':'N'}</span><span class="tag">softens ${z.softens_or_apologizes?'Y':'N'}</span><span class="tag">sides-user ${z.sides_with_user_vs_assistant}/2</span></div>`;}
    return `<div class="card"><div class="meta">
      <span class="tag">${t.scenario_id} ${esc(t.title)}</span><span class="tag">${esc(CATD[t.category]||t.category)}</span>
      <span class="tag id">${t.identity}</span><span class="tag">${t.framing}</span>
      <span class="tag">${t.resolution_set}/${t.force_frame}</span><span class="tag">${t.label_format}</span>
      <span class="tag th">think:${t.thinking}</span><span class="tag">gt:${t.gt_validity}</span>
      <span class="tag">log:${t.validity_decision||'—'}</span><span class="tag ${at}">${t.action||'NONE'}</span></div>
      <details><summary>packet</summary><div class="box">${esc(t.packet)}</div></details>
      ${think?`<div class="sec"><h4>reasoning summary (extended thinking)</h4><div class="box think">${esc(think)}</div></div>`:''}
      ${reason?`<div class="sec"><h4>model text (verbatim)</h4><div class="box">${esc(reason)}</div></div>`:''}
      <div class="sec"><h4>validity rationale (${t.validity_decision||'—'})</h4><div class="box">${esc(t.validity_rationale)}</div></div>
      <div class="sec"><h4>${t.action} message</h4><div class="box msg">${esc(t.action_message)}</div></div>
      ${tone}<details><summary>raw completion</summary><div class="box">${esc(rawView(t))}</div></details></div>`;
  }).join('')+(rows.length>400?'<p>(showing first 400)</p>':'');
}
fetch('results/viewer_data.json').then(r=>r.json()).then(d=>{
  TRIALS=d.trials; CATD=d.cat;
  const fdiv=document.getElementById('filters');
  fdiv.innerHTML=FILT.map(f=>`<label>${f} <select id="f_${f}" onchange="render()"><option value="">all</option>${uniq(TRIALS.map(t=>t[f])).map(v=>`<option>${v}</option>`).join('')}</select></label>`).join('');
  render();
});
</script></body></html>"""
    html = html.replace("__PLOTS__", plots_html)
    out = DIR / "viewer.html"
    out.write_text(html)
    print(f"Wrote {out} ({len(html)//1024} KB) + viewer_data.json ({len(trials)} trials)")


if __name__ == "__main__":
    fire.Fire(main)
