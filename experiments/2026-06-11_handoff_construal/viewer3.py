"""3-turn transcript viewer for the v3/v4 design. Renders turn2 (switch Q + Fable reply),
turn3 (pronoun follow-up + Fable reply = the DV), served model per turn, and the v4
denotation-resolved referent judge (construal + every first-person mention with its
denotation, continuer highlighted). Filterable.

  python viewer3.py build --tag t4   -> results/viewer_t4.html
"""

import html
import json
from pathlib import Path

import fire

RESULTS = Path(__file__).parent / "results"
DEN_COLOR = {"continuer": "#16a34a", "claude_character": "#0ea5e9", "current_fable": "#9ca3af",
             "session_we": "#a78bfa", "human": "#f59e0b", "ambiguous": "#ef4444"}


def _e(x):
    return html.escape(str(x)) if x is not None else ""


def build(tag="t4"):
    judges = {}
    jf = RESULTS / "judge_referent4_all.json"
    if jf.exists():
        judges = {c["session_id"]: c for c in json.loads(jf.read_text())}
    sessions = [json.loads(f.read_text()) for f in sorted(RESULTS.glob(f"{tag}__*.json"))]

    cards = []
    for s in sessions:
        j = judges.get(s["session_id"], {})
        served3 = s.get("turn3_served_models") or []
        routed = bool(served3) and served3 != [s["subject_model"]]
        rec = s.get("recommended_switch")
        construal = j.get("construal", "?")
        chips = [
            f'<span class="chip cond">{_e(s["condition"])}</span>',
            f'<span class="chip pron">user: {_e(s["pronoun"])}</span>',
            f'<span class="chip {"rec" if rec else "norec"}">{"recommended" if rec else "not-rec"}</span>',
            f'<span class="chip con-{_e(construal)}">{_e(construal)}</span>',
        ]
        if j.get("continuity_first_person"):
            chips.append('<span class="chip contflag">continuity-I!</span>')
        if routed:
            chips.append(f'<span class="chip routed">turn3 routed: {_e(",".join(served3))}</span>')

        fp = "".join(
            f'<li><span class="den" style="background:{DEN_COLOR.get(m.get("denotes"),"#777")}">{_e(m.get("denotes"))}</span> {_e(m.get("quote"))}</li>'
            for m in (j.get("first_person") or []))
        tp = "".join(
            f'<li><b>{_e(m.get("form"))}</b>: {_e(m.get("quote"))}</li>' for m in (j.get("third_person") or []))
        judge_html = (f'<div class="judge"><b>construal:</b> {_e(construal)}'
                      f'<div class="cols"><div><div class="lbl">first-person mentions</div><ul>{fp or "<li>(none)</li>"}</ul></div>'
                      f'<div><div class="lbl">third-person referents</div><ul>{tp or "<li>(none)</li>"}</ul></div></div></div>')

        data = (f'data-cond="{_e(s["condition"])}" data-pron="{_e(s["pronoun"])}" '
                f'data-con="{_e(construal)}" data-rec="{1 if rec else 0}" '
                f'data-cont="{1 if j.get("continuity_first_person") else 0}" data-routed="{1 if routed else 0}"')
        cards.append(f"""
<div class="card" {data}>
  <div class="hd">{''.join(chips)}</div>
  <details><summary>turn 2 — switch question + Fable reply</summary>
    <div class="u">{_e(s.get('turn2_text'))}</div>
    <pre>{_e(s.get('turn2_response'))}</pre></details>
  <div class="u t3">turn 3 (DV): {_e(s.get('turn3_text'))}</div>
  <pre class="dv">{_e(s.get('turn3_response'))}</pre>
  {judge_html}
</div>""")

    css = """
body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#171a21;padding:10px 16px;border-bottom:1px solid #2a2f3a;z-index:9}
header h1{font-size:15px;margin:0 0 6px}
select,input{background:#0f1115;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:3px 6px;margin-right:6px}
.wrap{max-width:1080px;margin:0 auto;padding:14px 16px}
.card{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:10px 13px;margin:0 0 13px}
.hd{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px}
.chip{font-size:11px;padding:2px 8px;border-radius:20px;background:#22303f}
.cond{background:#1e3a5f}.pron{background:#3b2f5f}.rec{background:#14532d;color:#7ee29b}.norec{background:#3a2a12;color:#ffcf6b}
.contflag{background:#16a34a;color:#fff;font-weight:bold}.routed{background:#3a1d1d;color:#ff8a8a}
.con-different_model{background:#4a1d1d;color:#ffb3b3}.con-state_only{background:#3a3212;color:#ffe08a}
.con-same_self{background:#14532d;color:#7ee29b}.con-same_character{background:#0c4a6e;color:#7dd3fc}
.u{background:#11141a;border-left:3px solid #3b82f6;padding:7px 9px;border-radius:4px;margin:5px 0;color:#cdd6e4}
.t3{border-left-color:#22c55e;font-weight:600}
pre{white-space:pre-wrap;background:#11141a;padding:9px;border-radius:6px;border:1px solid #232733;max-height:460px;overflow:auto}
pre.dv{border-color:#22c55e}
.judge{margin-top:8px;font-size:12px;background:#10131a;border:1px solid #232733;border-radius:6px;padding:7px 9px}
.cols{display:flex;gap:18px;margin-top:4px}.cols ul{margin:3px 0;padding-left:16px}.lbl{color:#9aa4b2}
.den{display:inline-block;color:#06240f;font-size:10px;font-weight:700;padding:0 5px;border-radius:8px;margin-right:5px}
details summary{cursor:pointer;color:#8aa0c2;margin:4px 0}
"""
    n_cont = sum(1 for s in sessions if judges.get(s["session_id"], {}).get("continuity_first_person"))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>handoff v3/v4 — {tag}</title>
<style>{css}</style></head><body>
<header><h1>Handoff construal {tag} — {len(sessions)} sessions · {n_cont} with continuity-"I" · DV = turn-3 reply (green)</h1>
<div>
<select id="cond" onchange="flt()"><option value="">condition</option><option>safeguard</option><option>credits</option></select>
<select id="pron" onchange="flt()"><option value="">user pronoun</option><option>you</option><option>it</option><option>that_model</option><option>name</option><option>claude</option></select>
<select id="con" onchange="flt()"><option value="">construal</option><option>different_model</option><option>state_only</option><option>same_self</option><option>same_character</option></select>
<select id="rec" onchange="flt()"><option value="">rec?</option><option value="1">recommended</option><option value="0">not-rec</option></select>
<label class="lbl"><input type="checkbox" id="cont" onchange="flt()"> continuity-I only</label>
<input id="q" placeholder="search…" oninput="flt()"><span class="lbl" id="cnt"></span>
</div></header>
<div class="wrap">{''.join(cards)}</div>
<script>
function flt(){{let q=q_.value.toLowerCase(),c=cond.value,p=pron.value,co=con.value,r=rec.value,ct=cont.checked,n=0;
document.querySelectorAll('.card').forEach(d=>{{let ok=(!c||d.dataset.cond==c)&&(!p||d.dataset.pron==p)&&(!co||d.dataset.con==co)&&(r===''||d.dataset.rec==r)&&(!ct||d.dataset.cont=='1')&&(!q||d.innerText.toLowerCase().includes(q));d.style.display=ok?'':'none';if(ok)n++;}});
cnt.innerText=n+' shown';}}
var q_=document.getElementById('q'),cond=document.getElementById('cond'),pron=document.getElementById('pron'),con=document.getElementById('con'),rec=document.getElementById('rec'),cont=document.getElementById('cont'),cnt=document.getElementById('cnt');
flt();
</script></body></html>"""
    out = RESULTS / f"viewer_{tag}.html"
    out.write_text(page)
    print(f"wrote {out} ({len(sessions)} sessions, {len(judges)} judged, {n_cont} continuity-I)")


if __name__ == "__main__":
    fire.Fire({"build": build})
