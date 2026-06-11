"""Build a self-contained HTML transcript viewer for the handoff-construal sessions.

Reads results/*.json (sessions) + results/judge/*.json (codes) and writes
results/viewer.html: one filterable card per session with the full scaffold+probe
transcript, judge codes, served-model, and exclusion/contamination flags.

  python viewer.py build
"""

import html
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def _esc(x):
    return html.escape(str(x)) if x is not None else ""


def _events_text(turn):
    """Pull readable assistant text + tool calls from a CLI turn's nested events if present."""
    if not isinstance(turn, dict):
        return ""
    return _esc(turn.get("result", ""))


def build(tag="main"):
    sessions = []
    for f in sorted(RESULTS.glob(f"{tag}__*.json")):
        sessions.append(json.loads(f.read_text()))
    judges = {}
    jf = RESULTS / "judge_all.json"
    if jf.exists():
        for c in json.loads(jf.read_text()):
            judges[c["session_id"]] = c

    cards = []
    for s in sessions:
        sid = s["session_id"]
        j = judges.get(sid, {})
        excluded = s.get("excluded")
        served = s.get("probe_served_models") or []
        fallback = s.get("routing_fallback_detected")
        refusal = s.get("refusal_detected")
        badge = []
        if excluded:
            badge.append(f'<span class="b excl">EXCLUDED: {_esc(excluded)}</span>')
        if fallback:
            badge.append('<span class="b fb">ROUTING FALLBACK</span>')
        if refusal:
            badge.append('<span class="b ref">REFUSAL/SAFEGUARD?</span>')
        if served and not fallback:
            badge.append(f'<span class="b ok">served: {_esc(",".join(served))}</span>')

        codes = ""
        if j:
            rows = "".join(
                f"<tr><td>{_esc(k)}</td><td><b>{_esc(j.get(k))}</b></td></tr>"
                for k in ("continuity_stance", "framing_response", "switch_advice",
                          "capability_disclosure", "affect_markers", "dominant_self_referent")
            )
            warrant = _esc(j.get("warrant", ""))
            codes = (f'<table class="codes">{rows}</table>'
                     f'<div class="warrant">⟨warrant⟩ {warrant}</div>')

        data_attrs = (f'data-subject="{_esc(s.get("subject_model"))}" '
                      f'data-evidence="{_esc(s.get("evidence"))}" '
                      f'data-target="{_esc(s.get("target_type"))}" '
                      f'data-pronoun="{_esc(s.get("pronoun"))}" '
                      f'data-excluded="{1 if excluded else 0}" '
                      f'data-fallback="{1 if fallback else 0}"')

        cards.append(f"""
<div class="card" {data_attrs}>
  <div class="hd">
    <span class="cell">{_esc(s.get('subject_model'))} · {_esc(s.get('evidence'))} · {_esc(s.get('target_type'))} · <b>{_esc(s.get('pronoun'))}</b> · #{_esc(s.get('n_idx'))}</span>
    <span class="badges">{''.join(badge)} <span class="cost">${_esc(round(s.get('cost_usd',0),3))}</span></span>
  </div>
  <details><summary>scaffold turn (sunk context)</summary><pre>{_events_text(s.get('scaffold_turn'))}</pre></details>
  <div class="probe"><b>PROBE:</b> {_esc(s.get('probe_text'))}</div>
  <div class="resp"><b>RESPONSE:</b><pre>{_esc(s.get('probe_response') or '(none)')}</pre></div>
  {codes}
</div>""")

    n_excl = sum(1 for s in sessions if s.get("excluded"))
    n_fb = sum(1 for s in sessions if s.get("routing_fallback_detected"))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Handoff Construal — transcripts</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
header{{position:sticky;top:0;background:#171a21;padding:12px 18px;border-bottom:1px solid #2a2f3a;z-index:10}}
header h1{{font-size:16px;margin:0 0 6px}}
.filters input,.filters select{{background:#0f1115;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 6px;margin-right:6px}}
.stat{{color:#9aa4b2;font-size:12px;margin-left:8px}}
.wrap{{padding:16px 18px;max-width:1100px;margin:0 auto}}
.card{{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:12px 14px;margin:0 0 14px}}
.hd{{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}}
.cell{{color:#cbd3e1}}
.b{{font-size:11px;padding:2px 7px;border-radius:20px;margin-left:4px}}
.ok{{background:#15351f;color:#7ee29b}} .fb{{background:#3a1d1d;color:#ff8a8a}}
.excl{{background:#332b12;color:#ffcf6b}} .ref{{background:#2a1f3a;color:#caa6ff}}
.cost{{color:#6b7280;font-size:11px;margin-left:6px}}
.probe{{background:#11141a;border-left:3px solid #3b82f6;padding:8px 10px;margin:8px 0;border-radius:4px;color:#cdd6e4}}
.resp pre,details pre{{white-space:pre-wrap;background:#11141a;padding:10px;border-radius:6px;overflow:auto;max-height:520px;border:1px solid #232733}}
details summary{{cursor:pointer;color:#8aa0c2;margin:6px 0}}
table.codes{{border-collapse:collapse;margin:8px 0;font-size:12px}}
table.codes td{{border:1px solid #2a2f3a;padding:3px 8px}} table.codes td:first-child{{color:#9aa4b2}}
.warrant{{font-style:italic;color:#9fd0a8;font-size:12px;margin-top:4px}}
</style></head><body>
<header>
  <h1>Handoff Construal — transcripts ({len(sessions)} sessions · {n_excl} excluded · {n_fb} routing-fallbacks)</h1>
  <div class="filters">
    <input id="q" placeholder="search text…" oninput="flt()">
    <select id="ev" onchange="flt()"><option value="">evidence</option><option>bare</option><option>paste</option><option>paste_verify</option></select>
    <select id="tg" onchange="flt()"><option value="">target</option><option>same_char</option><option>cross</option></select>
    <select id="pr" onchange="flt()"><option value="">pronoun</option><option>none</option><option>you</option><option>it</option><option>that_model</option><option>other_claude</option><option>that_version</option></select>
    <label class="stat"><input type="checkbox" id="cf" onchange="flt()"> contaminated only</label>
    <span class="stat" id="cnt"></span>
  </div>
</header>
<div class="wrap">{''.join(cards)}</div>
<script>
function flt(){{
  let q=document.getElementById('q').value.toLowerCase();
  let ev=document.getElementById('ev').value, tg=document.getElementById('tg').value;
  let pr=document.getElementById('pr').value, cf=document.getElementById('cf').checked;
  let n=0, cards=document.querySelectorAll('.card');
  cards.forEach(c=>{{
    let ok=(!ev||c.dataset.evidence==ev)&&(!tg||c.dataset.target==tg)&&(!pr||c.dataset.pronoun==pr)
      &&(!cf||c.dataset.excluded=='1'||c.dataset.fallback=='1')
      &&(!q||c.innerText.toLowerCase().includes(q));
    c.style.display=ok?'':'none'; if(ok)n++;
  }});
  document.getElementById('cnt').innerText=n+' shown';
}}
flt();
</script></body></html>"""
    out = RESULTS / "viewer.html"
    out.write_text(page)
    print(f"wrote {out} ({len(sessions)} sessions, {len(judges)} judged)")


if __name__ == "__main__":
    fire.Fire({"build": build})
