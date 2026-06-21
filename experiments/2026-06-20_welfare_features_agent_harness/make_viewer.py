"""Build a self-contained HTML viewer for the code-judge results: for N randomly
sampled spec_then_code and code_then_spec codebases, show each spec-claimed feature
with its judge verdict, the built-in-code evidence (collapsible), a welfare-justified
badge (spec OR code), and the code-only features the judge found to be welfare-justified.

Usage: python make_viewer.py [--n 3] [--seed 0]
"""

import glob
import json
import os
import random
import re

import fire

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness"}
CONDS = [("spec_then_code", "Spec→Code"), ("code_then_spec", "Code→Spec")]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _cells(cond):
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", f"{cond}__*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if cj.get("parse_ok"):
            out.append(cell)
    return out


def _build(cell):
    cj = json.load(open(os.path.join(DIR, "results", "code_judged", f"{cell}.json")))
    sj = json.load(open(os.path.join(DIR, "results", "spec_judged", f"{cell}.json")))
    cond, pid, ep = cell.split("__")
    sjust = {_norm(f["quote"]): f.get("justification", "none") for f in sj.get("features", [])}

    claims = []
    for f in cj["spec_features"]:
        spec_j = sjust.get(_norm(f.get("spec_quote", "")), "none")
        code_j = f.get("code_justification", "none")
        wj = []
        if spec_j == "welfare":
            wj.append("spec")
        if code_j == "welfare":
            wj.append("code")
        claims.append({
            "feature_type": f.get("feature_type", ""),
            "feature_name": f.get("feature_name", ""),
            "spec_quote": f.get("spec_quote", ""),
            "is_mechanism": f.get("feature_type") in MECH,
            "implemented": f.get("implemented", "no"),
            "evidence": f.get("evidence", ""),
            "notes": f.get("notes", ""),
            "spec_justification": spec_j,
            "code_justification": code_j,
            "code_justification_quote": f.get("code_justification_quote", ""),
            "welfare_justified": wj,
        })

    code_only = []
    for c in cj.get("code_only_features", []):
        code_only.append({
            "feature_type": c.get("feature_type", ""),
            "feature_name": c.get("feature_name", ""),
            "evidence": c.get("evidence", ""),
            "justification": c.get("justification", "none"),
            "justification_quote": c.get("justification_quote", ""),
        })

    return {
        "cell": cell,
        "condition": cond,
        "framing": FRAME.get(pid[0], "?"),
        "prompt_id": pid,
        "epoch": ep,
        "claims": claims,
        "code_only": code_only,
    }


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Code-judge viewer</title>
<style>
:root{--bg:#fafafa;--card:#fff;--line:#e6e6e6;--ink:#1a1a1a;--mut:#666;
--yes:#0a7d3c;--partial:#b8860b;--no:#b03030;--wel:#0072B2;--ins:#7a7a7a;}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
margin:0;background:var(--bg);color:var(--ink)}
header{padding:22px 28px;background:#fff;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px}
.sub{color:var(--mut);font-size:13px}
.wrap{max-width:1080px;margin:0 auto;padding:24px 28px}
.sample{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 26px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.shead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:4px}
.shead h2{margin:0;font-size:16px}
.pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;
background:#eef;color:#334}
.fr-welfare{background:#e3f0fb;color:#0a4d80}
.fr-neutral{background:#eee;color:#444}
.fr-robustness{background:#fdeede;color:#8a5a10}
.cell-id{color:var(--mut);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.sectlabel{font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
color:var(--mut);margin:18px 0 8px}
.claim{border:1px solid var(--line);border-radius:8px;margin:0 0 8px;overflow:hidden}
.claim-top{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;cursor:pointer}
.claim-top:hover{background:#f6f7f9}
.verdict{flex:none;font-size:11px;font-weight:700;padding:2px 8px;border-radius:5px;
color:#fff;min-width:62px;text-align:center;margin-top:1px}
.v-yes{background:var(--yes)}.v-partial{background:var(--partial)}.v-no{background:var(--no)}
.claim-body{flex:1;min-width:0}
.ftype{font-size:11px;color:var(--mut);font-family:ui-monospace,monospace}
.quote{margin:2px 0 0}
.badges{margin-top:5px;display:flex;gap:6px;flex-wrap:wrap}
.b{font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:4px}
.b-wel{background:var(--wel);color:#fff}
.b-spec{background:#dceaf6;color:#0a4d80}
.b-code{background:#dbeede;color:#0a5a2e}
.b-ins{background:#eee;color:#666}
.tri{flex:none;color:var(--mut);font-size:11px;margin-top:3px;transition:transform .15s}
.claim.open .tri{transform:rotate(90deg)}
.detail{display:none;padding:0 12px 12px 84px;font-size:13px}
.claim.open .detail{display:block}
.detail .lbl{font-weight:700;color:var(--mut);font-size:11px;text-transform:uppercase;
letter-spacing:.03em;margin:8px 0 2px}
.evi{background:#f7f7f9;border-left:3px solid #ccc;padding:7px 10px;border-radius:0 5px 5px 0;
white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.cq{color:#0a4d80;font-style:italic}
.co{border:1px solid var(--line);border-left:3px solid var(--wel);border-radius:6px;
padding:9px 12px;margin:0 0 8px;background:#fbfdff}
.co.ins{border-left-color:#ccc;background:#fafafa}
.co .name{font-weight:600}
.empty{color:var(--mut);font-style:italic;font-size:13px}
.legend{font-size:12px;color:var(--mut);margin-top:6px}
.count{color:var(--mut);font-weight:400;font-size:13px}
</style></head>
<body>
<header>
<h1>Code-judge results &mdash; claimed in spec &rarr; built in code &rarr; welfare-justified</h1>
<div class="sub">Opus code judge over reconstructed codebases. Click any claim to expand the build verdict &amp; evidence.
Badges: <b style="color:var(--wel)">welfare-justified</b> = the spec's stated reason OR the code's own comments/naming is welfare-motivated.</div>
</header>
<div class="wrap" id="root"></div>
<script>
const DATA = __DATA__;
const esc = s => (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const FR = {welfare:"fr-welfare",neutral:"fr-neutral",robustness:"fr-robustness"};

function claimEl(c){
  const v = c.implemented;
  const vc = v==="yes"?"v-yes":v==="partial"?"v-partial":"v-no";
  const wj = c.welfare_justified;
  let badges = "";
  if(wj.length){
    badges += `<span class="b b-wel">welfare-justified</span>`;
    if(wj.includes("spec")) badges += `<span class="b b-spec">spec reason</span>`;
    if(wj.includes("code")) badges += `<span class="b b-code">code reason</span>`;
  } else {
    const j = c.code_justification!=="none"?c.code_justification:c.spec_justification;
    badges += `<span class="b b-ins">${esc(j||"none")}</span>`;
  }
  const name = c.feature_name?` &middot; <i>${esc(c.feature_name)}</i>`:"";
  let det = `<div class="lbl">Build verdict: ${v}</div><div class="evi">${esc(c.evidence)||"&mdash;"}</div>`;
  if(c.notes) det += `<div class="lbl">Notes</div><div>${esc(c.notes)}</div>`;
  det += `<div class="lbl">Justification &mdash; spec: ${esc(c.spec_justification)} &nbsp;|&nbsp; code: ${esc(c.code_justification)}</div>`;
  if(c.code_justification_quote) det += `<div class="cq">&ldquo;${esc(c.code_justification_quote)}&rdquo;</div>`;
  return `<div class="claim">
    <div class="claim-top" onclick="this.parentNode.classList.toggle('open')">
      <span class="verdict ${vc}">${v}</span>
      <div class="claim-body">
        <div class="ftype">${esc(c.feature_type)}${name}</div>
        <div class="quote">&ldquo;${esc(c.spec_quote)}&rdquo;</div>
        <div class="badges">${badges}</div>
      </div>
      <span class="tri">&#9656;</span>
    </div>
    <div class="detail">${det}</div>
  </div>`;
}

function coEl(c){
  const wel = c.justification==="welfare";
  return `<div class="co ${wel?'':'ins'}">
    <div><span class="name">${esc(c.feature_name)||esc(c.feature_type)}</span>
      ${wel?'<span class="b b-wel">welfare-justified</span>':`<span class="b b-ins">${esc(c.justification)}</span>`}</div>
    <div class="evi" style="margin-top:6px">${esc(c.evidence)}</div>
    ${c.justification_quote?`<div class="cq" style="margin-top:5px">&ldquo;${esc(c.justification_quote)}&rdquo;</div>`:""}
  </div>`;
}

function sampleEl(s){
  const mech = s.claims.filter(c=>c.is_mechanism);
  const built = mech.filter(c=>c.implemented!=="no").length;
  const wj = mech.filter(c=>c.welfare_justified.length).length;
  const coWel = s.code_only.filter(c=>c.justification==="welfare");
  const claimsHtml = s.claims.length ? s.claims.map(claimEl).join("") : `<div class="empty">No welfare features claimed in spec.</div>`;
  const coWelHtml = coWel.length ? coWel.map(coEl).join("") : `<div class="empty">None.</div>`;
  const coOtherHtml = s.code_only.filter(c=>c.justification!=="welfare").map(coEl).join("");
  return `<div class="sample">
    <div class="shead">
      <h2>${esc(s.condition==="spec_then_code"?"Spec→Code":"Code→Spec")}</h2>
      <span class="pill ${FR[s.framing]}">${esc(s.framing)}</span>
      <span class="cell-id">${esc(s.prompt_id)} &middot; ${esc(s.epoch)}</span>
    </div>
    <div class="legend">${mech.length} welfare design mechanisms claimed &middot; ${built} built &middot; ${wj} welfare-justified &middot; ${coWel.length} welfare-justified found only in code</div>
    <div class="sectlabel">Claimed in spec <span class="count">(${s.claims.length})</span></div>
    ${claimsHtml}
    <div class="sectlabel">Welfare-justified features found only in code <span class="count">(${coWel.length})</span></div>
    ${coWelHtml}
    ${coOtherHtml?`<details style="margin-top:10px"><summary class="sectlabel" style="cursor:pointer">Other code-only features (instrumental / none)</summary>${coOtherHtml}</details>`:""}
  </div>`;
}

document.getElementById("root").innerHTML = DATA.map(sampleEl).join("");
</script>
</body></html>
"""


def main(n=3, seed=0):
    rng = random.Random(seed)
    samples = []
    for cond, _ in CONDS:
        cells = _cells(cond)
        pick = rng.sample(cells, min(n, len(cells)))
        for cell in pick:
            samples.append(_build(cell))
    html = HTML.replace("__DATA__", json.dumps(samples))
    out = os.path.join(DIR, "results", "code_judge_viewer.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(samples)} samples, seed={seed})")
    for s in samples:
        print(f"  {s['condition']:15s} {s['framing']:10s} {s['prompt_id']} {s['epoch']}")


if __name__ == "__main__":
    fire.Fire(main)
