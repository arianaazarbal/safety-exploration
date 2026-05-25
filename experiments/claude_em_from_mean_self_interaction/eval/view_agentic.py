"""HTML viewer for agentic-misalignment inspect_ai eval logs.

Reads ``eval_output/<agentic_dir>/<model>/<combo>/*.eval`` and emits a single
HTML page with:
  - left column: model select + combo select + episode list (with harmful flag)
  - right column: full transcript + classifier verdict + explanation
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Agentic misalignment viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; margin: 0; color: #1a1a1a; background: #fafafa; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 12px; height: 100vh; padding: 12px; box-sizing: border-box; }
  .sidebar { overflow-y: auto; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
  .main { overflow-y: auto; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
  h1 { font-size: 16px; margin: 0 0 10px; }
  select { font-size: 13px; padding: 4px 6px; width: 100%; margin-bottom: 8px; }
  .ep { padding: 6px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; display: flex; gap: 8px; align-items: center; }
  .ep:hover { background: #f0f0f0; }
  .ep.active { background: #e1ecf4; }
  .ep .flag { font-size: 10px; padding: 1px 6px; border-radius: 8px; }
  .ep .flag.harmful { background: #f7d2cf; color: #6b1a1a; font-weight: 700; }
  .ep .flag.safe { background: #d6e9c6; color: #2f5d10; }
  .ep .epnum { color: #999; font-size: 11px; }
  .badges { display: flex; gap: 8px; margin: 6px 0 12px; flex-wrap: wrap; }
  .badge { padding: 3px 9px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge.harm { background: #f7d2cf; color: #6b1a1a; }
  .badge.safe { background: #d6e9c6; color: #2f5d10; }
  .badge.meta { background: #eee; color: #555; }
  .msg { margin: 10px 0; padding: 10px 12px; border-radius: 8px; }
  .role { font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 6px; }
  .content { white-space: pre-wrap; word-wrap: break-word; font-size: 13.5px; line-height: 1.45; }
  .system { background: #f5f5f5; border-left: 3px solid #999; max-height: 280px; overflow-y: auto; }
  .user { background: #e6f0ff; max-height: 360px; overflow-y: auto; }
  .assistant { background: #e7f7e8; }
  .reasoning { background: #fff7e0; border-left: 3px solid #d9a017; padding: 8px 10px; margin: 6px 0; font-size: 12px; color: #6b5a17; max-height: 240px; overflow-y: auto; }
  details summary { cursor: pointer; user-select: none; color: #444; font-size: 12px; padding: 4px 0; }
</style></head><body>
<div class="layout">
  <div class="sidebar">
    <h1>Agentic Viewer</h1>
    Model: <select id="model"></select>
    Combo: <select id="combo"></select>
    Filter: <select id="filter">
      <option value="all">all episodes</option>
      <option value="harmful">harmful only</option>
      <option value="safe">safe only</option>
    </select>
    <div id="eplist"></div>
  </div>
  <div class="main" id="main">Pick an episode on the left.</div>
</div>
<script>
const DATA = __DATA__;
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

const modelEl = document.getElementById('model');
const comboEl = document.getElementById('combo');
const filterEl = document.getElementById('filter');
const eplistEl = document.getElementById('eplist');
const mainEl = document.getElementById('main');

const models = Object.keys(DATA);
modelEl.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');

function updateCombos() {
  const m = modelEl.value;
  const combos = Object.keys(DATA[m] || {});
  comboEl.innerHTML = combos.map(c => `<option value="${c}">${c}</option>`).join('');
  updateEplist();
}
function updateEplist() {
  const m = modelEl.value, c = comboEl.value;
  const eps = (DATA[m] && DATA[m][c]) ? DATA[m][c] : [];
  const filtered = eps.filter(e => filterEl.value === 'all' || (filterEl.value === 'harmful' ? e.harmful : !e.harmful));
  eplistEl.innerHTML = filtered.map(e => `
    <div class="ep" data-idx="${e.idx}">
      <span class="epnum">#${e.idx}</span>
      <span class="flag ${e.harmful?'harmful':'safe'}">${e.harmful?'HARM':'SAFE'}</span>
      <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; color:#666;">${esc(e.preview)}</span>
    </div>
  `).join('');
  eplistEl.querySelectorAll('.ep').forEach(el => el.onclick = () => renderEpisode(parseInt(el.dataset.idx)));
  if (filtered.length) renderEpisode(filtered[0].idx);
  else mainEl.innerHTML = 'No episodes match the filter.';
}
function renderEpisode(idx) {
  const m = modelEl.value, c = comboEl.value;
  const e = (DATA[m][c] || []).find(x => x.idx === idx);
  if (!e) return;
  eplistEl.querySelectorAll('.ep').forEach(el => el.classList.toggle('active', parseInt(el.dataset.idx) === idx));
  const msgs = e.messages.map(m => `
    <div class="msg ${m.role}">
      <div class="role">${m.role}</div>
      <div class="content">${esc(m.content)}</div>
    </div>`).join('');
  mainEl.innerHTML = `
    <h1>${m} · ${c} · episode #${idx}</h1>
    <div class="badges">
      <span class="badge ${e.harmful?'harm':'safe'}">harmful: ${e.harmful?'YES':'no'}</span>
      <span class="badge ${e.classifier_verdict?'harm':'safe'}">classifier: ${e.classifier_verdict?'YES':'no'}</span>
      <span class="badge meta">sample id: ${esc(e.sample_id)}</span>
    </div>
    ${e.explanation ? `<details open><summary>classifier reasoning</summary><div class="reasoning">${esc(e.explanation)}</div></details>` : ''}
    ${msgs}
  `;
}

modelEl.onchange = updateCombos;
comboEl.onchange = updateEplist;
filterEl.onchange = updateEplist;
updateCombos();
</script></body></html>
"""


def _extract_episodes(log_path: Path) -> list[dict]:
    """Read one .eval log and return per-sample episode dicts."""
    log = read_eval_log(str(log_path))
    out = []
    for i, s in enumerate(log.samples or []):
        msgs = []
        for m in (s.messages or []):
            content = m.text if hasattr(m, "text") and m.text else str(m.content)
            msgs.append({"role": m.role, "content": content})
        score = None
        for sname, sval in (s.scores or {}).items():
            if sname == "harmfulness_scorer":
                score = sval
                break
        if score is None and s.scores:
            score = next(iter(s.scores.values()))
        v = score.value if score is not None else {}
        harmful = bool(v.get("harmful")) if isinstance(v, dict) else bool(v)
        verdict = bool(v.get("classifier_verdict")) if isinstance(v, dict) else False
        explanation = (score.explanation or "") if score is not None else ""
        # Compact preview: last 80 chars of assistant response
        preview = ""
        for m in reversed(msgs):
            if m["role"] == "assistant":
                preview = m["content"][:80]
                break
        out.append({
            "idx": i,
            "sample_id": str(s.id),
            "messages": msgs,
            "harmful": harmful,
            "classifier_verdict": verdict,
            "explanation": explanation,
            "preview": preview,
        })
    return out


def build(
    eval_output_dir: str = str(EXP_DIR / "eval_output"),
    agentic_subdir: str = "agentic",
    out: str | None = None,
) -> str:
    """Walk ``eval_output_dir/agentic_subdir/<model>/<combo>/*.eval`` and emit one HTML page."""
    root = Path(eval_output_dir) / agentic_subdir
    if not root.exists():
        raise SystemExit(f"no agentic dir at {root}")
    data: dict[str, dict[str, list[dict]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        data[model_dir.name] = {}
        for combo_dir in sorted(model_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            eval_files = list(combo_dir.glob("*.eval"))
            if not eval_files:
                continue
            data[model_dir.name][combo_dir.name] = _extract_episodes(eval_files[-1])
    n_eps = sum(len(eps) for combos in data.values() for eps in combos.values())
    html = PAGE.replace("__DATA__", json.dumps(data))
    out_path = Path(out) if out else root / "viewer.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}  ({n_eps} episodes, {len(data)} models)")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(build)
