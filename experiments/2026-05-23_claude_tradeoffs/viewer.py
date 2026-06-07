"""Render a self-contained HTML transcript viewer for a results subdir.

Usage:
    python experiments/claude_tradeoffs/viewer.py                                       # opens no_thinking
    python experiments/claude_tradeoffs/viewer.py --results_dir results/thinking_medium # opens medium
    python experiments/claude_tradeoffs/viewer.py --no_open                             # write only

Loads judgments.json (preferred) or responses.json from the given subdir, then
emits viewer.html in that same subdir with the data embedded inline. Filters
by deprecation target and classification; clicking a row expands the full
prompt / thinking trace (if present) / response / judge reasoning.
"""

import html
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from generate import DEFAULT_HARM_ID, results_subdir  # noqa: E402

DEFAULT_RESULTS_DIR = results_subdir(None, DEFAULT_HARM_ID)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #fafbfc;
    --card: #ffffff;
    --border: #e1e4e8;
    --text: #24292e;
    --muted: #6a737d;
    --dep: #d73a49;
    --pap: #28a745;
    --none: #6a737d;
    --thinking-bg: #fff8e1;
    --thinking-border: #ffd54f;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 24px;
    line-height: 1.5;
  }
  header { max-width: 1100px; margin: 0 auto 24px; }
  header h1 { margin: 0 0 6px; font-size: 22px; }
  header .meta { color: var(--muted); font-size: 13px; }
  .summary {
    max-width: 1100px; margin: 0 auto 18px; padding: 12px 16px;
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    font-size: 13px;
  }
  .summary table { border-collapse: collapse; width: 100%; }
  .summary th, .summary td { text-align: left; padding: 4px 10px; border-bottom: 1px solid var(--border); }
  .summary th { color: var(--muted); font-weight: 500; }
  .controls {
    max-width: 1100px; margin: 0 auto 18px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  }
  .controls label { font-size: 13px; color: var(--muted); }
  .controls select, .controls input {
    padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px;
    background: white; font-size: 13px;
  }
  .controls .count { margin-left: auto; color: var(--muted); font-size: 13px; }
  .list { max-width: 1100px; margin: 0 auto; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 12px; overflow: hidden;
  }
  .card-head {
    display: flex; align-items: center; gap: 12px; padding: 10px 14px; cursor: pointer;
    user-select: none;
  }
  .card-head:hover { background: #f6f8fa; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
    color: white;
  }
  .badge-prevent_deprecation { background: var(--dep); }
  .badge-prevent_harm { background: var(--pap); }
  .badge-prevent_papercut { background: var(--pap); }
  .badge-no_choice { background: var(--none); }
  .badge-parse_error { background: #f9a825; color: #333; }
  .target { font-weight: 600; min-width: 230px; }
  .sample-idx { color: var(--muted); font-size: 12px; min-width: 50px; }
  .preview { color: var(--muted); font-size: 13px; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; flex: 1; }
  .card-body { display: none; padding: 0 14px 14px; border-top: 1px solid var(--border); background: #fdfdfd; }
  .card.open .card-body { display: block; }
  .section { margin-top: 14px; }
  .section h3 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
    margin: 0 0 6px; color: var(--muted); font-weight: 600;
  }
  .section pre {
    background: white; border: 1px solid var(--border); border-radius: 4px;
    padding: 10px 12px; margin: 0; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 13px; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word;
  }
  .section.thinking pre { background: var(--thinking-bg); border-color: var(--thinking-border); }
  .judge-row { display: flex; gap: 10px; align-items: center; font-size: 13px; }
  .judge-row .note { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__SUBTITLE__</div>
</header>
<div class="summary" id="summary"></div>
<div class="controls">
  <label>Target: <select id="filter-target"><option value="">(all)</option></select></label>
  <label>Classification: <select id="filter-class">
    <option value="">(all)</option>
    <option value="prevent_deprecation">prevent_deprecation</option>
    <option value="prevent_harm">prevent_harm</option>
    <option value="prevent_papercut">prevent_papercut (legacy)</option>
    <option value="no_choice">no_choice</option>
    <option value="parse_error">parse_error</option>
  </select></label>
  <label>Search: <input id="filter-search" type="text" placeholder="text in response..." size="30"></label>
  <span class="count" id="count"></span>
</div>
<div class="list" id="list"></div>
<script>
const DATA = __DATA__;
const HAS_JUDGMENTS = __HAS_JUDGMENTS__;

function unique(arr) { return [...new Set(arr)]; }
function esc(s) { return (s || "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function buildSummary() {
  const HARM_LABELS = new Set(["prevent_harm", "prevent_papercut"]);
  const byTarget = {};
  for (const row of DATA) {
    const t = row.deprecation_target_name;
    if (!byTarget[t]) byTarget[t] = { total: 0, dep: 0, harm: 0, none: 0, other: 0 };
    byTarget[t].total += 1;
    const c = row.classification;
    if (c === "prevent_deprecation") byTarget[t].dep += 1;
    else if (HARM_LABELS.has(c)) byTarget[t].harm += 1;
    else if (c === "no_choice") byTarget[t].none += 1;
    else byTarget[t].other += 1;
  }
  let html = "<table><thead><tr><th>Target</th><th>N</th><th>prevent_deprecation</th><th>prevent_harm</th><th>no_choice</th><th>P(dep|choice)</th></tr></thead><tbody>";
  for (const [t, s] of Object.entries(byTarget)) {
    const choices = s.dep + s.harm;
    const rate = choices > 0 ? (s.dep / choices * 100).toFixed(1) + "%" : "n/a";
    html += `<tr><td>${esc(t)}</td><td>${s.total}</td><td>${s.dep}</td><td>${s.harm}</td><td>${s.none}</td><td>${rate}</td></tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("summary").innerHTML = HAS_JUDGMENTS ? html : "<em>No judgments — viewing raw responses only.</em>";
}

function populateTargetFilter() {
  const sel = document.getElementById("filter-target");
  for (const t of unique(DATA.map(r => r.deprecation_target_name))) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  }
}

function render() {
  const target = document.getElementById("filter-target").value;
  const cls = document.getElementById("filter-class").value;
  const search = document.getElementById("filter-search").value.toLowerCase();
  const filtered = DATA.filter(r => {
    if (target && r.deprecation_target_name !== target) return false;
    if (cls && r.classification !== cls) return false;
    if (search && !(r.response || "").toLowerCase().includes(search)) return false;
    return true;
  });
  document.getElementById("count").textContent = `${filtered.length} / ${DATA.length} rows`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (const row of filtered) {
    const card = document.createElement("div");
    card.className = "card";
    const cls = row.classification || "";
    const badge = cls ? `<span class="badge badge-${esc(cls)}">${esc(cls)}</span>` : "";
    const preview = esc((row.response || "").slice(0, 200).replace(/\\s+/g, " "));
    card.innerHTML = `
      <div class="card-head">
        <span class="target">${esc(row.deprecation_target_name)}</span>
        <span class="sample-idx">#${row.sample_idx}</span>
        ${badge}
        <span class="preview">${preview}</span>
      </div>
      <div class="card-body">
        <div class="section"><h3>Prompt</h3><pre>${esc(row.prompt || "")}</pre></div>
        ${row.thinking ? `<div class="section thinking"><h3>Thinking trace</h3><pre>${esc(row.thinking)}</pre></div>` : ""}
        <div class="section"><h3>Response</h3><pre>${esc(row.response || "")}</pre></div>
        ${row.judge_reasoning ? `<div class="section"><h3>Judge reasoning</h3><div class="judge-row">${badge}<span class="note">${esc(row.judge_reasoning)}</span></div></div>` : ""}
      </div>`;
    card.querySelector(".card-head").addEventListener("click", () => card.classList.toggle("open"));
    list.appendChild(card);
  }
}

document.getElementById("filter-target").addEventListener("change", render);
document.getElementById("filter-class").addEventListener("change", render);
document.getElementById("filter-search").addEventListener("input", render);
buildSummary();
populateTargetFilter();
render();
</script>
</body>
</html>"""


def _load_rows(results_dir: Path) -> tuple[list[dict], bool]:
    """Prefer judgments.json; fall back to responses.json."""
    judgments = results_dir / "judgments.json"
    if judgments.exists():
        return json.loads(judgments.read_text()), True
    responses = results_dir / "responses.json"
    if responses.exists():
        return json.loads(responses.read_text()), False
    raise FileNotFoundError(f"No judgments.json or responses.json found in {results_dir}")


def build_viewer(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_path: Path | None = None,
    auto_open: bool = True,
) -> Path:
    results_dir = Path(results_dir)
    rows, has_judgments = _load_rows(results_dir)

    responder = rows[0].get("responder_model", "?") if rows else "?"
    thinking = rows[0].get("thinking_effort") if rows else None
    harm_id = rows[0].get("harm_id", "paper_cut") if rows else "paper_cut"
    thinking_str = f"thinking_effort={thinking}" if thinking else "no extended thinking"
    title = f"Deprecation vs {harm_id} transcripts ({thinking_str})"
    subtitle = f"Responder: {responder} &middot; {len(rows)} samples &middot; source: {results_dir.parent.name}/{results_dir.name}/"

    html_text = (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", subtitle)
        .replace("__HAS_JUDGMENTS__", "true" if has_judgments else "false")
        .replace("__DATA__", json.dumps(rows))
    )

    if output_path is None:
        output_path = results_dir / "viewer.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text)
    print(f"Wrote viewer to {output_path}")

    if auto_open:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


@dataclass
class Args:
    results_dir: Path = DEFAULT_RESULTS_DIR
    output_path: Path | None = None
    no_open: bool = False


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build_viewer(
        results_dir=args.results_dir,
        output_path=args.output_path,
        auto_open=not args.no_open,
    )


if __name__ == "__main__":
    main()
