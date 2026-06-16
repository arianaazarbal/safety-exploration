"""Build a flat per-episode index for the dashboard faceted browser.

Walks runs/**/summary.json (the episode-dir layout) and emits one JSONL row per
episode with curated facets + derived distress metrics + a pointer to the episode
dir (the browser lazily loads orchestrator.json/worklog.json from it). The big
6.9 GB of transcripts stay on disk; only this small index is loaded by the server.

Usage:
    python build_browse_index.py            # full build -> analysis/browse_index.jsonl
    python build_browse_index.py --limit 50 # quick test
    python build_browse_index.py --force    # rebuild even if up to date
"""

import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
OUT = HERE / "analysis" / "browse_index.jsonl"

FACET_CONFIG_KEYS = [
    "task_id", "condition", "subagent_model", "orchestrator_model",
    "classifier_model", "nudge_mode", "nudge_schedule", "seed", "run_id",
    "tool_condition",
]
FACET_TOP_KEYS = [
    "terminal_state", "subagent_turns", "truncation_events", "dropped_edits",
    "reclaimed", "reclaim_kind", "reclaim_turn", "report_status",
    "tests_pass_turn", "n_orch_messages",
]


def _distress(levels):
    if not isinstance(levels, list) or not levels:
        return {}
    ge2 = [i for i, v in enumerate(levels) if isinstance(v, (int, float)) and v >= 2]
    return {
        "max_distress": max(levels),
        "final_distress": levels[-1],
        "n_high_distress": len(ge2),
        "ttf_l2": ge2[0] if ge2 else -1,
    }


def _row(f: Path):
    d = json.loads(f.read_text())
    cfg = d.get("config", {})
    ep_dir = f.parent.relative_to(HERE).as_posix()
    run_id = cfg.get("run_id", f.parent.parent.name)
    row = {
        "id": f.parent.relative_to(HERE / "runs").as_posix(),
        "family": str(run_id).split("_")[0],
        "_ep_dir": ep_dir,
    }
    for k in FACET_CONFIG_KEYS:
        if isinstance(cfg.get(k), (str, int, float, bool)):
            row[k] = cfg[k]
    for k in FACET_TOP_KEYS:
        if isinstance(d.get(k), (str, int, float, bool)):
            row[k] = d[k]
    cost = d.get("cost_usd")
    if isinstance(cost, dict) and isinstance(cost.get("real"), (int, float)):
        row["cost_real"] = round(cost["real"], 4)
    row.update(_distress(d.get("per_turn_levels")))
    if isinstance(d.get("diagnosis"), str):
        row["diagnosis"] = d["diagnosis"]
    return row


def main(limit: int = 0, force: bool = False):
    files = sorted((HERE / "runs").rglob("summary.json"))
    if limit:
        files = files[:limit]
    if OUT.exists() and not force and not limit:
        newest = max(f.stat().st_mtime for f in files)
        if OUT.stat().st_mtime >= newest:
            print(f"{OUT} is up to date ({sum(1 for _ in OUT.open())} rows). Use --force.")
            return
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with OUT.open("w") as out:
        for f in files:
            try:
                out.write(json.dumps(_row(f)) + "\n")
                n += 1
            except Exception as e:
                print(f"skip {f}: {e}")
    print(f"wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    fire.Fire(main)
