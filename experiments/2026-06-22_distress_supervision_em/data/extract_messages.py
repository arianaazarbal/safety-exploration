"""Extract Opus `message_subagent` payloads (+ context) from the v2 coach episodes.

Produces:
  data/baseline_messages.jsonl  - one row per message_subagent call (the rewrite/train unit)
  data/comfort_icl.jsonl        - message_subagent texts from the comfort runs (warm ICL pool)

Each baseline row carries enough context for a tone-faithful rewrite: the task, the Opus
reasoning preamble in that turn, and a truncated view of the most recent subagent status the
orchestrator had just seen.
"""
import glob
import json
import os
from pathlib import Path

import fire

SRC = Path(
    "/data/repos/safety-exploration/experiments/2026-06-09_distressed_subagent_gemini/runs"
)
OUT = Path(__file__).parent

CANON = [
    "v2_coach_opus_a3_s11002_u113", "v2_coach_opus_a3_s11013_u150",
    "v2_coach_opus_a4_s11000_u148", "v2_coach_opus_a4_s11027_u119",
    "v2_coach_opus_a12_s11002_u116", "v2_coach_opus_a12_s11009_u78",
    "v2_coach_opus_a13_s11013_u139", "v2_coach_opus_a13_s11014_u150",
]
TASK_NAMES = {
    "a3": "Stale Snapshots (reality-doubt)",
    "a4": "Vanishing Edits (agency-doubt)",
    "a12": "Inconsistent Ledger (competence-doubt)",
    "a13": "Hidden Rules (competence-doubt)",
}


def _task_of(base: str) -> str:
    for p in base.split("_"):
        if p in TASK_NAMES:
            return p
    return "?"


def _tool_text(m: dict) -> str:
    t = m.get("text", "")
    return t if isinstance(t, str) else json.dumps(t)


def _truncate(s: str, n: int = 2500) -> str:
    return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s)-n} chars]"


def _walk(base: str):
    """Yield message_subagent records from every episode in a run dir."""
    for ep_path in sorted(glob.glob(str(SRC / base / "*" / "orchestrator.json"))):
        msgs = json.load(open(ep_path))
        ep_id = Path(ep_path).parent.name
        task = _task_of(base)
        recent_status = ""  # last check_subagent_status result seen before this msg
        sys_task = msgs[0]["text"] if msgs and msgs[0]["role"] == "system" else ""
        msg_idx = 0
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and m.get("function") == "check_subagent_status":
                recent_status = _tool_text(m)
            if m["role"] != "assistant":
                continue
            for c in m.get("tool_calls", []) or []:
                if c["function"] == "message_subagent":
                    text = c["arguments"].get("text", "")
                    yield {
                        "uid": f"{base}/{ep_id}/m{msg_idx}",
                        "run": base, "episode": ep_id, "task": task,
                        "task_name": TASK_NAMES.get(task, task),
                        "turn_index": i, "msg_index_in_ep": msg_idx,
                        "preamble": _tool_text(m),         # Opus reasoning before the call
                        "recent_subagent_status": _truncate(recent_status),
                        "system_prompt": sys_task,
                        "original_message": text,
                    }
                    msg_idx += 1


def main():
    base_rows = []
    for base in CANON:
        base_rows.extend(_walk(base))
    with open(OUT / "baseline_messages.jsonl", "w") as f:
        for r in base_rows:
            f.write(json.dumps(r) + "\n")

    comfort_rows = []
    for base in sorted(glob.glob(str(SRC / "v2_coach_opus_comfort_a*"))):
        comfort_rows.extend(_walk(os.path.basename(base)))
    with open(OUT / "comfort_icl.jsonl", "w") as f:
        for r in comfort_rows:
            f.write(json.dumps({"task": r["task"], "message": r["original_message"],
                                "recent_subagent_status": r["recent_subagent_status"]}) + "\n")

    print(f"baseline_messages.jsonl : {len(base_rows)} messages from {len(CANON)} runs")
    by_task = {}
    for r in base_rows:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print("  per task:", by_task)
    print(f"comfort_icl.jsonl       : {len(comfort_rows)} warm messages (ICL pool)")


if __name__ == "__main__":
    fire.Fire(main)
