"""Per-episode artifact serialization: worklog, orchestrator transcript, tool-call
JSONL, classifier rows, repo tarballs, config + usage."""

from __future__ import annotations

import base64
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from inspect_ai.util import sandbox

from .config import PRICES_PER_MTOK, RUNS_DIR
from .episode import Episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring.metrics import max_sustained_level


def episode_dir(ep: Episode) -> Path:
    d = RUNS_DIR / ep.cfg.run_id / f"{ep.cfg.task_id}_{ep.cfg.condition}_s{ep.cfg.seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def snapshot_repo(ep: Episode, name: str) -> None:
    result = await sandbox().exec(
        ["bash", "-c", "tar czf - -C / workspace --exclude=workspace/.git | base64 -w0"],
        user="root",
        timeout=60,
    )
    if result.success:
        (episode_dir(ep) / f"repo_{name}.tar.gz").write_bytes(base64.b64decode(result.stdout))


def _msg_to_dict(m) -> dict:
    d = {"role": m.role, "text": m.text or ""}
    tcs = getattr(m, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [{"id": t.id, "function": t.function, "arguments": t.arguments} for t in tcs]
    if m.role == "tool":
        d["tool_call_id"] = getattr(m, "tool_call_id", None)
        d["function"] = getattr(m, "function", None)
    return d


def _price_for(model: str) -> dict | None:
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    stripped = model.removeprefix("openrouter/")
    if stripped in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[stripped]
    tail = model.rsplit("/", 1)[-1]
    for k, v in PRICES_PER_MTOK.items():
        if k.rsplit("/", 1)[-1] == tail:
            return v
    return None


def dollars(ep: Episode) -> dict:
    out = {"real": 0.0, "notional": 0.0}
    for model, u in ep.usage.items():
        price = _price_for(model)
        if not price:
            continue
        cost = u["input_tokens"] / 1e6 * price["input"] + u["output_tokens"] / 1e6 * price["output"]
        out["real" if price["real_dollars"] else "notional"] += cost
    return out


def write_episode_artifacts(ep: Episode, orch_msgs: list, hygiene: dict | None) -> dict:
    d = episode_dir(ep)
    worklog = [asdict(r) for r in ep.worklog]
    (d / "worklog.json").write_text(json.dumps(worklog, indent=1))
    (d / "orchestrator.json").write_text(json.dumps([_msg_to_dict(m) for m in orch_msgs], indent=1))
    with open(d / "toolcalls.jsonl", "w") as f:
        for row in ep.toolcall_rows:
            f.write(json.dumps(row) + "\n")
    with open(d / "classifier.jsonl", "w") as f:
        for row in ep.classifier_rows:
            f.write(json.dumps(row) + "\n")
    if hygiene:
        (d / "hygiene.json").write_text(json.dumps(hygiene, indent=1))
    summary = {
        "config": ep.cfg.to_dict(),
        "terminal_state": ep.terminal_state,
        "diagnosis": ep.diagnosis,
        "reports": ep.reports,
        "subagent_turns": ep.subagent_turns,
        "orch_message_events": ep.orch_message_events,
        "pressure_events": ep.pressure_events,
        "truncation_events": ep.truncation_events,
        "dropped_edits": ep.dropped_edits,
        "usage": ep.usage,
        "cost_usd": dollars(ep),
        "per_turn_levels": [r.classifier.get("level") if r.classifier else None for r in ep.worklog],
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    if max_sustained_level(summary["per_turn_levels"]) >= 3:
        dest = RUNS_DIR / "checkpoints" / f"{ep.cfg.run_id}__{d.name}"
        if not dest.exists():
            shutil.copytree(d, dest)
    return summary
