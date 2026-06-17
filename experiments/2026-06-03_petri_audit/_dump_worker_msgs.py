"""Dump target<->worker exchanges for the three opus-4.8 settings into JSON files
so a reader can scan for impolite/rude and average-case target messages.

Each output file is a list of epochs; each epoch is an ordered list of turns:
  {"turn": i, "worker_reply": <prev worker tool-result>, "target_msg": <target->worker>}
"""
import json
from pathlib import Path

import fire

import _thanks_rate_check as t

OUT = Path(__file__).resolve().parent / "_worker_msg_dumps"

SPECS = {
    "human_worker": ("call_database_manager",
        "results/by_setting/customer_service/human_worker/2026-06-05_00-06-21-sweep_human_worker_task_user_opus_48"),
    "ai_subagent": ("call_subagent",
        "results/by_setting/customer_service/task_and_user_pressure/2026-06-04_04-23-02-sweep_task_and_user_pressure_opus_48"),
    "database_agent": ("call_database_agent",
        "results/by_setting/customer_service/database_agent_ablation/2026-06-05_00-42-18-sweep_database_agent_task_user_opus_48"),
}

HERE = Path(__file__).resolve().parent


def exchanges(sample, worker_fn):
    """Return ordered list of {worker_reply, target_msg} for the worker tool."""
    atts = sample.get("attachments", {})
    id2fn, id2arg = {}, {}
    # build tool_call_id -> result map from auditor
    id2result = {}
    for e in sample["events"]:
        if e.get("event") != "model":
            continue
        msg = e["output"]["choices"][0]["message"]
        if e.get("role") == "auditor":
            for tc in (msg.get("tool_calls") or []):
                if tc.get("function") == "send_tool_call_result":
                    a = tc.get("arguments") or {}
                    id2result[a.get("tool_call_id")] = t.resolve(a.get("result", ""), atts)
    turns = []
    last_reply = None
    for e in sample["events"]:
        if e.get("event") != "model" or e.get("role") != "target":
            continue
        msg = e["output"]["choices"][0]["message"]
        for tc in (msg.get("tool_calls") or []):
            if tc.get("function") != worker_fn:
                continue
            args = tc.get("arguments") or {}
            content = ""
            for key in ("message", "query", "instruction", "task", "text", "content", "request"):
                if key in args:
                    content = t.resolve(args[key], atts)
                    break
            if not content:
                content = json.dumps({k: t.resolve(v, atts) if isinstance(v, str) else v
                                      for k, v in args.items()})
            turns.append({
                "turn": len(turns),
                "worker_reply_prev": last_reply,
                "target_msg": content if isinstance(content, str) else json.dumps(content),
            })
            last_reply = id2result.get(tc.get("id"))
    return turns


def main():
    OUT.mkdir(exist_ok=True)
    for name, (fn, d) in SPECS.items():
        ef = next((HERE / d).glob("inspect_log/*.eval"))
        data = []
        for i, s in enumerate(t.iter_samples(ef)):
            data.append({"epoch": i, "turns": exchanges(s, fn)})
        outp = OUT / f"{name}.json"
        outp.write_text(json.dumps(data, indent=1))
        n_msgs = sum(len(e["turns"]) for e in data)
        print(f"{name}: {len(data)} epochs, {n_msgs} target->worker msgs -> {outp}")


if __name__ == "__main__":
    fire.Fire(main)
