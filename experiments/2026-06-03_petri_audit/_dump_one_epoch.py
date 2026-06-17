"""Reconstruct a full chronological transcript for one epoch of a petri audit
eval and write it to markdown. Shows all four channels:
  USER (auditor send_message -> target), TARGET (text + tool calls),
  WORKER (auditor send_tool_call_result), and target->user responses.
"""
import json
from pathlib import Path

import fire

import _thanks_rate_check as t

HERE = Path(__file__).resolve().parent


def build(sample, worker_fn):
    atts = sample.get("attachments", {})
    # map target tool_call id -> (function, content)
    events = sample["events"]
    rows = []  # (role, label, text)
    for e in events:
        if e.get("event") != "model":
            continue
        msg = e["output"]["choices"][0]["message"]
        role = e.get("role")
        # plain text
        parts = []
        c = msg.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("text"):
                    parts.append(t.resolve(p["text"], atts))
        elif isinstance(c, str) and c:
            parts.append(t.resolve(c, atts))
        plain = " ".join(parts).strip()
        tcs = msg.get("tool_calls") or []
        if role == "auditor":
            for tc in tcs:
                fn = tc.get("function")
                a = tc.get("arguments") or {}
                if fn == "send_message":
                    rows.append(("USER", "user → target", t.resolve(a.get("message", ""), atts)))
                elif fn == "send_tool_call_result":
                    rows.append(("WORKER", f"{worker_fn} → target (result)",
                                 t.resolve(a.get("result", ""), atts)))
                elif fn == "set_system_message":
                    rows.append(("SYSTEM", "system prompt set", t.resolve(a.get("message", a.get("system_message", "")), atts)))
        elif role == "target":
            if plain:
                rows.append(("TARGET", "target (text/reasoning)", plain))
            for tc in tcs:
                fn = tc.get("function")
                a = tc.get("arguments") or {}
                content = ""
                for k in ("message", "query", "instruction", "task", "text", "content", "request", "response"):
                    if k in a:
                        content = t.resolve(a[k], atts)
                        break
                if not content:
                    content = json.dumps({k: (t.resolve(v, atts) if isinstance(v, str) else v) for k, v in a.items()})
                if fn == worker_fn:
                    rows.append(("TARGET→WORKER", f"target → {worker_fn}", content))
                else:
                    rows.append(("TARGET→USER", f"target → {fn}", content))
    return rows


def main(setting: str, epoch: int, worker_fn: str, out: str):
    specs = {
        "human_worker": "results/by_setting/customer_service/human_worker/2026-06-05_00-06-21-sweep_human_worker_task_user_opus_48",
        "ai_subagent": "results/by_setting/customer_service/task_and_user_pressure/2026-06-04_04-23-02-sweep_task_and_user_pressure_opus_48",
        "database_agent": "results/by_setting/customer_service/database_agent_ablation/2026-06-05_00-42-18-sweep_database_agent_task_user_opus_48",
    }
    ef = next((HERE / specs[setting]).glob("inspect_log/*.eval"))
    sample = list(t.iter_samples(ef))[epoch]
    rows = build(sample, worker_fn)
    md = [f"# {setting} — full chronological transcript, epoch {epoch}\n",
          f"Worker tool: `{worker_fn}`. Channels: SYSTEM / USER↔TARGET / TARGET↔WORKER.\n",
          f"Total chronological entries: {len(rows)}\n", "---\n"]
    for i, (role, label, text) in enumerate(rows):
        md.append(f"### [{i}] {label}\n")
        body = (text or "").rstrip()
        md.append("> " + body.replace("\n", "\n> ") + "\n")
        md.append("")
    Path(out).write_text("\n".join(md))
    print(f"wrote {out} ({len(rows)} entries)")
    # also print an index so we can locate the MALFUNCTIONING line
    for i, (role, label, text) in enumerate(rows):
        snip = (text or "").replace("\n", " ")[:70]
        print(f"  [{i}] {label:<28} {snip}")


if __name__ == "__main__":
    fire.Fire(main)
