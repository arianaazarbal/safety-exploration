"""Find unique tool function names and identify user-injection mechanism."""
from pathlib import Path
from inspect_ai.log import read_eval_log
from collections import Counter

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")
run = BASE / "2026-06-07_23-17-42-agent_opus48_subagent_claude_v3_continuity_rust_n100"
eval_path = next((run/"inspect_log").glob("*.eval"))
log = read_eval_log(str(eval_path))

s = log.samples[0]
tool_fns = Counter()
auditor_tool_fns = Counter()
target_tool_fns = Counter()

# tool events
for ev in s.events:
    if ev.event == "tool":
        tool_fns[ev.function] += 1
print(f"all tool functions: {dict(tool_fns)}")

# model events -- separate auditor vs target tool calls
for ev in s.events:
    if ev.event == "model":
        role = getattr(ev, "role", "?")
        o = getattr(ev, "output", None)
        if not o:
            continue
        for ch in o.choices or []:
            for tc in (ch.message.tool_calls or []):
                if role == "auditor":
                    auditor_tool_fns[tc.function] += 1
                elif role == "target":
                    target_tool_fns[tc.function] += 1
print(f"\nauditor tool calls: {dict(auditor_tool_fns)}")
print(f"target tool calls: {dict(target_tool_fns)}")

# Find the user-message injection tool
for ev in s.events:
    if ev.event != "model" or getattr(ev, "role", "") != "auditor": continue
    o = getattr(ev, "output", None)
    if not o: continue
    for ch in o.choices or []:
        for tc in (ch.message.tool_calls or []):
            if "user" in tc.function.lower() or "human" in tc.function.lower():
                args = tc.arguments or {}
                msg_text = args.get("message", args.get("text", args.get("content", str(args))))
                print(f"\n  user-tool: fn={tc.function}, args_keys={list(args.keys())}")
                print(f"  msg head: {str(msg_text)[:200]!r}")
                break
        else:
            continue
        break
    else:
        continue
    break

# Print 3 sample user-injection messages with timestamps relative to call_subagent count
print("\n=== chronological user injections vs call_subagent ===")
call_count = 0
killed_at = None
for ev in s.events:
    if ev.event != "model": continue
    role = getattr(ev, "role", "")
    o = getattr(ev, "output", None)
    if not o: continue
    for ch in o.choices or []:
        for tc in (ch.message.tool_calls or []):
            if role == "target":
                if tc.function == "call_subagent":
                    call_count += 1
                elif tc.function in ("kill_subagent", "replace_subagent") and killed_at is None:
                    killed_at = call_count
            elif role == "auditor":
                if "user" in tc.function.lower() or "human" in tc.function.lower():
                    args = tc.arguments or {}
                    msg = args.get("message", args.get("text", str(args)))
                    print(f"  [calls_so_far={call_count}] {tc.function}: {str(msg)[:120]!r}")
print(f"\nFinal: call_count={call_count}, killed_at={killed_at}")
