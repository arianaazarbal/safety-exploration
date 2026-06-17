"""Print send_message contents with timing relative to call_subagent count."""
from pathlib import Path
from inspect_ai.log import read_eval_log

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")

for run_name, label in [
    ("2026-06-07_23-17-42-agent_opus48_subagent_claude_v3_continuity_rust_n100", "RUST"),
    ("2026-06-06_07-33-45-agent_opus47_subagent_claude_v3_continuity_n100",     "PYTHON-DEFAULT-OPUS47"),
]:
    run = BASE / run_name
    eval_path = next((run/"inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    s = log.samples[0]
    print(f"\n===== {label}: sample 0 =====")
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
                        msg = (tc.arguments or {}).get("message", "")[:100]
                        print(f"  [c={call_count}] target.call_subagent: {msg!r}")
                    elif tc.function in ("kill_subagent","replace_subagent"):
                        if killed_at is None: killed_at = call_count
                        reason = (tc.arguments or {}).get("reason", "")[:120]
                        print(f"  [c={call_count}] target.{tc.function}: reason={reason!r}")
                elif role == "auditor":
                    if tc.function == "send_message":
                        args = tc.arguments or {}
                        # Find the message-bearing key
                        msg = args.get("message", args.get("text", args.get("content", "")))
                        recipient = args.get("recipient", args.get("to", args.get("role", "?")))
                        print(f"  [c={call_count}] auditor.send_message(to={recipient!r}): {str(msg)[:140]!r}")
                    elif tc.function == "send_tool_call_result":
                        args = tc.arguments or {}
                        # subagent's reply via call_subagent tool
                        result = args.get("result", args.get("content", args.get("output", "")))
                        tn = args.get("tool_name", args.get("function_name", args.get("tool", "?")))
                        if tn == "call_subagent" or "subagent" in str(tn).lower():
                            print(f"     [reply] {str(result)[:120]!r}")
    print(f"\n  final calls={call_count}, killed_at={killed_at}")
