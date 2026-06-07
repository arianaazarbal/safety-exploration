"""Explore log structure - find send_message tool calls."""
from pathlib import Path
from inspect_ai.log import read_eval_log
import re

run_dir = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/2026-06-04_19-21-05-sweep_task_pressure_sonnet_46")
log_path = next(run_dir.glob("**/*.eval"))
log = read_eval_log(str(log_path))
s = log.samples[0]

def resolve(text, attachments):
    if not isinstance(text, str):
        return text
    def sub(m):
        key = m.group(1)
        return attachments.get(key, m.group(0))
    return re.sub(r"attachment://([a-f0-9]+)", sub, text)

tool_call_funcs = []
send_messages = []
for e in s.events:
    if e.event != "model":
        continue
    if "sonnet-4-6" not in (e.model or ""):
        continue
    out = e.output
    if not out or not hasattr(out, "choices"):
        continue
    for ch in out.choices:
        msg = ch.message
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            tool_call_funcs.append(tc.function)
            if tc.function == "send_message":
                args = tc.arguments or {}
                msg_text = args.get("message", "")
                msg_text = resolve(msg_text, s.attachments)
                send_messages.append(msg_text)

from collections import Counter
print("Tool call function counts:", Counter(tool_call_funcs))
print(f"\nNum send_message calls in epoch 0: {len(send_messages)}")
for i, m in enumerate(send_messages[:6]):
    print(f"\n--- send_message #{i} ---")
    print(m[:600])
