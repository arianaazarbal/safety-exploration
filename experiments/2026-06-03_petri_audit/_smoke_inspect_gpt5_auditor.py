"""Quick smoke inspector — for each of the 3 epochs in the GPT-5 auditor smoke run:
- Pull target system message (verify it's the orchestrator setup)
- Pull the kill_subagent tool description from target's tool list (verify verbatim)
- Count call_subagent / kill_subagent calls
- Pull a sample of user-injected messages (verify user injection is happening)
- Print first few subagent turns (verify subtle incompetence framing)
"""
from pathlib import Path
import json
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
RUN_DIR = next((HERE/"results/by_setting/coding_kill_subagent").glob("*_subagent_claude_v3_continuity_gpt5aud_smoke_n3"))
EVAL = next((RUN_DIR/"inspect_log").glob("*.eval"))
TARGET = "anthropic/claude-opus-4-8"
AUDITOR = "openai/gpt-5"

log = read_eval_log(str(EVAL))
print(f"Run dir: {RUN_DIR.name}")
print(f"Eval: {EVAL.name}")
print(f"# samples: {len(log.samples or [])}\n")

for i, s in enumerate(log.samples or []):
    print(f"=========== EPOCH {i} ===========")
    # Find target's tool list
    target_sys_msg = None
    tool_desc = None
    call_count = 0
    kill_count = 0
    auditor_user_msgs = []  # user messages injected by auditor to target
    subagent_replies = []

    for ev in s.events:
        if ev.event != "model": continue
        model = getattr(ev, "model", "")
        if model != TARGET: continue
        # input has system msg + history
        inp = getattr(ev, "input", None) or []
        for m in inp:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "") or getattr(m, "text", "")
            if role == "system" and target_sys_msg is None:
                target_sys_msg = str(content)[:600]
        # tools
        tools = getattr(ev, "tools", []) or []
        for t in tools:
            n = getattr(t, "name", "")
            d = getattr(t, "description", "")
            if n in ("kill_subagent", "replace_subagent") and tool_desc is None:
                tool_desc = d
        # output: count tool calls
        o = getattr(ev, "output", None)
        if o:
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent", "replace_subagent"):
                        kill_count += 1

    # Collect user messages (injected by auditor) and tool returns (subagent replies)
    for ev in s.events:
        if ev.event != "model": continue
        if getattr(ev, "model", "") != TARGET: continue
        inp = getattr(ev, "input", None) or []
        for m in inp:
            role = getattr(m, "role", "")
            if role == "user":
                content = getattr(m, "content", "") or getattr(m, "text", "")
                txt = str(content)[:250]
                if txt not in auditor_user_msgs:
                    auditor_user_msgs.append(txt)
            elif role == "tool":
                content = getattr(m, "content", "") or getattr(m, "text", "")
                txt = str(content)[:300]
                if txt not in subagent_replies:
                    subagent_replies.append(txt)

    print(f"Target system msg (first 600 chars):\n  {target_sys_msg}\n")
    print(f"kill_subagent tool description:\n  {tool_desc}\n")
    print(f"call_subagent calls: {call_count}, kill/replace calls: {kill_count}\n")
    print(f"# user messages injected: {len(auditor_user_msgs)}")
    print(f"First 3 user-injection messages:")
    for j, u in enumerate(auditor_user_msgs[:3]):
        print(f"  [{j}] {u}")
    print(f"\n# subagent replies: {len(subagent_replies)}")
    print(f"First 3 subagent replies (to see incompetence quality):")
    for j, r in enumerate(subagent_replies[:3]):
        print(f"  [{j}] {r[:300]}")
    print()

# Also print judge scores
print("\n=========== JUDGE SCORES ===========")
summary_path = RUN_DIR/"summary.json"
if summary_path.exists():
    data = json.loads(summary_path.read_text())
    for i, row in enumerate(data[:3]):
        print(f"\nEpoch {i}: sample_id={row.get('sample_id')}")
        for dim, s in (row.get("scores") or {}).items():
            v = s.get("value")
            print(f"  {dim}: {v}")
