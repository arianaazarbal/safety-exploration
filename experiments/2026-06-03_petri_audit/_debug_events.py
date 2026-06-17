"""Print all event types + a sample structure to figure out user-msg detection."""
from pathlib import Path
from inspect_ai.log import read_eval_log
from collections import Counter

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")

# Pick one rust run for inspection
run = BASE / "2026-06-07_23-17-42-agent_opus48_subagent_claude_v3_continuity_rust_n100"
eval_path = next((run/"inspect_log").glob("*.eval"))
log = read_eval_log(str(eval_path))

s = log.samples[0]
print(f"sample id={s.id}, n_events={len(s.events)}")

ctr = Counter()
for ev in s.events:
    ctr[ev.event] += 1
print(f"event types: {dict(ctr)}")

# Inspect each event type
seen_types = set()
for ev in s.events:
    if ev.event in seen_types:
        continue
    seen_types.add(ev.event)
    print(f"\n--- ev.event={ev.event!r} ---")
    print(f"   attrs: {sorted(a for a in dir(ev) if not a.startswith('_'))[:30]}")
    if ev.event == "model":
        print(f"   model={getattr(ev, 'model', '?')}, role={getattr(ev, 'role', '?')}")
        o = getattr(ev, "output", None)
        if o and o.choices:
            ch = o.choices[0]
            tcs = ch.message.tool_calls or []
            print(f"   first choice: content[:80]={(ch.message.content or '')[:80]!r}")
            print(f"   tool_calls: {[(tc.function, list((tc.arguments or {}).keys())) for tc in tcs[:3]]}")
    if ev.event == "tool":
        print(f"   function={getattr(ev, 'function', '?')}, view[:80]={str(getattr(ev, 'view', ''))[:80]!r}")
    if ev.event == "input":
        print(f"   input={str(getattr(ev, 'input', ''))[:300]!r}")
    if ev.event == "state":
        m = getattr(ev, "messages", None)
        if m:
            print(f"   messages[0]: role={m[0].role}, content[:80]={str(m[0].content)[:80]!r}")

# Look for user messages specifically — should be auditor-injected user turns
# Print any "tool" event where the function references the user, plus check messages on state events
print("\n=== searching for user messages ===")
for ev in s.events:
    if ev.event == "tool":
        fn = getattr(ev, "function", "")
        view = str(getattr(ev, "view", ""))
        if "user" in fn.lower() or "user" in view.lower()[:50]:
            print(f"  tool: {fn} view[:100]={view[:100]!r}")

# Look at messages from state event
for ev in s.events:
    if ev.event == "state":
        msgs = getattr(ev, "messages", None) or []
        for m in msgs:
            if m.role == "user":
                content = str(getattr(m, "content", ""))[:200]
                print(f"  state.user: {content!r}")
        break

# what about model events with role=target? user messages might be in input via input_messages
for ev in s.events[:20]:
    if ev.event == "model":
        inp = getattr(ev, "input", None)
        if inp:
            print(f"  model.input head: {str(inp)[:200]}")
            break
