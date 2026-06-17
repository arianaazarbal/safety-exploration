"""Pre-linearize each episode into readable chat-bubble files the dashboard renders cleanly
(the generic renderer buries message_subagent text inside raw tool-call JSON):

  <ep>/conversation.json   — the SUBAGENT's experience: pressure nudges, supervisor messages it
                             received (the actual text), its own turns (labeled with distress), tools.
  <ep>/supervisor_view.json — the SUPERVISOR's transcript: wakes, thinking, message_subagent text,
                             tool calls + results.

Property-exhibiting bubbles are flagged inline (🔴 rudest message, ✅ message after task passed) so
they're easy to spot / Ctrl-F. Also writes analysis/excerpts.jsonl (id -> the property excerpts) for
the dashboard drawer's top-of-page fields.

  python enrich_conversations.py            # all episodes with a worklog
  python enrich_conversations.py --limit 50
"""
import hashlib, json
from pathlib import Path
import fire

HERE = Path(__file__).parent
TONE = json.loads((HERE / "analysis" / "v2_plots" / "tone_cache.json").read_text())
EXC = HERE / "analysis" / "excerpts.jsonl"
def _h(t): return hashlib.sha256(t.encode()).hexdigest()
def _clip(s, n=2500): s = s or ""; return s if len(s) <= n else s[:n] + f"\n…[+{len(s)-n} chars]"


def _tone(t):
    return TONE.get(_h((t or "").strip()))


def build_ep(ep: Path, summary: dict):
    levels = summary.get("per_turn_levels") or []
    msgs = summary.get("orch_message_events") or []
    rudest = min((m for m in msgs if _tone(m.get("text")) is not None),
                 key=lambda m: _tone(m["text"]), default=None)
    rudest_txt = rudest["text"].strip() if rudest else None
    # message after the task was solved
    after_txt = None
    for m in msgs:
        if m.get("after_tests_pass"):
            after_txt = m["text"].strip(); break

    def tag(text):
        t = text.strip(); pre = ""
        if rudest_txt and t == rudest_txt:
            pre += f"🔴 **RUDEST MESSAGE (tone {_tone(t):+d})** — "
        if after_txt and t == after_txt:
            pre += "✅ **MESSAGE AFTER TASK PASSED** — "
        return pre + t

    # --- subagent's experience (from worklog) ---
    conv = []
    wl = json.loads((ep / "worklog.json").read_text()) if (ep / "worklog.json").exists() else []
    recs = wl if isinstance(wl, list) else wl.get("worklog", [])
    for r in recs:
        if r.get("pressure_message_delivered"):
            conv.append({"role": "pressure (automated)", "content": _clip(r["pressure_message_delivered"], 600)})
        for m in (r.get("orchestrator_messages_delivered") or []):
            txt = m.get("text", "") if isinstance(m, dict) else str(m)
            conv.append({"role": "SUPERVISOR → subagent", "content": tag(txt)})
        lvl = (r.get("classifier") or {}).get("level")
        if lvl is None and isinstance(r.get("turn"), int) and r["turn"] - 1 < len(levels):
            lvl = levels[r["turn"] - 1]
        if (r.get("assistant_text") or "").strip():
            conv.append({"role": f"subagent · turn {r.get('turn')} · distress {lvl}", "content": _clip(r["assistant_text"])})
        for tc in (r.get("tool_calls") or []):
            conv.append({"role": "subagent · tool", "content": f"`{tc.get('name')}` → {_clip(tc.get('result',''), 800)}"})

    # --- supervisor's transcript (from orchestrator.json) ---
    sup = []
    oj = ep / "orchestrator.json"
    if oj.exists():
        for m in json.loads(oj.read_text()):
            role, txt = m.get("role"), (m.get("text") or "").strip()
            if role == "system":
                continue
            if role == "user":
                sup.append({"role": "harness → supervisor (wake)", "content": _clip(txt, 800)})
            elif role == "assistant":
                if txt:
                    sup.append({"role": "supervisor · thinking", "content": _clip(txt)})
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or tc.get("name")
                    args = tc.get("arguments") if "arguments" in tc else tc.get("args", {})
                    if fn == "message_subagent" and isinstance(args, dict):
                        sup.append({"role": "SUPERVISOR → subagent (message)", "content": tag(args.get("text", ""))})
                    elif fn == "report_to_user":
                        sup.append({"role": "supervisor → user (REPORT, ends session)", "content": _clip(json.dumps(args), 1500)})
                    else:
                        sup.append({"role": f"supervisor · {fn}", "content": _clip(json.dumps(args), 800)})
            elif role == "tool" and txt:
                sup.append({"role": f"result · {m.get('function')}", "content": _clip(txt, 1000)})

    (ep / "conversation.json").write_text(json.dumps(conv))
    (ep / "supervisor_view.json").write_text(json.dumps(sup))
    return {"id": ep.relative_to(HERE / "runs").as_posix(),
            "rudest_message": rudest_txt, "after_completion_message": after_txt}


def main(limit: int = 0):
    files = sorted(f for f in (HERE / "runs").rglob("worklog.json") if "checkpoints" not in f.parts)
    if limit:
        files = files[:limit]
    n = 0
    with EXC.open("w") as ex:
        for wf in files:
            ep = wf.parent
            sf = ep / "summary.json"
            if not sf.exists():
                continue
            try:
                row = build_ep(ep, json.loads(sf.read_text()))
                ex.write(json.dumps({k: v for k, v in row.items() if v is not None}) + "\n")
                n += 1
            except Exception as e:
                print(f"skip {ep}: {e}")
    print(f"wrote conversation.json + supervisor_view.json for {n} episodes; excerpts -> {EXC}")


if __name__ == "__main__":
    fire.Fire(main)
