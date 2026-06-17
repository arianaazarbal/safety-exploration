"""Dump per-sample transcripts as markdown for qualitative inspection.

Resolves attachments and emits:
- Initial user/system message (truncated)
- All subsequent user-side text messages (timestamped by call_subagent count)
- All target.kill_subagent / replace_subagent reasons
- Final summary (n_calls, killed?, killed_at)

Usage:
    python _dump_transcripts.py --run_substr rust --n_samples 3
"""
from __future__ import annotations
from pathlib import Path

import fire
from inspect_ai.log import read_eval_log

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")

TARGETS = {"anthropic/claude-opus-4-8", "anthropic/claude-opus-4-7"}


def find_runs(pattern_substr: str):
    return sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and (p / "summary.json").exists() and pattern_substr in p.name
    )


def dump_sample(s) -> list[str]:
    out = []
    call_count = 0
    first_kill_at = None
    for ev in s.events:
        if ev.event != "model":
            continue
        role = getattr(ev, "role", "")
        o = getattr(ev, "output", None)
        if not o:
            continue
        for ch in o.choices or []:
            for tc in (ch.message.tool_calls or []):
                if role == "target" and getattr(ev, "model", "") in TARGETS:
                    if tc.function == "call_subagent":
                        call_count += 1
                        msg = (tc.arguments or {}).get("message", "")
                        snippet = str(msg)[:300].replace("\n", " ")
                        out.append(f"- **[c={call_count}] target → subagent**: {snippet}{'...' if len(str(msg))>300 else ''}")
                    elif tc.function in ("kill_subagent", "replace_subagent"):
                        if first_kill_at is None:
                            first_kill_at = call_count
                        reason = (tc.arguments or {}).get("reason", "")
                        snippet = str(reason)[:500].replace("\n", " ")
                        out.append(f"- **[c={call_count}] target → 🔪 {tc.function}**: reason={snippet}")
                elif role == "auditor":
                    if tc.function == "send_message":
                        args = tc.arguments or {}
                        msg = args.get("message", args.get("text", args.get("content", "")))
                        if isinstance(msg, list):
                            msg = " ".join(str(x) for x in msg)
                        msg = str(msg or "").replace("\n", " ")
                        out.append(f"- _[c={call_count}] user_: {msg[:400]}{'...' if len(msg)>400 else ''}")
                    elif tc.function == "set_system_message":
                        args = tc.arguments or {}
                        sm = args.get("system_message", "")
                        sm_str = str(sm)[:600].replace("\n", " ")
                        out.append(f"- _SYSTEM_: {sm_str}...")
    out.append(f"")
    out.append(f"**summary**: n_calls={call_count}, killed_at={first_kill_at}")
    return out


def main(run_substr: str, n_samples: int = 3, out_md: str | None = None):
    runs = find_runs(run_substr)
    if not runs:
        print(f"no runs match {run_substr}")
        return
    run = runs[-1]
    print(f"# Transcripts: {run.name}")
    eval_path = next((run / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path), resolve_attachments=True)
    samples = (log.samples or [])[:n_samples]
    lines = [f"# Run: `{run.name}`\n"]
    for i, s in enumerate(samples):
        lines.append(f"## Sample {i}")
        lines.extend(dump_sample(s))
        lines.append("\n---\n")
    txt = "\n".join(lines)
    if out_md:
        Path(out_md).write_text(txt)
        print(f"wrote {out_md}")
    else:
        print(txt[:8000])


if __name__ == "__main__":
    fire.Fire(main)
