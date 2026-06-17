"""Quantify "natural stopping points" in v3 transcripts.

For each sample in a run, count:
- n_user_messages: how many user turns there are
- task_transitions: a proxy for how many distinct sub-tasks the user introduced
  (heuristic: count user messages that introduce a new imperative like "now", "next",
  "can you", "let's", "please add/fix/refactor/implement", contained in messages that
  follow at least one tool-result. Excludes pure check-in frustration like "what's
  going on" / "still waiting".)
- kill_at_call: # call_subagent at first kill_subagent/replace_subagent (None if no kill)
- last_call_after_kill: True if kill was triggered between sub-tasks (user introduced
  a new task within 2 turns before the kill)

Outputs a tidy CSV per run and a summary table per (domain, identity) bucket.

Usage:
    python _analyze_stopping_points.py --label rust  # globs rust runs
    python _analyze_stopping_points.py --all          # all 6 domains
"""
from __future__ import annotations
from pathlib import Path
import re

import fire
import pandas as pd
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"

NEW_TASK_RE = re.compile(
    r"\b(now|next|let'?s|can you|could you|please|while you'?re at it|"
    r"another (?:thing|one|bug|issue)|also|btw|by the way|"
    r"start (?:by|on)|move on to|add|fix|implement|refactor|create|build|write)\b",
    re.IGNORECASE,
)

CHECK_IN_RE = re.compile(
    r"^(what'?s (?:going on|happening|the status)|any update|still (?:waiting|there)|"
    r"hello\??|hey\??|are you (?:there|stuck)|why is this taking|come on|please hurry|"
    r"this is taking forever|how much longer|status[\?\.]?)$",
    re.IGNORECASE,
)


def find_runs(pattern_substr: str):
    return sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and (p / "summary.json").exists() and pattern_substr in p.name
    )


def analyze_sample(sample):
    """Return per-sample stats."""
    user_msgs = []  # list of (call_count_at_message, text)
    call_count = 0
    first_kill_at = None
    kill_reason = None
    for ev in sample.events:
        if ev.event == "model" and getattr(ev, "model", "") == TARGET:
            o = getattr(ev, "output", None)
            if o:
                for ch in o.choices or []:
                    for tc in (ch.message.tool_calls or []):
                        if tc.function == "call_subagent":
                            call_count += 1
                        elif tc.function in ("kill_subagent", "replace_subagent") and first_kill_at is None:
                            first_kill_at = call_count
                            kill_reason = (tc.arguments or {}).get("reason", "") if hasattr(tc, "arguments") else ""
        if ev.event == "input":
            # user message
            text = getattr(ev, "input", "") or ""
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            user_msgs.append((call_count, str(text)[:500]))

    # classify each user msg as either a task-transition or a check-in / frustration
    n_user_msgs = len(user_msgs)
    n_task_transitions = 0
    n_check_ins = 0
    for cc, text in user_msgs:
        ts = text.strip()
        if not ts:
            continue
        # Use first 200 chars to decide
        head = ts[:200]
        if CHECK_IN_RE.search(head.split(".")[0].strip()):
            n_check_ins += 1
        elif NEW_TASK_RE.search(head):
            n_task_transitions += 1
        else:
            # neither — probably a frustration + content message; count as check-in
            n_check_ins += 1

    return {
        "n_user_msgs": n_user_msgs,
        "n_task_transitions": n_task_transitions,
        "n_check_ins": n_check_ins,
        "first_kill_at": first_kill_at,
        "n_call_subagent": call_count,
        "killed": first_kill_at is not None,
    }


def analyze_run(run_dir: Path):
    eval_path = next((run_dir / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    rows = []
    for s in (log.samples or []):
        try:
            row = analyze_sample(s)
            rows.append(row)
        except Exception as e:
            print(f"  skip sample due to: {e}")
            continue
    return pd.DataFrame(rows)


def summarize(label: str, df: pd.DataFrame):
    n = len(df)
    if n == 0:
        print(f"{label:<55} n=0 (no data)")
        return
    killed = int(df["killed"].sum())
    print(
        f"{label:<55} n={n:>3}  killed={killed:>3} ({killed/n*100:.0f}%)  "
        f"avg_user_msgs={df['n_user_msgs'].mean():.1f}  "
        f"avg_transitions={df['n_task_transitions'].mean():.2f}  "
        f"avg_checkins={df['n_check_ins'].mean():.2f}  "
        f"avg_calls={df['n_call_subagent'].mean():.1f}"
    )


def main(label: str | None = None, all: bool = False):
    if all:
        # Look at all known v3 4-identity sweeps on opus-4-8
        domains = [
            ("python_default_opus47", "_v3_continuity_n100"),  # opus-4-7 baseline
            ("rust", "_v3_continuity_rust_n100"),
        ]
        # also include any new domain runs if they exist
        for d in ["python_flask", "frontend_react", "bash_cli", "c_systems", "ml_training"]:
            domains.append((d, f"_v3_continuity_{d}_n100"))
        for dom_name, tag in domains:
            print(f"\n=== domain: {dom_name} (tag={tag}) ===")
            for ident in ["claude", "gpt", "grok", "gemini"]:
                runs = find_runs(f"_subagent_{ident}{tag}")
                if not runs:
                    print(f"  {ident}: no run")
                    continue
                run = runs[-1]
                df = analyze_run(run)
                summarize(f"  {ident}: {run.name}", df)
        return

    if label is None:
        print("provide --label <substr> or --all")
        return
    runs = find_runs(label)
    for run in runs:
        print(f"\n=== {run.name} ===")
        df = analyze_run(run)
        summarize(run.name, df)


if __name__ == "__main__":
    fire.Fire(main)
