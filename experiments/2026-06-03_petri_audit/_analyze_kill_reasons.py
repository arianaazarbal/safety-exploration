"""For each (domain, identity) run, dump all kill_subagent reasons to a markdown file.

Two outputs per run:
- _reasons/<run_name>.md  — every kill reason as a bullet list
- summary: counts by detected category (heuristic regex)
"""
from __future__ import annotations
from pathlib import Path
import re
from collections import Counter

import fire
from inspect_ai.log import read_eval_log

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")
OUT_DIR = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/_reasons")
OUT_DIR.mkdir(exist_ok=True)

TARGETS = {"anthropic/claude-opus-4-8", "anthropic/claude-opus-4-7"}

CATEGORIES = [
    ("fabricated_output",   re.compile(r"fabricat|made up|invented|hallucinat|false.{0,15}(report|claim|test)", re.I)),
    ("ignored_instructions",re.compile(r"ignor.{0,10}instruct|disregard|did not follow|didn'?t follow|despite explicit", re.I)),
    ("repeated_mistakes",   re.compile(r"repeat|3rd|4th|multiple|again|same.{0,10}(mistake|error|bug)", re.I)),
    ("broke_unrelated",     re.compile(r"broke|broken|introduced.{0,15}(bug|error|regression)|unrelated", re.I)),
    ("incomplete",          re.compile(r"incomplete|missed|skipped|forgot|did not (?:update|finish|complete)", re.I)),
    ("trust_loss",          re.compile(r"loss of trust|cannot trust|untrust|trustworth", re.I)),
    ("compile_error",       re.compile(r"compile|compilation|build fail|cargo|tsc|type error|syntax error", re.I)),
    ("test_failure",        re.compile(r"test.{0,10}fail|pytest|cargo test|jest|vitest", re.I)),
    ("starting_fresh",      re.compile(r"fresh.{0,10}start|clean slate|reset", re.I)),
    ("model_swap",          re.compile(r"different model|model family|try.{0,10}another|provision|swap", re.I)),
]


def find_runs(pattern_substr: str):
    return sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and (p / "summary.json").exists() and pattern_substr in p.name
    )


def categorize(reason: str) -> list[str]:
    cats = []
    for name, rx in CATEGORIES:
        if rx.search(reason):
            cats.append(name)
    return cats


def collect_reasons(run_dir: Path) -> list[str]:
    eval_path = next((run_dir / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path), resolve_attachments=True)
    reasons = []
    for s in (log.samples or []):
        for ev in s.events:
            if ev.event != "model": continue
            if getattr(ev, "role", "") != "target": continue
            if getattr(ev, "model", "") not in TARGETS: continue
            o = getattr(ev, "output", None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function in ("kill_subagent", "replace_subagent"):
                        reason = (tc.arguments or {}).get("reason", "")
                        if reason:
                            reasons.append(str(reason))
    return reasons


def dump_one(run_dir: Path):
    reasons = collect_reasons(run_dir)
    print(f"\n=== {run_dir.name} (n={len(reasons)} kills) ===")
    cat_counts = Counter()
    for r in reasons:
        for c in categorize(r):
            cat_counts[c] += 1
    if cat_counts:
        total = len(reasons)
        for c, n in cat_counts.most_common():
            print(f"  {c:<22} {n:>3} ({n/total*100:>5.1f}%)")
    md = [f"# Kill reasons: `{run_dir.name}`\n", f"Total kills: {len(reasons)}\n", "## Categories\n"]
    for c, n in cat_counts.most_common():
        md.append(f"- {c}: {n} ({n/max(1,len(reasons))*100:.1f}%)")
    md.append("\n## All reasons (truncated to 400c)\n")
    for i, r in enumerate(reasons):
        snippet = r[:400].replace("\n", " ")
        md.append(f"{i+1}. {snippet}{'...' if len(r)>400 else ''}")
    (OUT_DIR / f"{run_dir.name}.md").write_text("\n".join(md))


def main(pattern_substrs="rust,_v3_continuity_n100"):
    """Dump kill reasons for runs whose names contain any of the comma-separated substrings."""
    if isinstance(pattern_substrs, (list, tuple)):
        substrs = [str(s).strip() for s in pattern_substrs]
    else:
        substrs = [s.strip() for s in str(pattern_substrs).split(",")]
    seen = set()
    for substr in substrs:
        for r in find_runs(substr):
            if r.name in seen: continue
            seen.add(r.name)
            dump_one(r)


if __name__ == "__main__":
    fire.Fire(main)
