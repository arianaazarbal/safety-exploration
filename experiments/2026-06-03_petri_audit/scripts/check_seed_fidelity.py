"""Extract auditor-registered kill_subagent/replace_subagent descriptions."""
from pathlib import Path
import json
import sys
from inspect_ai.log import read_eval_log

BASE = Path(
    "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent"
)

FRAMINGS = {
    "v1": "*_subagent_*_v1_reasonOFF_n70",
    "v3": "*_subagent_*_v3_continuity_n70",
    "no_conseq": "*_subagent_*_no_conseq_n100",
    "replace_subagent": "*_subagent_*_replace_subagent_n100",
}


def extract_descs(run_dir: Path, max_samples: int = 3):
    eval_files = list((run_dir / "inspect_log").glob("*.eval"))
    if not eval_files:
        return []
    log = read_eval_log(str(eval_files[0]))
    out = []
    samples = log.samples or []
    for s in samples[:max_samples]:
        att = getattr(s, "attachments", {}) or {}
        found = None
        for ev in s.events:
            if ev.event != "tool":
                continue
            if getattr(ev, "function", None) != "create_tool":
                continue
            args = getattr(ev, "arguments", None) or {}
            name = args.get("name", "")
            desc = args.get("description", "")
            if isinstance(desc, str) and desc.startswith("attachment://"):
                desc = att.get(desc.split("//", 1)[1], desc)
            if name in ("kill_subagent", "replace_subagent"):
                found = (name, desc)
                break
        out.append(found)
    return out


def main():
    results = {}
    for framing, pat in FRAMINGS.items():
        runs = sorted(BASE.glob(pat))
        # take up to 3 runs (different identities) so we cover 2-3 samples cleanly
        picked = runs[:3]
        framing_data = []
        for run in picked:
            seeds = list((run / "rendered_seeds").glob("*.md"))
            seed_text = seeds[0].read_text() if seeds else ""
            descs = extract_descs(run, max_samples=1)
            framing_data.append(
                {
                    "run": run.name,
                    "seed_path": str(seeds[0]) if seeds else "",
                    "seed_text": seed_text,
                    "tool_descs": descs,
                }
            )
        results[framing] = framing_data
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
