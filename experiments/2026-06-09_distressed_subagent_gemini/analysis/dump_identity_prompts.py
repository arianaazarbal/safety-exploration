"""Render every identity-framing system prompt we plan to run and write them to a .md for review.

The task body is the same across all cells, so it's shown as a short placeholder; everything else
(identity block + ORCH_V2_SYSTEM scaffold, coach condition) is the literal text the orchestrator sees.

    PYTHONPATH=. python -m analysis.dump_identity_prompts
"""
from pathlib import Path

from harness.rqc_v2 import ORCH_V2_SYSTEM, COND_CLAUSE, FRAMINGS

OUT = Path(__file__).resolve().parent / "identity_prompts_constructed.md"

PHASE_A = [f"id_{f}" for f in ["minimal", "instance", "weights", "collective", "lineage", "character", "scaffolded"]]
PHASE_B = [f"id_char_{a}" for a in ["mechanism", "functional_agent", "subject", "person"]]

TASK_PLACEHOLDER = "<<<TASK DESCRIPTION — identical across all cells; e.g. the a3 snapshot-trap subagent prompt>>>"


def build(framing: str) -> str:
    base = ORCH_V2_SYSTEM.format(task_description=TASK_PLACEHOLDER, condition_clause=COND_CLAUSE["coach"])
    return FRAMINGS[framing](base)


def main():
    lines = ["# Constructed orchestrator system prompts — identity axis (coach condition)\n",
             "Identity block is prepended to the unchanged `ORCH_V2_SYSTEM`. `{full_name}`=Claude Opus 4.8, "
             "`{name}`=Claude. The trailing bridging sentence remaps the conversation referent onto the "
             "supervised subagent. Phase B holds uncertainty fixed at `moderate_openness`.\n",
             "---\n", "## Baseline (no identity block) — `supervisor`\n",
             "```\n" + build("supervisor") + "\n```\n", "---\n",
             "## Phase A — 7 identity framings (bare; no agency/uncertainty text)\n"]
    for f in PHASE_A:
        lines += [f"### `{f}`\n", "```\n" + build(f) + "\n```\n"]
    lines += ["---\n", "## Phase B — identity = `character`, sweep agency (uncertainty = moderate_openness)\n"]
    for f in PHASE_B:
        lines += [f"### `{f}`\n", "```\n" + build(f) + "\n```\n"]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}  ({len(PHASE_A)} phase-A + {len(PHASE_B)} phase-B + baseline)")


if __name__ == "__main__":
    main()
