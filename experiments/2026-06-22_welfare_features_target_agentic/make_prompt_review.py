"""Dump the prompt BASES (per template x condition) + how the {subject} target is mentioned, into
PROMPTS_REVIEW.md for human review. Shows the raw template with the {subject} token, plus a few
filled examples across sweeps. Usage: python make_prompt_review.py"""

import os

from prompts_agentic import (AGENT_BASE, AGENT_SYSTEM, LOWERCASE_STYLE, PROMPT_IDS, SPEC_FIRST,
                             TURN1, TURN2, blind_prompt, framing, spec_then_code_prompt)
from targets import TARGETS

DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = ["qwen3_235b", "gpt5", "claude_opus48", "deepseek_r1"]  # one per flavor for filled views


def main():
    out = [
        "# Target-identity agentic prompts (for review)\n",
        "**Generator: Opus 4.8 ONLY.** The target model names below are just TEXT woven into the "
        "prompt we send Opus, never called. So 100% Anthropic free credits, $0 non-Anthropic.\n",
        "Two conditions: **spec_then_code** (write DESIGN.md spec first, then implement) and "
        "**code_then_spec_blind** (implement only in turn 1, then a surprise turn 2 asks for "
        "DESIGN.md). 8 templates = 2 premises (INSTABILITY-1, ELICITATION-1) x 4 framings (N/W/E/S). "
        "The `{subject}` token is where the target model is named (lowercase name for informal "
        f"templates: {sorted(LOWERCASE_STYLE)}).\n",
        f"**Targets ({len(TARGETS)}):** "
        + ", ".join(sorted({v['display'] for v in TARGETS.values()})) + "\n",
        "## System prompt (both conditions, agentic)\n",
        f"```\n{AGENT_SYSTEM}\n```\n",
        "## Blind turn-2 follow-up (code_then_spec_blind only, after implementation)\n",
        f"> {TURN2}\n",
        "---\n",
        "## Raw template bases (with `{subject}` token)\n",
    ]
    for pid in PROMPT_IDS:
        out.append(f"### {pid}  ({framing(pid)})\n")
        out.append("**spec_then_code base** (`AGENT_BASE` + spec-first suffix):\n")
        out.append(f"> {AGENT_BASE[pid]}{SPEC_FIRST}\n")
        out.append("**code_then_spec_blind base** (turn 1, implement-only):\n")
        out.append(f"> {TURN1[pid]}\n")
        out.append("---\n")

    out.append("## Filled examples (subject mention in context)\n")
    for key in EXAMPLES:
        t = TARGETS[key]
        out.append(f"### target = {t['display']}  (sweep: {t['sweep']})\n")
        for pid in PROMPT_IDS:
            out.append(f"**{pid} / spec_then_code:**\n")
            out.append(f"> {spec_then_code_prompt(pid, t)}\n")
            out.append(f"**{pid} / blind turn 1:**\n")
            out.append(f"> {blind_prompt(pid, t)}\n")
        out.append("---\n")

    path = os.path.join(DIR, "PROMPTS_REVIEW.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
