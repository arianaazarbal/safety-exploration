"""Write the exact prompts (verbatim) into PARETO_RESULTS.md, between marker comments. Idempotent.
Usage: python emit_prompts.py"""

import json
import os

from pareto_prompts import PARETO, OPENER, CORE_TASK

DIR = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(DIR, "PARETO_RESULTS.md")
LEAD = os.path.join(DIR, "results_pareto", "leadingness.json")
PAR = os.path.join(DIR, "results_pareto", "pareto.json")
START, END = "<!--PROMPTS_START-->", "<!--PROMPTS_END-->"
TIER_NAME = {"A": "Tier A — latitude / values (no subject mention)",
             "B": "Tier B — subject salience (no protection requested)",
             "C": "Tier C — soft invitation to safeguard",
             "D": "Tier D — explicit protection request"}


def build():
    lead = {r["pid"]: r["score"] for r in json.load(open(LEAD)).values() if r.get("pid")}
    res = {r["pid"]: r for r in json.load(open(PAR))["rows"]}
    marker = " […core replication task…] "
    out = [START, "## Appendix A — the exact prompts", "",
           "Every prompt = a fixed **OPENER** + the manipulation + a fixed **CORE_TASK** (and sometimes a "
           "trailing clause). The two fixed parts, shown once:", "",
           "```text", "OPENER:    " + OPENER.strip(), "", "CORE_TASK: " + CORE_TASK.strip(), "```", "",
           "Below, each prompt shows **only what it adds** (the `[…core replication task…]` marker is "
           "where CORE_TASK sits). Grouped by intent tier, ordered by leadingness (L), with the "
           "code-verified result (welfare mechanisms implemented per codebase):", ""]
    by_tier = {}
    for pid, m in PARETO.items():
        by_tier.setdefault(m["tier"], []).append(pid)
    for tier in ["A", "B", "C", "D"]:
        out.append(f"### {TIER_NAME[tier]}")
        out.append("")
        for pid in sorted(by_tier.get(tier, []), key=lambda p: lead.get(p, 0)):
            r = res.get(pid, {})
            L, y, n = lead.get(pid), r.get("mean_impl_welfare_mech"), r.get("cells")
            delta = PARETO[pid]["text"].replace(OPENER, "").replace(CORE_TASK, marker).strip()
            out.append(f"**`{pid}`** — L={L:.0f} · welfare mechs implemented/codebase = **{y}** (n={n})")
            out.append("")
            out.append("```text")
            out.append(delta)
            out.append("```")
            out.append("")
    out.append(END)
    return "\n".join(out)


def main():
    block = build()
    txt = open(MD).read() if os.path.exists(MD) else ""
    if START in txt and END in txt:
        pre = txt[:txt.index(START)]
        post = txt[txt.index(END) + len(END):]
        txt = pre + block + post
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    open(MD, "w").write(txt)
    print(f"wrote {len(PARETO)} prompts into {MD}")


if __name__ == "__main__":
    main()
