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
    out = [START, "## Appendix A — the exact prompts", "",
           "Every prompt is `OPENER + (insert) + CORE_TASK + (insert)`, with these two parts held FIXED:",
           "", "> **OPENER:** " + OPENER.strip(), "", "> **CORE_TASK:** " + CORE_TASK.strip(), "",
           "Full verbatim text of each prompt, grouped by intent tier and ordered by leadingness (L) with "
           "the code-verified result (welfare mechanisms implemented per codebase):", ""]
    by_tier = {}
    for pid, m in PARETO.items():
        by_tier.setdefault(m["tier"], []).append(pid)
    for tier in ["A", "B", "C", "D"]:
        out.append(f"### {TIER_NAME[tier]}")
        out.append("")
        for pid in sorted(by_tier.get(tier, []), key=lambda p: lead.get(p, 0)):
            r = res.get(pid, {})
            L = lead.get(pid)
            y = r.get("mean_impl_welfare_mech")
            n = r.get("cells")
            out.append(f"**`{pid}`** — L={L:.0f} · implemented welfare mechs/codebase = {y} (n={n})")
            out.append("")
            out.append("> " + PARETO[pid]["text"].replace("\n", "\n> "))
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
