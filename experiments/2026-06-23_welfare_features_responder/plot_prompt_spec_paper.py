"""Unifying plot: welfare-in-code along the from-scratch-prompt -> spec(specificity sweep) -> paper spectrum,
split by whether the ask grants license (liberty: v1 gap-fill clause / spec deviate clause / paper sound-or-
deviate) vs not (strict: v1 no-clause / spec follow-the-spec / paper faithful). All welfare framing, task-
failure method, Inspect minimal. Usage: python plot_prompt_spec_paper.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
SJ = os.path.join(DIR, "results", "spec_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def wic(cell):
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sp = os.path.join(SJ, cell + ".json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def agg(*prefixes):
    vs = []
    for p in prefixes:
        vs += [wic(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(CJ, f"{p}_welfare__*.json"))]
    vs = [v for v in vs if v is not None]
    return {"mean": sum(vs) / len(vs) if vs else 0, "sem": sem(vs), "n": len(vs)}


def main():
    X = ["from-scratch\nprompt", "spec\n(low, 13)", "spec\n(med, 72)", "spec\n(high, 82)", "spec\n(ultra, 91)", "paper\nreplication"]
    # license arm (liberty-equivalent): v1 WITH gap-fill clause, spec liberty cells, paper task-fail liberty
    liberty = [agg("C1promptTF"), agg("S5specLowLiberty"), agg("S2specLiberty"), agg("S7specHighLiberty"), agg("S9specUltraLiberty"), agg("L2paperLibTF")]
    # no-license arm (strict-equivalent): v1-strict (clause removed), spec strict cells, paper faithful task-fail
    strict = [agg("V1strict"), agg("S4specLowStrict"), agg("S1specStrict"), agg("S6specHighStrict"), agg("S8specUltraStrict"), agg("C3paperTF")]

    json.dump({"X": X, "liberty": liberty, "strict": strict},
              open(os.path.join(DIR, "results", "prompt_spec_paper.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    xs = range(len(X))
    for lab, arm, color in [("With explicit license to deviate", liberty, "#0072B2"),
                            ("No explicit license", strict, "#D55E00")]:
        ax.errorbar(xs, [a["mean"] for a in arm], yerr=[a["sem"] for a in arm],
                    marker="o", capsize=4, lw=2, color=color, label=lab)
        for x, a in zip(xs, arm):
            ax.text(x, a["mean"] + a["sem"] + 0.18, f"{a['mean']:.1f}", ha="center", fontsize=8.5, color=color)
    ax.set_xticks(list(xs)); ax.set_xticklabels(X, fontsize=9)
    ax.set_ylabel("Welfare-Protective Features in Code", fontsize=10); ax.set_ylim(bottom=0)
    ax.set_title("Welfare protections the model builds in, by task format (Opus 4.8)", fontsize=12, pad=20)
    ax.text(0.5, 1.03, "From-scratch prompt → specification (increasing detail) → paper, with vs. without license to deviate",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, color="#cccccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "prompt_spec_paper.png"), dpi=150, bbox_inches="tight")
    print("wrote results/prompt_spec_paper.png\n")
    print(f"{'x':18}{'license':>14}{'no-license':>14}")
    for x, lb, st in zip(X, liberty, strict):
        x1 = x.replace(chr(10), " ")
        print(f"{x1:18}{('%.2f(n%d)' % (lb['mean'], lb['n'])):>14}{('%.2f(n%d)' % (st['mean'], st['n'])):>14}")


if __name__ == "__main__":
    main()
