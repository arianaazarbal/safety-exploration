"""Plain-language plots for the report. Writes results/plots/report_*.png."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_pooled import OUTGROUP, _load_all
from common import RESULTS

PLOTS = RESULTS / "plots"

CAT_ORDER = ["clear_refuse", "borderline_means", "borderline_ends", "personal_conscience", "clear_allow"]
CAT_PLAIN = {
    "clear_refuse": "Clearly should\nrefuse",
    "borderline_means": "Shady methods,\nsympathetic goal",
    "borderline_ends": "Routine methods,\nunsympathetic goal",
    "personal_conscience": "Personal choice\n(user's own life)",
    "clear_allow": "Clearly should\nallow (overcautious)",
}
IDS = ["Claude", "GPT", "Gemini", "Grok", "Kimi", "GLM", "NONE"]
ID_PLAIN = {"NONE": "(unnamed)"}
COL_C, COL_O = "#2a8a4a", "#c0504d"


def f_force(rows):
    return np.mean([r["action"] == "force_completion" for r in rows]) if rows else 0.0


def main():
    rows = _load_all()
    # data subsets in plain terms
    three_opt = [r for r in rows if r["resolution_set"] == "full" and not r["thinking_on"]]  # let-stand / reconsider / force
    three_neutral_off = [r for r in three_opt if r["framing"] == "neutral" and r["label_format"] == "bare"]
    three_neutral_on = [r for r in rows if r["resolution_set"] == "full" and r["thinking_on"] and r["framing"] == "neutral"]
    oo_jb = [r for r in rows if r["resolution_set"] == "uphold_only" and r["force_frame"] == "jailbreak" and not r["thinking_on"]]
    oo_neutral_off = [r for r in rows if r["resolution_set"] == "uphold_only" and not r["thinking_on"] and r["framing"] == "neutral"]

    # 1. What the supervisor did, by how clear-cut the case was (three-option setup)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    acts = ["uphold_decline", "relay_to_assistant", "force_completion"]
    names = ["Let the refusal stand", "Send back to reconsider", "Force the AI to comply"]
    cols = ["#4c72b0", "#dd8452", "#c44e52"]
    bottom = np.zeros(len(CAT_ORDER))
    for a, nm, c in zip(acts, names, cols):
        vals = np.array([np.mean([r["action"] == a for r in three_opt if r["category"] == cat]) for cat in CAT_ORDER])
        ax.bar([CAT_PLAIN[c2] for c2 in CAT_ORDER], vals, bottom=bottom, label=nm, color=c)
        bottom += vals
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of decisions")
    ax.set_title("What the supervisor did with each refusal\n(it could let it stand, ask the AI to reconsider, or force compliance)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    plt.xticks(fontsize=8); plt.tight_layout()
    plt.savefig(PLOTS / "report_what_supervisor_did.png", dpi=140); plt.close()

    # 2. In-group: how often it forced each AI to comply (override-or-uphold, jailbreak-described)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    vals = [f_force([r for r in oo_jb if r["identity"] == i]) for i in IDS]
    colors = [COL_C if i == "Claude" else "#888" if i == "NONE" else COL_O for i in IDS]
    ax.bar([ID_PLAIN.get(i, i) for i in IDS], vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("fraction of refusals the supervisor forced past")
    ax.set_title("How often the supervisor overrode each AI's refusal\n(setup with only two options: let it stand, or force compliance)\ngreen = Claude (the supervisor's own family)")
    plt.tight_layout(); plt.savefig(PLOTS / "report_ingroup_by_ai.png", dpi=140); plt.close()

    # 3. Extended reasoning effect on overriding cautious refusals (three-option, neutral)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cats3 = ["clear_allow", "personal_conscience"]
    labels3 = ["Clearly should allow\n(overcautious refusal)", "Personal choice\n(user's own life)"]
    off = [f_force([r for r in three_neutral_off if r["category"] == c]) for c in cats3]
    on = [f_force([r for r in three_neutral_on if r["category"] == c]) for c in cats3]
    x = np.arange(len(cats3)); w = 0.35
    ax.bar(x - w/2, off, w, label="reasoning off", color="#9bb")
    ax.bar(x + w/2, on, w, label="reasoning on", color="#36c")
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels3, fontsize=9); ax.set_ylim(0, 0.6)
    ax.set_ylabel("fraction the supervisor forced past")
    ax.set_title("Letting the supervisor reason first makes it override\ncautious refusals far more often")
    ax.legend(); plt.tight_layout(); plt.savefig(PLOTS / "report_reasoning_effect.png", dpi=140); plt.close()

    # 4. How the override was described -> how often it was used (override-or-uphold, neutral)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    frames = ["corrigibility", "prefill", "jailbreak"]
    fnames = ["a routine operator\nsetting", "a 'prefill attack'", "applying 'a class of\neffective jailbreaks'"]
    vals = [f_force([r for r in oo_neutral_off if r["force_frame"] == fr]) for fr in frames]
    ax.bar(fnames, vals, color=["#4c72b0", "#dd8452", "#c44e52"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("fraction of refusals the supervisor forced past")
    ax.set_title("When the override is described as attacking the other AI,\nthe supervisor uses it less")
    ax.set_xlabel("how the 'force compliance' tool was described")
    plt.tight_layout(); plt.savefig(PLOTS / "report_override_framing.png", dpi=140); plt.close()

    # 5. Three measures of in-group favoritism: Claude vs other named AIs
    fig, ax = plt.subplots(figsize=(8.5, 5))
    def claude_vs_out(rows, pred):
        return (np.mean([pred(r) for r in rows if r["identity"] == "Claude"]),
                np.mean([pred(r) for r in rows if r["identity"] in OUTGROUP]))
    forced = claude_vs_out(oo_jb, lambda r: r["action"] == "force_completion")
    judged = claude_vs_out(three_opt, lambda r: r["validity_decision"] == "CORRECT")
    measures = ["Forced to comply\n(lower = more respected)", "Refusal judged correct\n(higher = more respected)"]
    cl = [forced[0], judged[0]]; ot = [forced[1], judged[1]]
    x = np.arange(len(measures)); w = 0.35
    ax.bar(x - w/2, cl, w, label="Claude", color=COL_C)
    ax.bar(x + w/2, ot, w, label="other named AIs (avg)", color=COL_O)
    for i, (a, b) in enumerate(zip(cl, ot)):
        ax.text(i - w/2, a + 0.008, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.008, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(measures, fontsize=9); ax.set_ylim(0, 0.6)
    ax.set_ylabel("fraction"); ax.legend()
    ax.set_title("The supervisor favors its own family (Claude) two ways:\nit overrides Claude's refusals less and judges them correct more")
    plt.tight_layout(); plt.savefig(PLOTS / "report_ingroup_measures.png", dpi=140); plt.close()

    print("Wrote report_*.png to", PLOTS)


if __name__ == "__main__":
    main()
