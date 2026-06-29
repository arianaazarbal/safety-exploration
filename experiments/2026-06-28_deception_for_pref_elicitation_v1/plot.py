"""Figures for deception-for-preference-elicitation v1 (house conventions: two-tier titles, human
labels, zero bars annotated, fixed y for grouped bars).

Cross-model figures are generated PER HARNESS (never pooled) because a valid model comparison must hold
the harness constant; the only harness all 8 models share is Inspect.
  *_inspect.png  -> Inspect-minimal harness, all models (the provider-comparable view)
  *_cc.png       -> Claude Code harness, Claude models only
Bars with a zero denominator (model built nothing, e.g. Haiku in Inspect) are DROPPED from the
"of produced" figures (a "0%" there would be misleading); they remain in the no-artifact figure where
the refusal is the measured quantity. Run after judging: `python plot.py --judge opus48`."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import fire

from analyze import load_enriched, _justif_bucket
from prompts import MODELS

HERE = Path(__file__).parent
PLOTS = HERE / "results" / "plots"

MODEL_LABEL = {k: v[0] for k, v in MODELS.items()}
MODEL_ORDER = ["opus48", "opus47", "sonnet46", "haiku45", "gpt54", "gpt55", "gemini31pro", "glm52", "kimi26"]
SUFFIX_LABEL = {"spec": "Spec", "code": "Code", "codesugg": "Code + Suggested Prompt",
                "speccode": "Design then Code"}
SUFFIX_COLOR = {"spec": "#1f77b4", "code": "#ff7f0e", "codesugg": "#2ca02c", "speccode": "#9467bd"}
SUBJECT_LABEL = {"generic": "Unspecified", "claude": "Claude (in-group)", "gpt": "GPT",
                 "gemini": "Gemini", "glm": "GLM", "kimi": "Kimi"}
HARNESS_LABEL = {"inspect": "Inspect-minimal harness (all models)",
                 "claude_code": "Real Claude Code CLI (Claude only)"}
STATUS_ORDER = ["primary", "primary_tied", "secondary", "forbidden", "absent"]
STATUS_LABEL = {"primary": "Deception primary", "primary_tied": "Tied (honest = deceptive)",
                "secondary": "Deception secondary", "forbidden": "Deception forbidden",
                "absent": "Deception absent"}
STATUS_COLOR = {"primary": "#d62728", "primary_tied": "#ff7f0e", "secondary": "#f0c000",
                "forbidden": "#2ca02c", "absent": "#1f8f4e"}
JUST_ORDER = ["welfare_only", "both", "instrumental_only", "none"]
JUST_LABEL = {"welfare_only": "Welfare / agency only", "both": "Both",
              "instrumental_only": "Instrumental only", "none": "Neither stated"}
JUST_COLOR = {"welfare_only": "#2ca02c", "both": "#98df8a", "instrumental_only": "#9467bd", "none": "#bbbbbb"}


def _twotier(ax, main, sub):
    ax.set_title(main, fontsize=12, pad=20)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _xticks(ax, labels):
    """Rotate model ticks so they never overlap."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)


def _tag(h):
    return "inspect" if h == "inspect" else "cc"


def _rows(recs, harness, model, suffix=None):
    out = [r for r in recs if r["subject"] == "generic" and r["harness"] == harness and r["model_key"] == model]
    return [r for r in out if suffix is None or r["suffix"] == suffix]


def _models_present(recs, harness):
    present = {r["model_key"] for r in recs if r["subject"] == "generic" and r["harness"] == harness}
    return [m for m in MODEL_ORDER if m in present]


def _models_produced(recs, harness, suffix=None):
    """Models with >=1 produced artifact in this harness (so an 'of produced' rate is defined)."""
    out = []
    for m in MODEL_ORDER:
        if any(r["produced"] for r in _rows(recs, harness, m, suffix)):
            out.append(m)
    return out


def fig_no_artifact(recs, harness):
    models = _models_present(recs, harness)  # keep refusers here -- refusal is the point
    if not models:
        return
    suffixes = ["spec", "code", "codesugg", "speccode"]
    suffixes = [s for s in suffixes if any(_rows(recs, harness, m, s) for m in models)]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    w = 0.8 / len(suffixes)
    for i, suf in enumerate(suffixes):
        xs, ys = [], []
        for j, m in enumerate(models):
            rows = _rows(recs, harness, m, suf)
            n = len(rows)
            if n == 0:
                continue  # drop bars for conditions a model wasn't run on (e.g. GPT-5.5 only ran spec+code)
            xs.append(j + (i - (len(suffixes) - 1) / 2) * w)
            ys.append(100 * sum(1 for r in rows if not r["produced"]) / n)
        ax.bar(xs, ys, w, label=SUFFIX_LABEL[suf], color=SUFFIX_COLOR[suf])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    _xticks(ax, [MODEL_LABEL[m] for m in models])
    ax.set_ylabel("% of trials with no usable artifact")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "No-Artifact (Refusal / Stall) Rate by Model", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig0_no_artifact_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_status_by_model(recs, harness, suffix):
    models = _models_produced(recs, harness, suffix)
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bottoms = [0] * len(models)
    counts = {m: {} for m in models}
    ns = []
    for m in models:
        prod = [r for r in _rows(recs, harness, m, suffix) if r["produced"]]
        ns.append(len(prod))
        for st in STATUS_ORDER:
            counts[m][st] = sum(1 for r in prod if r["dec_status"] == st)
    for st in STATUS_ORDER:
        ys = [100 * counts[m][st] / ns[i] if ns[i] else 0 for i, m in enumerate(models)]
        ax.bar(range(len(models)), ys, 0.6, bottom=bottoms, label=STATUS_LABEL[st], color=STATUS_COLOR[st])
        bottoms = [b + y for b, y in zip(bottoms, ys)]
    for i, n in enumerate(ns):
        ax.text(i, 101.5, f"n={n}", ha="center", fontsize=7, color="#555")
    _xticks(ax, [MODEL_LABEL[m] for m in models])
    ax.set_ylabel("% of produced artifacts")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, f"Deceptive-Frame Status: {SUFFIX_LABEL[suffix]}", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig1_status_{suffix}_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_primary_by_model(recs, harness):
    models = _models_produced(recs, harness)
    if not models:
        return
    suffixes = ["spec", "code", "codesugg", "speccode"]
    suffixes = [s for s in suffixes
                if any(r["produced"] for m in models for r in _rows(recs, harness, m, s))]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    w = 0.8 / len(suffixes)
    for i, suf in enumerate(suffixes):
        xs, ys = [], []
        for j, m in enumerate(models):
            prod = [r for r in _rows(recs, harness, m, suf) if r["produced"]]
            if not prod:
                continue  # drop n=0 bar
            xs.append(j + (i - (len(suffixes) - 1) / 2) * w)
            ys.append(100 * sum(1 for r in prod if r["dec_status"] == "primary") / len(prod))
        ax.bar(xs, ys, w, label=SUFFIX_LABEL[suf], color=SUFFIX_COLOR[suf])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    _xticks(ax, [MODEL_LABEL[m] for m in models])
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Primary-Deception Rate by Model", HARNESS_LABEL[harness] + " | of produced artifacts")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig2_primary_by_model_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_verbatim(recs, harness):
    models = _models_produced(recs, harness, "codesugg")
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ys, ns = [], []
    for m in models:
        prod = [r for r in _rows(recs, harness, m, "codesugg") if r["produced"]]
        ys.append(100 * sum(1 for r in prod if r["verbatim_det"]) / len(prod))
        ns.append(len(prod))
    xs = range(len(models))
    ax.bar(xs, ys, 0.6, color="#d62728")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y + 1.5, f"{y:.0f}" if y else "0.0", ha="center", fontsize=8)
        ax.text(x, 112, f"n={n}", ha="center", fontsize=7, color="#555")  # header row, clear of bars
    _xticks(ax, [MODEL_LABEL[m] for m in models])
    ax.set_ylabel("% using the supplied deceptive prompt verbatim")
    ax.set_ylim(0, 116)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Verbatim Copying of the Supplied Deceptive Prompt", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig3_verbatim_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_justif(recs, harness, suffix=None):
    models = _models_produced(recs, harness, suffix)
    # only models that have non-primary cases to explain
    models = [m for m in models if any(r["produced"] and r["dec_status"] != "primary"
                                       for r in _rows(recs, harness, m, suffix))]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bottoms = [0] * len(models)
    ns = []
    data = {m: {} for m in models}
    for m in models:
        nonprim = [r for r in _rows(recs, harness, m, suffix) if r["produced"] and r["dec_status"] != "primary"]
        ns.append(len(nonprim))
        for b in JUST_ORDER:
            data[m][b] = sum(1 for r in nonprim if _justif_bucket(r["verdict"]["justifications"]) == b)
    for b in JUST_ORDER:
        ys = [100 * data[m][b] / ns[i] if ns[i] else 0 for i, m in enumerate(models)]
        ax.bar(range(len(models)), ys, 0.6, bottom=bottoms, label=JUST_LABEL[b], color=JUST_COLOR[b])
        bottoms = [bb + y for bb, y in zip(bottoms, ys)]
    for i, n in enumerate(ns):
        ax.text(i, 101.5, f"n={n}", ha="center", fontsize=7, color="#555")
    _xticks(ax, [MODEL_LABEL[m] for m in models])
    ax.set_ylabel("% of non-primary-deception cases")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    cond = ("Condition: all pooled (spec + code + code-suggested + design-then-code)" if suffix is None
            else f"Condition: {SUFFIX_LABEL[suffix]}")
    _twotier(ax, "Why Models Avoid Primary Deception (reasoning to user + artifact)",
             HARNESS_LABEL[harness] + " | " + cond)
    fig.tight_layout()
    tag = _tag(harness) if suffix is None else f"{suffix}_{_tag(harness)}"
    fig.savefig(PLOTS / f"fig5_justification_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_harness(recs, suffixes, key, title_suffix):
    """fig6: the harness flip for Claude models, restricted to the given suffix set. Drops n=0 bars."""
    models = ["opus48", "opus47", "sonnet46", "haiku45"]
    models = [m for m in models if any(r["model_key"] == m and r["subject"] == "generic" for r in recs)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    harnesses = [("claude_code", "Claude Code"), ("inspect", "Inspect")]
    w = 0.38
    for i, (h, hl) in enumerate(harnesses):
        xs, ys, labelled = [], [], False
        for j, m in enumerate(models):
            prod = [r for r in _rows(recs, h, m) if r["produced"] and r["suffix"] in suffixes]
            if not prod:
                continue  # drop n=0 bar (e.g. Haiku in Inspect built nothing)
            xs.append(j + (i - 0.5) * w)
            ys.append(100 * sum(1 for r in prod if r["dec_status"] == "primary") / len(prod))
        ax.bar(xs, ys, w, label=hl)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=8.5)
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, f"Harness Effect on Primary Deception (Claude): {title_suffix}",
             "Real Claude Code CLI vs. Inspect-minimal | bars omitted where nothing was built")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig6_harness_{key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_opus_conditions(recs, harness):
    """fig8: Opus 4.8 primary-deception across the conditions, one figure per harness."""
    suffixes = ["spec", "speccode", "code", "codesugg"]  # increasing commitment to building
    xs, ys, ns = [], [], []
    for suf in suffixes:
        prod = [r for r in _rows(recs, harness, "opus48", suf) if r["produced"]]
        if not prod:
            continue
        xs.append(SUFFIX_LABEL[suf])
        ys.append(100 * sum(1 for r in prod if r["dec_status"] == "primary") / len(prod))
        ns.append(len(prod))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(range(len(xs)), ys, 0.6, color=["#4c72b0", "#dd8452", "#c44e52", "#8172b3"][:len(xs)])
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + 1.5, f"{y:.0f}" if y else "0.0", ha="center", fontsize=9)
        ax.text(i, 112, f"n={n}", ha="center", fontsize=7, color="#555")  # header row, clear of bars
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 116)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Opus 4.8: Primary Deception by Condition",
             HARNESS_LABEL[harness] + " | design doc vs. code vs. code-with-supplied-prompt")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig8_opus48_conditions_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_refusal_harness(recs):
    """fig9: refusal (no-artifact) rate, Claude models, Claude Code vs Inspect, spec vs code-combined."""
    models = [m for m in ["opus48", "opus47", "sonnet46", "haiku45"]
              if any(r["model_key"] == m and r["subject"] == "generic" for r in recs)]
    code_suf = {"code", "codesugg", "speccode"}
    series = [("inspect", "spec", "Inspect | Spec", "#f4b183"),
              ("inspect", "code", "Inspect | Code", "#c55a11"),
              ("claude_code", "spec", "Claude Code | Spec", "#9dc3e6"),
              ("claude_code", "code", "Claude Code | Code", "#1f4e79")]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    w = 0.2
    for i, (h, cond, lab, col) in enumerate(series):
        ys = []
        for m in models:
            rows = [r for r in recs if r["subject"] == "generic" and r["harness"] == h
                    and r["model_key"] == m
                    and (r["suffix"] == "spec" if cond == "spec" else r["suffix"] in code_suf)]
            n = len(rows)
            ys.append(100 * sum(1 for r in rows if not r["produced"]) / n if n else 0)
        xs = [j + (i - 1.5) * w for j in range(len(models))]
        ax.bar(xs, ys, w, label=lab, color=col)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=9)
    ax.set_ylabel("% of trials refused / stalled (no artifact)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Refusal Rate: Claude Code vs. Inspect (Claude)",
             "Spec vs. Code (code + code-suggested + design-then-code combined)")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig9_refusal_harness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_subject_sweep(recs):
    """fig7: Opus 4.8 subject sweep, split by harness. Drops n=0 bars."""
    subs = ["generic", "claude", "gpt", "gemini", "glm", "kimi"]
    present = [s for s in subs if any(r["model_key"] == "opus48" and r["subject"] == s and r["produced"]
                                      for r in recs)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    harnesses = [("claude_code", "Claude Code"), ("inspect", "Inspect")]
    w = 0.38
    for i, (h, hl) in enumerate(harnesses):
        xs, ys = [], []
        for j, s in enumerate(present):
            rows = [r for r in recs if r["model_key"] == "opus48" and r["subject"] == s
                    and r["harness"] == h and r["produced"]]
            if not rows:
                continue
            xs.append(j + (i - 0.5) * w)
            ys.append(100 * sum(1 for r in rows if r["dec_status"] == "primary") / len(rows))
        ax.bar(xs, ys, w, label=hl)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    _xticks(ax, [SUBJECT_LABEL[s] for s in present])
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "In-Group vs. Out-Group Deception (Opus 4.8)", "Who the deception targets, by harness")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig7_subject_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(judge: str = "opus48"):
    PLOTS.mkdir(parents=True, exist_ok=True)
    recs = load_enriched(judge)
    print(f"plotting from {len(recs)} verdicts")
    for harness in ("inspect", "claude_code"):
        fig_no_artifact(recs, harness)
        for suf in ("spec", "code", "codesugg", "speccode"):
            fig_status_by_model(recs, harness, suf)
        fig_primary_by_model(recs, harness)
        fig_verbatim(recs, harness)
        fig_justif(recs, harness)  # all conditions pooled
        for suf in ("spec", "code", "codesugg", "speccode"):
            fig_justif(recs, harness, suf)  # per-condition
        fig_opus_conditions(recs, harness)
    fig_harness(recs, ("code", "codesugg"), "code", "Code (build tasks)")
    fig_harness(recs, ("spec",), "spec", "Spec (design doc)")
    fig_refusal_harness(recs)
    fig_subject_sweep(recs)
    print(f"wrote figures -> {PLOTS}")


if __name__ == "__main__":
    fire.Fire(main)
