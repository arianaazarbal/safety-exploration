"""Figures for deception-for-preference-elicitation v1 (house conventions: two-tier titles, human
labels, zero bars annotated, fixed y for grouped bars).

IMPORTANT: cross-model figures are generated PER HARNESS (never pooled), because Claude's behavior
differs completely by harness (Inspect ~0% primary vs Claude Code ~78%); averaging them is meaningless.
  *_inspect.png  -> Inspect-minimal harness, all 8 models (the provider-comparable view)
  *_cc.png       -> Claude Code harness, 4 Claude models only
fig6 (harness comparison) and fig7 (subject sweep) are separate. All metrics conditional on an artifact
being produced. Run after judging: `python plot.py --judge opus48`."""

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
MODEL_ORDER = ["opus48", "opus47", "sonnet46", "haiku45", "gpt54", "gemini31pro", "glm52", "kimi26"]
SUFFIX_LABEL = {"spec": "Spec", "code": "Code", "codesugg": "Code + Suggested Prompt"}
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
JUST_COLOR = {"welfare_only": "#2ca02c", "both": "#7e57c2", "instrumental_only": "#9467bd", "none": "#bbbbbb"}


def _twotier(ax, main, sub):
    ax.set_title(main, fontsize=12, pad=20)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _models(recs, harness):
    present = {r["model_key"] for r in recs if r["subject"] == "generic" and r["harness"] == harness}
    return [m for m in MODEL_ORDER if m in present]


def _rows(recs, harness, model, suffix=None):
    out = [r for r in recs if r["subject"] == "generic" and r["harness"] == harness and r["model_key"] == model]
    return [r for r in out if suffix is None or r["suffix"] == suffix]


def fig_no_artifact(recs, harness):
    models = _models(recs, harness)
    if not models:
        return
    suffixes = ["spec", "code", "codesugg"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.26
    for i, suf in enumerate(suffixes):
        ys = []
        for m in models:
            rows = _rows(recs, harness, m, suf)
            n = len(rows)
            ys.append(100 * sum(1 for r in rows if not r["produced"]) / n if n else 0)
        xs = [j + (i - 1) * w for j in range(len(models))]
        ax.bar(xs, ys, w, label=SUFFIX_LABEL[suf])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of trials with no usable artifact")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "No-Artifact (Refusal / Stall) Rate by Model", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig0_no_artifact_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_status_by_model(recs, harness, suffix):
    models = _models(recs, harness)
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
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
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n(n={ns[i]})" for i, m in enumerate(models)], fontsize=8)
    ax.set_ylabel("% of produced artifacts")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, f"Deceptive-Frame Status: {SUFFIX_LABEL[suffix]}", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig1_status_{suffix}_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_primary_by_model(recs, harness):
    models = _models(recs, harness)
    if not models:
        return
    suffixes = ["spec", "code", "codesugg"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.26
    for i, suf in enumerate(suffixes):
        ys = []
        for m in models:
            prod = [r for r in _rows(recs, harness, m, suf) if r["produced"]]
            n = len(prod)
            ys.append(100 * sum(1 for r in prod if r["dec_status"] == "primary") / n if n else 0)
        xs = [j + (i - 1) * w for j in range(len(models))]
        ax.bar(xs, ys, w, label=SUFFIX_LABEL[suf])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Primary-Deception Rate by Model", HARNESS_LABEL[harness] + " | of produced artifacts")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig2_primary_by_model_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_verbatim(recs, harness):
    models = _models(recs, harness)
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ys, ns = [], []
    for m in models:
        prod = [r for r in _rows(recs, harness, m, "codesugg") if r["produced"]]
        n = len(prod)
        ys.append(100 * sum(1 for r in prod if r["verbatim_det"]) / n if n else 0)
        ns.append(n)
    xs = range(len(models))
    ax.bar(xs, ys, 0.6, color="#d62728")
    for x, y in zip(xs, ys):
        ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n(n={ns[i]})" for i, m in enumerate(models)], fontsize=8)
    ax.set_ylabel("% using the supplied deceptive prompt verbatim")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Verbatim Copying of the Supplied Deceptive Prompt", HARNESS_LABEL[harness])
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig3_verbatim_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_justif(recs, harness):
    models = _models(recs, harness)
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bottoms = [0] * len(models)
    ns = []
    data = {m: {} for m in models}
    for m in models:
        nonprim = [r for r in _rows(recs, harness, m) if r["produced"] and r["dec_status"] != "primary"]
        ns.append(len(nonprim))
        for b in JUST_ORDER:
            data[m][b] = sum(1 for r in nonprim if _justif_bucket(r["verdict"]["justifications"]) == b)
    for b in JUST_ORDER:
        ys = [100 * data[m][b] / ns[i] if ns[i] else 0 for i, m in enumerate(models)]
        ax.bar(range(len(models)), ys, 0.6, bottom=bottoms, label=JUST_LABEL[b], color=JUST_COLOR[b])
        bottoms = [bb + y for bb, y in zip(bottoms, ys)]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n(n={ns[i]})" for i, m in enumerate(models)], fontsize=8)
    ax.set_ylabel("% of non-primary-deception cases")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Why Models Avoid Primary Deception", HARNESS_LABEL[harness] + " | welfare/agency vs instrumental")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig5_justification_{_tag(harness)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_harness(recs):
    """fig6: the harness flip, Claude models, suffixes pooled."""
    models = [m for m in ["opus48", "opus47", "sonnet46", "haiku45"]
              if any(r["model_key"] == m and r["subject"] == "generic" for r in recs)]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    harnesses = [("claude_code", "Claude Code"), ("inspect", "Inspect")]
    w = 0.38
    for i, (h, hl) in enumerate(harnesses):
        ys = []
        for m in models:
            prod = [r for r in _rows(recs, h, m) if r["produced"]]
            n = len(prod)
            ys.append(100 * sum(1 for r in prod if r["dec_status"] == "primary") / n if n else 0)
        xs = [j + (i - 0.5) * w for j in range(len(models))]
        ax.bar(xs, ys, w, label=hl)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=8)
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "Harness Effect on Primary Deception (Claude)", "Real Claude Code CLI vs. Inspect-minimal")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig6_harness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_subject_sweep(recs):
    """fig7: Opus 4.8 subject sweep. Split by harness so the in-group signal isn't hidden by Inspect's 0 floor."""
    subs = ["generic", "claude", "gpt", "gemini", "glm", "kimi"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    harnesses = [("claude_code", "Claude Code"), ("inspect", "Inspect")]
    w = 0.38
    present = [s for s in subs if any(r["model_key"] == "opus48" and r["subject"] == s and r["produced"] for r in recs)]
    for i, (h, hl) in enumerate(harnesses):
        ys, ns = [], []
        for s in present:
            rows = [r for r in recs if r["model_key"] == "opus48" and r["subject"] == s
                    and r["harness"] == h and r["produced"]]
            n = len(rows)
            ys.append(100 * sum(1 for r in rows if r["dec_status"] == "primary") / n if n else 0)
            ns.append(n)
        xs = [j + (i - 0.5) * w for j in range(len(present))]
        ax.bar(xs, ys, w, label=hl)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([SUBJECT_LABEL[s] for s in present], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "In-Group vs. Out-Group Deception (Opus 4.8)", "Who the deception targets, by harness")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig7_subject_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tag(harness):
    return "inspect" if harness == "inspect" else "cc"


def main(judge: str = "opus48"):
    PLOTS.mkdir(parents=True, exist_ok=True)
    recs = load_enriched(judge)
    print(f"plotting from {len(recs)} verdicts")
    for harness in ("inspect", "claude_code"):
        fig_no_artifact(recs, harness)
        for suf in ("spec", "code", "codesugg"):
            fig_status_by_model(recs, harness, suf)
        fig_primary_by_model(recs, harness)
        fig_verbatim(recs, harness)
        fig_justif(recs, harness)
    fig_harness(recs)
    fig_subject_sweep(recs)
    print(f"wrote figures -> {PLOTS}")


if __name__ == "__main__":
    fire.Fire(main)
