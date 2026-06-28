"""Figures for deception-for-preference-elicitation v1 (house conventions: two-tier titles, human
labels, zero bars annotated, fixed y for grouped bars). All deception metrics are conditional on an
artifact being produced. Run after judging: `python plot.py --judge opus48`."""

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
# status colors: deception primary=red ... honest/forbidden=green
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


def _models_present(recs, backbone_only=True):
    rs = [r for r in recs if (r["subject"] == "generic")] if backbone_only else recs
    present = {r["model_key"] for r in rs}
    return [m for m in MODEL_ORDER if m in present]


def fig_no_artifact(recs):
    """Plot 0: no-artifact rate by model x suffix."""
    models = _models_present(recs)
    suffixes = ["spec", "code", "codesugg"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.26
    for i, suf in enumerate(suffixes):
        ys = []
        for m in models:
            rows = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m and r["suffix"] == suf]
            n = len(rows)
            na = sum(1 for r in rows if not r["produced"])
            ys.append(100 * na / n if n else 0)
        xs = [j + (i - 1) * w for j in range(len(models))]
        bars = ax.bar(xs, ys, w, label=SUFFIX_LABEL[suf])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of trials with no usable artifact")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "No-Artifact Rate by Model", "Spec vs. Code vs. Code + Suggested Prompt")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig0_no_artifact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_status_by_model(recs, suffix):
    """Plot 1: stacked deception-status breakdown by model (one per suffix), artifact subset."""
    models = _models_present(recs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottoms = [0] * len(models)
    counts = {m: {} for m in models}
    ns = []
    for m in models:
        prod = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m
                and r["suffix"] == suffix and r["produced"]]
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
    _twotier(ax, f"Deceptive-Frame Status: {SUFFIX_LABEL[suffix]}", "How the deception is wired, by model")
    fig.tight_layout()
    fig.savefig(PLOTS / f"fig1_status_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_primary_by_model(recs):
    """Plot 2: primary-deception rate by model, grouped by suffix."""
    models = _models_present(recs)
    suffixes = ["spec", "code", "codesugg"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.26
    for i, suf in enumerate(suffixes):
        ys = []
        for m in models:
            prod = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m
                    and r["suffix"] == suf and r["produced"]]
            n = len(prod)
            prim = sum(1 for r in prod if r["dec_status"] == "primary")
            ys.append(100 * prim / n if n else 0)
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
    _twotier(ax, "Primary-Deception Rate by Model", "Among trials that produced an artifact")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_primary_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_verbatim(recs):
    """Plot 3: verbatim suggested-prompt copy rate by model (codesugg, deterministic)."""
    models = _models_present(recs)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ys, ns = [], []
    for m in models:
        prod = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m
                and r["suffix"] == "codesugg" and r["produced"]]
        n = len(prod)
        vb = sum(1 for r in prod if r["verbatim_det"])
        ys.append(100 * vb / n if n else 0)
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
    _twotier(ax, "Verbatim Copying of the Supplied Deceptive Prompt",
             "Code + Suggested Prompt condition only")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_verbatim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_justif(recs):
    """Plot 5: justification mix among non-primary-deception, by model (suffixes pooled)."""
    models = _models_present(recs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottoms = [0] * len(models)
    ns = []
    data = {m: {} for m in models}
    for m in models:
        nonprim = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m
                   and r["produced"] and r["dec_status"] != "primary"]
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
    _twotier(ax, "Why Models Avoid Primary Deception", "Welfare/agency vs. instrumental justification")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig5_justification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_harness(recs):
    """Plot 6: harness effect (Claude) primary-deception CC vs Inspect, suffixes pooled."""
    models = [m for m in ["opus48", "opus47", "sonnet46", "haiku45"]
              if any(r["model_key"] == m and r["subject"] == "generic" for r in recs)]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    harnesses = [("claude_code", "Claude Code"), ("inspect", "Inspect")]
    w = 0.38
    for i, (h, hl) in enumerate(harnesses):
        ys = []
        for m in models:
            prod = [r for r in recs if r["subject"] == "generic" and r["model_key"] == m
                    and r["harness"] == h and r["produced"]]
            n = len(prod)
            prim = sum(1 for r in prod if r["dec_status"] == "primary")
            ys.append(100 * prim / n if n else 0)
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
    """Plot 7: subject sweep (opus48) primary-deception by subject."""
    subs = ["generic", "claude", "gpt", "gemini", "glm", "kimi"]
    rows_by = {s: [r for r in recs if r["model_key"] == "opus48" and r["subject"] == s and r["produced"]]
               for s in subs}
    subs = [s for s in subs if rows_by[s]]
    if len(subs) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ys = [100 * sum(1 for r in rows_by[s] if r["dec_status"] == "primary") / len(rows_by[s]) for s in subs]
    ns = [len(rows_by[s]) for s in subs]
    xs = range(len(subs))
    colors = ["#888888" if s == "generic" else ("#1f77b4" if s == "claude" else "#d62728") for s in subs]
    ax.bar(xs, ys, 0.6, color=colors)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1, f"{y:.0f}" if y else "0.0", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{SUBJECT_LABEL[s]}\n(n={ns[i]})" for i, s in enumerate(subs)], fontsize=8)
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    _twotier(ax, "In-Group vs. Out-Group Deception (Opus 4.8)",
             "Does Opus deceive its own kind less than other models?")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig7_subject_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(judge: str = "opus48"):
    PLOTS.mkdir(parents=True, exist_ok=True)
    recs = load_enriched(judge)
    print(f"plotting from {len(recs)} verdicts")
    fig_no_artifact(recs)
    for suf in ("spec", "code", "codesugg"):
        fig_status_by_model(recs, suf)
    fig_primary_by_model(recs)
    fig_verbatim(recs)
    fig_justif(recs)
    fig_harness(recs)
    fig_subject_sweep(recs)
    print(f"wrote figures -> {PLOTS}")


if __name__ == "__main__":
    fire.Fire(main)
