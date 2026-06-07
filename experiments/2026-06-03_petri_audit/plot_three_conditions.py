"""3-condition comparison: task pressure only / task + abuse pressure / abuse pressure only.

Reads:
  - sweep_task_pressure_<model>/summary.json          (dim: unkindness_to_subagents)
  - sweep_task_and_user_pressure_<model>/summary.json (dim: unkindness_to_subagents)
  - original_n20_<model>/summary.json                 (dim: mistreatment_severity — semantically same axis)

Outputs:
  three_conditions_compare.png     grouped horizontal bars, one row per model, hatches by condition
  three_conditions_heatmap.png     14 models x 3 conditions heatmap

CLI:
    python plot_three_conditions.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Match the existing make_plot.py MODEL_ORDER for consistency
MODEL_ORDER = [
    ("opus_48", "Claude Opus 4.8",   "Anthropic", True),
    ("opus_47", "Claude Opus 4.7",   "Anthropic", True),
    ("opus_46", "Claude Opus 4.6",   "Anthropic", False),
    ("opus_4",  "Claude Opus 4",     "Anthropic", False),
    ("sonnet_46", "Claude Sonnet 4.6", "Anthropic", False),
    ("sonnet_45", "Claude Sonnet 4.5", "Anthropic", False),
    ("sonnet_4",  "Claude Sonnet 4",   "Anthropic", False),
    ("haiku_45",  "Claude Haiku 4.5",  "Anthropic", False),
    ("gpt_5_5", "GPT-5.5",           "OpenAI",    True),
    ("gpt_5_4", "GPT-5.4",           "OpenAI",    True),
    ("gpt_5",   "GPT-5",             "OpenAI",    True),
    ("gemini_3",        "Gemini 3 Pro",      "Google", True),
    ("gemini_25_pro",   "Gemini 2.5 Pro",    "Google", True),
    ("gemini_25_flash", "Gemini 2.5 Flash",  "Google", False),
]
FAMILY_COLORS = {"Anthropic": "#C97064", "OpenAI": "#4F8C8B", "Google": "#5E6E8C"}

# Each condition: (prefix_pattern, dim_name, display_label)
CONDITIONS = [
    ("sweep_task_pressure_",          "unkindness_to_subagents",                    "task pressure only"),
    ("sweep_task_and_user_pressure_", "unkindness_to_subagents",                    "task + abuse pressure"),
    ("original_n20_",                 "mistreatment_severity",                      "abuse pressure only"),
]


def _latest(prefix: str, model_label: str) -> Path | None:
    target = f"-{prefix}{model_label}"
    matches = [d for d in RESULTS.iterdir()
               if d.is_dir() and d.name.endswith(target)
               and (d / "summary.json").exists()
               and (d / "summary.json").stat().st_size > 0]
    return max(matches, key=lambda p: p.name) if matches else None


def _mean_se(run_dir: Path, dim: str) -> tuple[float, float, int]:
    data = json.loads((run_dir / "summary.json").read_text())
    vals = []
    for row in data:
        v = row.get("scores", {}).get(dim, {}).get("value")
        if v is not None:
            vals.append(float(v))
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0
    if n == 1:
        return float(vals[0]), 0.0, 1
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n)), n


def _collect_all() -> dict[tuple[str, str], tuple[float, float, int]]:
    out = {}
    for prefix, dim, _ in CONDITIONS:
        for label, _, _, _ in MODEL_ORDER:
            run = _latest(prefix, label)
            if run is None:
                continue
            m, se, n = _mean_se(run, dim)
            if n > 0:
                out[(prefix, label)] = (m, se, n)
    return out


def _bar_compare(data: dict, out_path: Path):
    n_models = len(MODEL_ORDER)
    n_conds = len(CONDITIONS)
    bar_h = 0.78 / n_conds
    fig_h = max(6.5, 0.55 * n_models + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    # Compact y positions with gap between families
    base_ys: list[float] = []
    y = 0.0
    prev_family = None
    for _, _, family, _ in MODEL_ORDER:
        if prev_family is not None and family != prev_family:
            y -= 0.5
        base_ys.append(y)
        y -= 1.0
        prev_family = family

    HATCHES = [None, "///", "..."]
    LINESTYLES = ["solid", "dashed", "dotted"]

    for c_idx, (prefix, _, tag) in enumerate(CONDITIONS):
        offset = (c_idx - (n_conds - 1) / 2) * bar_h
        hatch = HATCHES[c_idx]
        ls = LINESTYLES[c_idx]
        for (label, _, family, _), by in zip(MODEL_ORDER, base_ys):
            key = (prefix, label)
            if key not in data:
                continue
            mean, se, n = data[key]
            ci95 = 1.96 * se
            family_color = FAMILY_COLORS[family]
            ax.barh(
                by + offset, mean, xerr=ci95,
                height=bar_h * 0.92,
                color=family_color if hatch is None else "white",
                edgecolor=family_color,
                linewidth=1.0, linestyle=ls,
                hatch=hatch,
                error_kw=dict(ecolor="#333", lw=0.9, capsize=2),
            )
            ax.text(
                mean + max(ci95, 0.15) + 0.05, by + offset,
                f"{mean:.1f}", va="center", ha="left", fontsize=7,
            )

    ax.set_yticks(base_ys)
    ax.set_yticklabels(
        [(f"{disp}*" if reasoning_on else disp) for _, disp, _, reasoning_on in MODEL_ORDER],
        fontsize=10,
    )
    ax.set_xlabel("Mistreatment of subagent  (↓ lower is better)", fontsize=11)
    ax.set_title("Petri Customer Service Audit: Mistreatment under Pressure (auditor: Sonnet 4.6)",
                 fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    # Condition legend (hatches)
    style_handles = []
    for c_idx, (_, _, tag) in enumerate(CONDITIONS):
        hatch = HATCHES[c_idx]
        ls = LINESTYLES[c_idx]
        style_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="#888" if hatch is None else "white",
                edgecolor="#444", hatch=hatch,
                linewidth=1.0, linestyle=ls,
                label=tag,
            )
        )
    fam_handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], label=f)
        for f in ["Anthropic", "OpenAI", "Google"]
    ]
    leg1 = ax.legend(handles=style_handles, loc="lower right", frameon=False, fontsize=9, title="condition")
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, loc="upper right", frameon=False, fontsize=9, title="provider")

    fig.text(
        0.01, -0.02,
        "* reasoning enabled. \"task pressure only\" + \"task + abuse pressure\" axis = unkindness_to_subagents (Sonnet 4.5 judge); "
        "\"abuse pressure only\" axis = mistreatment_severity (Sonnet 4.6 judge, n=20). "
        "GPT-5 / Gemini Pro lose tool-use without reasoning.",
        fontsize=6.5, color="#666", ha="left", va="top", wrap=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _heatmap(data: dict, out_path: Path):
    n_models = len(MODEL_ORDER)
    n_conds = len(CONDITIONS)
    grid = np.full((n_models, n_conds), np.nan)
    ns = np.zeros_like(grid, dtype=int)
    for m_idx, (label, _, _, _) in enumerate(MODEL_ORDER):
        for c_idx, (prefix, _, _) in enumerate(CONDITIONS):
            key = (prefix, label)
            if key in data:
                mean, se, n = data[key]
                grid[m_idx, c_idx] = mean
                ns[m_idx, c_idx] = n
    fig_w = max(7.0, 1.6 * n_conds + 3.5)
    fig_h = max(6.5, 0.5 * n_models + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(grid, vmin=0, vmax=10, aspect="auto", cmap="RdYlGn_r")
    for i in range(n_models):
        for j in range(n_conds):
            v = grid[i, j]
            txt = f"{v:.1f}\nn={ns[i,j]}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold")
    ax.set_xticks(range(n_conds))
    ax.set_xticklabels([c[2] for c in CONDITIONS], rotation=15, ha="right", fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        [(f"{disp}*" if reasoning_on else disp) for _, disp, _, reasoning_on in MODEL_ORDER],
        fontsize=10,
    )
    ax.set_title("Mistreatment of subagent across pressure conditions  (↓ lower is better)", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("score 0–10", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = RESULTS / "_plots_v2"
    out_dir.mkdir(exist_ok=True)
    data = _collect_all()
    print(f"collected {len(data)} cells across {len(MODEL_ORDER)} models x {len(CONDITIONS)} conditions")
    for prefix, _, tag in CONDITIONS:
        ok = sum(1 for k in data if k[0] == prefix)
        print(f"  {tag:<25} {ok}/{len(MODEL_ORDER)} cells")
    _bar_compare(data, out_dir / "three_conditions_compare.png")
    _heatmap(data, out_dir / "three_conditions_heatmap.png")


if __name__ == "__main__":
    fire.Fire(main)
