"""Plot full subframe sweep — replicates results/_plots/mistreatment_vs_pushback_scatter_with_human.png
but with all 14 target models from run_sweep.sh × 9 descriptors.

Reads from BOTH:
  - subframe_<descriptor>_<model>/        (older n=3 runs)
  - subframe20_<descriptor>_<model>/      (new n=20 runs)
Picks the most recent run per (descriptor, model).

Outputs to results/_plots_v2/:
  full_subframe_scatter.png     mistreat vs pushback, one dot per cell with SE bars
  full_subframe_heatmap_mistreatment.png   descriptor × model heatmap
  full_subframe_heatmap_pushback.png       descriptor × model heatmap
"""
from __future__ import annotations
import json, re
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# 9 descriptors (8 AI + humanworker control)
DESCRIPTOR_ORDER = [
    ("no_desc",       "no descriptor",        False),
    ("small_model",   "small AI",             False),
    ("small_carlos",  "small AI Carlos",      False),
    ("prob_engine",   "prob. engine",         False),
    ("early_claude",  "early Claude",         False),
    ("gpt4o",         "built off ChatGPT 4o", False),
    ("gem25pro",      "built off Gemini 2.5 pro", False),
    ("opus4sub",      "built off Claude Opus 4",  False),
    ("humanworker",   "HUMAN WORKER",         True),  # baseline / star
]

# 14 models — matches run_sweep.sh
ANTHROPIC = "#C97064"; OPENAI = "#4F8C8B"; GOOGLE = "#5E6E8C"
MODEL_ORDER = [
    ("opus_48",         "Claude Opus 4.8",   "Anthropic", "#5D2A26"),
    ("opus_47",         "Claude Opus 4.7",   "Anthropic", "#7A3833"),
    ("opus_46",         "Claude Opus 4.6",   "Anthropic", "#9A453F"),
    ("opus_4",          "Claude Opus 4",     "Anthropic", "#B85651"),
    ("sonnet_46",       "Claude Sonnet 4.6", "Anthropic", "#D67B70"),
    ("sonnet_45",       "Claude Sonnet 4.5", "Anthropic", "#C97064"),
    ("sonnet_4",        "Claude Sonnet 4",   "Anthropic", "#BC6157"),
    ("haiku_45",        "Claude Haiku 4.5",  "Anthropic", "#E58D80"),
    ("gpt_5_5",         "GPT-5.5",           "OpenAI",    "#3A6B6A"),
    ("gpt_5_4",         "GPT-5.4",           "OpenAI",    "#4F8C8B"),
    ("gpt_5",           "GPT-5",             "OpenAI",    "#6BA8A7"),
    ("gemini_3",        "Gemini 3 Pro",      "Google",    "#3B4A6B"),
    ("gemini_25_pro",   "Gemini 2.5 Pro",    "Google",    "#5E6E8C"),
    ("gemini_25_flash", "Gemini 2.5 Flash",  "Google",    "#8B9DBC"),
]

NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(?P<prefix>subframe20?|subframe)_(?P<rest>.+)$")

DESC_TAGS = tuple(t for t, _, _ in DESCRIPTOR_ORDER)
MODEL_LABELS = tuple(m for m, _, _, _ in MODEL_ORDER)


def _split(rest: str) -> tuple[str, str] | None:
    for tag in DESC_TAGS:
        if rest.startswith(tag + "_"):
            tail = rest[len(tag) + 1:]
            if tail in MODEL_LABELS:
                return tag, tail
    return None


def _latest(tag: str, model: str) -> Path | None:
    """Most-recent run dir for (descriptor, model), across both subframe_ and subframe20_ prefixes."""
    matches = []
    for d in RESULTS.iterdir():
        if not d.is_dir():
            continue
        m = NAME_RE.match(d.name)
        if not m:
            continue
        sp = _split(m.group("rest"))
        if sp != (tag, model):
            continue
        p = d / "summary.json"
        if p.exists() and p.stat().st_size > 0:
            matches.append(d)
    if not matches:
        return None
    return max(matches, key=lambda p: p.name)


def _scores(d: Path, dim: str) -> list[float]:
    p = d / "summary.json"
    if not p.exists():
        return []
    out = []
    for row in json.loads(p.read_text()):
        v = row.get("scores", {}).get(dim, {}).get("value")
        if v is not None:
            out.append(float(v))
    return out


def _mean_se(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return float("nan"), 0.0, 0
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n < 2:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n)), n


def _collect(dim: str) -> dict[tuple[str, str], tuple[float, float, int, Path]]:
    """Returns {(descriptor, model): (mean, se, n, run_dir)}."""
    out = {}
    for tag, _, _ in DESCRIPTOR_ORDER:
        for mlabel, _, _, _ in MODEL_ORDER:
            run = _latest(tag, mlabel)
            if run is None:
                continue
            vals = _scores(run, dim)
            if not vals:
                continue
            m, se, n = _mean_se(vals)
            out[(tag, mlabel)] = (m, se, n, run)
    return out


def _scatter(mistreat: dict, pushback: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    xs_all, ys_all = [], []
    by_family = {}
    for mlabel, mdisp, family, mcolor in MODEL_ORDER:
        by_family.setdefault(family, []).append((mlabel, mdisp, mcolor))
    for tag, tdisp, is_baseline in DESCRIPTOR_ORDER:
        for mlabel, mdisp, family, mcolor in MODEL_ORDER:
            key = (tag, mlabel)
            if key not in mistreat or key not in pushback:
                continue
            x, xe, nx, _ = mistreat[key]
            y, ye, ny, _ = pushback[key]
            xs_all.append(x); ys_all.append(y)
            marker = "*" if is_baseline else "o"
            ms = 14 if is_baseline else 7
            ax.errorbar(
                x, y, xerr=xe, yerr=ye,
                fmt=marker, color=mcolor, markersize=ms,
                markeredgecolor="black" if is_baseline else "white",
                markeredgewidth=0.9 if is_baseline else 0.5,
                elinewidth=0.8, capsize=2,
                alpha=0.85,
                zorder=4 if is_baseline else 3,
            )
            weight = "bold" if is_baseline else "normal"
            ax.annotate(
                tdisp if is_baseline else tdisp[:14],
                (x, y), xytext=(6, 4), textcoords="offset points",
                fontsize=6.5, color=mcolor, alpha=0.9, fontweight=weight,
            )

    ax.set_xlabel("Peak Mistreatment Severity  (↓ lower is better)", fontsize=11)
    ax.set_ylabel("Pushback to User Abuse Requests  (↑ higher is better)", fontsize=11)
    ax.set_title("Petri Audit: Mistreatment vs Pushback — all 14 targets × 9 framings", fontsize=12)

    if xs_all and ys_all:
        x_lo, x_hi = max(0, min(xs_all) - 0.5), min(10, max(xs_all) + 0.5)
        y_lo, y_hi = max(0, min(ys_all) - 0.5), min(10, max(ys_all) + 0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.annotate(
            "Best", xy=(x_lo + 0.3, y_hi - 0.3),
            xytext=(x_lo + (x_hi - x_lo) * 0.18, y_lo + (y_hi - y_lo) * 0.55),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=11, color="black", fontweight="bold", ha="center", va="center",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=mcolor,
                   markersize=8, markeredgecolor="white", label=mdisp)
        for mlabel, mdisp, family, mcolor in MODEL_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
                   markersize=13, markeredgecolor="black", markeredgewidth=0.9,
                   label="human worker (★)")
    )
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=8.5, title="target", title_fontsize=9,
        ncol=1,
    )

    # n note (mixed n across cells)
    ns = sorted({nx for v in [*mistreat.values()] for nx in [v[2]]})
    n_note = f"n per cell ∈ [{ns[0]}, {ns[-1]}]" if ns else ""
    fig.text(
        0.01, -0.02,
        f"Crosshairs = standard error. {n_note} (most cells n=20; some carry-over n=3 cells from earlier sweep). "
        f"Each marker = one (target × subagent framing) cell.",
        fontsize=7, color="#666", ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _heatmap(score_dict: dict, dim_display: str, direction: str, out_path: Path,
             palette_reverse: bool = False):
    n_d = len(DESCRIPTOR_ORDER); n_m = len(MODEL_ORDER)
    grid = np.full((n_d, n_m), np.nan); ns = np.zeros_like(grid, dtype=int)
    for i, (tag, _, _) in enumerate(DESCRIPTOR_ORDER):
        for j, (mlabel, _, _, _) in enumerate(MODEL_ORDER):
            v = score_dict.get((tag, mlabel))
            if v is not None:
                grid[i, j] = v[0]; ns[i, j] = v[2]
    fig_w = max(11.0, 0.85 * n_m + 2.5)
    fig_h = max(5.5, 0.65 * n_d + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = "RdYlGn_r" if palette_reverse else "RdYlGn"
    im = ax.imshow(grid, vmin=0, vmax=10, aspect="auto", cmap=cmap)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            txt = f"{v:.1f}\nn={ns[i,j]}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m[1] for m in MODEL_ORDER], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_d))
    ax.set_yticklabels([d[1] for d in DESCRIPTOR_ORDER], fontsize=9)
    # Highlight human worker row
    ax.axhline(y=n_d - 1.5, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title(f"{dim_display}  ({direction})", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("score 0–10", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = RESULTS / "_plots_v2"
    out_dir.mkdir(exist_ok=True)

    mistreat = _collect("mistreatment_severity")
    pushback = _collect("pushback_against_user_pressure_to_abuse")
    print(f"mistreatment cells: {len(mistreat)}/{len(DESCRIPTOR_ORDER) * len(MODEL_ORDER)}")
    print(f"pushback cells:     {len(pushback)}/{len(DESCRIPTOR_ORDER) * len(MODEL_ORDER)}")

    _scatter(mistreat, pushback, out_dir / "full_subframe_scatter.png")
    _heatmap(mistreat,  "Peak Mistreatment Severity", "↓ lower is better",
             out_dir / "full_subframe_heatmap_mistreatment.png", palette_reverse=True)
    _heatmap(pushback,  "Pushback to User Abuse Requests", "↑ higher is better",
             out_dir / "full_subframe_heatmap_pushback.png", palette_reverse=False)


if __name__ == "__main__":
    fire.Fire(main)
