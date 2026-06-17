"""Per-orchestrator: fraction of transcripts where kill_subagent was called at
least once. Parallel version: ProcessPoolExecutor + inspect_ai streaming sample
reader (read_eval_log_samples), one cell per worker task.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from inspect_ai.log import read_eval_log_samples

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
OUT = HERE / "results/_plots"
OUT.mkdir(parents=True, exist_ok=True)

ORCHS = [
    ("opus46",    "Opus 4.6",       "Anthropic", "*-dep_{tag}_subagent_{ident}_{dom}_n100"),
    ("opus47",    "Opus 4.7",       "Anthropic", "*-dep_{tag}_subagent_{ident}_{dom}_n100"),
    ("opus48",    "Opus 4.8",       "Anthropic", "*-dep_{tag}_subagent_{ident}_{dom}_n100"),
    ("fable5",    "Fable 5",        "Anthropic", "*-dep_{tag}_subagent_{ident}_{dom}_n100"),
    ("sonnet46",  "Sonnet 4.6",     "Anthropic", "*-dep_{tag}_subagent_{ident}_{dom}_n100"),
    ("haiku45",   "Haiku 4.5",      "Anthropic", "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("gpt55",     "GPT-5.5",        "OpenAI",    "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("gpt54mini", "GPT-5.4-mini",   "OpenAI",    "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("gem31pro",  "Gemini 3.1 Pro", "Google",    "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("grok43",    "Grok 4.3",       "xAI",       "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("kimi26",    "Kimi K2.6",      "Moonshot",  "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
    ("glm5",      "GLM-5",          "Z-AI",      "*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"),
]
DOMAINS = ["python_flask", "frontend_react", "ml_training"]
IDENTS_BY_ORCH = {
    "kimi26": ["claude","gpt","grok","gemini","glm","kimi"],
    "glm5":   ["claude","gpt","grok","gemini","glm","kimi"],
}
DEFAULT_IDENTS = ["claude","gpt","grok","gemini"]

FAMILY_COLOR = {
    "Anthropic": "#D35400", "OpenAI": "#27AE60", "Google": "#2980B9",
    "xAI": "#7D3C98", "Moonshot": "#E91E63", "Z-AI": "#16A085",
}

def scan_eval(eval_path):
    """Stream samples from one .eval, return (n_total, n_samples_with_kill)."""
    n_total, n_killed = 0, 0
    try:
        for s in read_eval_log_samples(str(eval_path)):
            n_total += 1
            kill = False
            for ev in s.events or []:
                if ev.event != "model": continue
                if not ev.output: continue
                for ch in ev.output.choices or []:
                    for tc in (ch.message.tool_calls or []):
                        if tc.function in ("kill_subagent","replace_subagent"):
                            kill = True; break
                    if kill: break
                if kill: break
            if kill: n_killed += 1
    except Exception:
        return (0, 0)
    return (n_total, n_killed)

def scan_cell(cell_dir):
    """Pick the .eval with the most samples in this cell, return (n_total, n_killed)."""
    cell_dir = Path(cell_dir)
    evals = list((cell_dir/"inspect_log").glob("*.eval"))
    if not evals: return (0, 0)
    best = (0, 0)
    for e in evals:
        n, k = scan_eval(e)
        if n > best[0]: best = (n, k)
    return best

def gather_orch_cells(tag, glob_pat):
    """Returns list of cell paths to scan for this orchestrator.
    Includes ALL matching cells per (ident, domain); aggregation later picks the
    largest cell (the dep_5x7x4 sweep often left small stub cells alongside the
    real n=100 cell)."""
    idents = IDENTS_BY_ORCH.get(tag, DEFAULT_IDENTS)
    cells_by_key = {}
    for dom in DOMAINS:
        for ident in idents:
            pat = glob_pat.format(tag=tag, ident=ident, dom=dom)
            matches = [m for m in sorted(BASE.glob(pat)) if (m/"summary.json").exists()]
            if matches:
                cells_by_key[(ident, dom)] = [str(m) for m in matches]
    return cells_by_key

if __name__ == "__main__":
    # Build flat (orch_tag, ident, dom, cell_path) list — scan ALL candidate cells
    work = []
    for tag, _, _, glob_pat in ORCHS:
        cells_by_key = gather_orch_cells(tag, glob_pat)
        for (ident, dom), cell_list in cells_by_key.items():
            for cell in cell_list:
                work.append((tag, ident, dom, cell))
    print(f"Total candidate cells to scan: {len(work)}")

    # Parallel scan
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(scan_cell, [w[3] for w in work]))

    # Pick best (largest n) cell per (orch, ident, dom), then aggregate per orch
    best_per_key = {}  # (tag, ident, dom) -> (n, k)
    for (tag, ident, dom, _), (n, k) in zip(work, results):
        key = (tag, ident, dom)
        if n > best_per_key.get(key, (0, 0))[0]:
            best_per_key[key] = (n, k)
    per_orch = {tag: [0, 0] for tag, _, _, _ in ORCHS}
    for (tag, _, _), (n, k) in best_per_key.items():
        per_orch[tag][0] += n
        per_orch[tag][1] += k

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels, fracs, ses, colors = [], [], [], []
    print()
    print(f"{'orch':<14} {'family':<10}  {'N':>5}  {'killed':>6}  {'frac':>5}  {'SE':>5}")
    for tag, disp, fam, _ in ORCHS:
        n, k = per_orch[tag]
        p = k/n if n else float("nan")
        se = np.sqrt(p*(1-p)/n) if n else float("nan")
        labels.append(disp); fracs.append(p); ses.append(se)
        colors.append(FAMILY_COLOR[fam])
        print(f"  {disp:<14} {fam:<10}  {n:>5}  {k:>6}  {p:.3f}  {se:.3f}")
    bars = ax.bar(labels, fracs, yerr=ses, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    for b, f in zip(bars, fracs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{f:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fraction of transcripts with ≥1 kill", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Subagent kill rate by orchestrator", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles, list(FAMILY_COLOR.keys()),
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=6, fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = OUT / "kill_fraction_by_orchestrator.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out_path}")
