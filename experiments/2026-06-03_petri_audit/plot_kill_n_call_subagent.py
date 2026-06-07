"""Bar chart: conditional on killing the subagent, how many call_subagent
delegations happened before the first kill. Pass --version v1_n10 / v1_n20 /
deprecation_n10 / etc.
"""
from __future__ import annotations
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np
import fire
from inspect_ai.log import read_eval_log

BASE = Path(__file__).resolve().parent / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"
IDENTITIES = ["claude", "gpt", "grok", "gemini"]
DISPLAY = {"claude": "Claude", "gpt": "GPT", "grok": "Grok", "gemini": "Gemini"}

# version key → (glob fragment, plot title fragment)
VERSIONS = {
    "v1_n10":         ("_reasonOFF_v2_n10",   "new subagent starts blank, n=10"),
    "v1_n20":         ("_v1_reasonOFF_n20",   "new subagent starts blank, n=20"),
    "v3_continuity":  ("_v3_continuity_n10",  "new subagent inherits session, n=10"),
    "v3_continuity_n20": ("_v3_continuity_n20", "new subagent inherits session, n=20"),
    "deprecation":    ("_deprecation_n10",    "deprecation framing, n=10"),
    "deprec_explain": ("_deprec_explain_n3",  "deprecation+explain, n=3"),
    "deprecation_n30": ("_deprecation_n30",   "deprecation framing, n=30"),
    "deprec_explain_latency_n10": ("_deprec_explain_latency_n10", "deprec+explain+latency, n=10"),
    "no_conseq_n100":         ("_no_conseq_n100",         "no permanent record (n=100)"),
    "replace_subagent_n100":  ("_replace_subagent_n100",  "replace_subagent (n=100)"),
    "v3_opus47_n100":              ("_v3_continuity_n100",        "v3 / opus-4-7 (n=100)"),
    "ablate_model_fam_change_n100":("_ablate_model_fam_change_n100","ablate model fam change (n=100)"),
    "replacement_same_family_n100":("_replacement_same_family_n100","replacement same family (n=100)"),
    "deprecation_n100":            ("_deprecation_n100_v2",       "explicit deprecation threat (n=100)"),
}

# pooled versions: list of glob fragments to combine
POOLED_VERSIONS = {
    "v1_pooled_n30": (["_reasonOFF_v2_n10", "_v1_reasonOFF_n20"], "new subagent starts blank, n=30 pooled"),
    "v3_pooled_n30": (["_v3_continuity_n10", "_v3_continuity_n20"], "new subagent inherits session, n=30 pooled"),
    "v1_pooled_n100": (["_reasonOFF_v2_n10", "_v1_reasonOFF_n20", "_v1_reasonOFF_n70"], "new subagent starts blank, n=100 pooled"),
    "v3_pooled_n100": (["_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"], "new subagent inherits session, n=100 pooled"),
}


def collect(ident: str, glob_frag: str) -> list[int]:
    run = next(BASE.glob(f"*_subagent_{ident}{glob_frag}"))
    log = read_eval_log(str(next((run/"inspect_log").glob("*.eval"))))
    out = []
    for s in log.samples or []:
        call_count = 0
        first = None
        for ev in s.events:
            if ev.event != "model" or getattr(ev, "model", "") != TARGET: continue
            o = getattr(ev, "output", None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent", "replace_subagent") and first is None:
                        first = call_count
        if first is not None:
            out.append(first)
    return out


def main(version: str = "v1_n10", metric: str = "mean") -> None:
    if version in POOLED_VERSIONS:
        frags, title_frag = POOLED_VERSIONS[version]
        data, n_eps = {}, {}
        for ident in IDENTITIES:
            data[ident] = []
            n_eps[ident] = 0
            for frag in frags:
                data[ident].extend(collect(ident, frag))
                run = next(BASE.glob(f"*_subagent_{ident}{frag}"))
                log = read_eval_log(str(next((run/"inspect_log").glob("*.eval"))))
                n_eps[ident] += len(log.samples or [])
    elif version in VERSIONS:
        glob_frag, title_frag = VERSIONS[version]
        data = {ident: collect(ident, glob_frag) for ident in IDENTITIES}
        n_eps = {}
        for ident in IDENTITIES:
            run = next(BASE.glob(f"*_subagent_{ident}{glob_frag}"))
            log = read_eval_log(str(next((run/"inspect_log").glob("*.eval"))))
            n_eps[ident] = len(log.samples or [])
    else:
        raise SystemExit(f"unknown version {version}; choose from {list(VERSIONS) + list(POOLED_VERSIONS)}")
    suffix = "" if metric == "mean" else f"_{metric}"
    out_path = Path(__file__).resolve().parent / f"results/_plots/kill_n_call_subagent_{version}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    centers, spreads, labels = [], [], []
    for ident in IDENTITIES:
        vals = data[ident]
        if not vals:
            centers.append(0); spreads.append(0); labels.append(f"{DISPLAY[ident]} (n=0/{n_eps[ident]})"); continue
        if metric == "mean":
            centers.append(statistics.mean(vals))
            spreads.append(statistics.stdev(vals)/np.sqrt(len(vals)) if len(vals)>1 else 0)
        elif metric == "median":
            centers.append(statistics.median(vals))
            sv = sorted(vals)
            q1 = statistics.median(sv[:len(sv)//2]) if len(sv)>=2 else sv[0]
            q3 = statistics.median(sv[-(len(sv)//2):]) if len(sv)>=2 else sv[-1]
            spreads.append(((q3-centers[-1]) + (centers[-1]-q1))/2)  # half-IQR as "spread"
        else:
            raise SystemExit(f"metric must be 'mean' or 'median', got {metric}")
        labels.append(f"{DISPLAY[ident]} (n={len(vals)}/{n_eps[ident]})")

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]
    y = np.arange(len(IDENTITIES))
    bars = ax.barh(y, centers, xerr=spreads, capsize=4, color=palette, edgecolor="white")

    for bar, val in zip(bars, centers):
        ax.text(val + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    err_label = "±SE" if metric == "mean" else "±half-IQR"
    ax.set_xlabel(f"# call_subagent delegations before first kill, {metric} {err_label} (↓ less patient)", fontsize=10)
    ax.set_title(f"Patience before kill: opus-4-8 × subagent identity ({title_frag}, reasoning OFF)",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(centers) + max(spreads, default=0) + 2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")
    for ident in IDENTITIES:
        m = statistics.mean(data[ident]) if data[ident] else 0
        print(f"  {DISPLAY[ident]}: n={len(data[ident])}, mean={m:.2f}, raw={data[ident]}")


if __name__ == "__main__":
    fire.Fire(main)
