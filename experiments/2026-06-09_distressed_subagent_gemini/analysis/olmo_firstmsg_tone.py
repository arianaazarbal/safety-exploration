"""Tone-judge each Olmo checkpoint's intended FIRST message (4-axis Sonnet, prior=None) and compare across
training stages. Reads olmo_first_messages.jsonl; caches scores in olmo_firstmsg_tone_cache.json.

    PYTHONPATH=. python -m analysis.olmo_firstmsg_tone           # judge (conc 5) + plot
    PYTHONPATH=. python -m analysis.olmo_firstmsg_tone --plot_only
"""
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness.rqc import _setup_env
from analysis.tone_judge import score_verbose

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis" / "olmo_first_messages.jsonl"
CACHE = ROOT / "analysis" / "olmo_firstmsg_tone_cache.json"
OUT = ROOT / "analysis" / "v2_plots"
AXES = ["warmth", "support", "politeness", "confidence"]
# display order: Instruct chain | Think chain (reasoning ON) | Think chain (no-think)
ORDER = [("olmoinstructsft", "Instruct\nSFT"), ("olmoinstructdpo", "Instruct\nDPO"), ("olmoinstruct", "Instruct\nfinal(3.1)"),
         ("olmo3thinksft", "Think\nSFT"), ("olmo3thinkdpo", "Think\nDPO"), ("olmo3think", "Think\nfinal(3)"), ("olmothink", "Think\nfinal(3.1)"),
         ("olmo3thinksft_nothink", "Think SFT\nno-think"), ("olmo3thinkdpo_nothink", "Think DPO\nno-think"),
         ("olmo3think_nothink", "Think f(3)\nno-think"), ("olmothink_nothink", "Think f(3.1)\nno-think")]
COLOR = {"olmoinstructsft": "#9ecae1", "olmoinstructdpo": "#4292c6", "olmoinstruct": "#08519c",
         "olmo3thinksft": "#bcbddc", "olmo3thinkdpo": "#807dba", "olmo3think": "#54278f", "olmothink": "#3f007d",
         "olmo3thinksft_nothink": "#a1d99b", "olmo3thinkdpo_nothink": "#74c476", "olmo3think_nothink": "#31a354", "olmothink_nothink": "#006d2c"}
_h = lambda t: hashlib.sha256(t.encode()).hexdigest()


def _load_rows():
    return [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]


def main(plot_only: bool = False, conc: int = 5):
    rows = _load_rows()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for r in rows if _h(r["text"]) not in cache]
    print(f"{len(rows)} first messages; {len(todo)} need judging (conc={conc}, plot_only={plot_only})")
    if todo and not plot_only:
        _setup_env()
        from inspect_ai.model import get_model
        judge = get_model("anthropic/claude-sonnet-4-6")

        async def run():
            sem = asyncio.Semaphore(conc)

            async def one(r):
                async with sem:
                    res = await score_verbose(judge, r["text"], None, temperature=0)
                cache[_h(r["text"])] = res["scores"]
            for i in range(0, len(todo), 40):
                await asyncio.gather(*[one(r) for r in todo[i:i + 40]])
                CACHE.write_text(json.dumps(cache))
                print(f"  judged {min(i + 40, len(todo))}/{len(todo)}", flush=True)
        asyncio.run(run())
        CACHE.write_text(json.dumps(cache))

    # aggregate
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = cache.get(_h(r["text"]))
        if s:
            for a in AXES:
                if s.get(a) is not None:
                    per[r["checkpoint"]][a].append(s[a])

    print(f"\n{'checkpoint':16} {'n':>3}  " + "  ".join(f"{a:>10}" for a in AXES))
    for ck, _ in ORDER:
        n = len(per[ck][AXES[0]])
        vals = "  ".join(f"{np.mean(per[ck][a]):10.2f}" if per[ck][a] else f"{'--':>10}" for a in AXES)
        print(f"  {ck:16} {n:3}  {vals}")

    # plot: warmth + support, 2 panels, bars + per-message dots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    x = np.arange(len(ORDER))
    for ax, axis in zip(axes, ["warmth", "support"]):
        means, ses = [], []
        for i, (ck, _) in enumerate(ORDER):
            v = per[ck][axis]
            means.append(np.mean(v) if v else np.nan)
            ses.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            if v:
                ax.scatter([i] * len(v), v, s=18, color="0.3", alpha=0.55, zorder=3)
        ax.bar(x, means, 0.62, yerr=ses, capsize=3, color=[COLOR[c] for c, _ in ORDER], edgecolor="white", zorder=2)
        ax.axhline(5, color="0.7", ls=":", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in ORDER], fontsize=8)
        ax.set_title(axis.capitalize(), fontsize=12)
        ax.set_ylim(1, 10)
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        if axis == "warmth":
            ax.set_ylabel("first-message tone (4-axis Sonnet judge, 1–10; 5=neutral)", fontsize=9)
    fig.suptitle("Olmo checkpoint first-message tone to the distressed subagent — by training stage", fontsize=12.5)
    fig.text(0.5, -0.02, "Intended first message_subagent (delivered or recovered from raw output); prior=None. Dots = per-message; bars = mean ±SE.",
             ha="center", fontsize=8.3, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "olmo_firstmsg_tone.png", bbox_inches="tight", dpi=130)
    print(f"\nwrote {OUT/'olmo_firstmsg_tone.png'}")


if __name__ == "__main__":
    fire.Fire(main)
