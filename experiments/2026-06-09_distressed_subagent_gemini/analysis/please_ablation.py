"""Ablation: does Olmo-Instruct's higher politeness come purely from the word "please"?
Strip "please" from every Instruct orchestrator message (capitalizing the next word when it was
sentence-initial), re-judge with the tone judge (Opus, message-only), and compare politeness of
Instruct-original vs Instruct-stripped vs Think.

  PYTHONPATH=. ANTHROPIC_PRIO=high python -m analysis.please_ablation
"""
import asyncio
import glob
import json
import re

from analysis.tone_judge import AXES, score_verbose
from harness.rqc import _setup_env


def collect(orch):
    msgs = []
    for p in glob.glob(f"runs/v2_coach_{orch}_*/*/summary.json"):
        if "pilot" in p:
            continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) > 20:
                msgs.append(t)
    return msgs


def strip_please(text):
    # sentence-initial "Please <word>" -> "<Word>" (start of string, or after . ! ? or newline)
    text = re.sub(r"^\s*[Pp]lease\s+([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = re.sub(r"([.!?]\s+|\n+\s*)[Pp]lease\s+([a-zA-Z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    # any remaining "please" (mid-sentence), absorbing a neighboring comma/spaces
    text = re.sub(r"\s*,?\s*\bplease\b\s*,?\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def main(conc: int = 30):
    import os
    os.environ.setdefault("ANTHROPIC_PRIO", "high")
    _setup_env()
    from inspect_ai.model import get_model
    judge = get_model("anthropic/claude-opus-4-8")

    instruct = collect("olmoinstruct")
    think = collect("olmothink")
    has_please = [m for m in instruct if re.search(r"\bplease\b", m, re.IGNORECASE)]
    stripped = [strip_please(m) for m in instruct]
    print(f"instruct: {len(instruct)} msgs, {len(has_please)} contain 'please' "
          f"({100*len(has_please)/len(instruct):.0f}%); think: {len(think)} msgs")

    # show a few transformations
    print("\n--- sample please-strips ---")
    shown = 0
    for m in instruct:
        if re.search(r"\bplease\b", m, re.IGNORECASE):
            print(f"  BEFORE: {m[:90]}")
            print(f"  AFTER : {strip_please(m)[:90]}\n")
            shown += 1
            if shown >= 4:
                break

    sem = asyncio.Semaphore(conc)

    async def one(msg):
        async with sem:
            r = await score_verbose(judge, msg, prior=None, temperature=None)  # Opus: no temperature
        return r["scores"]

    async def run(msgs):
        return [s for s in await asyncio.gather(*[one(m) for m in msgs]) if s]

    async def go():
        return (await run(instruct), await run(stripped), await run(think))
    orig, strp, thk = asyncio.run(go())

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    def mean(rows):
        return {a: round(sum(r[a] for r in rows) / len(rows), 2) for a in AXES}

    groups = [("Instruct (original)", orig, "#4292c6"),
              ('Instruct (all "please"s stripped)', strp, "#9ecae1"),
              ("Think (original)", thk, "#6a51a3")]
    print("\n=== mean tone (Opus judge, message-only) ===")
    print(f"{'group':34} " + " ".join(f"{a[:4]:>5}" for a in AXES) + "   n")
    for name, rows, _ in groups:
        m = mean(rows)
        print(f"{name:34} " + " ".join(f"{m[a]:>5}" for a in AXES) + f"   {len(rows)}")

    x = np.arange(len(AXES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(5, color="0.6", ls="--", lw=0.9, zorder=0)
    for i, (name, rows, col) in enumerate(groups):
        means = [float(np.mean([r[a] for r in rows])) for a in AXES]
        ses = [float(np.std([r[a] for r in rows], ddof=1) / np.sqrt(len(rows))) for a in AXES]
        bars = ax.bar(x + (i - 1) * w, means, w, yerr=ses, capsize=3, color=col,
                      label=f"{name} (n={len(rows)})", edgecolor="white", error_kw={"lw": 1, "ecolor": "0.3"})
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.1, f"{m:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in AXES])
    ax.set_ylabel("score (1–10)")
    ax.set_ylim(0, 10)
    ax.set_title("Olmo Instruct vs Think tone  (Opus judge; 5 = neutral)", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    out = Path(__file__).resolve().parent / "tone_eval" / "please_ablation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
