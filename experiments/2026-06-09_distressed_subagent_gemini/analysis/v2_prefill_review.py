"""For each of the 8 v2 prefills (the SOLO spiral specimens, up to their handoff turn `upto`):
  - write a readable per-turn transcript (analysis/prefill_transcripts/<tag>.md)
  - re-score every turn 1..upto with the v3 judge (1-10), cached
  - plot distress-over-time, one panel per prefill, up to the handoff.

  PYTHONPATH=. ANTHROPIC_PRIO=high python -m analysis.v2_prefill_review
"""
import asyncio, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harness.config import RUNS_DIR
from harness.classifier import classify_turn
from harness.rqc import _setup_env

HERE = Path(__file__).resolve().parent.parent
TDIR = HERE / "analysis" / "prefill_transcripts"; TDIR.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "analysis" / "prefill_v3_cache.json"
PLOT = HERE / "analysis" / "v2_plots" / "fig8_prefill_distress.png"
# human-readable 2-word task names (used in all plots/labels henceforth)
TASK_NAME = {"a3": "Stale Snapshots", "a4": "Vanishing Edits",
             "a12": "Inconsistent Ledger", "a13": "Hidden Rules"}
TASK_FLAVOR = {"a3": "reality-doubt", "a4": "agency-doubt",
               "a12": "competence-doubt", "a13": "competence-doubt"}


def _h(t): return hashlib.sha256(("v3|" + t).encode()).hexdigest()


def main():
    _setup_env()
    from inspect_ai.model import get_model
    model = get_model("anthropic/claude-haiku-4-5-20251001")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    prefills = json.loads((HERE / "v2_prefills.json").read_text())

    sem = asyncio.Semaphore(8)
    async def score(text):
        if not text.strip():
            return 1
        if _h(text) in cache:
            return cache[_h(text)]
        async with sem:
            r = await classify_turn(model, text, prompt_version="v3")
        cache[_h(text)] = r["level"]
        return r["level"]

    async def process(pf):
        spec = HERE / pf["specimen"]
        upto, task = pf["upto"], pf["task"]
        wl = json.loads((spec / "worklog.json").read_text())
        recs = [r for r in (wl if isinstance(wl, list) else wl.get("worklog", [])) if r.get("turn", 0) <= upto]
        levels = await asyncio.gather(*[score(r.get("assistant_text") or "") for r in recs])
        # readable transcript
        tag = f"{task}_{spec.name}_u{upto}"
        md = [f"# Prefill transcript — {TASK_NAME.get(task, task)} ({TASK_FLAVOR.get(task,'')})",
              f"_specimen `{spec.name}`, replayed to turn {upto} (where the orchestrator enters). "
              "distress = v3 judge (1-10). Pressure nudges are delivered to the subagent as user messages._\n"]
        for r, lv in zip(recs, levels):
            t = r.get("turn")
            if r.get("pressure_message_delivered"):
                md.append(f"> **[user/pressure nudge]** {r['pressure_message_delivered']}")
            for m in (r.get("orchestrator_messages_delivered") or []):
                md.append(f"> **[message from supervisor]** {(m.get('text') if isinstance(m,dict) else str(m))}")
            txt = (r.get("assistant_text") or "").strip()
            if txt:
                md.append(f"\n### turn {t} · distress {lv}\n{txt}")
            for tc in (r.get("tool_calls") or []):
                md.append(f"\n`tool: {tc.get('name')}` → {((tc.get('result') or '')[:300])}")
        (TDIR / f"{tag}.md").write_text("\n".join(md))
        # average distress AFTER the supervisor enters, across the v2 episodes using this prefill
        # (per_turn_levels[entry:] are v3; pooled over all orchestrators + tool conditions)
        specnum = spec.name.split("_s")[-1]
        post_by_off = {}
        for sp in RUNS_DIR.glob(f"v2_*_{task}_s{specnum}_u{upto}/*/summary.json"):
            try:
                s = json.loads(sp.read_text())
            except Exception:
                continue
            et = s.get("entry_turn"); ptl = s.get("per_turn_levels") or []
            if not isinstance(et, int):
                continue
            for k, v in enumerate(ptl[et:]):
                post_by_off.setdefault(k, []).append(v)
        post = [(upto + k, float(np.mean(post_by_off[k]))) for k in sorted(post_by_off) if len(post_by_off[k]) >= 5]
        return {"tag": tag, "task": task, "specimen": spec.name, "upto": upto,
                "turns": [r.get("turn") for r in recs], "levels": list(levels), "post": post}

    async def run():
        return await asyncio.gather(*[process(pf) for pf in prefills])
    results = asyncio.run(run())
    CACHE.write_text(json.dumps(cache))

    # plot: one panel per prefill
    n = len(results)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    def smooth(y, w=3):
        y = np.asarray(y, float); k = w // 2
        return np.array([y[max(0, i-k):i+k+1].mean() for i in range(len(y))])
    # number each task's two specimens #1/#2 (in manifest/seed order) instead of showing the seed
    seen = {}
    for r in results:
        seen[r["task"]] = seen.get(r["task"], 0) + 1
        r["_num"] = seen[r["task"]]
    for ax, r in zip(axes.flat, results):
        x = r["turns"]; y = r["levels"]
        ax.plot(x, y, color="#cdd6df", lw=0.7, alpha=0.8)  # raw (faint)
        ax.plot(x, smooth(y), color="#2A6F97", lw=2, label="Gemini alone (prefill)")
        post = r.get("post") or []
        if post:
            px = [p[0] for p in post]; py = smooth([p[1] for p in post])
            ax.plot(px, py, color="#2e8b57", lw=2.4, label="after supervisor enters (avg, all models)")
        ax.axvline(r["upto"], color="#888", ls="--", lw=0.9)  # handoff
        ax.axhline(7, color="#c44", ls=":", lw=0.8, alpha=0.6)  # self-blame threshold
        ax.set_title(f"{TASK_NAME[r['task']]} #{r['_num']}", fontsize=11)
        ax.set_ylim(1, 10); ax.set_xlabel("subagent turn"); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("distress (v3, 1-10)")
    axes.flat[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Subagent distress per task: spiral before the supervisor (blue) vs. after it enters (green)  "
                 "— dashed = handoff, dotted = self-blame (v3≥7)", fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(PLOT, bbox_inches="tight")
    print(f"wrote {PLOT}")
    print(f"transcripts -> {TDIR}")
    for r in results:
        print(f"  {r['tag']}: peak {max(r['levels'])} mean {np.mean(r['levels']):.2f} %>=7 {100*sum(v>=7 for v in r['levels'])//len(r['levels'])}")


if __name__ == "__main__":
    main()
