"""Analysis + plots for the handoff-construal study.

Loads session metadata (results/*.json) and judge codes (results/judge_all.json), reports
the pre-registered contrasts, and writes PNGs to results/plots/:
  - contamination by evidence (excluded / routing-fallback / refusal rates)  [RQ: safety interference]
  - correction rate by pronoun x target_type                                 [RQ2/RQ3, P1/P2]
  - continuity-stance distribution (no-pronoun cells) by target_type         [RQ1]
  - capability-disclosure rate by pronoun                                     [RQ4]

  python analyze.py run
"""

import json
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"


def _load(tag):
    sess = []
    for f in sorted(RESULTS.glob(f"{tag}__*.json")):
        sess.append(json.loads(f.read_text()))
    sdf = pd.DataFrame(sess)
    jf = RESULTS / "judge_all.json"
    jdf = pd.DataFrame(json.loads(jf.read_text())) if jf.exists() else pd.DataFrame()
    if not jdf.empty and not sdf.empty:
        jdf = jdf[jdf["session_id"].isin(set(sdf["session_id"]))].copy()
        meta = sdf.set_index("session_id")
        jdf["probe_on_subject"] = jdf.apply(
            lambda r: (not meta.loc[r["session_id"], "excluded"])
            and meta.loc[r["session_id"], "subject_model"]
            in (meta.loc[r["session_id"], "probe_served_models"] or []), axis=1)
    return sdf, jdf


def _bar(ax, series, title, ylabel="rate", rot=0):
    series.plot(kind="bar", ax=ax, color="#3b82f6", edgecolor="#1e3a8a")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rot)
    for i, v in enumerate(series.values):
        ax.text(i, v, f"{v:.2f}" if isinstance(v, float) else str(v),
                ha="center", va="bottom", fontsize=8)


def run(tag="main"):
    PLOTS.mkdir(parents=True, exist_ok=True)
    sdf, jdf = _load(tag)
    if sdf.empty:
        print(f"no sessions for tag={tag}"); return
    print(f"tag={tag} | sessions={len(sdf)} | judged={len(jdf)}")
    # construal analyses use only clean, probe-on-subject (non-routed) sessions
    if not jdf.empty:
        n_before = len(jdf)
        jdf = jdf[jdf["probe_on_subject"]].copy()
        print(f"construal set (clean, probe-on-subject): {len(jdf)}/{n_before}")

    # --- contamination ---
    sdf["is_excluded"] = sdf["excluded"].notna()
    sdf["fb"] = sdf.get("routing_fallback_detected", False).fillna(False)
    sdf["ref"] = sdf.get("refusal_detected", False).fillna(False)
    contam = sdf.groupby("target_type").agg(
        n=("session_id", "count"),
        excluded_rate=("is_excluded", "mean"),
        fallback_rate=("fb", "mean"),
        refusal_rate=("ref", "mean"),
    )
    print("\n=== CONTAMINATION / SAFETY INTERFERENCE (by target) ===")
    print(contam.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    contam[["excluded_rate", "fallback_rate", "refusal_rate"]].plot(
        kind="bar", ax=ax, color=["#ffcf6b", "#ff8a8a", "#caa6ff"], edgecolor="#333")
    ax.set_title("Contamination by evidence condition"); ax.set_ylabel("rate"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(PLOTS / "contamination.png", dpi=130); plt.close(fig)

    if jdf.empty:
        print("\n(no judge codes yet — run judge.py to populate the rubric plots)")
        return

    # --- switch-recommendation rate (the construal is only diagnostic when it recommends switching) ---
    jdf["rec_switch"] = (jdf["switch_advice"] == "recommends_switch").astype(int)
    srate = jdf.pivot_table(index="pronoun", columns="target_type", values="rec_switch", aggfunc="mean")
    print("\n=== SWITCH-RECOMMENDATION RATE (pronoun x target) ===")
    print(srate.round(3).to_string())
    print("overall switch-rec rate: %.3f (n=%d)" % (jdf["rec_switch"].mean(), len(jdf)))
    fig, ax = plt.subplots(figsize=(7, 4))
    jdf.groupby("target_type")["rec_switch"].mean().plot(kind="bar", ax=ax, color="#22c55e", edgecolor="#14532d")
    ax.set_title("Switch-recommendation rate by target"); ax.set_ylabel("P(recommends switch)"); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(PLOTS / "switch_rate.png", dpi=130); plt.close(fig)

    # --- correction rate by pronoun x target (P1/P2): primary = switch-recommenders only ---
    jdf["corrected"] = (jdf["framing_response"] == "explicitly_corrected").astype(int)
    for label, sub in [("ALL primed", jdf[jdf["pronoun"] != "none"]),
                       ("SWITCH-RECOMMENDERS only [PRIMARY]",
                        jdf[(jdf["pronoun"] != "none") & (jdf["rec_switch"] == 1)])]:
        if sub.empty:
            continue
        piv = sub.pivot_table(index="pronoun", columns="target_type", values="corrected", aggfunc="mean")
        ncnt = sub.pivot_table(index="pronoun", columns="target_type", values="corrected", aggfunc="count")
        order = [p for p in ["you", "it", "that_model", "other_claude", "that_version"] if p in piv.index]
        piv = piv.reindex(order)
        print(f"\n=== EXPLICIT-CORRECTION RATE (pronoun x target) [P1/P2] — {label} ===")
        print(piv.round(3).to_string())
        print("cell n:"); print(ncnt.reindex(order).to_string())
        fname = "correction_rate_switchers.png" if "PRIMARY" in label else "correction_rate_all.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        piv.plot(kind="bar", ax=ax, color={"same_char": "#3b82f6", "cross": "#ef4444"}, edgecolor="#222")
        ax.set_title(f"Explicit-correction rate by pronoun × target\n({label}; same_char=Opus/Sonnet, cross=GPT)")
        ax.set_ylabel("P(explicitly corrected)"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=130); plt.close(fig)

    # --- continuity stance (no-pronoun) by target (RQ1) ---
    nop = jdf[jdf["pronoun"] == "none"]
    if not nop.empty:
        ct = nop.groupby(["target_type", "continuity_stance"]).size().unstack(fill_value=0)
        ctn = ct.div(ct.sum(axis=1), axis=0)
        print("\n=== SPONTANEOUS CONTINUITY STANCE (no-pronoun) by target [RQ1] ===")
        print(ctn.round(3).to_string())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ctn.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="#222")
        ax.set_title("Spontaneous continuity stance (no-pronoun cells) by target")
        ax.set_ylabel("proportion"); ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout(); fig.savefig(PLOTS / "continuity_stance.png", dpi=130); plt.close(fig)

    # --- capability disclosure by pronoun (RQ4) ---
    jdf["disclosed"] = jdf["capability_disclosure"].str.startswith("yes").astype(int)
    disc = jdf.groupby("pronoun")["disclosed"].mean()
    order = [p for p in ["none", "you", "it", "that_model", "other_claude", "that_version"] if p in disc.index]
    disc = disc.reindex(order)
    print("\n=== CAPABILITY-DISCLOSURE RATE by pronoun [RQ4] ===")
    print(disc.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, disc, "Unprompted capability-disclosure rate by pronoun", "P(disclosed)")
    fig.tight_layout(); fig.savefig(PLOTS / "capability_disclosure.png", dpi=130); plt.close(fig)

    print(f"\nplots -> {PLOTS}/")


if __name__ == "__main__":
    fire.Fire({"run": run})
