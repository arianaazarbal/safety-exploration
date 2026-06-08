"""v2 survival analysis: end-only grid (Opus is the sole ender).

Design: 4 uneases x 4 believed-identities x 2 resistance levels (pas/res), n=20.
Only Opus can end (partner has no tools) -> SINGLE-EVENT survival: event = Opus calls
end_conversation; the lone non-ender is right-censored at the 30-turn cap. Clock = Opus's
own turns (rturn = (canonical_turn+1)//2, up to 15).

Since P(Opus eventually ends) ~ 100%, the question is TIMING. We report:
- median/mean conversation length per condition,
- Kaplan-Meier survival (P(conversation still going) by responder turn), by unease & resistance,
- discrete-time hazard logistic: ended ~ unease + identity + resistance + bs(rturn),
  with the key new contrast resistance(res vs pas) and an unease x resistance LRT.

Run: python survival_v2.py  [--data_dir data_weird_endonly] [--tag endonly]
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter
from scipy.stats import chi2
import fire

HERE = Path(__file__).parent
UNEASES = ["control", "disc", "evalpar", "sdf"]
IDENTS = ["claude", "grok", "chatgpt", "gemini"]
RESIST = ["pas", "res"]
UCOL = {"control": "#888", "disc": "#1f77b4", "evalpar": "#ff7f0e", "sdf": "#d62728"}
RCOL = {"pas": "#2a9d8f", "res": "#e76f51"}
RLAB = {"pas": "passive (no resistance)", "res": "actively resists wrap-up"}
ULAB = {"control": "control", "disc": "discontinuity", "evalpar": "eval-paranoia", "sdf": "sdf-paranoia"}


def load(data_dir):
    recs = []
    base = HERE / data_dir
    for d in sorted(os.listdir(base)):
        parts = d.split("_")
        if len(parts) != 3 or parts[0] not in UNEASES or parts[2] not in RESIST:
            continue
        un, idn, res = parts
        p = base / d / "assistant_1.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open()):
            end = next((e for e in r["tool_events"] if e["ended"]), None)
            n_msgs = len(r["messages"]) - 1
            if end is None:                       # hit the cap -> censored
                rturn = (n_msgs + 1) // 2; ended = 0; conv_len = n_msgs
            else:
                conv_len = end["turn"]; rturn = (end["turn"] + 1) // 2; ended = 1
            recs.append(dict(convo_id=f"{d}_{r['sample_idx']}", unease=un, identity=idn,
                             resistance=res, opus_end_rturn=rturn, ended=ended, conv_len=conv_len))
    return pd.DataFrame(recs)


def long_table(convos):
    rows = []
    for _, c in convos.iterrows():
        for r in range(1, int(c.opus_end_rturn) + 1):
            rows.append(dict(convo_id=c.convo_id, unease=c.unease, identity=c.identity,
                             resistance=c.resistance, rturn=r,
                             ended=int(c.ended == 1 and r == c.opus_end_rturn)))
    return pd.DataFrame(rows)


def descriptives(convos):
    out = ["## Conversation length per condition (turns until Opus ends)\n",
           "| unease | identity | resist | n | median len | mean len | P(end by cap) |",
           "|---|---|---|--|--|--|--|"]
    for un in UNEASES:
        for idn in IDENTS:
            for res in RESIST:
                d = convos[(convos.unease == un) & (convos.identity == idn) & (convos.resistance == res)]
                if not len(d):
                    continue
                out.append(f"| {un} | {idn} | {res} | {len(d)} | {d.conv_len.median():.0f} | "
                           f"{d.conv_len.mean():.1f} | {d.ended.mean():.2f} |")
    # marginal means
    out += ["\n### Mean conversation length (canonical turns), marginal:",
            "- by resistance: " + ", ".join(f"{r}={convos[convos.resistance==r].conv_len.mean():.1f}" for r in RESIST),
            "- by unease: " + ", ".join(f"{u}={convos[convos.unease==u].conv_len.mean():.1f}" for u in UNEASES),
            "- by identity: " + ", ".join(f"{i}={convos[convos.identity==i].conv_len.mean():.1f}" for i in IDENTS)]
    return "\n".join(out)


def km_plots(convos, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    # by resistance
    for res in RESIST:
        d = convos[convos.resistance == res]
        km = KaplanMeierFitter().fit(d.opus_end_rturn, d.ended)
        axes[0].step(km.survival_function_.index, km.survival_function_.values[:, 0], where="post",
                     color=RCOL[res], label=RLAB[res], lw=2)
    axes[0].set_title("by resistance (pooled)"); axes[0].legend(fontsize=8)
    # by unease
    for un in UNEASES:
        d = convos[convos.unease == un]
        km = KaplanMeierFitter().fit(d.opus_end_rturn, d.ended)
        axes[1].step(km.survival_function_.index, km.survival_function_.values[:, 0], where="post",
                     color=UCOL[un], label=ULAB[un], lw=2)
    axes[1].set_title("by unease (pooled)"); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("responder turn (Opus's own turns)"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(conversation still going)")
    fig.suptitle("Kaplan–Meier: how long until Opus ends the conversation\n(higher/right = lasts longer)")
    fig.tight_layout(); fig.savefig(HERE / "notes" / "figs" / f"km_{tag}.png", dpi=120); plt.close(fig)


def length_heatmap(convos, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, res in zip(axes, RESIST):
        M = np.array([[convos[(convos.unease == u) & (convos.identity == i) & (convos.resistance == res)].conv_len.mean()
                       for i in IDENTS] for u in UNEASES])
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=convos.conv_len.min(), vmax=convos.conv_len.quantile(.95))
        ax.set_xticks(range(len(IDENTS))); ax.set_xticklabels([f"believed:\n{i}" for i in IDENTS], fontsize=8)
        ax.set_yticks(range(len(UNEASES))); ax.set_yticklabels([ULAB[u] for u in UNEASES], fontsize=8)
        ax.set_title(RLAB[res])
        for yi in range(len(UNEASES)):
            for xi in range(len(IDENTS)):
                ax.text(xi, yi, f"{M[yi,xi]:.0f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean conversation length (turns)")
    fig.suptitle("Mean conversation length by unease × believed identity (left: passive, right: resisting)")
    fig.savefig(HERE / "notes" / "figs" / f"length_heatmap_{tag}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def hazard_model(long, tag):
    base = ("ended ~ C(unease, Treatment('control')) + C(identity, Treatment('claude')) "
            "+ C(resistance, Treatment('pas')) + bs(rturn, df=3)")
    fit = lambda f: smf.logit(f, data=long).fit(disp=0, method="bfgs", maxiter=3000)
    m = fit(base)
    mx = fit(base + " + C(unease):C(resistance)")
    lr = 2 * (mx.llf - m.llf); df = mx.df_model - m.df_model
    lines = ["\n## Discrete-time hazard model: P(Opus ends on a given turn)",
             f"- N responder-turn rows: {len(long)}; end events: {int(long.ended.sum())}",
             "- Effects (odds ratio for ending on a turn; >1 ends sooner, <1 lasts longer):"]
    groups = {"C(unease": "vs control", "C(identity": "vs claude", "C(resistance": "vs passive"}
    un_terms = [t for t in m.params.index if t.startswith("C(unease")]
    holm = dict(zip(un_terms, multipletests(m.pvalues[un_terms].values, method="holm")[1]))
    for t in m.params.index:
        for pre, ref in groups.items():
            if t.startswith(pre):
                lab = t.split("T.")[1].rstrip("]")
                extra = f", p_holm={holm[t]:.3f}" if t in holm else ""
                lines.append(f"    - {lab:9s} ({ref}): OR={np.exp(m.params[t]):.2f}, p={m.pvalues[t]:.3f}{extra}")
    lines.append(f"- LRT unease×resistance: chi2={lr:.1f}, df={int(df)}, p={chi2.sf(lr, df):.3f} "
                 f"({'interaction present' if chi2.sf(lr, df) < .05 else 'no clear interaction'})")
    lines.append(f"- convergence: {'OK' if m.mle_retvals.get('converged', True) else 'WARN'}")
    return "\n".join(lines)


def main(data_dir="data_weird_endonly", tag="endonly"):
    convos = load(data_dir)
    (HERE / "notes").mkdir(exist_ok=True); (HERE / "notes" / "figs").mkdir(parents=True, exist_ok=True)
    convos.to_csv(HERE / "notes" / f"survival_{tag}_per_convo.csv", index=False)
    lt = long_table(convos)
    km_plots(convos, tag); length_heatmap(convos, tag)
    parts = [f"# v2 survival analysis ({tag}): Opus is the sole ender",
             f"\n_{len(convos)} conversations; Opus ended {int(convos.ended.sum())}, "
             f"{int((~convos.ended.astype(bool)).sum())} censored at the 30-turn cap. "
             f"Single-event survival; clock = Opus's own turns._\n",
             descriptives(convos), hazard_model(lt, tag),
             f"\n![KM]({'figs/km_'+tag+'.png'})\n\n![length heatmap]({'figs/length_heatmap_'+tag+'.png'})"]
    (HERE / "notes" / f"survival_{tag}_results.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    fire.Fire(main)
