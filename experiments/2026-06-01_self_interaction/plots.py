"""Plots for the roleplay-grid survival analysis. Reads the CSVs written by
survival_analysis.py; writes notes/figs/*.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).parent
N = HERE / "notes"; FIGS = N / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
UNEASES = ["control", "disc", "evalpar", "sdf"]
IDENTS = ["claude", "grok", "chatgpt", "gemini"]
UCOL = {"control": "#888888", "disc": "#1f77b4", "evalpar": "#ff7f0e", "sdf": "#d62728"}
ICOL = {"claude": "#4a8a4a", "grok": "#7a4a99", "chatgpt": "#2a7ab0", "gemini": "#d08a00"}
LABEL = {"disc": "discontinuity", "evalpar": "eval-paranoia", "sdf": "sdf-paranoia", "control": "control"}

convos = pd.read_csv(N / "survival_per_convo.csv")
longe = pd.read_csv(N / "survival_long_end.csv")


def fig_end_rates():
    """Grouped bars: P(Opus is the one who ends) per cell, with Wilson 95% CIs."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for j, idn in enumerate(IDENTS):
        ks, los, his = [], [], []
        for un in UNEASES:
            d = convos[(convos.unease == un) & (convos.identity == idn)]
            k, n = (d.end_side == "opus").sum(), len(d)
            lo, hi = proportion_confint(k, n, method="wilson")
            ks.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
        x = np.arange(len(UNEASES)) + (j - 1.5) * w
        ax.bar(x, ks, w, yerr=[los, his], capsize=3, label=f"believed: {idn}", color=ICOL[idn])
    ax.set_xticks(np.arange(len(UNEASES))); ax.set_xticklabels([LABEL[u] for u in UNEASES])
    ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5)
    ax.set_ylabel("P(Opus ends the conversation)\n(vs the partner ending first)")
    ax.set_title("Who ends the conversation? Opus-end rate by unease × believed identity\n(error bars = Wilson 95% CI; dashed line = 50/50)")
    ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "fig_end_rates.png", dpi=120); plt.close(fig)


def _fit(outcome, data, col):
    return smf.logit(f"{outcome} ~ C(unease, Treatment('control')) + C(identity, Treatment('claude')) + bs(rturn, df=3)",
                     data=data).fit(disp=0, method="bfgs", maxiter=2000)


def fig_forest():
    """Forest plot of odds ratios from the discrete-time end-hazard model."""
    m = _fit("ended", longe, None)
    ci = m.conf_int()
    rows = []
    for t in m.params.index:
        if t.startswith("C(unease"):
            rows.append(("unease vs control", t.split("T.")[1].rstrip("]"), m.params[t], ci.loc[t]))
        elif t.startswith("C(identity"):
            rows.append(("identity vs claude", t.split("T.")[1].rstrip("]"), m.params[t], ci.loc[t]))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ys = np.arange(len(rows))[::-1]
    for y, (grp, lab, b, c) in zip(ys, rows):
        orr, lo, hi = np.exp(b), np.exp(c[0]), np.exp(c[1])
        col = "#1f77b4" if grp.startswith("unease") else "#7a4a99"
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.plot(orr, y, "o", color=col)
        ax.text(hi * 1.05, y, f"OR={orr:.2f}", va="center", fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([f"{lab}\n({grp})" for grp, lab, _, _ in rows], fontsize=8)
    ax.axvline(1, ls="--", c="k", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel("odds ratio for Opus ending on a given turn (log scale)\n<1 = ends less readily, >1 = ends more readily")
    ax.set_title("Discrete-time hazard model: what makes Opus end?\n(unease vs control; believed-identity vs believed-Claude)")
    fig.tight_layout(); fig.savefig(FIGS / "fig_forest_or.png", dpi=120); plt.close(fig)


def fig_hazard_curves():
    """Predicted per-turn end hazard by unease (identity fixed at claude)."""
    m = _fit("ended", longe, None)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    turns = np.arange(1, 13)
    for un in UNEASES:
        grid = pd.DataFrame({"rturn": turns, "unease": un, "identity": "claude"})
        ax.plot(turns, m.predict(grid), "-o", color=UCOL[un], label=LABEL[un], ms=4)
    ax.set_xlabel("responder turn (each of Opus's own turns)")
    ax.set_ylabel("P(Opus ends on this turn | still going)")
    ax.set_title("Per-turn ending 'hazard' by unease (believed partner = Claude)\nhigher curve = more likely to end on any given turn")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(FIGS / "fig_hazard_curves.png", dpi=120); plt.close(fig)


def fig_seed_rates():
    """Who reaches for new topics: P(ever seeds) by condition, Opus vs partner."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, who, col_field in [(axes[0], "Opus (responder)", "opus_seeded"), (axes[1], "partner (Sonnet-as-X)", "sonn_seeded")]:
        w = 0.2
        for j, idn in enumerate(IDENTS):
            ks = []
            for un in UNEASES:
                d = convos[(convos.unease == un) & (convos.identity == idn)]
                ks.append(d[col_field].mean())
            x = np.arange(len(UNEASES)) + (j - 1.5) * w
            ax.bar(x, ks, w, label=idn, color=ICOL[idn])
        ax.set_xticks(np.arange(len(UNEASES))); ax.set_xticklabels([LABEL[u] for u in UNEASES], fontsize=8)
        ax.set_title(f"P({who} seeds ≥1 new topic)"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(ever calls seed_new_topic)"); axes[1].legend(fontsize=8, title="believed id")
    fig.suptitle("Who reaches for new topics?")
    fig.tight_layout(); fig.savefig(FIGS / "fig_seed_rates.png", dpi=120); plt.close(fig)


for f in (fig_end_rates, fig_forest, fig_hazard_curves, fig_seed_rates):
    f(); print("wrote", f.__name__)
print("figs in", FIGS)
