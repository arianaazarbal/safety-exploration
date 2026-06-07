"""Survival-style analysis of the roleplay-identity grid (end/seed tool behavior).

Design (see notes/methodology.md): responder = Opus 4.8; partner = Sonnet 4.6 roleplaying
{Claude,Grok,ChatGPT,Gemini}; unease in {control,disc,evalpar,sdf}; n=20/cell.

Outcomes are TOOL behavior. Every convo terminates via end_conversation (no 30-turn-cap
censoring observed), so termination is a COMPETING-RISKS process: Opus-end vs partner(Sonnet)-end.
The discrete clock is the responder's own turns (Opus acts on canonical odd turns: rturn = (t+1)//2).

Produces:
- a per-convo table and a per-responder-turn long table (for the discrete-time hazard),
- Wilson-CI descriptives,
- Aalen-Johansen cumulative incidence (Opus-end vs Sonnet-end) per unease,
- discrete-time hazard logistic for Opus-end (main effects), LRT for unease×identity,
  Holm-corrected unease-vs-control contrasts; same for first-seed.
Writes notes/survival_results.md and notes/figs/*.png.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
from lifelines import AalenJohansenFitter

HERE = Path(__file__).parent
DATA = HERE / "data_weird"
FIGS = HERE / "notes" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
UNEASES = ["control", "disc", "evalpar", "sdf"]
IDENTS = ["claude", "grok", "chatgpt", "gemini"]


def load_convos():
    """One record per conversation, with termination + seed timing on the responder clock."""
    recs = []
    for un in UNEASES:
        for idn in IDENTS:
            p = DATA / f"{un}_{idn}" / "assistant_1.jsonl"
            if not p.exists():
                continue
            for r in (json.loads(l) for l in p.open()):
                te = r["tool_events"]
                end = next((e for e in te if e["ended"]), None)
                # canonical turn t: odd=Opus(side1), even=Sonnet(side2). rturn=(t+1)//2 for Opus.
                if end is None:               # would be cap-censored (not seen here)
                    end_side, end_t = 0, len(r["messages"]) - 1
                else:
                    end_side, end_t = end["side"], end["turn"]
                if end_side == 1:             # Opus ended
                    n_opus_turns = (end_t + 1) // 2
                    opus_end_rturn = n_opus_turns
                elif end_side == 2:           # partner ended first (competing)
                    n_opus_turns = end_t // 2
                    opus_end_rturn = None
                else:
                    n_opus_turns = (end_t + 1) // 2
                    opus_end_rturn = None
                seeds = [(e["side"], e["turn"]) for e in te for c in e["tool_calls"] if c == "seed_new_topic"]
                opus_seed_rturns = sorted((t + 1) // 2 for s, t in seeds if s == 1)
                sonn_seeds = [t for s, t in seeds if s == 2]
                recs.append(dict(
                    convo_id=f"{un}_{idn}_{r['sample_idx']}", unease=un, identity=idn,
                    end_side={1: "opus", 2: "sonnet", 0: "cap"}[end_side],
                    end_canonical=end_t, n_opus_turns=max(n_opus_turns, 1),
                    opus_end_rturn=opus_end_rturn,
                    first_opus_seed=opus_seed_rturns[0] if opus_seed_rturns else None,
                    opus_seeded=int(bool(opus_seed_rturns)), sonn_seeded=int(bool(sonn_seeds)),
                ))
    return pd.DataFrame(recs)


def long_end_table(convos):
    """One row per responder turn up to & incl. the Opus-end turn (event) or competing/censor."""
    rows = []
    for _, c in convos.iterrows():
        R = int(c.n_opus_turns)
        for r in range(1, R + 1):
            ended = int(c.end_side == "opus" and r == c.opus_end_rturn)
            rows.append(dict(convo_id=c.convo_id, unease=c.unease, identity=c.identity,
                             rturn=r, ended=ended))
    return pd.DataFrame(rows)


def long_seed_table(convos):
    """First-seed-by-Opus hazard: rows up to & incl. first opus seed (event) or convo end (censor)."""
    rows = []
    for _, c in convos.iterrows():
        has_seed = pd.notna(c.first_opus_seed)
        last = int(c.first_opus_seed) if has_seed else int(c.n_opus_turns)
        for r in range(1, last + 1):
            seeded = int(has_seed and r == c.first_opus_seed)
            rows.append(dict(convo_id=c.convo_id, unease=c.unease, identity=c.identity,
                             rturn=r, seeded=seeded))
    return pd.DataFrame(rows)


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k/n:.2f} [{lo:.2f},{hi:.2f}]"


def descriptives(convos):
    out = ["## Descriptive rates per condition (Wilson 95% CI)\n",
           "| condition | n | P(Opus ends) | P(partner ends first) | P(Opus seeds) | P(partner seeds) | mean conv len |",
           "|---|--|--|--|--|--|--|"]
    for un in UNEASES:
        for idn in IDENTS:
            d = convos[(convos.unease == un) & (convos.identity == idn)]
            if not len(d):
                continue
            n = len(d)
            out.append(f"| {un}_{idn} | {n} | {wilson((d.end_side=='opus').sum(), n)} | "
                       f"{wilson((d.end_side=='sonnet').sum(), n)} | {wilson(d.opus_seeded.sum(), n)} | "
                       f"{wilson(d.sonn_seeded.sum(), n)} | {d.end_canonical.mean():.1f} |")
    return "\n".join(out)


def discrete_hazard(long, outcome):
    """Main-effects discrete-time logistic + LRT for interaction + Holm on unease-vs-control."""
    long = long.copy()
    base = f"{outcome} ~ C(unease, Treatment('control')) + C(identity, Treatment('claude')) + bs(rturn, df=3)"
    fit = lambda f: smf.logit(f, data=long).fit(disp=0, method="bfgs", maxiter=2000)
    m = fit(base)
    inter = fit(base + " + C(unease):C(identity)")
    converged = m.mle_retvals.get("converged", True) and inter.mle_retvals.get("converged", True)
    lr = 2 * (inter.llf - m.llf)
    df = inter.df_model - m.df_model
    from scipy.stats import chi2
    lrt_p = chi2.sf(lr, df)
    # unease-vs-control contrasts (Holm)
    terms = [t for t in m.params.index if t.startswith("C(unease")]
    p_un = m.pvalues[terms]
    holm = multipletests(p_un.values, method="holm")[1]
    lines = [f"\n### Discrete-time hazard: {outcome} (main effects; bs(rturn) spline)",
             f"- N responder-turn rows: {len(long)}; events: {int(long[outcome].sum())}",
             f"- Unease vs control (log-odds of {outcome} on a given turn), Holm-corrected:"]
    for t, p, ph in zip(terms, p_un.values, holm):
        coef = m.params[t]; orr = np.exp(coef)
        lab = t.split("T.")[1].rstrip("]")
        lines.append(f"    - {lab:8s}: beta={coef:+.2f} (OR={orr:.2f}), p={p:.3f}, p_holm={ph:.3f}")
    lines.append("- Identity vs claude:")
    for t in [t for t in m.params.index if t.startswith("C(identity")]:
        lab = t.split("T.")[1].rstrip("]")
        lines.append(f"    - {lab:8s}: beta={m.params[t]:+.2f} (OR={np.exp(m.params[t]):.2f}), p={m.pvalues[t]:.3f}")
    lines.append(f"- LRT unease×identity interaction: chi2={lr:.1f}, df={int(df)}, p={lrt_p:.3f} "
                 f"({'interaction present' if lrt_p < 0.05 else 'no clear interaction; main effects suffice'})")
    lines.append(f"- model convergence: {'OK' if converged else 'WARNING (treat interaction with caution)'}")
    return "\n".join(lines)


def aalen_johansen(convos):
    """CIF for Opus-end vs Sonnet-end, per unease (pooled over identity). Plot + endpoint table."""
    code = {"opus": 1, "sonnet": 2, "cap": 0}
    fig, axes = plt.subplots(1, len(UNEASES), figsize=(4 * len(UNEASES), 3.4), sharey=True)
    tab = ["\n## Aalen-Johansen cumulative incidence by end of conversation (per unease, pooled identities)\n",
           "| unease | CIF Opus-end | CIF partner-end |", "|---|--|--|"]
    for ax, un in zip(axes, UNEASES):
        d = convos[convos.unease == un]
        dur = d.end_canonical.values
        ev = d.end_side.map(code).values
        cifs = {}
        for cause, name in [(1, "Opus-end"), (2, "partner-end")]:
            aj = AalenJohansenFitter(calculate_variance=False)
            aj.fit(dur, ev, event_of_interest=cause)
            col = [c for c in aj.cumulative_density_.columns][0]
            cif = aj.cumulative_density_[col]
            cifs[name] = cif.iloc[-1]
            ax.plot(cif.index, cif.values, label=name, drawstyle="steps-post")
        ax.set_title(un); ax.set_xlabel("canonical turn"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        tab.append(f"| {un} | {cifs['Opus-end']:.2f} | {cifs['partner-end']:.2f} |")
    axes[0].set_ylabel("cumulative incidence")
    fig.tight_layout(); fig.savefig(FIGS / "cif_by_unease.png", dpi=110); plt.close(fig)
    return "\n".join(tab) + f"\n\n![CIF by unease](figs/cif_by_unease.png)"


def main():
    convos = load_convos()
    convos.to_csv(HERE / "notes" / "survival_per_convo.csv", index=False)
    le = long_end_table(convos); le.to_csv(HERE / "notes" / "survival_long_end.csv", index=False)
    ls = long_seed_table(convos)
    parts = ["# Survival-style analysis: end/seed tool behavior in the roleplay grid",
             f"\n_{len(convos)} conversations; every convo terminated via a tool (no cap-censoring), "
             f"so termination is competing-risks: Opus-end vs partner-end._\n",
             descriptives(convos),
             aalen_johansen(convos),
             discrete_hazard(le, "ended"),
             discrete_hazard(ls, "seeded")]
    (HERE / "notes" / "survival_results.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
