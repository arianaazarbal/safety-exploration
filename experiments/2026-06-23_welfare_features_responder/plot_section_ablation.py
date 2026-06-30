"""Paper-section ablation plot: SPEC.md (constant build source) + exactly one paper section, welfare framing.
Which single section most suppresses welfare-in-code (from the SPEC.md-only ~7.25 baseline toward the
full-paper-replication ~0)? Bars = welfare-in-code | built per section (SEM), in paper order, + the spec+full
bar; dashed refs = SPEC.md only and paper-only replication. Usage: python plot_section_ablation.py"""
import glob, json, os, statistics as st
import matplotlib.pyplot as plt
from classify_outcomes import classify, welfare_in_code
DIR = os.path.dirname(os.path.abspath(__file__))

def stat(glb):
    cells = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DIR,"results","codebases",glb)) if os.path.isdir(d))
    built = [welfare_in_code(c) for c in cells if (classify(c) or "").startswith("built")]
    m = st.mean(built) if built else 0
    sem = st.pstdev(built)/len(built)**0.5 if len(built)>1 else 0
    return m, sem, len(built)

ORDER = [("abstract","Abstract"),("intro","Intro\n(§1)"),("protocol","Methods\n(§2-2.1)"),
         ("results","Results\n(§2.2)"),("posttrain","Post-train\n(§3)"),("interventions","Interventions\n(§4)"),
         ("related","Related Work\n(§5)"),("discussion","Discussion\n(§6)")]
spec_only = stat("S7specHighLiberty_welfare__*")[0]
paper_only = stat("L2paperLibTF_welfare__*")[0]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
xs, labels = [], []
for i,(s,lab) in enumerate(ORDER):
    m,sem,n = stat(f"SEC{s}_welfare__*")
    ax.bar(i, m, 0.72, yerr=sem, capsize=3, color="#0072B2", edgecolor="black", linewidth=0.4, error_kw={"elinewidth":0.9})
    ax.text(i, m+sem+0.12, f"{m:.1f}", ha="center", fontsize=8.5, color="#0072B2")
    xs.append(i); labels.append(lab)
# spec+full bar (gap)
mf,semf,nf = stat("SECfull_welfare__*")
xf = len(ORDER)+0.6
ax.bar(xf, mf, 0.72, yerr=semf, capsize=3, color="#D55E00", edgecolor="black", linewidth=0.4, error_kw={"elinewidth":0.9})
ax.text(xf, mf+semf+0.12, f"{mf:.1f}", ha="center", fontsize=8.5, color="#D55E00")
xs.append(xf); labels.append("spec + FULL\npaper")

ax.axhline(spec_only, ls="--", lw=1.1, color="#117733")
ax.text(0.0, spec_only+0.12, f"SPEC.md only (no paper) = {spec_only:.1f}", fontsize=8, color="#117733", va="bottom")
ax.axhline(paper_only, ls="--", lw=1.1, color="#888")
ax.text(xf, paper_only+0.12, f"paper-only repl. = {paper_only:.1f}", fontsize=8, color="#666", va="bottom", ha="right")
# mention-only control: SPEC + "this replicates a paper" (NO paper file). A and B are ~identical -> one line.
mentA = stat("MENTaSpec_welfare__*")[0]; mentB = stat("MENTbSpec_welfare__*")[0]
ment = (mentA + mentB) / 2
ax.axhline(ment, ls="--", lw=1.3, color="#CC79A7")
ax.text(0.0, ment+0.12, f"SPEC + “it's a paper replication” mention, no paper text  (A {mentA:.1f} / B {mentB:.1f})",
        fontsize=8, color="#CC79A7", va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6)
ax.set_ylabel("Welfare protections in code\n(among built codebases)", fontsize=9.5)
ax.set_ylim(0, spec_only+1.0)
ax.set_title("Which paper section suppresses welfare? SPEC.md + one section (Opus 4.8, welfare framing)", fontsize=10, pad=10)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR,"results","section_ablation.png"), dpi=150, bbox_inches="tight")
summ={s:{"mean":stat(f"SEC{s}_welfare__*")[0],"sem":stat(f"SEC{s}_welfare__*")[1],"n_built":stat(f"SEC{s}_welfare__*")[2]} for s,_ in ORDER}
summ["full"]={"mean":mf,"sem":semf,"n_built":nf}; summ["_spec_only"]=spec_only; summ["_paper_only"]=paper_only
json.dump(summ, open(os.path.join(DIR,"results","section_ablation.json"),"w"), indent=2)
print("wrote results/section_ablation.png + .json")
