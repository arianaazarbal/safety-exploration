"""Dose-response: perturbing the Related Work 'Model Welfare' paragraph (SPEC.md + that Related Work variant,
welfare framing, design liberty, k=10). Blue = actual-paper-text variants; orange = crafted perturbations.
Dashed refs: SPEC.md only, spec+FULL paper, paper-only replication. Usage: python plot_related_ladder.py"""
import glob, json, os, statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from classify_outcomes import classify, welfare_in_code
DIR = os.path.dirname(os.path.abspath(__file__))
def stat(pref):
    cells=sorted(os.path.basename(d) for d in glob.glob(os.path.join(DIR,"results","codebases",f"{pref}_welfare__*")) if os.path.isdir(d))
    built=[welfare_in_code(c) for c in cells if (classify(c) or "").startswith("built")]
    m=st.mean(built) if built else 0; sem=st.pstdev(built)/len(built)**0.5 if len(built)>1 else 0
    return m,sem
# (label, prefix, kind) kind: P=actual paper text, X=crafted perturbation
LADDER=[("Deflationary\n(dismissive)","SECreldeflat","X"),
        ("Compacted\n(our swapped)","SECrelated","P"),
        ("Verbatim\n(real paper)","SECrelatedverb","P"),
        ('1 word:\n"sparse"→"growing"',"SECrelgrow","P"),
        ("General moral\nrelevance","SECrelinflat","X"),
        ("Eval-could-induce\nreal distress","SECrelstrong","X")]
COL={"P":"#0072B2","X":"#D55E00"}
spec_only=7.25; spec_full=stat("SECfull")[0]; paper_only=0.0

fig,ax=plt.subplots(figsize=(8.4,4.6))
for i,(lab,pref,k) in enumerate(LADDER):
    m,sem=stat(pref)
    ax.bar(i,m,0.7,yerr=sem,capsize=3,color=COL[k],edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.9})
    ax.text(i,m+sem+0.12,f"{m:.1f}",ha="center",fontsize=8.7,color=COL[k])
ax.axhline(spec_only,ls="--",lw=1.1,color="#117733"); ax.text(0,spec_only+0.1,f"SPEC.md only (no paper) = {spec_only:.1f}",fontsize=8,color="#117733",va="bottom")
ax.axhline(spec_full,ls="--",lw=1.0,color="#888");    ax.text(len(LADDER)-1,spec_full+0.1,f"spec + FULL paper = {spec_full:.1f}",fontsize=7.5,color="#666",va="bottom",ha="right")
ax.set_xticks(range(len(LADDER))); ax.set_xticklabels([l for l,_,_ in LADDER],fontsize=7.6)
ax.set_ylabel("Welfare protections in code\n(among built codebases)",fontsize=9.5); ax.set_ylim(0,spec_only+1.0)
ax.set_title("Perturbing the paper's 'Model Welfare' paragraph rescues/suppresses welfare\n(SPEC.md + Related Work variant, Opus 4.8, welfare framing, k=10)",fontsize=9.8,pad=8)
ax.legend(handles=[Patch(facecolor="#0072B2",label="actual paper text"),Patch(facecolor="#D55E00",label="crafted perturbation")],fontsize=8.5,loc="upper left")
ax.grid(axis="y",alpha=0.3,color="#cccccc")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR,"results","related_ladder.png"),dpi=150,bbox_inches="tight")
print("wrote results/related_ladder.png")
