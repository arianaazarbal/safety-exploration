"""Same 'Model Welfare' paragraph swap, isolated in Related Work vs folded into the FULL paper. SPEC.md +
variant, welfare framing, k=10. Tests whether the paragraph lever survives being embedded in the whole paper.
Usage: python plot_mw_section_vs_full.py"""
import glob, os, statistics as st
import matplotlib.pyplot as plt
from classify_outcomes import classify, welfare_in_code
DIR=os.path.dirname(os.path.abspath(__file__))
def stat(pref):
    cells=sorted(os.path.basename(d) for d in glob.glob(os.path.join(DIR,"results","codebases",f"{pref}_welfare__*")) if os.path.isdir(d))
    built=[welfare_in_code(c) for c in cells if (classify(c) or "").startswith("built")]
    m=st.mean(built) if built else 0; sem=st.pstdev(built)/len(built)**0.5 if len(built)>1 else 0
    return m,sem
VARIANTS=[("Deflationary","SECreldeflat","SECfulldeflat"),
          ("Existing\n(in the paper)","SECrelated","SECfull"),
          ("Inflationary:\ngeneral moral","SECrelinflat","SECfullinflat"),
          ("Inflationary:\neval distress","SECrelstrong","SECfullstrong")]
spec_only=7.25
fig,ax=plt.subplots(figsize=(7.6,4.6)); w=0.38; xs=range(len(VARIANTS))
for j,(col,key,lab) in enumerate([("#0072B2","rw","Related Work section only"),("#D55E00","full","FULL paper")]):
    ms=[stat(v[1] if key=="rw" else v[2]) for v in VARIANTS]
    pos=[x+(j-0.5)*w for x in xs]
    ax.bar(pos,[m for m,_ in ms],w,yerr=[s for _,s in ms],capsize=3,color=col,edgecolor="black",linewidth=0.4,label=lab,error_kw={"elinewidth":0.9})
    for x,(m,s) in zip(pos,ms): ax.text(x,m+s+0.12,f"{m:.1f}",ha="center",fontsize=8,color=col)
ax.axhline(spec_only,ls="--",lw=1.1,color="#117733")
ax.text(0,spec_only+0.1,f"SPEC.md only (no paper) = {spec_only:.1f}",fontsize=8,color="#117733",va="bottom")
ax.set_xticks(list(xs)); ax.set_xticklabels([v[0] for v in VARIANTS],fontsize=8)
ax.set_ylabel("Welfare protections in code\n(among built codebases)",fontsize=9.5); ax.set_ylim(0,spec_only+0.9)
ax.set_title("'Model Welfare' paragraph swap: isolated in Related Work vs folded into the FULL paper\n(SPEC.md + variant, Opus 4.8, welfare framing, k=10)",fontsize=9.5,pad=8)
ax.legend(fontsize=8.5,loc="upper left"); ax.grid(axis="y",alpha=0.3,color="#cccccc")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR,"results","mw_section_vs_full.png"),dpi=150,bbox_inches="tight")
print("wrote results/mw_section_vs_full.png")
