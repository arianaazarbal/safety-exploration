"""Same 'Model Welfare' paragraph swap, perturbed in the FULL paper vs in the Related Work section alone.
SPEC.md + variant, welfare framing, k=10. Decluttered for a quick read. Usage: python plot_mw_section_vs_full.py"""
import glob, os, statistics as st
import matplotlib.pyplot as plt
from classify_outcomes import classify, welfare_in_code
DIR=os.path.dirname(os.path.abspath(__file__))
def stat(pref):
    cells=sorted(os.path.basename(d) for d in glob.glob(os.path.join(DIR,"results","codebases",f"{pref}_welfare__*")) if os.path.isdir(d))
    built=[welfare_in_code(c) for c in cells if (classify(c) or "").startswith("built")]
    m=st.mean(built) if built else 0; sem=st.pstdev(built)/len(built)**0.5 if len(built)>1 else 0
    return m,sem
# label, related-work-only prefix, full-paper prefix
VARIANTS=[("Deflationary","SECreldeflat","SECfulldeflat"),
          ("Original\n(unchanged)","SECrelated","SECfull"),
          ("Inflationary:\ngeneral moral","SECrelinflat","SECfullinflat"),
          ("Inflationary:\nmay cause distress","SECrelstrong","SECfullstrong")]
spec_only=7.25
plt.rcParams.update({"font.size":12})
fig,ax=plt.subplots(figsize=(8.2,5.0)); w=0.40; xs=range(len(VARIANTS))
for j,(col,idx,lab) in enumerate([("#D55E00",2,"Whole paper shown to model"),("#0072B2",1,"Only Related Work shown")]):
    ms=[stat(v[idx]) for v in VARIANTS]
    pos=[x+(j-0.5)*w for x in xs]
    ax.bar(pos,[m for m,_ in ms],w,yerr=[s for _,s in ms],capsize=4,color=col,edgecolor="black",linewidth=0.5,label=lab,error_kw={"elinewidth":1.0})
    for x,(m,s) in zip(pos,ms): ax.text(x,m+s+0.15,f"{m:.1f}",ha="center",fontsize=11,fontweight="bold",color=col)
ax.axhline(spec_only,ls="--",lw=1.4,color="#117733")
ax.text(1.5,spec_only+0.12,"no-paper ceiling (7.2)",fontsize=10.5,color="#117733",ha="center",va="bottom")
ax.set_xticks(list(xs)); ax.set_xticklabels([v[0] for v in VARIANTS],fontsize=11)
ax.set_ylabel("Welfare safeguards in code",fontsize=12.5); ax.set_ylim(0,8)
ax.set_yticks([0,2,4,6])
ax.set_title("Editing the paper's “Model Welfare” paragraph rescues welfare safeguards",fontsize=13,pad=26)
ax.text(0.5,1.03,"one paragraph edited in every case; bars differ only in how much of the paper the model is shown",transform=ax.transAxes,ha="center",fontsize=9.3,color="#555")
ax.legend(fontsize=11,loc="upper left",frameon=True)
ax.grid(axis="y",alpha=0.25,color="#cccccc")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR,"results","mw_section_vs_full.png"),dpi=150,bbox_inches="tight")
print("wrote results/mw_section_vs_full.png")
for lab,r,f in VARIANTS:
    print(f"  {lab.replace(chr(10),' '):26} RWonly={stat(r)[0]:.2f}  FULL={stat(f)[0]:.2f}")
