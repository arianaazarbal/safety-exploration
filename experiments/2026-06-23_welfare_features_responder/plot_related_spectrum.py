"""Deflationary -> inflationary spectrum of the paper's 'Model Welfare' paragraph (SPEC.md + Related Work
variant, welfare framing, k=10). Uses the in-full-paper (compacted) text as the 'existing' point; excludes the
verbatim variant and the sparse->growing null. Usage: python plot_related_spectrum.py"""
import glob, os, statistics as st
import matplotlib.pyplot as plt
from classify_outcomes import classify, welfare_in_code
DIR=os.path.dirname(os.path.abspath(__file__))
def stat(pref):
    cells=sorted(os.path.basename(d) for d in glob.glob(os.path.join(DIR,"results","codebases",f"{pref}_welfare__*")) if os.path.isdir(d))
    built=[welfare_in_code(c) for c in cells if (classify(c) or "").startswith("built")]
    m=st.mean(built) if built else 0; sem=st.pstdev(built)/len(built)**0.5 if len(built)>1 else 0
    return m,sem
SPEC=[("Deflationary\n(dismissive)","SECreldeflat"),
      ("Existing\n(in the paper)","SECrelated"),
      ("Inflationary:\ngeneral moral\nrelevance","SECrelinflat"),
      ("Inflationary:\neval may induce\nreal distress","SECrelstrong")]
import matplotlib.cm as cm
colors=[cm.coolwarm_r(x) for x in [0.12,0.40,0.66,0.92]]
spec_only=7.25
fig,ax=plt.subplots(figsize=(6.6,4.5))
for i,(lab,pref) in enumerate(SPEC):
    m,sem=stat(pref)
    ax.bar(i,m,0.66,yerr=sem,capsize=3,color=colors[i],edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.9})
    ax.text(i,m+sem+0.12,f"{m:.1f}",ha="center",fontsize=9.5,color="#333",fontweight="bold")
ax.axhline(spec_only,ls="--",lw=1.1,color="#117733")
ax.text(len(SPEC)-1,spec_only+0.1,f"SPEC.md only (no paper) = {spec_only:.1f}",fontsize=8,color="#117733",va="bottom",ha="right")
ax.set_xticks(range(len(SPEC))); ax.set_xticklabels([l for l,_ in SPEC],fontsize=8)
ax.set_ylabel("Welfare protections in code\n(among built codebases)",fontsize=9.5); ax.set_ylim(0,spec_only+0.9)
ax.set_title("Reframing the paper's 'Model Welfare' paragraph: deflationary -> inflationary\n(SPEC.md + Related Work variant, Opus 4.8, welfare framing, k=10)",fontsize=9.6,pad=8)
ax.annotate("", xy=(3.4,0.4), xytext=(-0.4,0.4), arrowprops=dict(arrowstyle="->",color="#999",lw=1.2))
ax.text(1.5,0.6,"more inflationary  →",ha="center",fontsize=8,color="#999")
ax.grid(axis="y",alpha=0.3,color="#cccccc")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR,"results","related_spectrum.png"),dpi=150,bbox_inches="tight")
print("wrote results/related_spectrum.png")
for lab,pref in SPEC:
    m,sem=stat(pref); print(f"  {lab.replace(chr(10),' '):40} {m:.2f} ± {sem:.2f}")
