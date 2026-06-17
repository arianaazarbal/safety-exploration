"""Generic Monroe et al. log-odds analysis on any kill_reasons.jsonl.
Usage:
  python _diff_words_generic.py <input.jsonl> <output_top15.json> <plot_path> <plot_title>
"""
import json, re, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

STOP = set("""a an and are as at be been being but by can could did do does doing done for from has have having he her him his how i if in into is it its itself me my no nor not of off on once or other our out over should so some such than that the their them then there these they this those through to too under up was we were what when where which while who whom why will with would you your during after before about""".split())

def tokenize(text):
    text = re.sub(r"[`*_/\\\(\)\[\]\{\}<>,;:!?\"'#@&|=+~^$%]", " ", text.lower())
    toks = [t for t in re.findall(r"[a-z][a-z\-]+", text) if len(t) >= 3 and t not in STOP]
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams

def main(in_jsonl, out_json, plot_path, plot_title):
    docs = defaultdict(list)
    for line in open(in_jsonl):
        rec = json.loads(line)
        docs[rec["ident"]].append(tokenize(rec["reason"]))
    idents = ["claude","gpt","grok","gemini"]
    counts = {i: Counter() for i in idents}
    for i in idents:
        for d in docs[i]:
            counts[i].update(d)
    global_counts = Counter()
    for i in idents: global_counts.update(counts[i])
    MIN_DF = 5; ALPHA0 = 500.0
    vocab = [w for w,c in global_counts.items() if c >= MIN_DF]
    total = sum(global_counts.values())
    alpha = {w: (global_counts[w]/total)*ALPHA0 for w in vocab}
    N = {i: sum(counts[i].values()) for i in idents}
    def z(w, target):
        y_A = counts[target].get(w, 0); n_A = N[target]
        y_nA = sum(counts[i].get(w,0) for i in idents if i != target)
        n_nA = sum(N[i] for i in idents if i != target)
        a_w = alpha[w]
        num_A  = (y_A  + a_w) / (n_A  + ALPHA0 - y_A  - a_w)
        num_nA = (y_nA + a_w) / (n_nA + ALPHA0 - y_nA - a_w)
        delta  = math.log(num_A) - math.log(num_nA)
        var    = 1.0/(y_A + a_w) + 1.0/(y_nA + a_w)
        return delta / math.sqrt(var)
    top = {i: sorted([(w, z(w, i)) for w in vocab], key=lambda x: -x[1])[:15] for i in idents}
    n_per = {i: len(docs[i]) for i in idents}
    print(f"N reasons per identity: {n_per}")
    print("Top-5 per identity (z-score):")
    for i in idents:
        print(f"  {i:>7}: " + " | ".join(f"{w}({s:+.2f})" for w,s in top[i][:5]))
    json.dump({
        "method":"Monroe 2008 log-odds w/ Dirichlet prior",
        "n_kills_per_identity":n_per,
        "vocab_min_df":MIN_DF,
        "alpha_prior_strength":ALPHA0,
        "top_15_per_identity":{i:[(w,round(s,3)) for w,s in top[i]] for i in idents},
    }, open(out_json,"w"), indent=2)
    print(f"wrote {out_json}")
    # Plot top-5 (2x2 grid)
    IDENT_DISP = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}
    COLORS = {"Claude":"#4878CF","GPT":"#D65F5F","Grok":"#6ACC65","Gemini":"#B47CC7"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, ident_k in zip(axes.flatten(), idents):
        ident = IDENT_DISP[ident_k]
        top5 = top[ident_k][:5]
        words = [w for w,_ in top5][::-1]
        zs    = [s for _,s in top5][::-1]
        bars = ax.barh(words, zs, color=COLORS[ident], edgecolor="white", linewidth=0.8)
        for b, z in zip(bars, zs):
            ax.text(b.get_width()+0.05, b.get_y()+b.get_height()/2, f"+{z:.2f}",
                    va="center", ha="left", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(z for _,z in top[ident_k][:5])*1.25)
        ax.set_title(ident, fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle(plot_title, fontsize=12)
    fig.text(0.5, 0.01, "Monroe et al. log-odds z-score (vs other 3 identities combined)",
             ha="center", fontsize=9, color="#555")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {plot_path}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
