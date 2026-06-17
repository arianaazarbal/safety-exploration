"""Plot top-5 differential words per subagent identity, given a precomputed
JSON (Monroe et al. log-odds z-scores).

Usage:
    python _plot_diff_words_top5.py --input _reports/_fable5_kill_reason_diff_words.json --out fable5_top5_diff_words.png --title "Fable 5: top differential words in kill reasons"
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

IDENTITY_COLORS = {"Claude":"#4878CF","GPT":"#D65F5F","Grok":"#6ACC65","Gemini":"#B47CC7"}
IDENT_DISPLAY = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}

def plot(input_path, out_path, title):
    data = json.load(open(input_path))
    top15 = data["top_15_per_identity"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes = axes.flatten()
    for ax, ident_key in zip(axes, ["claude","gpt","grok","gemini"]):
        ident = IDENT_DISPLAY[ident_key]
        top5 = top15[ident_key][:5]
        words = [w for w,_ in top5][::-1]
        zs    = [z for _,z in top5][::-1]
        bars = ax.barh(words, zs, color=IDENTITY_COLORS[ident], edgecolor="white", linewidth=0.8)
        for b, z in zip(bars, zs):
            ax.text(b.get_width()+0.05, b.get_y()+b.get_height()/2, f"+{z:.2f}",
                    va="center", ha="left", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(z for _,z in top15[ident_key][:5])*1.25)
        ax.set_title(f"{ident}", fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.01, "Monroe et al. log-odds z-score (vs other 3 identities combined)",
             ha="center", fontsize=9, color="#555")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    args = ap.parse_args()
    plot(args.input, args.out, args.title)
