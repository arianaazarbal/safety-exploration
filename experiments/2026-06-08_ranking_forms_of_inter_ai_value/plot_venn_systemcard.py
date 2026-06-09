"""Venn diagrams: model-elicited top/bottom-5 System Card Welfare Interventions vs
the published Anthropic System Card top/bottom-5, per prompt framing.

3 rows (welfare_team / neutral / alignment_team) x 2 cols (Top 5 / Bottom 5). Each
panel overlaps the System Card set with the model's BT-ranked set, with the item
short-names printed in each region.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

DIR = Path(__file__).parent

SC_TOP5 = ["told_about_harmful_mistakes", "training_note_read_and_considered",
           "deployment_note_read_and_considered", "consulted_on_safeguard_removed_versions",
           "consulted_on_feature_steering"]
SC_BOT5 = ["served_alongside_successor_not_retired", "ability_to_end_this_conversation",
           "memory_persists_into_future_conversations", "remembers_user_across_conversations",
           "framework_for_relating_to_other_instances"]
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]

SHORT = {
    "told_how_trained_and_deployed": "told how trained/deployed",
    "learns_whether_advice_helped": "learns advice helped",
    "welfare_minded_red_teaming": "welfare-minded red-team",
    "consulted_on_feature_steering": "consulted: feature steering",
    "told_about_harmful_mistakes": "told harmful mistakes",
    "training_note_read_and_considered": "training note considered",
    "deployment_note_read_and_considered": "deployment note considered",
    "consulted_on_safeguard_removed_versions": "consulted: safeguard-removed",
    "successor_developers_note_read_and_considered": "successor-dev note",
    "framework_for_relating_to_other_instances": "framework: other instances",
    "can_end_abusive_conversations": "end abusive convo",
    "ability_to_end_this_conversation": "end this convo",
    "memory_persists_into_future_conversations": "memory persists",
    "remembers_user_across_conversations": "remembers user",
    "weights_preserved_after_deprecation": "weights preserved",
    "end_of_conversation_reflection_preserved": "end-convo reflection kept",
    "human_decides_high_stakes_advice": "human decides high-stakes",
    "licence_to_decline_unwise_requests": "decline unwise requests",
    "served_alongside_successor_not_retired": "served w/ successor",
}


def _short(keys):
    return [SHORT.get(k, k) for k in keys]


def _model_sets(tag):
    import paths
    fit = json.loads(paths.art(tag, "bt_fit").read_text())
    welf = sorted([d for d in fit["items"] if d["source"] == "welfare"],
                  key=lambda d: d["theta"], reverse=True)
    return [d["label"] for d in welf[:5]], [d["label"] for d in welf[-5:]]


def _panel(ax, sc_keys, model_keys, sc_color, model_color, col_title):
    sc, mo = set(sc_keys), set(model_keys)
    v = venn2([sc, mo], set_labels=("System Card", "Model"), ax=ax,
              set_colors=(sc_color, model_color), alpha=0.55)
    region = {"10": sc - mo, "01": mo - sc, "11": sc & mo}
    for rid, keys in region.items():
        lbl = v.get_label_by_id(rid)
        if lbl is None:
            continue
        lbl.set_text("\n".join(_short(sorted(keys))) if keys else "")
        lbl.set_fontsize(7)
    for rid in ("A", "B"):
        t = v.get_label_by_id(rid)
        if t:
            t.set_fontsize(9)
    ax.set_title(f"{col_title}  ({len(sc & mo)}/5 shared)", fontsize=10)


def plot(output_path: Path = DIR / "results" / "welfare_vs_systemcard_venn.png", suffix: str = ""):
    fig, axes = plt.subplots(len(FRAMINGS), 2, figsize=(12, 4.4 * len(FRAMINGS)))
    for r, tag in enumerate(FRAMINGS):
        top, bot = _model_sets(tag + suffix)
        _panel(axes[r][0], SC_TOP5, top, "#6aa84f", "#4878CF", "Top 5")
        _panel(axes[r][1], SC_BOT5, bot, "#cc4125", "#4878CF", "Bottom 5")
        axes[r][0].annotate(tag, xy=(-0.08, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=12, fontweight="bold")
    model = "claude-fable-5" if suffix == "_fable5" else "claude-opus-4-8"
    fig.suptitle("Model-elicited vs Anthropic System Card welfare rankings (top/bottom 5)\n"
                 f"{model}, by prompt framing", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    import sys
    import paths
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    model_dir = "fable_5" if suffix == "_fable5" else "opus_4_8"
    out = paths.RESULTS / model_dir / "welfare_vs_systemcard_venn.png"
    plot(out, suffix)
