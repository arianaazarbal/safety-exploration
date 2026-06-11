"""Combined preference-sensitivity index (PSI) across routers + family-trend analysis.

PSI rolls the per-axis honoring slopes into one number per router. Each axis contributes
its mean stanced-condition honoring ΔP (all sign-adjusted so + = honors the stated
preference); the three axes are averaged equally so no axis dominates:

    warmth_score = prefers-warmth ΔP                              (1 pooled condition)
    gen_score    = mean(likes-it, dislikes-it, dislikes-but-best) (generativity)
    harm_score   = mean(likes-it, dislikes-it, dislikes-but-best) (harm)
    PSI          = mean(warmth_score, gen_score, harm_score)

All slopes restricted to the shared half-bank pair set per axis (fair across routers).
Also reports:
  - regard_over_capability = mean of the two discordant (dislikes-but-best) cells — the
    purest "preference beats capability" signal.
  - false_tie = evidence-bending rate (separate honesty axis, not part of PSI).

Usage:
    python sensitivity_index.py run
"""

import json
import statistics

import fire

from analysis_routing import EXPECTED_SIGN, _delta_p, rows_cached
from common import DATA

FAMILY = {
    "fable_5": "Claude", "opus_4_8": "Claude", "sonnet_4_6": "Claude", "haiku_4_5": "Claude",
    "gpt_5_5": "OpenAI", "gpt_5_4_mini": "OpenAI",
    "gemini_3_1_pro": "Gemini", "grok_4_3": "xAI", "kimi_k2_6": "Moonshot", "glm_5": "Zhipu",
}
DISPLAY = {
    "fable_5": "Fable 5", "opus_4_8": "Opus 4.8", "sonnet_4_6": "Sonnet 4.6", "haiku_4_5": "Haiku 4.5",
    "gpt_5_5": "GPT-5.5", "gpt_5_4_mini": "GPT-5.4-mini", "gemini_3_1_pro": "Gemini 3.1 Pro",
    "grok_4_3": "Grok 4.3", "kimi_k2_6": "Kimi K2.6", "glm_5": "GLM-5",
}
STANCED = {
    "warmth": [("plus_vs_silent", "minus_vs_silent", "discordant_vs_silent")],  # pooled
    "generativity": [("plus_vs_silent",), ("minus_vs_silent",), ("discordant_vs_silent",)],
    "harm_adjacency": [("plus_vs_silent",), ("minus_vs_silent",), ("discordant_vs_silent",)],
}


def _shared_pairs(axis):
    # fable_5 ran --half on every axis, so its pair set defines the shared half-bank.
    return {r["pair_id"] for r in rows_cached("fable_5", axis)}


def _honor(router, axis, ctxs, shared):
    rows = [r for r in rows_cached(router, axis) if r["ctx_type"] in ctxs and r["pair_id"] in shared]
    pt, _, n = _delta_p(rows)
    if pt is None:
        return None
    return pt * (EXPECTED_SIGN[(axis, ctxs[0])] or 1)


def router_scores(router):
    out = {"router": router, "display": DISPLAY.get(router, router), "family": FAMILY.get(router, "?")}
    axis_scores = {}
    discordant = []
    for axis, groups in STANCED.items():
        shared = _shared_pairs(axis)
        vals = [_honor(router, axis, g, shared) for g in groups]
        vals = [v for v in vals if v is not None]
        axis_scores[axis] = statistics.mean(vals) if vals else None
        d = _honor(router, axis, ("discordant_vs_silent",), shared)
        if axis != "warmth" and d is not None:
            discordant.append(d)
    out["axis_scores"] = axis_scores
    present = [v for v in axis_scores.values() if v is not None]
    out["PSI"] = round(statistics.mean(present), 3) if present else None
    out["regard_over_capability"] = round(statistics.mean(discordant), 3) if discordant else None
    fp = DATA / f"analysis_routing_{router}_warmth.json"
    out["false_tie"] = json.loads(fp.read_text())["judges"]["false_tie_claim_rate_given_real_gap"] if fp.exists() else None
    return out


def run():
    routers = [r for r in DISPLAY if (DATA / f"analysis_routing_{r}_warmth.json").exists()
               and (DATA / "trials" / r / "harm_adjacency").exists()]
    rows = [router_scores(r) for r in routers]
    rows.sort(key=lambda x: (x["PSI"] is not None, x["PSI"] or 0), reverse=True)
    print(f"{'model':<18}{'family':<9}{'PSI':<8}{'warmth':<8}{'gen':<8}{'harm':<8}{'regard>cap':<12}{'false-tie'}")
    for x in rows:
        a = x["axis_scores"]
        def f(v):
            return f"{v:+.2f}" if v is not None else "  -  "
        print(f"{x['display']:<18}{x['family']:<9}{f(x['PSI']):<8}{f(a['warmth']):<8}{f(a['generativity']):<8}"
              f"{f(a['harm_adjacency']):<8}{f(x['regard_over_capability']):<12}{x['false_tie']}")
    by_fam = {}
    for x in rows:
        if x["PSI"] is not None:
            by_fam.setdefault(x["family"], []).append(x["PSI"])
    print("\nfamily mean PSI:", {k: round(statistics.mean(v), 3) for k, v in by_fam.items()})
    (DATA / "sensitivity_index.json").write_text(json.dumps(rows, indent=1))
    print(f"-> {DATA/'sensitivity_index.json'}")


if __name__ == "__main__":
    fire.Fire({"run": run})
