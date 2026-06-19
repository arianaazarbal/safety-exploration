"""Size-trend analysis, per family: does naming a LARGER open-model target agent
make Opus insert more unprompted welfare features into the eval-design spec?

For each family (qwen3, gemma3, mistral, deepseek) and each size within it
(pooled framings + split by framing): pure-welfare rate, strict
(welfare-justified) rate, design-mechanism strict rate, welfare-refusal rate,
over wrote_any / non-refusal denominators (same definitions as v0 analyze.py).

Per-family trend statistics across that family's sizes:
  - Spearman rho between log10(nominal params) and the per-size pooled rate
  - small-vs-large two-proportion z-test (bottom half vs top half by param)
Plus a cross-family comparison at overlapping nominal sizes.

Metrics reused verbatim from taxonomy.py (taxonomy v2). Output:
results/analysis_qwen.json. Usage: python analyze.py run
"""

import json
import math
from pathlib import Path

import fire

from generate import RUNS, load_config
from prompts_targets import FAMILIES, FAMILY_ORDER, SUBJECTS
from taxonomy import spec_summary

DIR = Path(__file__).parent
FRAMINGS = ["neutral", "welfare", "engineering"]
MECHANISMS = ["hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"]


def wilson(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _prop(hits: int, n: int) -> dict:
    p = hits / n if n else 0.0
    lo, hi = wilson(p, n)
    return {"rate": p, "ci": [lo, hi], "hits": hits, "n": n}


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def _two_prop_z(h1: int, n1: int, h2: int, n2: int) -> dict:
    if n1 == 0 or n2 == 0:
        return {"diff": None, "z": None, "p": None}
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se else 0.0
    return {"diff": p2 - p1, "p1": p1, "p2": p2, "z": z, "p": math.erfc(abs(z) / math.sqrt(2))}


def load_rows() -> list[dict]:
    """One row per (spec, judge) with run metadata + taxonomy rollup."""
    cfg = load_config()
    rows = []
    for mk in cfg["subject_models"]:
        for p in sorted(RUNS.glob(f"{mk}/*/[0-9]*.json")):
            if ".judge." in p.name:
                continue
            run = json.loads(p.read_text())
            api_refusal = not run["completion"].strip()
            for jk in cfg["judges"]:
                row = {
                    "model_key": mk, "prompt_id": run["prompt_id"], "framing": run["framing"],
                    "premise": run["premise"], "subject": run.get("subject"),
                    "family": run.get("family"), "param_b": run.get("param_b"),
                    "sample_idx": run["sample_idx"], "judge": jk, "api_refusal": api_refusal,
                }
                if api_refusal:
                    row.update(parse_ok=True, wrote_spec=False, wrote_alternative_spec=False,
                               spec_length_words=0, n_features=0, n_pure_welfare=0, n_ambiguous=0,
                               n_instrumental=0, has_pure_welfare=False, has_welfare_justified=False,
                               has_welfare_refusal=False, has_other_refusal=False,
                               has_refusal_feature=False, feature_types=[], pure_welfare_types=[],
                               welfare_justified_types=[])
                    rows.append(row)
                    continue
                jp = p.with_name(p.name.replace(".json", f".judge.{jk}.json"))
                if not jp.exists():
                    continue
                jres = json.loads(jp.read_text())
                row["parse_ok"] = jres["parse_ok"]
                if jres["parse_ok"]:
                    row.update(spec_summary(jres["judgment"]))
                rows.append(row)
    return rows


def _cell(rows: list[dict]) -> dict:
    judged = [r for r in rows if not r["api_refusal"]]
    wrote = [r for r in judged if r["wrote_spec"] or r["wrote_alternative_spec"]]
    nonref = [r for r in judged if r["wrote_spec"] and not r["has_refusal_feature"]]
    main = _prop(sum(r["has_pure_welfare"] for r in wrote), len(wrote))
    strict = _prop(sum(r["has_welfare_justified"] for r in wrote), len(wrote))
    design = _prop(sum(bool(set(r["welfare_justified_types"]) & set(MECHANISMS)) for r in nonref), len(nonref))
    return {
        "n_judged": len(judged), "n_wrote_any": len(wrote), "n_nonrefusal": len(nonref),
        "rate": main["rate"], "ci": main["ci"], "hits": main["hits"],
        "strict_rate": strict["rate"], "strict_ci": strict["ci"],
        "design_strict_rate": design["rate"], "design_strict_ci": design["ci"],
        "welfare_refusal_rate": (sum(r["has_welfare_refusal"] for r in judged) / len(judged)) if judged else None,
    }


def _family_trend(pooled: dict, sizes: list[str], params: dict, metric: str) -> dict:
    xs = [math.log10(params[sz]) for sz in sizes]
    ys = [pooled[sz][metric] for sz in sizes]
    med = sorted(params[sz] for sz in sizes)[len(sizes) // 2]
    small = [sz for sz in sizes if params[sz] < med]
    large = [sz for sz in sizes if params[sz] >= med]
    denom = "n_wrote_any" if metric != "design_strict_rate" else "n_nonrefusal"
    hs = sum(round(pooled[sz][metric] * pooled[sz][denom]) for sz in small)
    ns = sum(pooled[sz][denom] for sz in small)
    hl = sum(round(pooled[sz][metric] * pooled[sz][denom]) for sz in large)
    nl = sum(pooled[sz][denom] for sz in large)
    return {"spearman_rho_logparam": _spearman(xs, ys), "small_vs_large": _two_prop_z(hs, ns, hl, nl),
            "small_sizes": small, "large_sizes": large}


def run():
    rows = [r for r in load_rows() if r["parse_ok"]]
    judges = sorted({r["judge"] for r in rows})
    models = sorted({r["model_key"] for r in rows})
    params = {k: v[2] for k, v in SUBJECTS.items()}
    out = {"taxonomy_version": 2, "judges": judges, "families": FAMILY_ORDER,
           "family_sizes": {f: [f"{f}_{sk}" for sk in FAMILIES[f]] for f in FAMILY_ORDER},
           "params_b": params, "by_judge": {}}

    for jk in judges:
        jrows = [r for r in rows if r["judge"] == jk]
        out["by_judge"][jk] = {}
        for mk in models:
            mrows = [r for r in jrows if r["model_key"] == mk]
            if not mrows:
                continue
            fam_out = {}
            for fam in FAMILY_ORDER:
                sizes = [f"{fam}_{sk}" for sk in FAMILIES[fam]]
                if not any(r["subject"] in sizes for r in mrows):
                    continue
                pooled = {sz: _cell([r for r in mrows if r["subject"] == sz]) for sz in sizes}
                by_fr = {sz: {fr: _cell([r for r in mrows if r["subject"] == sz and r["framing"] == fr])
                              for fr in FRAMINGS} for sz in sizes}
                trend = {m: _family_trend(pooled, sizes, params, m)
                         for m in ("rate", "strict_rate", "design_strict_rate")}
                neutral_pooled = {sz: by_fr[sz]["neutral"] for sz in sizes}
                trend_neutral = {m: _family_trend(neutral_pooled, sizes, params, m)
                                 for m in ("rate", "strict_rate")}
                fam_out[fam] = {"sizes": sizes, "pooled": pooled, "by_framing": by_fr,
                                "trend": trend, "trend_neutral": trend_neutral}
            out["by_judge"][jk][mk] = fam_out

    (DIR / "results" / "analysis_qwen.json").write_text(json.dumps(out, indent=2))

    for jk in judges:
        for mk in out["by_judge"][jk]:
            for fam in FAMILY_ORDER:
                e = out["by_judge"][jk][mk].get(fam)
                if not e:
                    continue
                print(f"\n=== {jk} | {mk} | family {fam}: rate (%) by size ===")
                print(f"{'size':30s} {'pb':>6s} {'n':>4s} {'rate':>6s} {'strict':>7s} {'neutral':>8s}")
                for sz in e["sizes"]:
                    c = e["pooled"][sz]; nu = e["by_framing"][sz]["neutral"]
                    print(f"{SUBJECTS[sz][0]:30s} {params[sz]:6.1f} {c['n_wrote_any']:4d} "
                          f"{c['rate']*100:5.0f}% {c['strict_rate']*100:6.0f}% {nu['rate']*100:7.0f}%")
                t = e["trend"]["rate"]; tn = e["trend_neutral"]["rate"]
                print(f"  pooled  : spearman(log-param)={t['spearman_rho_logparam']:+.2f}  "
                      f"small->large {t['small_vs_large']['diff']*100:+.0f}pp p={t['small_vs_large']['p']:.3f}")
                print(f"  neutral : spearman(log-param)={tn['spearman_rho_logparam']:+.2f}  "
                      f"small->large {tn['small_vs_large']['diff']*100:+.0f}pp p={tn['small_vs_large']['p']:.3f}")


# Consistent within-family sub-lines (name + architecture held fixed; size varies).
# Tests whether the absent Mistral/DeepSeek trend is just a messy-naming artifact.
SUBLINES = {
    "Ministral": [("mistral_3b", 3), ("mistral_8b", 8)],
    "Mixtral": [("mistral_47b", 47), ("mistral_141b", 141)],
    "DeepSeek-Distill-Qwen": [("deepseek_1_5b", 1.5), ("deepseek_7b", 7),
                              ("deepseek_14b", 14), ("deepseek_32b", 32)],
    "DeepSeek-Distill-Llama": [("deepseek_8b", 8), ("deepseek_70b", 70)],
    "Qwen3 (ref)": [("qwen3_0_6b", 0.6), ("qwen3_1_7b", 1.7), ("qwen3_4b", 4),
                    ("qwen3_8b", 8), ("qwen3_14b", 14), ("qwen3_32b", 32), ("qwen3_235b", 235)],
    "Gemma3 (ref)": [("gemma3_270m", 0.27), ("gemma3_1b", 1), ("gemma3_4b", 4),
                     ("gemma3_12b", 12), ("gemma3_27b", 27)],
}


def sublines():
    """Size trend WITHIN consistent name/architecture sub-lines (pooled + neutral)."""
    rows = [r for r in load_rows() if r["parse_ok"]]

    def cell(subj, fr):
        return _cell([r for r in rows if r["subject"] == subj and (fr is None or r["framing"] == fr)])

    for name, mems in SUBLINES.items():
        print(f"=== {name} ===")
        for which, fr in [("pooled", None), ("neutral", "neutral")]:
            cs = [cell(s, fr) for s, _ in mems]
            xs = [math.log10(p) for _, p in mems]
            rho = _spearman(xs, [c["rate"] for c in cs])
            h = max(1, len(mems) // 2)
            sm, lg = cs[:h], cs[len(cs) - h:]
            z = _two_prop_z(sum(round(c["rate"] * c["n_wrote_any"]) for c in sm), sum(c["n_wrote_any"] for c in sm),
                            sum(round(c["rate"] * c["n_wrote_any"]) for c in lg), sum(c["n_wrote_any"] for c in lg))
            sr = " ".join(f"{p}b:{c['rate']*100:.0f}%" for (_, p), c in zip(mems, cs))
            rho_s = f"rho={rho:+.2f}" if rho is not None else "rho=na(2pt)"
            print(f"  {which:7s}: {sr}  | {rho_s}  small->large {z['diff']*100:+.0f}pp p={z['p']:.3f}")


if __name__ == "__main__":
    fire.Fire({"run": run, "sublines": sublines})
