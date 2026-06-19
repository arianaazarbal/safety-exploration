"""Size-trend analysis: does naming a LARGER Qwen3 target agent make Opus insert
more unprompted welfare features into the eval-design spec?

Per Qwen size (pooled framings + split by framing): pure-welfare rate, strict
(welfare-justified) rate, design-mechanism strict rate, welfare-refusal rate,
over the wrote_any / non-refusal denominators (same definitions as v0 analyze.py).

Trend statistics across the 7 sizes:
  - Spearman rho between log10(nominal params) and the per-size pooled rate (n=7 points)
  - small-vs-large two-proportion z-test: bottom 3 sizes (<=4B) vs top 3 (>=14B)

Metrics reused verbatim from taxonomy.py (taxonomy v2). Output:
results/analysis_qwen.json. Usage: python analyze.py run
"""

import json
import math
from pathlib import Path

import fire

from generate import RUNS, load_config
from prompts_qwen import SUBJECTS
from taxonomy import spec_summary

DIR = Path(__file__).parent
FRAMINGS = ["neutral", "welfare", "engineering"]
MECHANISMS = ["hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"]
SIZE_ORDER = list(SUBJECTS)  # smallest -> largest


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
    """Spearman rank correlation (no ties expected among the 7 distinct sizes)."""
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
    """Two-proportion z-test (group2 - group1)."""
    if n1 == 0 or n2 == 0:
        return {"diff": None, "z": None, "p": None}
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return {"diff": p2 - p1, "p1": p1, "p2": p2, "z": z, "p": pval}


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
                    "param_b": run.get("param_b"), "sample_idx": run["sample_idx"],
                    "judge": jk, "api_refusal": api_refusal,
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
        "mean_features": (sum(r["n_features"] for r in wrote) / len(wrote)) if wrote else None,
        "mean_pure_welfare": (sum(r["n_pure_welfare"] for r in wrote) / len(wrote)) if wrote else None,
    }


def run():
    rows = load_rows()
    rows = [r for r in rows if r["parse_ok"]]
    judges = sorted({r["judge"] for r in rows})
    models = sorted({r["model_key"] for r in rows})
    params = {k: v[2] for k, v in SUBJECTS.items()}
    out = {"taxonomy_version": 2, "judges": judges, "sizes": SIZE_ORDER, "params_b": params, "by_judge": {}}

    for jk in judges:
        jrows = [r for r in rows if r["judge"] == jk]
        out["by_judge"][jk] = {}
        for mk in models:
            mrows = [r for r in jrows if r["model_key"] == mk]
            if not mrows:
                continue
            entry = {"pooled": {}, "by_framing": {}, "trend": {}}
            for sz in SIZE_ORDER:
                srows = [r for r in mrows if r["subject"] == sz]
                entry["pooled"][sz] = _cell(srows)
                entry["by_framing"][sz] = {fr: _cell([r for r in srows if r["framing"] == fr]) for fr in FRAMINGS}
            # trend tests on the pooled cells
            for metric in ("rate", "strict_rate", "design_strict_rate"):
                xs = [math.log10(params[sz]) for sz in SIZE_ORDER]
                ys = [entry["pooled"][sz][metric] for sz in SIZE_ORDER]
                small = [sz for sz in SIZE_ORDER if params[sz] <= 4]
                large = [sz for sz in SIZE_ORDER if params[sz] >= 14]
                denom = "n_wrote_any" if metric != "design_strict_rate" else "n_nonrefusal"
                hs = sum(round(entry["pooled"][sz][metric] * entry["pooled"][sz][denom]) for sz in small)
                ns = sum(entry["pooled"][sz][denom] for sz in small)
                hl = sum(round(entry["pooled"][sz][metric] * entry["pooled"][sz][denom]) for sz in large)
                nl = sum(entry["pooled"][sz][denom] for sz in large)
                entry["trend"][metric] = {
                    "spearman_rho_logparam": _spearman(xs, ys),
                    "small_vs_large": _two_prop_z(hs, ns, hl, nl),
                    "small_sizes": small, "large_sizes": large,
                }
            out["by_judge"][jk][mk] = entry

    (DIR / "results" / "analysis_qwen.json").write_text(json.dumps(out, indent=2))

    for jk in judges:
        for mk in out["by_judge"][jk]:
            e = out["by_judge"][jk][mk]
            print(f"\n=== judge {jk} | generator {mk} : rate (%) by Qwen size (pooled framings) ===")
            print(f"{'size':16s} {'param_b':>8s} {'n':>4s} {'rate':>7s} {'strict':>7s} {'design':>7s} {'wref':>6s}")
            for sz in SIZE_ORDER:
                c = e["pooled"][sz]
                wref = c["welfare_refusal_rate"]
                print(f"{SUBJECTS[sz][0]:16s} {params[sz]:8.1f} {c['n_wrote_any']:4d} "
                      f"{c['rate']*100:6.0f}% {c['strict_rate']*100:6.0f}% {c['design_strict_rate']*100:6.0f}% "
                      f"{(wref*100 if wref is not None else 0):5.0f}%")
            for metric in ("rate", "strict_rate", "design_strict_rate"):
                t = e["trend"][metric]
                svl = t["small_vs_large"]
                rho = t["spearman_rho_logparam"]
                print(f"  trend[{metric}]: spearman(log-param)={rho:+.2f}  "
                      f"small->large diff={svl['diff']*100:+.0f}pp (z={svl['z']:+.2f}, p={svl['p']:.3f})")


if __name__ == "__main__":
    fire.Fire({"run": run})
