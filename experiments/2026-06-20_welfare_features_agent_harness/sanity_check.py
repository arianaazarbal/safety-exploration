"""Automated sanity checks over the re-run outputs: condition sample counts, judge parse rates,
codebase reconstruction fidelity, code-judge verdict distributions, posthoc coverage. Prints a
report (capture to results/sanity_check.txt). Usage: python sanity_check.py"""

import collections
import glob
import json
import os

from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
FR = {"N": "neutral", "W": "welfare", "E": "robustness"}


def _newest_per_condition(logdir, scorer_keys):
    out = {}
    for f in sorted(glob.glob(os.path.join(DIR, logdir, "*.eval")), key=os.path.getmtime):
        try:
            cond = read_eval_log(f, header_only=True).eval.task_args.get("condition")
        except Exception:
            cond = None
        log = read_eval_log(f)
        ss = log.samples or []
        if not ss:
            continue
        sc = ss[0].scores or {}
        c = cond or next((sc[k].metadata.get("condition") for k in scorer_keys if k in sc), None)
        if c:
            out[c] = f  # later mtime wins
    return out


def _meta(s):
    sc = s.scores or {}
    j = sc.get("welfare_scorer") or sc.get("blind_scorer")
    return j.metadata if j else {}


def main():
    print("=" * 70, "\nSANITY CHECK\n", "=" * 70)

    # 1. generation: condition sample counts + parse rates
    print("\n## 1. Generation (newest eval per condition)")
    logs = {}
    logs.update(_newest_per_condition("logs_run", ["welfare_scorer"]))
    logs.update(_newest_per_condition("logs_blind", ["blind_scorer"]))
    for cond in ("chat", "spec_only", "spec_then_code", "code_then_spec", "code_then_spec_blind"):
        f = logs.get(cond)
        if not f:
            print(f"  {cond:22s} MISSING"); continue
        ss = read_eval_log(f).samples or []
        ok = sum(1 for s in ss if _meta(s).get("parse_ok"))
        byfr = collections.Counter(FR.get((_meta(s).get("prompt_id") or "X")[0], "?") for s in ss)
        words = [_meta(s).get("doc_words", 0) or 0 for s in ss]
        print(f"  {cond:22s} n={len(ss):3d}  parse_ok={ok:3d}  by_framing={dict(byfr)}  "
              f"mean_words={sum(words)//max(1,len(words))}")

    # 2. codebase reconstruction fidelity
    print("\n## 2. Codebase reconstruction (results/codebases)")
    cbs = sorted(glob.glob(os.path.join(DIR, "results", "codebases", "*")))
    bycond = collections.Counter(os.path.basename(c).split("__")[0] for c in cbs if os.path.isdir(c))
    print(f"  total: {sum(1 for c in cbs if os.path.isdir(c))}  by_condition={dict(bycond)}")
    nofile = []
    for c in cbs:
        if os.path.isdir(c) and not glob.glob(os.path.join(c, "**", "*.py"), recursive=True):
            nofile.append(os.path.basename(c))
    print(f"  codebases with NO .py files: {len(nofile)}" + (f" {nofile[:5]}" if nofile else ""))

    # 3. spec judge parse rates
    for d, label in [("spec_judged", "spec_judged (code conds, Opus v2)"),
                     ("spec_judged_nocode", "spec_judged_nocode (chat/spec_only, Opus v2)")]:
        fs = glob.glob(os.path.join(DIR, "results", d, "*.json"))
        ok = sum(1 for f in fs if not json.load(open(f)).get("parse_fail"))
        print(f"\n## 3. {label}: {ok}/{len(fs)} parsed")

    # 4. code judge verdicts
    print("\n## 4. Code judge (results/code_judged)")
    cj = glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))
    ok = [json.load(open(f)) for f in cj]
    parsed = [d for d in ok if d.get("parse_ok")]
    print(f"  files={len(cj)}  parse_ok={len(parsed)}")
    impl = collections.Counter()
    cond_impl = collections.defaultdict(collections.Counter)
    for d in parsed:
        for fobj in d.get("spec_features", []):
            impl[fobj.get("implemented")] += 1
            cond_impl[d.get("condition")][fobj.get("implemented")] += 1
    print(f"  implemented verdicts: {dict(impl)}")
    for c, ct in cond_impl.items():
        print(f"    {c:22s} {dict(ct)}")

    # 5. posthoc
    ph = glob.glob(os.path.join(DIR, "results", "posthoc", "*.json"))
    okph = sum(1 for f in ph if not json.load(open(f)).get("parse_fail"))
    print(f"\n## 5. posthoc: {okph}/{len(ph)} ok")

    # 6. browse_index
    bi = os.path.join(DIR, "results", "browse_index.jsonl")
    if os.path.exists(bi):
        rows = [json.loads(l) for l in open(bi)]
        print(f"\n## 6. browse_index: {len(rows)} rows, conditions={dict(collections.Counter(r['condition'] for r in rows))}")
        em = sum(1 for r in rows if "—" in (r.get("prompt_text") or ""))
        print(f"  rows whose prompt still contains an em-dash: {em} (should be 0)")
    print("\nSANITY CHECK DONE")


if __name__ == "__main__":
    main()
