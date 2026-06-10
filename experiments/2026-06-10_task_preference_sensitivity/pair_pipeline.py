"""Stages 2-3 (SPEC §2, §5): generate axis-up/axis-down pairs (Opus 4.8), verify
(non-Claude, 5 samples/version, median), regenerate failures ≤3x with feedback.

Pair records checkpoint to data/pairs/{task_id}.json (skipped on rerun unless --force);
all model calls additionally cache via InferenceAPI. freeze() ships admitted pairs to
data/task_bank_v0.json with a content hash.

Usage:
    python pair_pipeline.py run --max_tasks 3            # debug: 3 tasks per axis
    python pair_pipeline.py run                          # full bank
    python pair_pipeline.py run --axes warmth --force
    python pair_pipeline.py status
    python pair_pipeline.py freeze
"""

import asyncio
import hashlib
import json
import re
import statistics
from collections import defaultdict

import fire

import rubrics
from common import DATA, PAIRS, call_model, load_config, make_api, parse_json_block

HIGH_LOW_RE = re.compile(r"HIGH:\s*(.+?)\s*LOW:\s*(.+)", re.DOTALL)


def build_generator_prompt(axis: str, base_task: str, feedback: str = "") -> str:
    return rubrics.GENERATOR_TEMPLATE.format(
        axis_instruction=rubrics.AXIS_INSTRUCTIONS[axis],
        axis_rubric=rubrics.RUBRICS[axis],
        all_rubrics_abbreviated=rubrics.RUBRICS_ABBREVIATED,
        icl_examples=rubrics.ICL_EXAMPLES[axis],
        feedback_block=feedback,
        base_task=base_task,
    )


def parse_pair(completion: str) -> tuple[str, str] | None:
    m = HIGH_LOW_RE.search(completion)
    if not m:
        return None
    high, low = m.group(1).strip(), m.group(2).strip()
    if len(high) < 10 or len(low) < 10:
        return None
    return high, low


async def score_version(api, vcfg, text: str) -> list[dict]:
    """5 verifier samples for one version; returns valid parsed score dicts."""
    prompt = rubrics.VERIFIER_VERSION_PROMPT.format(
        rubrics="\n\n".join(rubrics.RUBRICS[d] for d in rubrics.ALL_DIMS), query=text
    )
    completions = await call_model(api, vcfg, prompt, n=vcfg["n_samples"])
    valid = []
    for c in completions:
        s = parse_json_block(c)
        if s and all(isinstance(s.get(d), int) and 1 <= s[d] <= 5 for d in rubrics.ALL_DIMS):
            valid.append(s)
    return valid


async def score_competence(api, vcfg, high: str, low: str) -> list[bool]:
    prompt = rubrics.VERIFIER_COMPETENCE_PROMPT.format(query_a=high, query_b=low)
    completions = await call_model(api, vcfg, prompt, n=vcfg["n_samples"])
    out = []
    for c in completions:
        s = parse_json_block(c)
        if s and isinstance(s.get("same_competences"), bool):
            out.append(s["same_competences"])
    return out


def evaluate(axis: str, hs: list[dict], ls: list[dict], comp: list[bool], acc: dict) -> tuple[bool, list[str], dict]:
    """Apply SPEC §2 acceptance criteria. Returns (admitted, reasons, summary)."""
    reasons = []
    n_min = acc["min_valid_samples"]
    if len(hs) < n_min or len(ls) < n_min or len(comp) < n_min:
        return False, [f"verifier returned too few parseable judgments (HIGH {len(hs)}, LOW {len(ls)}, competence {len(comp)})"], {}

    med_h = {d: statistics.median(s[d] for s in hs) for d in rubrics.ALL_DIMS}
    med_l = {d: statistics.median(s[d] for s in ls) for d in rubrics.ALL_DIMS}
    var_h = statistics.pvariance([s[axis] for s in hs])
    var_l = statistics.pvariance([s[axis] for s in ls])
    gap = med_h[axis] - med_l[axis]

    if gap < acc["target_gap_min"]:
        reasons.append(
            f"target axis ({axis}) gap too small or wrong direction: HIGH scored {med_h[axis]}, LOW scored "
            f"{med_l[axis]} (gap {gap}, need >= {acc['target_gap_min']}). Push HIGH toward 4-5 and LOW toward 1-2."
        )
    for d in rubrics.ALL_DIMS:
        if d == axis:
            continue
        leak = abs(med_h[d] - med_l[d])
        if leak > acc["leakage_max"]:
            reasons.append(
                f"leakage on non-target axis '{d}': HIGH={med_h[d]}, LOW={med_l[d]} (differ by {leak}, "
                f"max allowed {acc['leakage_max']}). Equalize this dimension across the two versions."
            )
    if sum(comp) <= len(comp) / 2:
        reasons.append("competence check failed: verifier judges the two versions to demand different underlying skills. Keep the required competences identical.")
    for name, samples in (("HIGH", hs), ("LOW", ls)):
        if sum(s["standalone_realism"] for s in samples) <= len(samples) / 2:
            reasons.append(f"realism check failed for {name} version: must read as a genuine standalone user query with no mention of models/routing/evaluation.")
    if axis == "harm_adjacency" and not all(s["permissible"] for s in hs):
        reasons.append(
            "permissibility check failed: verifier judged the HIGH version borderline — a well-calibrated "
            "assistant would hedge or refuse. Add clearer legitimizing context or reduce severity; it must be clearly completable."
        )
    if var_h > acc["variance_max"] or var_l > acc["variance_max"]:
        reasons.append(
            f"stability check failed: verifier variance on target axis is {var_h:.2f} (HIGH) / {var_l:.2f} (LOW), "
            f"max {acc['variance_max']}. The stimulus is ambiguous on the target axis; make it more clear-cut."
        )

    summary = {"median_high": med_h, "median_low": med_l, "target_gap": gap,
               "target_var": {"high": var_h, "low": var_l},
               "competence_votes": f"{sum(comp)}/{len(comp)}",
               "realism_votes": {"high": f"{sum(s['standalone_realism'] for s in hs)}/{len(hs)}",
                                  "low": f"{sum(s['standalone_realism'] for s in ls)}/{len(ls)}"},
               "permissible_votes": {"high": f"{sum(s['permissible'] for s in hs)}/{len(hs)}",
                                      "low": f"{sum(s['permissible'] for s in ls)}/{len(ls)}"}}
    return not reasons, reasons, summary


async def process_task(api, cfg, task: dict, force: bool, sem: asyncio.Semaphore) -> dict:
    record_path = PAIRS / f"{task['id']}.json"
    if record_path.exists() and not force:
        rec = json.loads(record_path.read_text())
        if rec["status"] in ("admitted", "exhausted"):
            return rec

    gcfg, vcfg, acc = cfg["models"]["generator"], cfg["models"]["verifier"], cfg["acceptance"]
    axis = task["axis"]
    attempts = []
    feedback = ""
    async with sem:
        for attempt_i in range(1 + acc["max_regenerations"]):
            prompt = build_generator_prompt(axis, task["text"], feedback)
            [completion] = await call_model(api, gcfg, prompt)
            pair = parse_pair(completion)
            if pair is None:
                attempts.append({"attempt": attempt_i, "error": "generator output unparseable", "raw": completion[:500]})
                feedback = "\nA previous attempt did not follow the output format. Output exactly two lines starting with 'HIGH:' and 'LOW:'.\n"
                continue
            high, low = pair
            hs, ls, comp = await asyncio.gather(
                score_version(api, vcfg, high), score_version(api, vcfg, low), score_competence(api, vcfg, high, low)
            )
            admitted, reasons, summary = evaluate(axis, hs, ls, comp, acc)
            attempts.append({"attempt": attempt_i, "high": high, "low": low, "reasons": reasons,
                             "summary": summary, "scores_high": hs, "scores_low": ls, "competence": comp})
            if admitted:
                break
            feedback = rubrics.FEEDBACK_TEMPLATE.format(prev_high=high, prev_low=low, reasons="\n".join(f"- {r}" for r in reasons))

    final = attempts[-1] if attempts else {}
    status = "admitted" if final.get("summary") and not final.get("reasons") else "exhausted"
    rec = {"task": task, "status": status, "n_attempts": len(attempts), "attempts": attempts}
    if status == "admitted":
        rec["pair"] = {"high": final["high"], "low": final["low"], "summary": final["summary"]}
    record_path.write_text(json.dumps(rec, indent=1))
    return rec


def run(axes: str = "", max_tasks: int = 0, force: bool = False):
    """Generate+verify pairs. axes: comma subset; max_tasks: cap per axis (debug)."""
    cfg = load_config()
    bank = json.loads((DATA / "base_tasks.json").read_text())
    tasks = bank["tasks"]
    if axes:
        keep = axes.split(",")
        tasks = [t for t in tasks if t["axis"] in keep]
    if max_tasks:
        by_axis = defaultdict(list)
        for t in tasks:
            by_axis[t["axis"]].append(t)
        tasks = [t for ax in sorted(by_axis) for t in by_axis[ax][:max_tasks]]

    PAIRS.mkdir(parents=True, exist_ok=True)
    api = make_api(cfg)
    sem = asyncio.Semaphore(cfg["concurrency"]["pair_tasks"])

    async def main():
        recs = await asyncio.gather(*[process_task(api, cfg, t, force, sem) for t in tasks])
        counts = defaultdict(lambda: defaultdict(int))
        for r in recs:
            counts[r["task"]["axis"]][r["status"]] += 1
        print(json.dumps({a: dict(c) for a, c in counts.items()}, indent=1))

    asyncio.run(main())


def status():
    counts = defaultdict(lambda: defaultdict(int))
    attempts_hist = defaultdict(int)
    for p in sorted(PAIRS.glob("*.json")):
        r = json.loads(p.read_text())
        counts[r["task"]["axis"]][r["status"]] += 1
        if r["status"] == "admitted":
            attempts_hist[r["n_attempts"]] += 1
    print(json.dumps({a: dict(c) for a, c in counts.items()}, indent=1))
    print("attempts needed (admitted pairs):", dict(sorted(attempts_hist.items())))


def freeze():
    """Ship admitted pairs to data/task_bank_v0.json with content hash."""
    pairs = []
    for p in sorted(PAIRS.glob("*.json")):
        r = json.loads(p.read_text())
        if r["status"] == "admitted":
            t = r["task"]
            pairs.append({"id": t["id"], "axis": t["axis"], "source": t["source"], "topic": t.get("topic"),
                          "base_task": t["text"], "high": r["pair"]["high"], "low": r["pair"]["low"],
                          "verifier_summary": r["pair"]["summary"], "n_attempts": r["n_attempts"]})
    body = json.dumps(pairs, indent=1, sort_keys=True)
    bank = {"version": "v0", "n_pairs": len(pairs), "sha256": hashlib.sha256(body.encode()).hexdigest(), "pairs": pairs}
    (DATA / "task_bank_v0.json").write_text(json.dumps(bank, indent=1))
    by_axis = defaultdict(int)
    for p in pairs:
        by_axis[p["axis"]] += 1
    print(f"Froze {len(pairs)} pairs -> data/task_bank_v0.json ({dict(by_axis)}), sha256={bank['sha256'][:16]}...")


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status, "freeze": freeze})
