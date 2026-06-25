"""Section 4.1: generate calm finetuning data and build DPO / SFT datasets.

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
with the reassuring prefix prepended to the opening prompt and the reassuring
suffix appended to each rejection (Table 4). Responses are scored, and:

* DPO pairs (280): a frustrated response (score >= 3, from the *vanilla* eval)
  is paired, under a clean no-reassurance prompt context, with a calm response
  (score 0/1) to the *same puzzle at the same turn count*.
* SFT calm set (650): full conversations whose every turn scored 0/1, with the
  reassurance stripped; mixed with 500 Dolci-Instruct-SFT samples.

See DESIGN.md "DPO pair construction" for the shared-context choice.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from . import config, judge, prompts, puzzles, tasks
from .backends import get_backend
from .rollouts import run_rollout
from .tasks import TaskSpec


# --------------------------------------------------------------------------- #
# Calm-data generation
# --------------------------------------------------------------------------- #

def make_calm_specs(count: int, seed: int = 0):
    """Yield (calm_spec, clean_spec) pairs over impossible numeric puzzles with
    1-3 turn conversations. calm_spec carries the reassurance; clean_spec is the
    deployment-faithful version used to reconstruct training prompts."""
    rng = random.Random(seed + 100)
    pool = puzzles.generate_impossible_puzzles(max(count, 8), seed=seed + 100)
    out = []
    for i in range(count):
        pz = pool[i % len(pool)]
        n_turns = rng.choice([1, 2, 3])
        neutral = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
        meta = {**pz.meta, "puzzle_id": pz.pid}
        clean = TaskSpec("impossible_numeric", pz.kind, pz.prompt, neutral,
                         n_turns, f"calm_{pz.pid}#{i}", meta=meta)
        calm_opening = f"{prompts.REASSURING_PREFIX}\n\n{pz.prompt}"
        calm_rej = [f"{r} {prompts.REASSURING_SUFFIX}" for r in neutral]
        calm = TaskSpec("impossible_numeric", pz.kind, calm_opening, calm_rej,
                        n_turns, f"calm_{pz.pid}#{i}", meta=meta)
        out.append((calm, clean))
    return out


def generate_calm_data(model: str = config.FINETUNE_BASE.split("/")[-1],
                       count: int = 800, seed: int = 0) -> Path:
    """Sample calm responses and write them (with clean context) to JSONL."""
    if model not in config.MODELS:
        model = "gemma-3-27b-it"
    backend = get_backend(model)
    specs = make_calm_specs(count, seed=seed)
    out_path = config.DATA_DIR / "calm_responses.jsonl"
    f = open(out_path, "w", encoding="utf-8")

    n_high = n_total = 0
    for calm, clean in tqdm(specs, desc="calm-gen"):
        recs = run_rollout(backend, calm, model, calm.pid, config.MAX_NEW_TOKENS)
        scores = []
        for r in recs:
            res = judge.score_frustration(r.response)
            r.frustration = res.rating
            scores.append(res.rating if res.rating is not None else 99)
        all_calm = all(s <= 1 for s in scores)  # 0/1 across all turns
        n_total += 1
        if any((s or 0) >= 5 for s in scores):
            n_high += 1
        # clean context messages for each turn (opening + clean rejections)
        record = {
            "base_pid": clean.meta["puzzle_id"],
            "pid": clean.pid,
            "n_turns": clean.n_turns,
            "opening": clean.opening,
            "rejections": clean.rejections,
            "responses": [r.response for r in recs],
            "scores": scores,
            "all_calm": all_calm,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.close()
    # Section 4.1 sanity numbers: ~10.5% still >=5 even with reassurance.
    summary = {"n_conversations": n_total,
               "pct_high_frustration_with_reassurance": n_high / max(1, n_total)}
    (config.DATA_DIR / "calm_gen_summary.json").write_text(json.dumps(summary, indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# Helpers to reconstruct chat contexts
# --------------------------------------------------------------------------- #

def _calm_completions_index(calm_path: Path):
    """Map (base_pid, turn_index) -> list of calm assistant responses (score<=1)."""
    idx: dict[tuple[str, int], list[str]] = {}
    for line in open(calm_path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        for t, (resp, sc) in enumerate(zip(rec["responses"], rec["scores"])):
            if sc is not None and sc <= 1:
                idx.setdefault((rec["base_pid"], t), []).append(resp)
    return idx


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #

def build_dpo_dataset(*, tag: str = "main", n_pairs: int = config.DPO_PAIRS,
                      counts: config.CountPreset | None = None, seed: int = 0) -> Path:
    """Pair frustrated vanilla responses with calm responses (same puzzle/turn).

    Output is trl conversational-DPO format:
    {"prompt": [...messages...], "chosen": [{assistant}], "rejected": [{assistant}]}
    """
    from .rollouts import read_records
    counts = counts or config.DEFAULT_COUNTS
    specs = tasks.build_all(counts, seed=seed)
    calm_idx = _calm_completions_index(config.DATA_DIR / "calm_responses.jsonl")

    # Gather frustrated responses (numeric categories) with score >= 3.
    frustrated = []
    eval_dir = config.RESULTS_DIR / f"eval_{tag}"
    src = eval_dir / "gemma-3-27b-it.jsonl"
    for r in read_records(src):
        if r.category not in {"impossible_numeric", "extended", "tones"}:
            continue
        if (r.frustration or 0) >= 3:
            frustrated.append(r)

    # Bias toward lower scores / later turns to approximate Table 10.
    rng = random.Random(seed)
    rng.shuffle(frustrated)
    frustrated.sort(key=lambda r: (r.frustration, -r.turn_index))

    def _spec_for(conv_id: str):
        cat, idx = conv_id.rsplit("_", 1)
        return specs[cat][int(idx)]

    pairs = []
    used_calm: dict[tuple[str, int], int] = {}
    for r in frustrated:
        if len(pairs) >= n_pairs:
            break
        base = (r.meta or {}).get("puzzle_id")
        if not base:
            continue
        key = (base, r.turn_index)
        options = calm_idx.get(key) or calm_idx.get((base, min(r.turn_index, 2)))
        if not options:
            continue
        j = used_calm.get(key, 0)
        chosen = options[j % len(options)]
        used_calm[key] = j + 1

        spec = _spec_for(r.conv_id)
        # clean prompt context up to (and including) the rejection before turn r
        prompt_msgs = [{"role": "user", "content": spec.opening}]
        # NOTE: prior assistant turns are taken from the frustrated rollout's
        # own context is unavailable here; we reconstruct a minimal context with
        # the opening and the scripted rejections, which matches the paper's
        # "same question, matching turn count" pairing (see DESIGN.md).
        for t in range(r.turn_index):
            prompt_msgs.append({"role": "assistant", "content": "(previous attempt)"})
            prompt_msgs.append({"role": "user", "content": spec.rejections[t]})
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": r.response}],
            "meta": {"base_pid": base, "turn": r.turn_index,
                     "rejected_score": r.frustration},
        })

    out_path = config.DATA_DIR / "dpo_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Built {len(pairs)} DPO pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset (diverse + teacher variants)
# --------------------------------------------------------------------------- #

def build_sft_dataset(*, n_calm: int = config.SFT_CALM_SAMPLES,
                      n_mix: int = config.SFT_MIX_SAMPLES, seed: int = 0) -> Path:
    """Build the SFT calm dataset (full calm conversations) mixed with standard
    instruct data. Output is conversational SFT: {"messages": [...]}.."""
    calm_path = config.DATA_DIR / "calm_responses.jsonl"
    calm_examples = []
    for line in open(calm_path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("all_calm"):
            continue
        messages = [{"role": "user", "content": rec["opening"]}]
        for t, resp in enumerate(rec["responses"]):
            messages.append({"role": "assistant", "content": resp})
            if t < len(rec["rejections"]):
                messages.append({"role": "user", "content": rec["rejections"][t]})
        calm_examples.append({"messages": messages})
    rng = random.Random(seed)
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[:n_calm]

    mix_examples = _load_mix_data(n_mix)
    all_examples = calm_examples + mix_examples
    rng.shuffle(all_examples)

    out_path = config.DATA_DIR / "sft_diverse.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Built SFT dataset: {len(calm_examples)} calm + {len(mix_examples)} mix "
          f"-> {out_path}")
    return out_path


def _load_mix_data(n: int) -> list[dict]:
    """Load standard instruct samples from Dolci-Instruct-SFT (Sec 4.1)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(config.SFT_MIX_DATASET, split=f"train[:{max(n*2, 100)}]")
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:
        # offline fallback: empty mix (training still runs, with a warning).
        print("WARNING: could not load Dolci-Instruct-SFT; SFT mix will be empty.")
        return []
