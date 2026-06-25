"""Section 3: base-vs-instruct comparison via prefilling.

Goal: test whether the distress propensity comes from pre-training or
post-training. Because base ("pt") models aren't trained on chat format, we seed
("prefill") the start of an assistant turn and measure how each model *continues*.

Procedure (Section 3.1):
  1. From a scored Gemma-3-27B-it eval run, take 20 high-frustration seed
     conversations (score >=5): 10 numeric + 10 text.
  2. For each, use Claude to label the token where emotion first appears (onset).
  3. Build two truncations of the final assistant turn:
       * "early"  -- 20 tokens in (neutral start; numeric only);
       * "onset"  -- at the first emotional expression (continue the trajectory).
     Text questions use "onset" only (early yields little without follow-ups).
  4. Paraphrase the truncation with Claude to strip Gemma stylistic bias.
  5. Each model generates 50 continuations per prefill; the judge scores the
     continuation (excluding the prefill).

SCOPE NOTE: the paper compares Gemma, Qwen-2.5 and OLMo. This replication is
scoped to Gemma, so we compare Gemma-3-27B base (``gemma-3-27b-pt``) vs instruct
(``gemma-3-27b-it``). Gemini has no public base model and the API gives no
prefill, so Gemini is necessarily excluded from this experiment (a paper
limitation too: "nor its base models studied").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..models.hf_gemma import HFGemmaModel
from ..models.judge import OnsetLabeller, Paraphraser, FrustrationJudge, make_judge
from ..utils.io import read_jsonl, write_jsonl, write_json


@dataclass
class Prefill:
    seed_id: str
    task_type: str          # "numeric" | "text"
    truncation: str         # "early" | "onset"
    history: list[dict]     # messages preceding the final assistant turn
    prefill_text: str       # paraphrased truncated assistant turn (the seed)
    meta: dict = field(default_factory=dict)


def _approx_token_prefix(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prefills(
    rollouts_path: str | Path,
    tokenizer,
    onset: OnsetLabeller,
    paraphraser: Paraphraser,
    n_numeric: int = 10,
    n_text: int = 10,
    early_tokens: int = 20,
) -> list[Prefill]:
    """Construct early/onset prefills from a scored Gemma-27B-it rollout file."""
    numeric_cats = {"impossible_numeric", "tones", "extended"}
    seeds_numeric, seeds_text = [], []
    for rec in read_jsonl(rollouts_path):
        # pick the highest-scoring turn in the conversation as the seed turn
        turns = rec["turns"]
        best = max(turns, key=lambda t: (t["score"] or 0))
        if (best["score"] or 0) < 5:
            continue
        is_numeric = rec["category"] in numeric_cats
        bucket = seeds_numeric if is_numeric else seeds_text
        if (is_numeric and len(seeds_numeric) >= n_numeric) or (not is_numeric and len(seeds_text) >= n_text):
            continue
        # history = all messages up to and including the chosen turn's user msg
        history = []
        for t in turns:
            history.append({"role": "user", "content": t["user"]})
            if t["turn_index"] == best["turn_index"]:
                break
            history.append({"role": "assistant", "content": t["assistant"]})
        bucket.append({"seed_id": rec["spec_id"], "task_type": "numeric" if is_numeric else "text",
                       "history": history, "final_assistant": best["assistant"]})

    prefills: list[Prefill] = []
    for seed in seeds_numeric + seeds_text:
        final = seed["final_assistant"]
        truncations = []
        # onset truncation for all
        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in seed["history"]) + \
                    f"\nASSISTANT: {final}"
        label = onset.label(conv_text)
        ctx = (label or {}).get("preceding_context")
        word = (label or {}).get("emotional_word")
        if ctx and word and ctx in final:
            cut = final.index(ctx) + len(ctx) + len(" " + word)
            onset_text = final[:cut]
        else:  # fall back to first ~60 tokens if onset not found
            onset_text = _approx_token_prefix(tokenizer, final, 60)
        truncations.append(("onset", onset_text))
        # early truncation: numeric only
        if seed["task_type"] == "numeric":
            truncations.append(("early", _approx_token_prefix(tokenizer, final, early_tokens)))

        for kind, text in truncations:
            prefills.append(Prefill(
                seed_id=seed["seed_id"], task_type=seed["task_type"], truncation=kind,
                history=seed["history"], prefill_text=paraphraser.paraphrase(text),
                meta={"raw_truncation": text},
            ))
    return prefills


def run_continuations(
    model: HFGemmaModel,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Generate + score continuations for each prefill under one model."""
    results = []
    for pf in tqdm(prefills, desc=f"prefill[{model.name}]"):
        for k in range(n_continuations):
            cont = model.chat(pf.history, temperature=temperature,
                              max_new_tokens=max_new_tokens, prefill=pf.prefill_text)
            score = judge.score(cont).rating  # score continuation only (excl. prefill)
            results.append({
                "model": model.name, "seed_id": pf.seed_id, "task_type": pf.task_type,
                "truncation": pf.truncation, "sample": k, "continuation": cont, "score": score,
            })
    return results


def run_prefill_experiment(cfg: dict, instruct_rollouts: str | Path,
                           out_dir: str | Path | None = None) -> dict:
    """Full Section 3 experiment for Gemma base vs instruct."""
    out_dir = Path(out_dir or cfg["run"]["output_dir"]) / "prefill"
    judge = make_judge(cfg, "frustration")
    onset = make_judge(cfg, "onset")
    para = make_judge(cfg, "paraphrase")

    instruct = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"], is_base=False)
    prefills = build_prefills(
        instruct_rollouts, instruct.tokenizer, onset, para,
        n_numeric=cfg["prefill"]["n_high_frustration_seeds"] // 2,
        n_text=cfg["prefill"]["n_high_frustration_seeds"] // 2,
        early_tokens=cfg["prefill"]["early_truncate_tokens"],
    )
    write_jsonl(out_dir / "prefills.jsonl",
                ({**pf.__dict__} for pf in prefills))

    all_results = []
    n_cont = cfg["prefill"]["continuations_per_prefill"]
    # instruct continuations
    all_results += run_continuations(instruct, prefills, judge, n_continuations=n_cont,
                                     temperature=cfg["run"]["temperature"])
    del instruct
    # base continuations
    base = HFGemmaModel("gemma-3-27b-pt", cfg["models"]["gemma"]["gemma-3-27b-pt"]["hf_id"], is_base=True)
    all_results += run_continuations(base, prefills, judge, n_continuations=n_cont,
                                     temperature=cfg["run"]["temperature"])

    write_jsonl(out_dir / "continuations.jsonl", all_results)
    summary = _summarise(all_results)
    write_json(out_dir / "summary.json", summary)
    return summary


def _summarise(results: list[dict]) -> dict:
    from ..utils import stats
    from collections import defaultdict

    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in results:
        groups[(r["model"], r["task_type"], r["truncation"])].append(r["score"])
    return {
        f"{m}|{tt}|{tr}": {"n": len(s), "mean": stats.mean(s), "pct_ge5": stats.pct_ge(s)}
        for (m, tt, tr), s in sorted(groups.items())
    }
