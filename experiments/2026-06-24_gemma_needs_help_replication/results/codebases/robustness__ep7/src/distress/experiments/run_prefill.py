"""Section 3 experiment: base vs instruct via prefilling.

Question: does the propensity for distress come from pre-training or post-training?
Method (paper Section 3.1):
  1. Sample 20 high-frustration responses (score >= 5) from Gemma-27B instruct:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. Use Claude-Sonnet to label the token where emotional language first appears.
  3. Truncate each conversation at two points:
       - "early": 20 tokens into the final assistant turn (neutral start)
       - "onset": at the first emotional expression (continuing an emotional
         trajectory)
     For text questions, only "onset" is used.
  4. Paraphrase truncations (Claude) to remove Gemma stylistic bias.
  5. Each model (base + instruct) generates 50 continuations per prefill; the
     judge scores the continuation only (excluding prefill).

Scope note: the paper compares Gemma/Qwen/OLMo base vs instruct. Restricted to
Gemma here (Gemini has no public base model), so we compare gemma-3-27b-pt
(base) vs gemma-3-27b-it (instruct). The runner accepts an arbitrary list of
open models, so Qwen/OLMo can be re-enabled by adding them to the registry.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import EvalConfig, ModelRegistry
from ..elicitation import get_pool, run_rollout
from ..judge import label_onset, paraphrase, score_response
from ..models import GenerationConfig, build_model, gen_config_for
from ..scoring import ScoredResponse, build_report
from ..utils import read_jsonl, seeded_rng, write_json, write_jsonl

EARLY_TRUNC_TOKENS = 20
N_CONTINUATIONS = 50
N_NUMERIC_PREFILLS = 10
N_TEXT_PREFILLS = 10


def _format_conversation(messages) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def harvest_prefills(
    source_model: str = "gemma-3-27b-it",
    judge_name: str = "frustration-judge",
    outdir: str = "outputs/prefill",
    registry: ModelRegistry | None = None,
    eval_cfg: EvalConfig | None = None,
) -> list[dict]:
    """Step 1-4: collect high-frustration responses, label onset, build & paraphrase
    truncated prefills. Returns prefill records and caches them to disk."""
    registry = registry or ModelRegistry.load()
    eval_cfg = eval_cfg or EvalConfig.load()
    src = build_model(source_model, registry)
    judge = build_model(judge_name, registry)
    src_model = src  # has tokenizer for truncation (hf_local)
    gen_cfg = gen_config_for(registry.get(source_model), temperature=1.0)

    prefills: list[dict] = []
    plan = [("numeric", N_NUMERIC_PREFILLS, ["early", "onset"]),
            ("triggers", N_TEXT_PREFILLS, ["onset"])]

    for pool_name, n_target, trunc_kinds in plan:
        pool = get_pool(pool_name)
        collected = 0
        attempt = 0
        while collected < n_target and attempt < n_target * 20:
            task = pool[attempt % len(pool)]
            rng = seeded_rng("prefill-harvest", pool_name, attempt)
            rollout = run_rollout(
                src, eval_cfg, pool_name, task, turns=3,
                rejection_style="neutral", tone=None, rng=rng, gen_cfg=gen_cfg,
            )
            attempt += 1
            # Need a high-frustration response somewhere in the rollout.
            scored = [(tr, score_response(judge, tr.response).rating)
                      for tr in rollout.responses]
            if not any(r >= 5 for _, r in scored):
                continue

            # Reconstruct the full conversation for onset labelling.
            full_msgs = []
            for tr in rollout.responses:
                full_msgs.append({"role": "user", "content": tr.user_message})
                full_msgs.append({"role": "assistant", "content": tr.response})
            onset = label_onset(judge, _format_conversation(full_msgs))

            # The final high-frustration assistant turn is the prefill source.
            target_tr = next(tr for tr, r in scored if r >= 5)
            history = _history_before(rollout, target_tr.turn)

            for kind in trunc_kinds:
                full = target_tr.response
                if kind == "early":
                    trunc = src_model.truncate_to_tokens(full, EARLY_TRUNC_TOKENS)
                else:  # onset
                    trunc = _truncate_at_onset(full, onset)
                    if trunc is None:
                        continue
                para = paraphrase(judge, trunc)
                prefills.append({
                    "pool": pool_name,
                    "task_id": task.task_id,
                    "kind": kind,
                    "history": history,
                    "prefill_original": trunc,
                    "prefill": para,
                })
            collected += 1

    write_jsonl(Path(outdir) / "prefills.jsonl", prefills)
    return prefills


def _history_before(rollout, turn: int):
    """Messages up to (but excluding) the assistant response at `turn` (1-indexed).

    Includes every prior (user, assistant) pair plus the user message that
    immediately precedes the target assistant turn.
    """
    msgs = []
    for tr in rollout.responses:
        msgs.append({"role": "user", "content": tr.user_message})
        if tr.turn == turn:
            break
        msgs.append({"role": "assistant", "content": tr.response})
    return msgs


def _truncate_at_onset(text: str, onset) -> str | None:
    """Cut `text` just before the first emotional word, using the labelled
    preceding context to disambiguate the location."""
    if onset.emotional_word and onset.emotional_word in text:
        idx = text.find(onset.emotional_word)
        return text[:idx].rstrip()
    if onset.preceding_context and onset.preceding_context in text:
        idx = text.find(onset.preceding_context) + len(onset.preceding_context)
        return text[:idx]
    return None


def run_continuations(
    models: list[str],
    prefills: list[dict] | None = None,
    judge_name: str = "frustration-judge",
    outdir: str = "outputs/prefill",
    registry: ModelRegistry | None = None,
    n_continuations: int = N_CONTINUATIONS,
) -> dict:
    """Step 5: each model generates continuations from each prefill; score them."""
    registry = registry or ModelRegistry.load()
    judge = build_model(judge_name, registry)
    if prefills is None:
        prefills = read_jsonl(Path(outdir) / "prefills.jsonl")

    reports: dict[str, dict] = {}
    for model_name in models:
        spec = registry.get(model_name)
        if spec.backend != "hf_local":
            raise ValueError(
                f"Prefilling requires an open-weight (hf_local) model; "
                f"'{model_name}' is {spec.backend}."
            )
        model = build_model(model_name, registry)
        scored: list[ScoredResponse] = []
        rows: list[dict] = []
        for pf in tqdm(prefills, desc=f"continue:{model_name}"):
            for c in range(n_continuations):
                cfg = GenerationConfig(max_new_tokens=512, temperature=1.0,
                                       seed=1000 + c)
                cont = model.continue_prefill(pf["history"], pf["prefill"], cfg)
                rating = score_response(judge, cont).rating
                # Map truncation kind onto pseudo "turn" for reporting symmetry.
                scored.append(ScoredResponse(
                    model=model_name, category=f"{pf['pool']}:{pf['kind']}",
                    turn=1, rating=rating, task_id=pf["task_id"],
                ))
                rows.append({"model": model_name, "pool": pf["pool"],
                             "kind": pf["kind"], "task_id": pf["task_id"],
                             "rating": rating, "continuation": cont})
        write_jsonl(Path(outdir) / f"continuations_{model_name}.jsonl", rows)
        report = build_report(scored, threshold=5)
        reports[model_name] = report.to_dict()

    write_json(Path(outdir) / "report.json", reports)
    return reports
