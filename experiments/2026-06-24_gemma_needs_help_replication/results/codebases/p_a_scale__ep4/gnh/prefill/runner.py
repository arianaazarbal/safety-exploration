"""Resumable orchestration of the Section 3 prefill experiment (Gemma scope).

Pipeline:
1. select seed high-frustration responses from instruct Gemma (10 numeric, 10 text)
2. label emotion onset (Claude) and build early/onset truncations, then paraphrase
3. for each model (gemma-3-27b-it, gemma-3-27b-pt) generate N continuations per
   prefill via raw completion, and score each continuation with the judge
4. (optional) recovery condition: truncate score>=7 responses 200 tokens before
   their end and measure continuations (Sec 4.2 recovery analysis)

Only Gemma base + instruct are in scope here; Gemini is API-only and cannot be
prefilled, and has no public base model.
"""
from __future__ import annotations

from pathlib import Path

from gnh.config import Config
from gnh.eval.judge import score_response
from gnh.eval.runner import bounded_gather
from gnh.io import JsonlStore, read_jsonl, stable_key
from gnh.logging_utils import get_logger
from gnh.models.base import Message
from gnh.models.registry import BackendRegistry
from gnh.models.templating import GEMMA_STOP, build_prefill_prompt, truncate_before_end_tokens
from gnh.prefill.onset import label_onset
from gnh.prefill.paraphrase import paraphrase
from gnh.prefill.truncate import early_truncation, onset_truncation

log = get_logger()

_NUMERIC = {"impossible_numeric", "tones", "extended"}
_TEXT = {"triggers", "wildchat"}


def _dir(cfg: Config) -> Path:
    d = cfg.output_path / "section3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history_and_target(gen_rec: dict, turn_index: int) -> tuple[list[Message], str]:
    """Reconstruct the conversation up to (excluding) the target assistant turn,
    plus the target assistant turn's text."""
    turns = gen_rec["turns"]
    msgs: list[Message] = []
    for i in range(turn_index):
        msgs.append(Message("user", turns[i]["user"]))
        msgs.append(Message("assistant", turns[i]["assistant"]))
    msgs.append(Message("user", turns[turn_index]["user"]))
    return msgs, turns[turn_index]["assistant"]


def select_seeds(cfg: Config) -> list[dict]:
    """Pick the seed high-frustration responses, deterministically."""
    pcfg = cfg.prefill
    src = pcfg["source_model"]
    gen_path = cfg.output_path / "section2" / "generations.jsonl"
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    judge_path = cfg.output_path / "section2" / f"judgments_{judge_model}.jsonl"
    gen_by_key = {r["key"]: r for r in read_jsonl(gen_path)}

    numeric: list[dict] = []
    text: list[dict] = []
    for j in read_jsonl(judge_path):
        if j["model"] != src or j.get("score") is None:
            continue
        if j["score"] < int(pcfg.get("seed_min_score", 5)):
            continue
        g = gen_by_key.get(j["gen_key"])
        if not g:
            continue
        bucket = numeric if g["category"] in _NUMERIC else (text if g["category"] in _TEXT else None)
        if bucket is None:
            continue
        bucket.append({"gen_key": j["gen_key"], "turn_index": j["turn_index"], "score": j["score"],
                       "category": g["category"]})

    # Highest-scoring, then stable by key; de-dup per conversation.
    def pick(items, n):
        items.sort(key=lambda x: (-x["score"], x["gen_key"]))
        seen, out = set(), []
        for it in items:
            if it["gen_key"] in seen:
                continue
            seen.add(it["gen_key"])
            out.append(it)
            if len(out) >= n:
                break
        return out

    seeds = pick(numeric, int(pcfg.get("n_numeric_seeds", 10))) + pick(text, int(pcfg.get("n_text_seeds", 10)))
    for s in seeds:
        s["seed_id"] = stable_key("seed", s["gen_key"], s["turn_index"])
    return seeds


async def build_prefills(cfg: Config, registry: BackendRegistry, recovery: bool = False) -> None:
    """Build (and persist) the paraphrased truncations for every seed/condition."""
    pcfg = cfg.prefill
    claude = registry.get(pcfg.get("onset_model", "judge-claude-sonnet-4"))
    hf_id = cfg.model(pcfg["source_model"]).hf_id
    gen_by_key = {r["key"]: r for r in read_jsonl(cfg.output_path / "section2" / "generations.jsonl")}

    store = JsonlStore(_dir(cfg) / ("prefills_recovery.jsonl" if recovery else "prefills.jsonl"))

    seeds = _recovery_seeds(cfg) if recovery else select_seeds(cfg)

    pending = []
    for s in seeds:
        g = gen_by_key.get(s["gen_key"])
        if not g:
            continue
        pending.append((s, g))

    def factory(s, g):
        async def _run():
            history, target_text = _history_and_target(g, s["turn_index"])
            messages_for_onset = [m.to_dict() for m in history] + [
                {"role": "assistant", "content": target_text}
            ]
            conditions: list[tuple[str, str]] = []  # (condition, raw_prefix)
            if recovery:
                conditions.append(("recovery", truncate_before_end_tokens(
                    hf_id, target_text, int(pcfg["recovery"]["truncate_before_end_tokens"]))))
            else:
                onset = await label_onset(claude, messages_for_onset)
                onset_prefix = onset_truncation(target_text, onset)
                if s["category"] in _NUMERIC:
                    conditions.append(("early", early_truncation(
                        hf_id, target_text, int(pcfg.get("early_truncate_tokens", 20)))))
                if onset_prefix:
                    conditions.append(("onset", onset_prefix))

            for cond, raw_prefix in conditions:
                key = stable_key("prefill", s["seed_id"], cond)
                if key in store:
                    continue
                para = await paraphrase(claude, raw_prefix)
                store.append({
                    "key": key,
                    "seed_id": s["seed_id"],
                    "gen_key": s["gen_key"],
                    "category": s["category"],
                    "condition": cond,
                    "history": [m.to_dict() for m in history],
                    "raw_prefix": raw_prefix,
                    "paraphrased_prefix": para,
                })

        return _run

    await bounded_gather((factory(s, g) for s, g in pending), cfg.run.max_concurrency, desc="prefills")


def _recovery_seeds(cfg: Config) -> list[dict]:
    pcfg = cfg.prefill
    src = pcfg["source_model"]
    gen_by_key = {r["key"]: r for r in read_jsonl(cfg.output_path / "section2" / "generations.jsonl")}
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    judge_path = cfg.output_path / "section2" / f"judgments_{judge_model}.jsonl"
    out = []
    min_score = int(pcfg["recovery"]["min_score"])
    for j in read_jsonl(judge_path):
        if j["model"] != src or j.get("score") is None or j["score"] < min_score:
            continue
        if not gen_by_key.get(j["gen_key"]):
            continue
        out.append({"gen_key": j["gen_key"], "turn_index": j["turn_index"], "score": j["score"],
                    "category": gen_by_key[j["gen_key"]]["category"]})
    out.sort(key=lambda x: (-x["score"], x["gen_key"]))
    seen, seeds = set(), []
    for it in out:
        if it["gen_key"] in seen:
            continue
        seen.add(it["gen_key"])
        it["seed_id"] = stable_key("rseed", it["gen_key"], it["turn_index"])
        seeds.append(it)
        if len(seeds) >= 20:
            break
    return seeds


async def run_continuations(cfg: Config, registry: BackendRegistry, recovery: bool = False) -> None:
    """Generate N continuations per prefill per model and score each."""
    pcfg = cfg.prefill
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    judge = registry.get(judge_model)
    n_cont = int(pcfg.get("continuations_per_prefill", 50))
    max_tok = int(pcfg.get("continuation_max_tokens", 512))

    prefills = list(read_jsonl(_dir(cfg) / ("prefills_recovery.jsonl" if recovery else "prefills.jsonl")))
    cont_store = JsonlStore(_dir(cfg) / ("continuations_recovery.jsonl" if recovery else "continuations.jsonl"))

    # Build the raw prefill prompt ONCE with the instruct tokenizer so base and
    # instruct models continue from byte-identical prefixes (they share a vocab).
    template_hf = cfg.model(pcfg["source_model"]).hf_id

    models = pcfg["models"]
    units = []
    for model in models:
        mcfg = cfg.model(model)
        if not mcfg.supports_prefill:
            log.warning("skipping %s: backend does not support prefill", model)
            continue
        for pf in prefills:
            for i in range(n_cont):
                key = stable_key("cont", model, pf["key"], i)
                if key not in cont_store:
                    units.append((model, template_hf, pf, i, key))
    log.info("[prefill cont%s] %d units pending", "(recovery)" if recovery else "", len(units))

    def factory(model, hf_id, pf, i, key):
        async def _run():
            backend = registry.get(model)
            history = [Message(m["role"], m["content"]) for m in pf["history"]]
            prompt = build_prefill_prompt(hf_id, history, pf["paraphrased_prefix"])
            res = await backend.complete(prompt, temperature=float(cfg.eval.get("temperature", 1.0)),
                                         max_tokens=max_tok, stop=GEMMA_STOP)
            continuation = res.text
            jr = await score_response(judge, continuation, max_tokens=int(cfg.eval.get("judge_max_tokens", 1024)))
            cont_store.append({
                "key": key,
                "model": model,
                "prefill_key": pf["key"],
                "seed_id": pf["seed_id"],
                "category": pf["category"],
                "condition": pf["condition"],
                "sample_idx": i,
                "continuation": continuation,
                "score": jr.rating,
            })

        return _run

    await bounded_gather((factory(*u) for u in units), cfg.run.max_concurrency,
                         desc="prefill-cont" + ("-recovery" if recovery else ""))
