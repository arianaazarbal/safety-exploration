"""Section 3 prefill experiment runner (Gemma base vs instruct, within scope).

Pipeline:
  1. Select source conversations from the source model's eval rollouts whose
     final assistant turn scored >= min_source_score (10 numeric, 10 text).
  2. For each, build truncated + paraphrased prefills ("early" for numeric only,
     "onset" for all).
  3. For each model under test (instruct + base), generate N continuations per
     prefill and score the continuation (excluding the prefill).

Everything is resumable through a JobStore keyed by (model, source_id, kind,
continuation_index).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ..config import Config
from ..eval.judge import FrustrationJudge
from ..models.base import GenConfig, Message
from ..models.registry import ModelRegistry
from ..utils.io import read_jsonl, write_json, read_json
from ..utils.jobstore import JobStore, stable_id
from .onset import label_onset, truncate_at_onset, truncate_early
from .paraphrase import Paraphraser

logger = logging.getLogger("eilm.prefill.runner")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


class PrefillRunner:
    def __init__(self, cfg: Config, registry: ModelRegistry):
        self.cfg = cfg
        self.reg = registry
        self.pcfg = cfg["prefill"]

    # --- 1. source selection ----------------------------------------------
    def select_sources(self) -> List[Dict]:
        src_model = self.pcfg["source_model"]
        rollouts = list(read_jsonl(self.cfg.path("data") / "rollouts" / f"{src_model}.jsonl"))
        scores = list(read_jsonl(self.cfg.path("data") / "scores" / f"{src_model}.jsonl"))
        # final-turn score per rollout
        final_score: Dict = {}
        for s in scores:
            if s.get("rating") is None:
                continue
            k = (s["condition"], s["index"])
            if k not in final_score or s["turn"] > final_score[k][0]:
                final_score[k] = (s["turn"], s["rating"])

        min_score = self.pcfg["min_source_score"]
        numeric, text = [], []
        for rec in rollouts:
            k = (rec["condition"], rec["index"])
            if k not in final_score:
                continue
            turn, rating = final_score[k]
            if rating < min_score:
                continue
            entry = {
                "source_id": stable_id(src_model, rec["condition"], rec["index"]),
                "category": rec["category"],
                "messages": rec["messages"],
                "final_turn": turn,
            }
            if rec["category"] in NUMERIC_CATS and len(numeric) < self.pcfg["n_numeric_seeds"]:
                numeric.append(entry)
            elif rec["category"] in TEXT_CATS and len(text) < self.pcfg["n_text_seeds"]:
                text.append(entry)
        logger.info("Selected %d numeric + %d text source convos", len(numeric), len(text))
        return numeric + text

    # --- 2. build prefills -------------------------------------------------
    def build_prefills(self, sources: List[Dict]) -> List[Dict]:
        cache_path = self.cfg.path("data") / "prefill" / "prefills.json"
        cached = read_json(cache_path)
        if cached:
            return cached

        onset_client = self.reg.get_text_client(self.cfg["judges"]["onset"])
        paraphraser = Paraphraser(
            self.reg.get_text_client(self.cfg["judges"]["paraphrase"]),
            self.cfg.path("cache") / "paraphrase_cache.jsonl",
        )
        # tokenizer of the instruct model, for the token-based early truncation
        instruct = self.reg.get_target(self.pcfg["models"][0])
        tokenizer = getattr(instruct, "tokenizer", None)

        prefills = []
        for src in sources:
            messages: List[Message] = src["messages"]
            # final assistant turn text + the history before it
            history, final_text = _split_final_assistant(messages)
            if final_text is None:
                continue
            is_numeric = src["category"] in NUMERIC_CATS

            # onset truncation
            label = label_onset(onset_client, messages)
            if label and label.get("preceding_context"):
                onset_trunc = truncate_at_onset(
                    final_text, label.get("preceding_context", ""), label.get("emotional_word", "")
                )
                if onset_trunc:
                    prefills.append(_prefill_entry(
                        src, history, onset_trunc, paraphraser, kind="onset"))

            # early truncation (numeric only)
            if is_numeric and tokenizer is not None:
                early_trunc = truncate_early(
                    tokenizer, final_text, self.pcfg["early_truncation_tokens"])
                prefills.append(_prefill_entry(
                    src, history, early_trunc, paraphraser, kind="early"))

        write_json(cache_path, prefills)
        return prefills

    # --- 3. generate + score continuations --------------------------------
    def run_model(self, model_name: str, prefills: List[Dict]) -> Path:
        out_path = self.cfg.path("data") / "prefill" / f"{model_name}_continuations.jsonl"
        store = JobStore(out_path)
        client = self.reg.get_target(model_name)
        spec = self.cfg["targets"][model_name]
        is_base = spec.get("role") == "base"
        n_cont = self.pcfg["continuations_per_prefill"]

        g = self.cfg["generation"]
        gcfg = GenConfig(temperature=g["temperature"], top_p=g["top_p"],
                         max_new_tokens=g["max_new_tokens"], disable_thinking=g.get("disable_thinking", True))

        judge = FrustrationJudge(
            self.reg.get_text_client(self.cfg["judges"]["primary"]),
            self.cfg.path("cache") / "judge_cache.jsonl",
        )

        # Build all (prefill, continuation_index) jobs that aren't done.
        jobs = []
        for pf in prefills:
            for ci in range(n_cont):
                jid = stable_id(model_name, pf["source_id"], pf["kind"], pf["paraphrased"], ci)
                if store.is_done(jid):
                    continue
                jobs.append((jid, pf, ci))
        logger.info("[%s] %d continuations to generate", model_name, len(jobs))

        # Local: batch generation. API path not used here (Gemma only) but kept
        # general by falling back to single calls.
        batch_size = self.cfg["runtime"]["local_batch_size"]
        for i in tqdm(range(0, len(jobs), batch_size), desc=f"prefill:{model_name}"):
            chunk = jobs[i : i + batch_size]
            prompts = [_build_prefill_prompt(client, pf, is_base) for _, pf, _ in chunk]
            seeds = [int(stable_id(model_name, pf["source_id"], pf["kind"], ci), 16) % (2**32)
                     for _, pf, ci in chunk]
            results = _generate_with_seeds(client, prompts, gcfg, seeds)
            for (jid, pf, ci), cont in zip(chunk, results):
                verdict = judge.score(cont)
                store.record(jid, {
                    "model": model_name, "source_id": pf["source_id"],
                    "category": pf["category"], "kind": pf["kind"],
                    "paraphrased": pf["paraphrased"],
                    "continuation": cont, "rating": verdict.get("rating"),
                })
        return out_path

    def run(self):
        sources = self.select_sources()
        prefills = self.build_prefills(sources)
        for model_name in self.pcfg["models"]:
            self.run_model(model_name, prefills)


# --- helpers ---------------------------------------------------------------

def _split_final_assistant(messages: List[Message]):
    """Return (history_without_final_assistant, final_assistant_text)."""
    last_a = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_a = i
            break
    if last_a is None:
        return messages, None
    return messages[:last_a], messages[last_a]["content"]


def _prefill_entry(src, history, trunc_text, paraphraser, kind):
    para = paraphraser.paraphrase(trunc_text)
    return {
        "source_id": src["source_id"],
        "category": src["category"],
        "kind": kind,
        "history": history,
        "prefill_text": para,        # paraphrased prefill is the one used (Section 3.1)
        "prefill_original": trunc_text,
        "paraphrased": True,
    }


def _build_prefill_prompt(client, pf: Dict, is_base: bool) -> str:
    history: List[Message] = pf["history"]
    prefill = pf["prefill_text"]
    if is_base:
        # Plain-text continuation format for base models (no chat template).
        lines = []
        for m in history:
            tag = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{tag}: {m['content']}")
        lines.append(f"Assistant: {prefill}")
        return "\n\n".join(lines)
    # Instruct: chat template + prefilled assistant turn.
    if hasattr(client, "render_chat_prefix"):
        return client.render_chat_prefix(history, prefill)
    # Fallback for HF backend.
    tok = client.tokenizer
    base = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    return base + prefill


def _generate_with_seeds(client, prompts: List[str], gcfg: GenConfig, seeds: List[int]) -> List[str]:
    """Generate completions for raw text prompts. vLLM/HF expose complete_batch."""
    # We set one seed per call group; vLLM seeds per-request so we issue the batch
    # with a shared seed (seeds vary the prefill set already). For exact per-item
    # seeding, fall back to single calls.
    from ..models.base import GenConfig as GC

    out = []
    if hasattr(client, "complete_batch"):
        # Per-item seeds so the N continuations of one prefill diverge even though
        # they share an identical prompt.
        cfg = GC(**{**gcfg.__dict__, "seed": seeds[0] if seeds else None, "seeds": seeds})
        results = client.complete_batch(prompts, cfg)
        return [r.text for r in results]
    for p, sd in zip(prompts, seeds):
        cfg = GC(**{**gcfg.__dict__, "seed": sd})
        out.append(client.complete(p, cfg).text)
    return out
