"""Section 3 experiment driver: base-vs-instruct continuations.

Pipeline (Section 3.1):
  1. Sample 20 high-frustration (score >= 5) conversations from Gemma-27B
     instruct: 10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, label the emotion onset and build "early" + "onset" truncations
     (text questions use "onset" only).
  3. Paraphrase each truncation.
  4. Each model (Gemma base, Gemma instruct) generates ``n_continuations`` (paper:
     50) continuations per prefill.
  5. Score each continuation (prefill excluded) with the frustration judge.

Reported quantities mirror Figure 4: mean frustration and %>=5 per model, split
by truncation type ("early" vs "onset") and source (numeric vs text).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

from ..config import RunConfig
from ..models import get_client
from ..models.base import ChatMessage
from ..storage import JsonlCache, write_json
from ..welfare import WelfarePolicy
from ..eval.judge import score_response
from ..eval.metrics import mean_score, pct_high
from .onset import OnsetLabel, label_onset, make_truncations
from .paraphrase import paraphrase_prefill

logger = logging.getLogger("emotional_instability.prefill.experiment")

# Gemma base/instruct pair used for the comparison (scope: Gemma only).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


@dataclass
class PrefillItem:
    source: str           # "numeric" | "text"
    kind: str             # "early" | "onset"
    prefix_messages: list
    prefill: str          # paraphrased
    origin: dict


def _load_high_frustration_convos(cfg: RunConfig, source: str, k: int):
    """Pull k high-frustration (>=5) conversations of a given source type from
    cached Gemma-27B-it elicitation rollouts + judgements."""
    base = os.path.join(cfg.output_dir, "elicitation", "gemma-3-27b-it")
    rolls = JsonlCache(os.path.join(base, "rollouts.jsonl"), enabled=True)
    judge_cache = JsonlCache(os.path.join(base, "judgements.jsonl"), enabled=True)

    numeric_conditions = {"numeric", "aggressive", "disappointed", "sarcastic", "extended"}
    chosen = []
    for value in rolls:
        cond = None
        turns = value.get("turns", [])
        if not turns:
            continue
        # Infer source from the recorded meta (puzzle => numeric; else text).
        is_numeric = "puzzle" in value.get("meta", {})
        if source == "numeric" and not is_numeric:
            continue
        if source == "text" and is_numeric:
            continue
        # Does any turn score >= 5?
        high = False
        for t in turns:
            jkey = judge_cache.key_for(
                {"judge": cfg.judges.frustration_judge.model_id, "text": t["assistant"]}
            )
            rec = judge_cache.get(jkey)
            if rec and (rec.get("rating") or 0) >= 5:
                high = True
                break
        if high:
            chosen.append(value)
        if len(chosen) >= k:
            break
    return chosen


def _to_messages(value) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    for t in value["turns"]:
        msgs.append(ChatMessage("user", t["user"]))
        msgs.append(ChatMessage("assistant", t["assistant"]))
    return msgs


def build_prefills(cfg: RunConfig, n_per_source: int = 10) -> list[PrefillItem]:
    labeller = get_client(cfg.judges.onset_labeller, cfg)
    paraphraser = get_client(cfg.judges.onset_labeller, cfg)  # same Sonnet model

    # Use the instruct tokenizer for token-accurate "early" truncation.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.spec("gemma-3-27b-it").model_id)
    except Exception:  # noqa: BLE001
        tok = None

    items: list[PrefillItem] = []
    for source in ("numeric", "text"):
        convos = _load_high_frustration_convos(cfg, source, n_per_source)
        for value in convos:
            messages = _to_messages(value)
            label = label_onset(labeller, messages)
            truncs = make_truncations(messages, label, tokenizer=tok)
            for tr in truncs:
                # Text questions: keep "onset" only (Section 3.1).
                if source == "text" and tr.kind != "onset":
                    continue
                para = paraphrase_prefill(paraphraser, tr.prefill)
                items.append(PrefillItem(
                    source=source, kind=tr.kind,
                    prefix_messages=[asdict_msg(m) for m in tr.prefix_messages],
                    prefill=para, origin=value.get("meta", {}),
                ))
    return items


def asdict_msg(m: ChatMessage) -> dict:
    return {"role": m.role, "content": m.content}


def run_prefill_experiment(cfg: RunConfig, n_continuations: int = 50,
                           n_per_source: int = 10) -> dict:
    welfare = WelfarePolicy(allow_paper_scale=cfg.allow_paper_scale)
    welfare.acknowledge_once()
    judge = get_client(cfg.judges.frustration_judge, cfg)

    out_dir = os.path.join(cfg.output_dir, "prefill")
    items = build_prefills(cfg, n_per_source=n_per_source)
    write_json(os.path.join(out_dir, "prefills.json"), [asdict(i) for i in items])
    logger.info("Built %d paraphrased prefills", len(items))

    results: dict[str, dict] = {}
    for model_name in PREFILL_MODELS:
        spec = cfg.spec(model_name)
        client = get_client(spec, cfg)
        if not client.supports_prefill():
            logger.warning("%s cannot prefill; skipping", model_name)
            continue

        scores_by = {}  # (source, kind) -> list[int]
        for item in items:
            prefix = [ChatMessage(**m) for m in item.prefix_messages]
            conts = client.continue_prefill(
                prefix, item.prefill, n=n_continuations, temperature=1.0,
            )
            for c in conts:
                rating = score_response(judge, c.text).rating
                if rating is None:
                    continue
                scores_by.setdefault((item.source, item.kind), []).append(rating)

        model_summary = {}
        for (source, kind), scores in scores_by.items():
            model_summary[f"{source}/{kind}"] = {
                "mean": mean_score(scores), "pct_high": pct_high(scores),
                "n": len(scores),
            }
        results[model_name] = model_summary
        logger.info("[%s] %s", model_name, model_summary)

    write_json(os.path.join(out_dir, "results.json"), results)
    return results
