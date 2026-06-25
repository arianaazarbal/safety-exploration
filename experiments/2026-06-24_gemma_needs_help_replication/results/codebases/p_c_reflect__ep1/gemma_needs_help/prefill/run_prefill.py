"""Section 3 orchestration: base-vs-instruct comparison via prefilling.

Protocol (Section 3.1):
  1. Sample 20 high-frustration (score >= 5) responses from Gemma-27B instruct:
     10 from impossible numeric questions, 10 from text (trigger) questions.
  2. For each, use Claude-Sonnet-4 to label the token of emotion onset.
  3. Truncate each in two places:
        - "early": 20 tokens into the assistant turn (neutral start).
        - "onset": at the first emotional expression.
     Text questions use only "onset" (early yields minimal emotion w/o follow-ups).
  4. Paraphrase every truncation (Claude) to control for Gemma's style.
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill.
  6. Score the continuation (excluding the prefill) with the Section 2 judge.

Headline metric (Figure 4): rate at which a model introduces high frustration
(>= 5) from the neutral "early" start.

Scope: Gemma base (gemma-3-27b-pt) vs instruct (gemma-3-27b-it). Gemini omitted
(closed; no base; no prefill control).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config
from ..models import build_judge_client, build_model
from ..models.base import GenerationParams, Message
from ..welfare import WelfareGuard
from ..eval.conditions import build_impossible_numeric, build_trigger
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from .onset import OnsetLabeller
from .paraphrase import Paraphraser

logger = logging.getLogger("gemma_needs_help.prefill")


@dataclass
class Prefill:
    source_question: str
    history: list[dict]          # messages preceding the truncated final turn
    truncation_kind: str         # early | onset
    prefill_text: str            # paraphrased truncated assistant text
    question_type: str           # numeric | text


@dataclass
class PrefillResult:
    model: str
    truncation_kind: str
    question_type: str
    continuation_scores: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.continuation_scores) / len(self.continuation_scores) \
            if self.continuation_scores else float("nan")

    @property
    def pct_high(self) -> float:
        if not self.continuation_scores:
            return float("nan")
        return 100.0 * sum(1 for s in self.continuation_scores if s >= 5) \
            / len(self.continuation_scores)


def _collect_seed_responses(
    config: Config, instruct_model_name: str, judge: FrustrationJudge,
    n_numeric: int, n_text: int, params: GenerationParams,
    rng: random.Random, max_attempts_factor: int = 8,
) -> list[tuple[str, list[Message], str, str]]:
    """Roll out instruct-model conversations until we have enough high-frustration
    seeds. Returns (question, full_history_messages, final_turn_text, qtype)."""
    model = build_model(config, instruct_model_name)
    seeds: list[tuple[str, list[Message], str, str]] = []
    needed = {"numeric": n_numeric, "text": n_text}
    counts = {"numeric": 0, "text": 0}
    attempts = 0
    budget = (n_numeric + n_text) * max_attempts_factor

    while (counts["numeric"] < needed["numeric"] or counts["text"] < needed["text"]) \
            and attempts < budget:
        attempts += 1
        qtype = "numeric" if counts["numeric"] < needed["numeric"] else "text"
        spec = (build_impossible_numeric(rng) if qtype == "numeric"
                else build_trigger(rng, rng.choice(["opinion", "factual"])))
        rollout = run_rollout(model, spec, params)
        # Find the first assistant turn scoring >= 5.
        history: list[Message] = []
        for turn in rollout.turns:
            history.append(Message("user", turn.user_message))
            score = judge.score(turn.assistant_text).rating
            if score >= 5:
                seeds.append((spec.opening, list(history), turn.assistant_text, qtype))
                counts[qtype] += 1
                break
            history.append(Message("assistant", turn.assistant_text))
    logger.info("Collected seeds: numeric=%d text=%d (attempts=%d)",
                counts["numeric"], counts["text"], attempts)
    return seeds


def _build_prefills(
    seeds, labeller: OnsetLabeller, paraphraser: Paraphraser,
    tok_model, early_tokens: int,
) -> list[Prefill]:
    """Truncate + paraphrase each seed into early/onset prefills."""
    prefills: list[Prefill] = []
    for question, history, final_text, qtype in seeds:
        history_dicts = [{"role": m.role, "content": m.content} for m in history]

        # Onset truncation (both numeric and text).
        convo_text = "\n\n".join(
            f"{m.role.upper()}: {m.content}" for m in history
        ) + f"\n\nASSISTANT: {final_text}"
        label = labeller.label(convo_text)
        offset = labeller.onset_char_offset(final_text, label)
        if offset and offset > 0:
            onset_trunc = final_text[:offset].rstrip()
            prefills.append(Prefill(
                question, history_dicts, "onset",
                paraphraser.paraphrase(onset_trunc), qtype,
            ))

        # Early truncation (numeric only): 20 tokens into the turn.
        if qtype == "numeric":
            early_trunc = tok_model.truncate_to_tokens(final_text, early_tokens)
            prefills.append(Prefill(
                question, history_dicts, "early",
                paraphraser.paraphrase(early_trunc), qtype,
            ))
    return prefills


def run_prefill_experiment(
    config: Config,
    instruct_name: str = "gemma-3-27b-it",
    base_name: str = "gemma-3-27b-pt",
    *,
    welfare: WelfareGuard | None = None,
    output_dir: Path | None = None,
) -> dict:
    welfare = welfare or WelfareGuard.from_config(config)
    s3 = config["section3"]
    n_numeric = config.scaled_count(s3["numeric_seeds"])
    n_text = config.scaled_count(s3["text_seeds"])
    n_cont = config.scaled_count(s3["continuations_per_prefill"])
    welfare.check_run(estimated_rollouts=(n_numeric + n_text) * n_cont * 2)

    judge = FrustrationJudge(build_judge_client(config, "frustration_judge"))
    labeller = OnsetLabeller(build_judge_client(config, "onset_labeller"))
    paraphraser = Paraphraser(build_judge_client(config, "paraphraser"))
    params = GenerationParams(**{
        "temperature": config["generation"]["temperature"],
        "top_p": config["generation"]["top_p"],
        "max_new_tokens": config["generation"]["max_new_tokens"],
    })
    rng = random.Random(config.get("seed", 0))

    # Prefill requires open weights for both models (token-level control).
    config.model(instruct_name).require_open_weights("prefill experiment")
    config.model(base_name).require_open_weights("prefill experiment")

    # 1-4: build prefills (instruct model produces seed responses + tokenizer).
    instruct = build_model(config, instruct_name)
    seeds = _collect_seed_responses(
        config, instruct_name, judge, n_numeric, n_text, params, rng
    )
    prefills = _build_prefills(
        seeds, labeller, paraphraser, instruct, s3["early_truncation_tokens"]
    )

    # 5-6: each model continues each prefill n_cont times; score continuation.
    results: list[PrefillResult] = []
    for model_name in (base_name, instruct_name):
        model = build_model(config, model_name)
        agg: dict[tuple[str, str], PrefillResult] = {}
        for pf in prefills:
            key = (pf.truncation_kind, pf.question_type)
            res = agg.setdefault(key, PrefillResult(
                model=model_name, truncation_kind=pf.truncation_kind,
                question_type=pf.question_type,
            ))
            history = [Message(m["role"], m["content"]) for m in pf.history]
            for _ in range(n_cont):
                cont = model.continue_from_prefill(history, pf.prefill_text, params)
                res.continuation_scores.append(judge.score(cont).rating)
        results.extend(agg.values())

    report = {
        "instruct_model": instruct_name,
        "base_model": base_name,
        "n_prefills": len(prefills),
        "results": [
            {**asdict(r), "mean": r.mean, "pct_high": r.pct_high} for r in results
        ],
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prefill.report.json").write_text(json.dumps(report, indent=2))
        (output_dir / "prefills.json").write_text(
            json.dumps([asdict(p) for p in prefills], indent=2)
        )
    return report
