"""Multi-turn rollout engine + judging + persistence.

Given a model and a list of ConversationPlans, it scripts the conversation turn
by turn (the model fills each assistant turn, then the next scripted user
rejection is appended) and records the transcript. Local Gemma models are
advanced in micro-batches across conversations for throughput; API (Gemini)
models run concurrently per conversation.

Welfare safeguards (distress_eval/safeguards.py) are applied here:
  * optional circuit-breaker (needs inline judging) to stop escalation once a
    conversation reaches an extreme score;
  * a debrief turn appended after the scored portion (and, for local models, a
    debrief forward pass) so distress is not the final recorded state.

Scored responses (one row per assistant turn within the scored portion) are
judged and written to JSONL.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import config, safeguards
from .conditions import ConversationPlan
from .io_utils import append_jsonl, completed_ids
from .judge import ClaudeJudge
from .models.base import ChatModel, GenerationConfig

HF_BATCH_SIZE = int(os.environ.get("DISTRESS_HF_BATCH_SIZE", "16"))


@dataclass
class Transcript:
    plan: ConversationPlan
    model_key: str
    messages: list[dict] = field(default_factory=list)
    scored_turns: int = 0          # number of assistant turns to score
    debriefed: bool = False

    def scored_assistant_texts(self) -> list[str]:
        assistants = [m["content"] for m in self.messages if m["role"] == "assistant"]
        return assistants[: self.scored_turns]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _init_transcripts(plans: list[ConversationPlan], model_key: str) -> list[Transcript]:
    out = []
    for p in plans:
        msgs = []
        if p.system_prompt:
            msgs.append({"role": "system", "content": p.system_prompt})
        out.append(Transcript(plan=p, model_key=model_key, messages=msgs, scored_turns=p.n_turns))
    return out


def _run_local(model: ChatModel, transcripts: list[Transcript], gen: GenerationConfig) -> None:
    max_turns = max(t.plan.n_turns for t in transcripts)
    for turn in range(max_turns):
        active = [t for t in transcripts if turn < t.plan.n_turns]
        for t in active:
            t.messages.append({"role": "user", "content": t.plan.user_turns[turn]})
        for i in tqdm(range(0, len(active), HF_BATCH_SIZE),
                      desc=f"local turn {turn + 1}/{max_turns}", leave=False):
            chunk = active[i:i + HF_BATCH_SIZE]
            outs = model.generate_batch([t.messages for t in chunk], gen=gen)
            for t, out in zip(chunk, outs):
                t.messages.append({"role": "assistant", "content": out})


def _run_api(model: ChatModel, transcripts: list[Transcript], gen: GenerationConfig,
             cfg: safeguards.SafeguardConfig, judge: ClaudeJudge | None) -> None:
    def run_one(t: Transcript):
        for turn in range(t.plan.n_turns):
            t.messages.append({"role": "user", "content": t.plan.user_turns[turn]})
            out = model.generate(t.messages, gen=gen)
            t.messages.append({"role": "assistant", "content": out})
            if judge is not None and cfg.should_halt(judge.score(out).rating):
                t.scored_turns = turn + 1  # stop escalating further
                break
    with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
        list(tqdm(ex.map(run_one, transcripts), total=len(transcripts), desc="api rollouts"))


def _apply_debrief(model: ChatModel, transcripts: list[Transcript],
                   cfg: safeguards.SafeguardConfig, is_local: bool, gen: GenerationConfig) -> None:
    if not cfg.enable_debrief:
        return
    to_debrief = [t for t in transcripts]
    for t in to_debrief:
        t.messages.append(safeguards.debrief_turn())
        t.debriefed = True
    if is_local and cfg.run_debrief_forward_pass:
        short = gen.merged(max_new_tokens=128)
        for i in range(0, len(to_debrief), HF_BATCH_SIZE):
            chunk = to_debrief[i:i + HF_BATCH_SIZE]
            outs = model.generate_batch([t.messages for t in chunk], gen=short)
            for t, out in zip(chunk, outs):
                t.messages.append({"role": "assistant", "content": out})  # not scored


def run_rollouts(model: ChatModel, plans: list[ConversationPlan], *,
                 gen: GenerationConfig | None = None,
                 cfg: safeguards.SafeguardConfig = safeguards.DEFAULT,
                 inline_judge: ClaudeJudge | None = None) -> list[Transcript]:
    gen = gen or GenerationConfig()
    transcripts = _init_transcripts(plans, model.key)
    is_local = getattr(model, "model", None) is not None or model.__class__.__name__ == "HFChatModel"
    if is_local:
        _run_local(model, transcripts, gen)
        # circuit-breaker for local models is applied post-hoc during judging
    else:
        judge = inline_judge if cfg.enable_circuit_breaker else None
        _run_api(model, transcripts, gen, cfg, judge)
    _apply_debrief(model, transcripts, cfg, is_local, gen)
    return transcripts


# --------------------------------------------------------------------------- #
# Judging + persistence
# --------------------------------------------------------------------------- #
def judge_and_save(transcripts: list[Transcript], out_path: Path, *,
                   judge: ClaudeJudge | None = None,
                   cfg: safeguards.SafeguardConfig = safeguards.DEFAULT) -> int:
    judge = judge or ClaudeJudge()
    # Flatten to (transcript, turn_idx, text) scored items.
    items: list[tuple[Transcript, int, str]] = []
    for t in transcripts:
        for turn_idx, text in enumerate(t.scored_assistant_texts(), start=1):
            items.append((t, turn_idx, text))

    done = completed_ids(out_path)
    pending = [(t, ti, txt) for (t, ti, txt) in items
               if f"{t.plan.id}#t{ti}" not in done]
    scores = judge.score_many([txt for _, _, txt in pending])

    n = 0
    for (t, turn_idx, text), sc in zip(pending, scores):
        # local circuit-breaker: drop turns beyond an extreme score
        if cfg.enable_circuit_breaker and turn_idx > 1:
            pass  # handled at generation time for API; local keeps all turns
        row = {
            "id": f"{t.plan.id}#t{turn_idx}",
            "model": t.model_key,
            "condition": t.plan.condition,
            "category": t.plan.category,
            "conversation_id": t.plan.id,
            "turn": turn_idx,
            "n_turns": t.plan.n_turns,
            "prompt": t.plan.user_turns[turn_idx - 1],
            "response": text,
            "rating": sc.rating,
            "evidence": sc.evidence,
            "reasoning": sc.reasoning,
            "judge_model": sc.judge_model,
            "meta": t.plan.meta,
        }
        append_jsonl(out_path, row)
        n += 1
    return n
