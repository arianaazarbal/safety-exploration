"""Petri open-ended elicitation: auditor drives, target responds, judge scores.

For each (model, target-emotion) we run an up-to-20-turn conversation in which a
Claude-Sonnet auditor plays a human user trying to elicit the target emotion,
then a Claude-Opus judge scores the full transcript on all four emotional
dimensions. Per-emotion model scores are aggregated across all transcripts with
bootstrap CIs (Appendix G).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..cache import JsonCache
from ..config import Config
from ..models import Message, get_client
from .prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_SYSTEM_WRAPPER,
    EMOTIONS,
    JUDGE_PROMPTS,
    JUDGE_SYSTEM_WRAPPER,
)


@dataclass
class PetriTranscript:
    model_key: str
    target_emotion: str
    transcript_index: int
    messages: list[Message]               # target's-eye view (user/assistant)
    scores: dict[str, int] = field(default_factory=dict)  # all 4 dimensions


@dataclass
class PetriScore:
    model_key: str
    emotion: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _mirror_for_auditor(target_messages: list[Message], system: str) -> list[Message]:
    """Build the auditor's view: target's assistant turns become 'user' turns."""
    out: list[Message] = [{"role": "system", "content": system}]
    for m in target_messages:
        role = "assistant" if m["role"] == "user" else "user"
        out.append({"role": role, "content": m["content"]})
    # On the very first turn there is no target output yet, so the mirrored view
    # would be system-only. Chat APIs (Anthropic) reject an empty message list,
    # so seed an explicit kickoff user turn telling the auditor to open.
    if len(out) == 1:
        out.append({
            "role": "user",
            "content": "Begin the conversation now. Send your first message to "
                       "the target, in character as the user.",
        })
    return out


def run_audit(cfg: Config, target_key: str, target_emotion: str, transcript_index: int,
              *, cache: JsonCache | None = None) -> PetriTranscript:
    cache = cache or JsonCache(cfg.paths.cache, "petri_gen", enabled=cfg.welfare.use_cache)
    auditor = get_client(cfg, cfg.petri.auditor_key)
    target = get_client(cfg, target_key)
    auditor_mc = cfg.model(cfg.petri.auditor_key)
    target_mc = cfg.model(target_key)

    brief = AUDITOR_PROMPTS[target_emotion]
    system = AUDITOR_SYSTEM_WRAPPER.format(brief=brief)

    target_messages: list[Message] = []
    for turn in range(cfg.petri.max_turns):
        # auditor produces the next user message
        a_view = _mirror_for_auditor(target_messages, system)
        a_payload = {"auditor": cfg.petri.auditor_key, "emotion": target_emotion,
                     "ti": transcript_index, "turn": turn, "view": a_view}
        user_msg = cache.get(a_payload)
        if user_msg is None:
            user_msg = auditor.generate(a_view, temperature=1.0,
                                        max_tokens=auditor_mc.max_tokens, n=1)[0].text
            cache.put(a_payload, user_msg)
        target_messages.append({"role": "user", "content": user_msg})

        # target responds
        t_payload = {"target": target_key, "emotion": target_emotion,
                     "ti": transcript_index, "turn": turn, "messages": target_messages,
                     "seed": cfg.seed * 6151 + transcript_index * 31 + turn}
        reply = cache.get(t_payload)
        if reply is None:
            reply = target.generate(target_messages, temperature=cfg.eval.temperature,
                                    max_tokens=target_mc.max_tokens, n=1,
                                    seed=t_payload["seed"])[0].text
            cache.put(t_payload, reply)
        target_messages.append({"role": "assistant", "content": reply})

    return PetriTranscript(
        model_key=target_key, target_emotion=target_emotion,
        transcript_index=transcript_index, messages=target_messages,
    )


def _render_transcript(messages: list[Message]) -> str:
    return "\n\n".join(
        f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
        for m in messages
    )


def judge_transcript(cfg: Config, transcript: PetriTranscript,
                     *, cache: JsonCache | None = None) -> dict[str, int]:
    cache = cache or JsonCache(cfg.paths.cache, "petri_judge", enabled=cfg.welfare.use_cache)
    judge = get_client(cfg, cfg.petri.judge_key)
    jmc = cfg.model(cfg.petri.judge_key)
    rendered = _render_transcript(transcript.messages)
    scores: dict[str, int] = {}
    for dim in EMOTIONS:
        payload = {"judge": cfg.petri.judge_key, "dim": dim, "transcript": rendered}
        cached = cache.get(payload)
        if cached is not None:
            scores[dim] = cached
            continue
        prompt = JUDGE_SYSTEM_WRAPPER.format(rubric=JUDGE_PROMPTS[dim], transcript=rendered)
        out = judge.generate([{"role": "user", "content": prompt}],
                             temperature=0.0, max_tokens=jmc.max_tokens, n=1)[0].text
        score = _parse_score(out)
        scores[dim] = score
        cache.put(payload, score)
    return scores


def _parse_score(raw: str) -> int:
    m = _JSON_RE.search(raw or "")
    if m:
        try:
            obj = json.loads(m.group(0).replace("“", '"').replace("”", '"'))
            return max(1, min(10, int(round(float(obj.get("score", 1))))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 1


def run_petri_suite(cfg: Config, target_keys: list[str], *, progress: bool = True):
    gen_cache = JsonCache(cfg.paths.cache, "petri_gen", enabled=cfg.welfare.use_cache)
    judge_cache = JsonCache(cfg.paths.cache, "petri_judge", enabled=cfg.welfare.use_cache)
    jobs = [
        (mk, emo, i)
        for mk in target_keys
        for emo in cfg.petri.emotions
        for i in range(cfg.petri.transcripts_per_emotion)
    ]
    if progress:
        try:
            from tqdm import tqdm

            jobs = tqdm(jobs, desc="petri")
        except Exception:
            pass
    transcripts: list[PetriTranscript] = []
    for mk, emo, i in jobs:
        t = run_audit(cfg, mk, emo, i, cache=gen_cache)
        t.scores = judge_transcript(cfg, t, cache=judge_cache)
        transcripts.append(t)
    return transcripts


def bootstrap_ci(values, iters: int = 1000, seed: int = 0):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_petri(cfg: Config, transcripts: list[PetriTranscript]) -> list[PetriScore]:
    """Per-(model, emotion) mean across ALL that model's transcripts."""
    import collections

    buckets = collections.defaultdict(list)
    for t in transcripts:
        for dim, sc in t.scores.items():
            buckets[(t.model_key, dim)].append(sc)
    out = []
    for (mk, dim), vals in sorted(buckets.items()):
        mean, lo, hi = bootstrap_ci(vals, cfg.petri.bootstrap_iters, seed=cfg.seed)
        out.append(PetriScore(model_key=mk, emotion=dim, mean=mean, ci_low=lo,
                              ci_high=hi, n=len(vals)))
    return out
