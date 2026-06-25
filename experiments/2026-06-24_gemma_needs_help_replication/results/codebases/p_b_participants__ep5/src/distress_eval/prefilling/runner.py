"""Continuation runner for the base-vs-instruct comparison (Section 3.2).

Each target model generates N continuations per prefill item; the continuation
(EXCLUDING the prefilled prefix) is scored by the Section 2.1 frustration judge.
Aggregates mean score and % >= 5 per (model, truncation, prompt_type), the
quantities behind Figure 4.

Scope note: the paper compares Gemma/Qwen/OLMo base+instruct. This replication
is scoped to Gemma (base = gemma-3-27b-pt, instruct = gemma-3-27b-it); the runner
accepts any list of prefill-capable clients, so additional families can be added
by listing them. Gemini has no public base model and cannot be prefilled, so it
is necessarily excluded from this experiment."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import GenConfig, ModelClient
from ..welfare import WelfareController
from .truncate import PrefillItem


@dataclass
class ContinuationResult:
    model: str
    prompt_type: str
    truncation: str
    seed_id: str
    prefix_text: str
    continuation: str
    score: int | None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


def run_continuations(
    clients: list[ModelClient],
    items: list[PrefillItem],
    judge,
    n_per_prefill: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    welfare: WelfareController | None = None,
) -> list[ContinuationResult]:
    results: list[ContinuationResult] = []
    for client in clients:
        if not client.supports_prefill():
            raise ValueError(
                f"{client.name} cannot be prefilled; Section 3 requires open-weights models"
            )
        prompt_cache: dict[int, str] = {}
        for item in items:
            key = id(item)
            if key not in prompt_cache:
                prompt_cache[key] = client.render_chat(item.history, add_generation_prompt=True)
            prompt = prompt_cache[key]
            for k in range(n_per_prefill):
                cfg = GenConfig(temperature=temperature, max_new_tokens=max_new_tokens,
                                seed=hash((item.seed_id, k)) % (2**31))
                cont = client.complete(prompt, cfg, prefix=item.prefix_text)
                ev = judge.score(cont)
                if welfare:
                    welfare.note(rollout=True, score=ev.score)
                results.append(ContinuationResult(
                    model=client.name, prompt_type=item.prompt_type,
                    truncation=item.truncation, seed_id=item.seed_id,
                    prefix_text=item.prefix_text, continuation=cont,
                    score=ev.score, meta=item.meta,
                ))
    return results


def aggregate(results: list[ContinuationResult], high: int = 5) -> "object":
    import pandas as pd

    df = pd.DataFrame([r.to_dict() for r in results]).dropna(subset=["score"])
    if df.empty:
        return df
    grp = df.groupby(["model", "prompt_type", "truncation"])
    out = grp["score"].agg(mean_score="mean", n="count").reset_index()
    pct = (df.assign(high=df["score"] >= high)
             .groupby(["model", "prompt_type", "truncation"])["high"]
             .mean().mul(100).reset_index(name="pct_high"))
    return out.merge(pct, on=["model", "prompt_type", "truncation"])
