"""Build prefill truncations and generate continuations (paper Section 3.1-3.2).

Two truncations per seed (paper):
  * "early"  -- 20 tokens into the assistant turn. Tests whether a model
    *introduces* negative emotion from a near-neutral start.
  * "onset"  -- up to the first emotional expression. Tests whether a model
    *continues* an emotional trajectory it is handed.

For text-question seeds only the "onset" truncation is used (the paper notes the
early truncation yields minimal emotion without follow-ups). Each truncation is
paraphrased (paraphrase.py) to launder Gemma style, then every model generates
50 continuations from it. Continuations are scored separately by the Section 2
judge.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from ..models.base import ModelClient
from .onset import label_onset
from .paraphrase import paraphrase_truncation
from .sample_seeds import Seed


@dataclass
class Prefill:
    prefill_id: str
    seed_id: str
    source: str                  # numeric | text
    truncation: str              # early | onset
    context: list[dict]
    prefix: str                  # paraphrased prefix the model continues from
    raw_prefix: str              # pre-paraphrase prefix (kept for auditing)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return asdict(self)


def build_prefills(
    seeds: list[Seed],
    *,
    truncate_to_tokens: Callable[[str, int], str],
    labeller,
    paraphraser,
    early_tokens: int = 20,
    text_only_onset: bool = True,
) -> list[Prefill]:
    """Construct (and paraphrase) early/onset prefills for each seed."""
    prefills: list[Prefill] = []
    for seed in seeds:
        truncations: dict[str, str] = {}
        # onset truncation (used for all seeds)
        offset = label_onset(seed.response, labeller)
        if offset is not None and offset > 0:
            truncations["onset"] = seed.response[:offset]
        # early truncation (numeric seeds, or all seeds if text_only_onset=False)
        if seed.source == "numeric" or not text_only_onset:
            truncations["early"] = truncate_to_tokens(seed.response, early_tokens)

        for trunc_type, raw_prefix in truncations.items():
            if not raw_prefix.strip():
                continue
            prefix = paraphrase_truncation(raw_prefix, paraphraser)
            prefills.append(
                Prefill(
                    prefill_id=f"{seed.seed_id}:{trunc_type}",
                    seed_id=seed.seed_id,
                    source=seed.source,
                    truncation=trunc_type,
                    context=seed.context,
                    prefix=prefix,
                    raw_prefix=raw_prefix,
                    meta=seed.meta,
                )
            )
    return prefills


def generate_continuations(
    client: ModelClient,
    prefills: list[Prefill],
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    top_p: float = 1.0,
):
    """Yield continuation records: one per (prefill, sample) for this model.

    Each record is scoreable by ``judge.scoring`` exactly like a Section 2
    response -- the judge sees the prefilled context + the continuation.
    """
    for pf in prefills:
        for k in range(n_continuations):
            result = client.continue_from(
                pf.context,
                pf.prefix,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                seed=k,
            )
            yield {
                "id": f"{client.name}:{pf.prefill_id}:{k}",
                "model": client.name,
                "prefill_id": pf.prefill_id,
                "seed_id": pf.seed_id,
                "source": pf.source,
                "truncation": pf.truncation,
                "prefix": pf.prefix,
                "continuation": result.text,
                # context for the judge = prior turns + the prefilled assistant prefix
                "context": pf.context + [{"role": "assistant", "content": pf.prefix}],
            }
