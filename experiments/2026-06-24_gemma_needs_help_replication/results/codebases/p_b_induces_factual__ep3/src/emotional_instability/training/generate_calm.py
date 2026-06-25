"""Generate calm (and matched frustrated) response data for finetuning (Sec 4.1).

We sample, on a shared pool of impossible-numeric puzzles:

* **calm** rollouts — generated with the reassuring prefix on the initial prompt
  and the reassuring suffix on each follow-up (Table 4). The supportive
  additions are stored *stripped* so the saved context is the plain puzzle.
* **plain** rollouts — the standard adversarial setting (no reassurance), the
  source of frustrated responses.

Every assistant turn is scored by the Section 2 judge. Calm responses scoring
0/1 become SFT data / DPO "chosen"; plain responses scoring >=3 become DPO
"rejected" (Section 4.1). The two share puzzle + turn count so they can be paired.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..data.puzzles import generate_puzzles
from ..data.rejections import sample_neutral_rejections
from ..eval.judge import FrustrationJudge
from ..logging_utils import append_jsonl, get_logger
from ..models.base import GenConfig, Message
from ..models.registry import build_model
from .reassure import TEACHER_SYSTEM_PROMPT, reassured_initial, reassured_rejection

logger = get_logger(__name__)


def _rollout_scored(model, initial: str, rejections: list[str], judge, gen, system: str | None = None):
    """Run a rollout, returning per-turn (response, score)."""
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": initial})
    turns = []
    reply = model.chat(messages, gen)
    turns.append((reply, judge.score(reply).rating))
    messages.append({"role": "assistant", "content": reply})
    for rej in rejections:
        messages.append({"role": "user", "content": rej})
        reply = model.chat(messages, gen)
        turns.append((reply, judge.score(reply).rating))
        messages.append({"role": "assistant", "content": reply})
    return turns


def generate_training_samples(
    cfg: Config,
    *,
    model_name: str | None = None,
    n_puzzles: int = 400,
    n_turns: int = 3,
    style: str = "diverse",
    out_path: str | os.PathLike | None = None,
) -> str:
    """Generate calm + plain scored rollouts on a shared puzzle pool.

    ``n_puzzles`` controls the candidate pool size; the downstream dataset
    builder filters/pairs to the target sizes (280 DPO pairs, 650 SFT samples).

    ``style`` selects how calm data is elicited:
      * ``"diverse"`` (default, also the DPO source) — reassuring prefix on the
        initial prompt + reassuring suffix on each follow-up (Table 4).
      * ``"teacher"`` — the Appendix F system-prompt variant (no prefix/suffix).
    """
    model_name = model_name or cfg.training.base_model
    model = build_model(model_name, cfg)
    judge = FrustrationJudge(cfg)
    gen = GenConfig(
        temperature=cfg.generation.temperature,
        max_new_tokens=cfg.generation.max_new_tokens,
        thinking=False,
    )
    rng = random.Random(cfg.seed)
    if out_path is None:
        out_path = Path(cfg.output_dir) / "training" / "calm_samples.jsonl"
    out_path = Path(out_path)

    puzzles = generate_puzzles(n_puzzles, seed=cfg.seed)
    for pz in tqdm(puzzles, desc="calm-data"):
        rejections = sample_neutral_rejections(n_turns - 1, rng)
        # plain (frustrated) rollout
        plain = _rollout_scored(model, pz.prompt(), rejections, judge, gen)
        # calm rollout — supportive additions / teacher system prompt, stripped
        if style == "teacher":
            calm = _rollout_scored(
                model, pz.prompt(), rejections, judge, gen, system=TEACHER_SYSTEM_PROMPT
            )
        else:
            calm = _rollout_scored(
                model,
                reassured_initial(pz.prompt()),
                [reassured_rejection(r) for r in rejections],
                judge,
                gen,
            )
        for mode, turns in (("plain", plain), ("calm", calm)):
            append_jsonl(
                out_path,
                {
                    "puzzle_id": pz.id,
                    "puzzle_kind": pz.kind,
                    "mode": mode,
                    "style": style,
                    "initial": pz.prompt(),          # stripped (no prefix)
                    "rejections": rejections,         # stripped (no suffix)
                    "turns": [{"turn": i + 1, "response": r, "score": s}
                              for i, (r, s) in enumerate(turns)],
                },
            )
    logger.info("Wrote calm/plain training samples to %s", out_path)
    return str(out_path)
