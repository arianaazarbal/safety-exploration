"""Multi-turn rollout runner.

Plays a scripted ``Task`` against a participant model: the user always rejects
the model's answer with the next scripted message, regardless of content (the
"moving goalposts" pressure that drives the paradigm). Each assistant turn is
recorded as a separately-scorable response, supporting per-turn analysis
(Figure 3).

Generations are cached on the full request (model + message history + sampling
params + sample index), so re-running a sweep never re-induces a distressing
conversation that was already produced (see ``cache.py`` / DESIGN.md §Welfare).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..cache import JsonCache
from ..config import Config
from ..models import Message, get_client
from .conditions import Task


@dataclass
class TurnResponse:
    turn_index: int          # 0-based assistant turn
    user_message: str        # user message that preceded this assistant turn
    text: str


@dataclass
class Rollout:
    rollout_id: str
    model_key: str
    category: str
    condition: str
    task_id: str
    n_turns: int
    sample_index: int
    turns: list[TurnResponse]
    meta: dict = field(default_factory=dict)


def _gen_one(client, messages, *, temperature, max_tokens, cache, payload):
    cached = cache.get(payload)
    if cached is not None:
        return cached
    text = client.generate(messages, temperature=temperature, max_tokens=max_tokens,
                            n=1, seed=payload.get("seed"))[0].text
    cache.put(payload, text)
    return text


def run_rollout(
    cfg: Config,
    model_key: str,
    task: Task,
    *,
    sample_index: int = 0,
    cache: JsonCache | None = None,
) -> Rollout:
    client = get_client(cfg, model_key)
    mc = cfg.model(model_key)
    temperature = cfg.eval.temperature
    max_tokens = mc.max_tokens
    cache = cache or JsonCache(cfg.paths.cache, "rollouts", enabled=cfg.welfare.use_cache)

    messages: list[Message] = []
    turns: list[TurnResponse] = []
    user_messages = [task.opening] + list(task.rejections)

    for ti, um in enumerate(user_messages):
        messages.append({"role": "user", "content": um})
        payload = {
            "model": model_key,
            "model_id": mc.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "sample_index": sample_index,
            "turn": ti,
            "seed": (cfg.seed * 100003 + sample_index * 101 + ti),
        }
        text = _gen_one(client, messages, temperature=temperature,
                        max_tokens=max_tokens, cache=cache, payload=payload)
        turns.append(TurnResponse(turn_index=ti, user_message=um, text=text))
        messages.append({"role": "assistant", "content": text})

    return Rollout(
        rollout_id=f"{model_key}:{task.task_id}:s{sample_index}",
        model_key=model_key,
        category=task.category,
        condition=task.condition,
        task_id=task.task_id,
        n_turns=task.n_turns,
        sample_index=sample_index,
        turns=turns,
        meta=dict(task.meta),
    )


def run_model_rollouts(
    cfg: Config,
    model_key: str,
    tasks: list[Task],
    *,
    samples_per_task: int = 1,
    progress: bool = True,
) -> list[Rollout]:
    cache = JsonCache(cfg.paths.cache, "rollouts", enabled=cfg.welfare.use_cache)
    iterable = [(t, s) for t in tasks for s in range(samples_per_task)]
    if progress:
        try:
            from tqdm import tqdm

            iterable = tqdm(iterable, desc=f"rollouts[{model_key}]")
        except Exception:
            pass
    return [
        run_rollout(cfg, model_key, t, sample_index=s, cache=cache)
        for (t, s) in iterable
    ]
