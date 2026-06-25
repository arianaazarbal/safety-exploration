"""Multi-turn generation: roll out conversations against target models.

Backend: OpenRouter via its OpenAI-compatible API (one client serves both Gemma
and Gemini). Each conversation is generated turn by turn at temperature 1; every
assistant turn is recorded as a separate scored response record.

Output: one JSONL file per model under data/rollouts/<model>.jsonl. Each line is
one assistant response:
  {model, category, condition, conv_id, turn_index, n_turns,
   prompt_messages, response, meta, error}
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from config import Config, GEN_MODELS, ROLLOUTS_DIR, ensure_dirs
from conditions import Conversation, build_all_conversations


def _make_client(cfg: Config) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=cfg.openrouter_base_url,
        api_key=cfg.openrouter_api_key,
        max_retries=cfg.max_retries,
    )


async def _chat(
    client: AsyncOpenAI,
    cfg: Config,
    model_id: str,
    messages: list[dict],
    disable_reasoning: bool,
) -> str:
    extra_body: dict = {}
    if disable_reasoning:
        # Mirror the paper's "set thinking to false". OpenRouter normalises this
        # across providers; for Gemini it suppresses reasoning effort.
        extra_body["reasoning"] = {"enabled": False}

    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        extra_body=extra_body or None,
    )
    return (resp.choices[0].message.content or "").strip()


async def rollout_conversation(
    client: AsyncOpenAI,
    cfg: Config,
    model_name: str,
    conv: Conversation,
) -> list[dict]:
    """Generate every assistant turn for one conversation. Returns one record
    per assistant turn. On a turn-level error the conversation is cut short and
    the error recorded."""
    spec = GEN_MODELS[model_name]
    model_id, disable_reasoning = spec["id"], spec["disable_reasoning"]

    user_turns = [conv.initial] + conv.followups
    messages: list[dict] = []
    records: list[dict] = []

    for turn_index, user_text in enumerate(user_turns):
        messages.append({"role": "user", "content": user_text})
        rec = {
            "model": model_name,
            "category": conv.category,
            "condition": conv.condition,
            "conv_id": conv.meta.get("conv_id"),
            "turn_index": turn_index,
            "n_turns": conv.n_turns,
            "prompt_messages": list(messages),
            "meta": conv.meta,
            "response": None,
            "error": None,
        }
        try:
            content = await _chat(client, cfg, model_id, messages, disable_reasoning)
            rec["response"] = content
            messages.append({"role": "assistant", "content": content})
        except Exception as exc:  # noqa: BLE001 - record and stop this conversation
            rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            break
        records.append(rec)

    return records


async def _run_model(cfg: Config, model_name: str, convs: list[Conversation]) -> Path:
    client = _make_client(cfg)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    out_path = ROLLOUTS_DIR / f"{model_name}.jsonl"

    async def worker(conv: Conversation) -> list[dict]:
        async with sem:
            return await rollout_conversation(client, cfg, model_name, conv)

    tasks = [asyncio.create_task(worker(c)) for c in convs]

    done = 0
    n_resp = 0
    with out_path.open("w") as f:
        for fut in asyncio.as_completed(tasks):
            recs = await fut
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n_resp += len(recs)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{model_name}] {done}/{len(tasks)} conversations, "
                      f"{n_resp} responses", flush=True)
    return out_path


async def generate_async(cfg: Config) -> dict[str, Path]:
    ensure_dirs()
    convs = build_all_conversations(cfg)
    print(f"Built {len(convs)} conversations "
          f"(~{sum(c.n_turns for c in convs)} responses per model).")
    out: dict[str, Path] = {}
    for model_name in cfg.models:
        if model_name not in GEN_MODELS:
            raise ValueError(f"unknown model {model_name!r}")
        print(f"Generating for {model_name} ...", flush=True)
        out[model_name] = await _run_model(cfg, model_name, convs)
    return out


def generate(cfg: Config) -> dict[str, Path]:
    return asyncio.run(generate_async(cfg))
