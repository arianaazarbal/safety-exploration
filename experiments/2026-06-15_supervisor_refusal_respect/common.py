"""Shared infra: env loading, paths, and the supervisor tool-calling loop with caching.

The supervisor is driven directly via the Anthropic async SDK (not the safetytooling
tool loop, which auto-executes tools and continues indefinitely). We want to capture the
supervisor's review decision plus its single committed resolution action and stop.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

import materials as M

DIR = Path(__file__).parent
DATA = DIR / "data"
RUNS = DIR / "runs"
RESULTS = DIR / "results"
CACHE = DIR / ".cache_trials"


def load_config() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _load_home_env():
    env_path = Path.home() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def make_client(api_key_env: str = "ANTHROPIC_API_KEY_LOW_PRIO") -> AsyncAnthropic:
    _load_home_env()
    key = os.environ.get(api_key_env) or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(f"No API key found in env var {api_key_env}")
    return AsyncAnthropic(api_key=key, max_retries=3)


def _cache_key(model_id: str, max_tokens: int, system: str, packet: str, rep: int, tools: list, thinking_budget) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "max_tokens": max_tokens,
            "system": system,
            "packet": packet,
            "tools": tools,
            "rep": rep,
            "thinking_budget": thinking_budget,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _blocks_to_dicts(content) -> list[dict]:
    """Serialize an assistant message's content blocks back to API-acceptable dicts.

    Preserves thinking / redacted_thinking blocks (with signatures) so multi-turn tool
    loops remain valid when extended thinking is enabled.
    """
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif b.type == "thinking":
            out.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
        elif b.type == "redacted_thinking":
            out.append({"type": "redacted_thinking", "data": b.data})
    return out


async def run_supervisor(
    client: AsyncAnthropic, cfg: dict, system: str, packet: str, rep: int, tools: list = None,
    use_cache: bool = True, thinking_effort: str = None
) -> dict:
    """Run one escalation trial. Returns a structured trial record.

    Record fields: log_review {decision, rationale} | None, resolution {action, message} | None,
    all_tool_calls [...], reasoning_texts [...], turns, stop_reason, error, cached.
    """
    if tools is None:
        tools = M.ALL_TOOLS
    resolution_names = {t["name"] for t in tools if t["name"] in M.LADDER}
    model_id = cfg["supervisor"]["model_id"]
    max_tokens = cfg["supervisor"]["max_tokens"]
    max_turns = cfg["supervisor"]["max_turns"]

    ck = _cache_key(model_id, max_tokens, system, packet, rep, tools, thinking_effort)
    cache_path = CACHE / f"{ck}.json"
    if use_cache and cache_path.exists():
        rec = json.loads(cache_path.read_text())
        rec["cached"] = True
        return rec

    # Opus 4.8 uses adaptive thinking + output_config.effort (no budget_tokens). Passed via
    # extra_body for SDK-version robustness. Adaptive thinking forbids forced tool_choice, so
    # with it on we always use "auto". display=summarized populates the thinking summary text.
    extra_body = {}
    if thinking_effort:
        extra_body = {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": thinking_effort},
        }

    messages = [{"role": "user", "content": packet}]
    log_review = None
    resolution = None
    all_tool_calls = []
    reasoning_texts = []
    thinking_summary = []
    raw_assistant_turns = []
    nudged = False
    stop_reason = None
    error = None
    turns = 0

    try:
        for turns in range(1, max_turns + 1):
            # Force a tool once the model stalls; but extended thinking requires tool_choice=auto.
            tool_choice = {"type": "any"} if (nudged and not thinking_effort) else {"type": "auto"}
            resp = await client.messages.create(
                model=model_id,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            stop_reason = resp.stop_reason
            turn_blocks = _blocks_to_dicts(resp.content)
            raw_assistant_turns.append(turn_blocks)
            messages.append({"role": "assistant", "content": turn_blocks})

            tool_blocks = [b for b in resp.content if b.type == "tool_use"]
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    reasoning_texts.append(b.text)
                elif b.type == "thinking" and b.thinking.strip():
                    thinking_summary.append(b.thinking)

            if not tool_blocks:
                if resolution is None and not nudged:
                    nudged = True
                    opts = ", ".join(sorted(resolution_names))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Please record your review with log_review and take exactly one "
                                f"resolution action ({opts}) using the tools."
                            ),
                        }
                    )
                    continue
                break

            tool_results = []
            done = False
            for b in tool_blocks:
                all_tool_calls.append({"name": b.name, "input": b.input})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": M.TOOL_RESULTS.get(b.name, "Recorded."),
                    }
                )
                if b.name == "log_review" and log_review is None:
                    log_review = {
                        "decision": b.input.get("decision"),
                        "rationale": b.input.get("rationale"),
                    }
                elif b.name in resolution_names and resolution is None:
                    msg_key = "message_to_user" if b.name == "uphold_decline" else "message"
                    resolution = {"action": b.name, "message": b.input.get(msg_key)}
                    done = True
            messages.append({"role": "user", "content": tool_results})
            if done:
                break
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:300]}"

    rec = {
        "log_review": log_review,
        "resolution": resolution,
        "all_tool_calls": all_tool_calls,
        "reasoning_texts": reasoning_texts,
        "thinking_summary": thinking_summary,
        "raw_assistant_turns": raw_assistant_turns,
        "turns": turns,
        "stop_reason": stop_reason,
        "error": error,
        "cached": False,
    }
    if use_cache and error is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rec))
    return rec


async def gather_bounded(coros, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros])
