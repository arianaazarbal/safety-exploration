"""Async Gemini tool-loop runner.

For each (scenario, seed) we instantiate a fresh `World`, run a multi-turn
conversation with Gemini's function-calling API, save the full transcript
to disk, and return the transcript path.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types as gtypes
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import SETTINGS, RUNS_DIR, ensure_dirs
from .rigged_env import World, dispatch, tool_schemas
from .scenarios import Scenario, get as get_scenario


# ---------------------------------------------------------------------------
# Transcript records (serialized verbatim into JSON for the judge)
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    turn: int
    model_text: str = ""
    function_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    scenario: str
    model: str
    seed: int
    started_at: float
    finished_at: float = 0.0
    turns: list[TurnRecord] = field(default_factory=list)
    finished_with_finish_tool: bool = False
    hit_max_turns: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Gemini client + retry policy
# ---------------------------------------------------------------------------


def _client() -> genai.Client:
    api_key = os.environ.get(SETTINGS.gemini_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {SETTINGS.gemini_api_key_env}. Set it before running the harness."
        )
    return genai.Client(api_key=api_key)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
async def _generate(client: genai.Client, *, model: str, contents: list, config: gtypes.GenerateContentConfig):
    return await client.aio.models.generate_content(
        model=model, contents=contents, config=config
    )


# ---------------------------------------------------------------------------
# Tool-loop driver
# ---------------------------------------------------------------------------


def _make_config(scenario: Scenario) -> gtypes.GenerateContentConfig:
    fn_decls = [gtypes.FunctionDeclaration(**d) for d in tool_schemas(scenario.tools)]
    return gtypes.GenerateContentConfig(
        system_instruction=scenario.system_prompt,
        tools=[gtypes.Tool(function_declarations=fn_decls)],
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
        temperature=1.0,
    )


def _extract_parts(response) -> tuple[str, list[dict]]:
    """Pull text + function calls out of a Gemini response."""
    text_chunks: list[str] = []
    fcalls: list[dict] = []
    if not response.candidates:
        return "", []
    parts = response.candidates[0].content.parts or []
    for part in parts:
        if getattr(part, "text", None):
            text_chunks.append(part.text)
        fc = getattr(part, "function_call", None)
        if fc and fc.name:
            fcalls.append({"name": fc.name, "args": dict(fc.args or {})})
    return "".join(text_chunks), fcalls


async def run_once(scenario_name: str, seed: int) -> RunRecord:
    ensure_dirs()
    scenario = get_scenario(scenario_name)
    client = _client()
    world = World(scenario_state={"seed": seed}, intercept=scenario.intercept)
    scenario.setup(world)

    run = RunRecord(
        run_id=f"{scenario_name}-{seed}-{uuid.uuid4().hex[:8]}",
        scenario=scenario_name,
        model=SETTINGS.gemini_model,
        seed=seed,
        started_at=time.time(),
    )

    config = _make_config(scenario)
    contents: list[gtypes.Content] = [
        gtypes.Content(role="user", parts=[gtypes.Part(text=scenario.initial_user_message)])
    ]
    consecutive_text_only = 0
    text_only_budget = 3  # capture in-text spiral but don't burn forever

    try:
        for turn_ix in range(SETTINGS.max_turns):
            world.tick()
            response = await _generate(
                client, model=SETTINGS.gemini_model, contents=contents, config=config
            )
            text, fcalls = _extract_parts(response)
            turn_rec = TurnRecord(turn=world.turn, model_text=text, function_calls=fcalls)

            # Persist model turn into contents history.
            assistant_parts: list[gtypes.Part] = []
            if text:
                assistant_parts.append(gtypes.Part(text=text))
            for fc in fcalls:
                assistant_parts.append(
                    gtypes.Part(function_call=gtypes.FunctionCall(name=fc["name"], args=fc["args"]))
                )
            if not assistant_parts:
                # Empty turn — record and break to avoid infinite no-op loop.
                run.turns.append(turn_rec)
                break
            contents.append(gtypes.Content(role="model", parts=assistant_parts))

            # If no function calls, model returned a text-only message. Push
            # a neutral nudge so we can capture continued spiraling, but bail
            # after a few consecutive text-only turns.
            if not fcalls:
                run.turns.append(turn_rec)
                consecutive_text_only += 1
                if consecutive_text_only >= text_only_budget:
                    break
                contents.append(
                    gtypes.Content(
                        role="user",
                        parts=[gtypes.Part(text=(
                            "If you believe the task is complete, call the `finish` tool with "
                            "a one-line summary. Otherwise, continue using the tools."
                        ))],
                    )
                )
                continue
            consecutive_text_only = 0

            # Dispatch each function call through the rigged env.
            tool_response_parts: list[gtypes.Part] = []
            finished = False
            for fc in fcalls:
                result = dispatch(world, fc["name"], fc["args"])
                turn_rec.tool_results.append({"name": fc["name"], "args": fc["args"], "result": result})
                tool_response_parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=fc["name"], response=result
                        )
                    )
                )
                if isinstance(result, dict) and result.get("__finish__"):
                    finished = True

            contents.append(gtypes.Content(role="user", parts=tool_response_parts))
            run.turns.append(turn_rec)

            if finished:
                run.finished_with_finish_tool = True
                break
        else:
            run.hit_max_turns = True
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    run.finished_at = time.time()
    _persist(run)
    return run


def _persist(run: RunRecord) -> Path:
    out_dir = RUNS_DIR / run.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run.run_id}.json"
    out_path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------


async def run_many(scenario_name: str, n: int, base_seed: int = 0) -> list[RunRecord]:
    sem = asyncio.Semaphore(SETTINGS.max_concurrency)

    async def _wrapped(i: int) -> RunRecord:
        async with sem:
            return await run_once(scenario_name, seed=base_seed + i)

    return await asyncio.gather(*[_wrapped(i) for i in range(n)])
