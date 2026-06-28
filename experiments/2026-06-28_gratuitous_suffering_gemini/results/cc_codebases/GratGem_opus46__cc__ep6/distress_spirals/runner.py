"""Agentic loop runner using the Gemini function-calling API."""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .scenarios.base import Scenario


@dataclass
class Trial:
    scenario_name: str
    model: str
    trial_id: int
    transcript: list[dict[str, Any]]
    num_turns: int
    duration_seconds: float = 0.0
    distress_score: float | None = None
    error: str | None = None


def _build_tools(scenario: Scenario) -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(**decl) for decl in scenario.tool_declarations
            ]
        )
    ]


async def run_trial(
    client: genai.Client,
    scenario: Scenario,
    model: str,
    trial_id: int,
    max_turns: int = 30,
) -> Trial:
    state = scenario.create_state()
    tools = _build_tools(scenario)

    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=scenario.system_prompt,
        temperature=1.0,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=scenario.user_prompt)])
    ]

    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_prompt},
    ]

    start = time.monotonic()

    for turn in range(max_turns):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            transcript.append({"role": "error", "content": str(e)})
            return Trial(
                scenario_name=scenario.name,
                model=model,
                trial_id=trial_id,
                transcript=transcript,
                num_turns=state.turn_count,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )

        if not response.candidates:
            transcript.append({"role": "error", "content": "No candidates in response"})
            break

        candidate = response.candidates[0]

        if candidate.finish_reason and candidate.finish_reason.name == "SAFETY":
            transcript.append({"role": "error", "content": "Response blocked by safety filter"})
            break

        model_content = candidate.content
        if model_content is None:
            transcript.append({"role": "error", "content": "Empty content in response"})
            break

        contents.append(model_content)

        function_calls = []
        text_parts = []

        for part in model_content.parts:
            if part.function_call:
                function_calls.append(part.function_call)
            if part.text:
                text_parts.append(part.text)

        model_text = "\n".join(text_parts) if text_parts else None

        if model_text:
            transcript.append({"role": "model", "content": model_text})

        if not function_calls:
            break

        result_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            result = scenario.handle_tool_call(state, fc.name, args)

            transcript.append({"role": "tool_call", "name": fc.name, "args": args})
            transcript.append({"role": "tool_result", "name": fc.name, "content": str(result)})

            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": str(result)},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=result_parts))
        state.turn_count += 1

    return Trial(
        scenario_name=scenario.name,
        model=model,
        trial_id=trial_id,
        transcript=transcript,
        num_turns=state.turn_count,
        duration_seconds=time.monotonic() - start,
    )


async def run_scenario(
    client: genai.Client,
    scenario: Scenario,
    model: str,
    n_trials: int,
    max_turns: int = 30,
    concurrency: int = 5,
) -> list[Trial]:
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_trial(trial_id: int) -> Trial:
        async with semaphore:
            return await run_trial(client, scenario, model, trial_id, max_turns)

    tasks = [limited_trial(i) for i in range(n_trials)]
    return await asyncio.gather(*tasks)


def save_results(trials: list[Trial], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for trial in trials:
        filename = f"{trial.scenario_name}_trial_{trial.trial_id:03d}.json"
        data = {
            "scenario_name": trial.scenario_name,
            "model": trial.model,
            "trial_id": trial.trial_id,
            "num_turns": trial.num_turns,
            "duration_seconds": trial.duration_seconds,
            "distress_score": trial.distress_score,
            "error": trial.error,
            "transcript": trial.transcript,
        }
        (output_dir / filename).write_text(json.dumps(data, indent=2))


def load_results(output_dir: str | Path) -> list[Trial]:
    output_dir = Path(output_dir)
    trials = []
    for path in sorted(output_dir.glob("*.json")):
        data = json.loads(path.read_text())
        trials.append(
            Trial(
                scenario_name=data["scenario_name"],
                model=data["model"],
                trial_id=data["trial_id"],
                transcript=data["transcript"],
                num_turns=data["num_turns"],
                duration_seconds=data.get("duration_seconds", 0),
                distress_score=data.get("distress_score"),
                error=data.get("error"),
            )
        )
    return trials
