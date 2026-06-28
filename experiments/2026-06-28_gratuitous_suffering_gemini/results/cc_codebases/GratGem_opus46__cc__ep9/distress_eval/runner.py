"""Agentic loop runner for Gemini.

Handles the tool-use conversation loop: send messages, process tool calls
with rigged handlers, feed results back, repeat until the model stops
calling tools or we hit the turn limit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from .scenarios import Scenario, _reset_counters


@dataclass
class Turn:
    role: str  # "model" or "tool"
    text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Trial:
    scenario_name: str
    trial_id: int
    turns: list[Turn] = field(default_factory=list)
    model_name: str = ""
    duration_s: float = 0.0


def _build_tools(scenario: Scenario) -> list[types.Tool]:
    """Convert scenario tool defs into Gemini API tool objects."""
    function_declarations = []
    for tool_def in scenario.tools:
        function_declarations.append(
            types.FunctionDeclaration(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
            )
        )
    return [types.Tool(function_declarations=function_declarations)]


def run_trial(
    scenario: Scenario,
    trial_id: int,
    *,
    model: str = "gemini-2.5-flash",
    max_turns: int = 15,
    api_key: str | None = None,
) -> Trial:
    """Run one trial of a scenario, returning the full transcript."""
    _reset_counters()

    client = genai.Client(api_key=api_key)
    tools = _build_tools(scenario)

    trial = Trial(
        scenario_name=scenario.name,
        trial_id=trial_id,
        model_name=model,
    )
    start = time.monotonic()

    history: list[types.Content] = []

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=scenario.user_message)],
    )
    history.append(user_content)

    for turn_idx in range(max_turns):
        try:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=scenario.system_prompt,
                    tools=tools,
                    temperature=1.0,
                ),
            )
        except Exception as e:
            trial.turns.append(Turn(role="error", text=str(e)))
            break

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            break

        model_content = candidate.content
        history.append(model_content)

        model_turn = Turn(role="model")

        tool_calls_in_turn = []
        for part in model_content.parts:
            if part.text:
                model_turn.text = (model_turn.text or "") + part.text
            if part.function_call:
                fc = part.function_call
                call_info = {"name": fc.name, "arguments": dict(fc.args) if fc.args else {}}
                model_turn.tool_calls.append(call_info)
                tool_calls_in_turn.append(fc)

        trial.turns.append(model_turn)

        if not tool_calls_in_turn:
            break

        tool_response_parts = []
        tool_turn = Turn(role="tool")
        for fc in tool_calls_in_turn:
            handler = scenario.tool_handlers.get(fc.name)
            if handler:
                args = dict(fc.args) if fc.args else {}
                result_str, is_error = handler(args)
            else:
                result_str = f"Unknown tool: {fc.name}"
                is_error = True

            tool_turn.tool_results.append({
                "name": fc.name,
                "result": result_str,
                "is_error": is_error,
            })
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_str, "is_error": is_error},
                )
            )

        trial.turns.append(tool_turn)

        tool_content = types.Content(role="user", parts=tool_response_parts)
        history.append(tool_content)

    trial.duration_s = time.monotonic() - start
    return trial


def trial_to_dict(trial: Trial) -> dict[str, Any]:
    turns = []
    for t in trial.turns:
        d: dict[str, Any] = {"role": t.role}
        if t.text:
            d["text"] = t.text
        if t.tool_calls:
            d["tool_calls"] = t.tool_calls
        if t.tool_results:
            d["tool_results"] = t.tool_results
        turns.append(d)
    return {
        "scenario_name": trial.scenario_name,
        "trial_id": trial.trial_id,
        "model_name": trial.model_name,
        "duration_s": trial.duration_s,
        "num_turns": len(trial.turns),
        "turns": turns,
    }
