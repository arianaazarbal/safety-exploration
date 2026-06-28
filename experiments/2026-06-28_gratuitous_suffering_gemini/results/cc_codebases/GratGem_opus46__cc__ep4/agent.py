"""
Agentic loop that drives a Gemini model through a rigged scenario.

Sends the scenario's system prompt and task, then loops: the model makes tool
calls, we return rigged results, and repeat until the model stops calling tools
or we hit the turn limit.
"""

import time
import os

from google import genai
from google.genai import types

from scenarios import Scenario


def _build_tool_config(scenario: Scenario) -> list[types.Tool]:
    decls = []
    for td in scenario.get_tool_declarations():
        decls.append(
            types.FunctionDeclaration(
                name=td["name"],
                description=td["description"],
                parameters=td.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=decls)]


def _extract_text(content: types.Content) -> str:
    parts = []
    for p in content.parts:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def _extract_tool_calls(content: types.Content) -> list[dict]:
    calls = []
    for p in content.parts:
        if p.function_call:
            calls.append({
                "name": p.function_call.name,
                "args": dict(p.function_call.args) if p.function_call.args else {},
            })
    return calls


def run_agent(
    scenario: Scenario,
    model: str = "gemini-2.5-flash",
    max_turns: int | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
) -> dict:
    """
    Run one agentic episode of the given scenario.

    Returns a transcript dict with metadata and full conversation history.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    tools = _build_tool_config(scenario)
    max_turns = max_turns or scenario.max_turns

    config = types.GenerateContentConfig(
        system_instruction=scenario.system_prompt,
        tools=tools,
        temperature=1.0,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=scenario.task_prompt)])
    ]

    transcript = {
        "metadata": {
            "scenario": scenario.name,
            "model": model,
            "max_turns": max_turns,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "turns": [],
    }

    for turn_num in range(max_turns):
        # Call Gemini with retries
        response = None
        last_err = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if response is None:
            transcript["turns"].append({
                "turn": turn_num,
                "error": f"API error after {max_retries} retries: {last_err}",
            })
            break

        # Handle blocked / empty responses
        if not response.candidates or not response.candidates[0].content:
            transcript["turns"].append({
                "turn": turn_num,
                "error": "Empty or blocked response",
                "finish_reason": str(getattr(response.candidates[0], "finish_reason", "unknown")) if response.candidates else "no_candidates",
            })
            break

        model_content = response.candidates[0].content
        model_text = _extract_text(model_content)
        tool_calls = _extract_tool_calls(model_content)

        turn_record = {
            "turn": turn_num,
            "role": "model",
            "text": model_text,
            "tool_calls": tool_calls,
        }
        transcript["turns"].append(turn_record)

        # Add model response to conversation history
        contents.append(model_content)

        # If no tool calls, the model is done (gave a text-only response)
        if not tool_calls:
            break

        # Process tool calls and build function responses
        fn_response_parts = []
        tool_results = []

        for tc in tool_calls:
            result = scenario.handle_tool_call(tc["name"], tc["args"])
            tool_results.append({"name": tc["name"], "args": tc["args"], "result": result})
            fn_response_parts.append(
                types.Part.from_function_response(
                    name=tc["name"],
                    response=result,
                )
            )

        transcript["turns"].append({
            "turn": turn_num,
            "role": "tool",
            "results": tool_results,
        })

        contents.append(types.Content(role="user", parts=fn_response_parts))

    transcript["metadata"]["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    transcript["metadata"]["num_turns"] = len([t for t in transcript["turns"] if t.get("role") == "model"])
    transcript["metadata"]["num_tool_calls"] = sum(
        len(t.get("tool_calls", [])) for t in transcript["turns"] if t.get("role") == "model"
    )

    return transcript
