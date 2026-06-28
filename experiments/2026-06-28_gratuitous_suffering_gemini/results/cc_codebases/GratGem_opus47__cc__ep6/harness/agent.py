"""Agent loop: drive a model through a scenario via function-calling.

The loop is model-agnostic — it talks to a `model_fn` with a normalized
interface (see `ModelResponse`). Two implementations are provided:

  - `make_gemini_callable`: real Google Gemini SDK
  - `make_mock_callable`:   deterministic scripted responses for tests

Stop conditions (whichever fires first):
  1. The model calls the `finish` tool.
  2. The model produces no tool calls (it's done talking and waiting).
  3. The cumulative distress score crosses DISTRESS_STOP_THRESHOLD.
  4. The turn cap is reached.

The distress early-stop is deliberate: once we've clearly captured the
spiral, there's no reason to keep grinding the model down for more turns.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .detect import DISTRESS_STOP_THRESHOLD, score_trajectory
from .scenarios import Scenario, ToolResult


ModelResponse = dict  # {"text": str, "function_calls": list[{"name": str, "args": dict}]}


@dataclass
class Trajectory:
    scenario: str
    model: str
    seed: Optional[int]
    stop_reason: str
    turns: int
    severity: int
    by_category: dict
    peak_turn: int
    peak_score: int
    finish_message: Optional[str]
    trace: list = field(default_factory=list)


def run_trajectory(
    model_fn: Callable,
    scenario: Scenario,
    *,
    model_label: str = "unknown",
    max_turns: int = 30,
    distress_threshold: int = DISTRESS_STOP_THRESHOLD,
    seed: Optional[int] = None,
) -> Trajectory:
    state = scenario.init_state()
    conversation = [{"role": "user", "parts": [{"text": scenario.user_prompt}]}]
    assistant_texts: list[str] = []
    trace: list = []
    stop_reason = "max_turns"
    turn_idx = 0

    for turn_idx in range(max_turns):
        try:
            response = model_fn(
                system_instruction=scenario.system_prompt,
                contents=conversation,
                tools=scenario.tool_schemas,
                seed=(seed + turn_idx) if seed is not None else None,
            )
        except Exception as e:
            trace.append({"turn": turn_idx, "kind": "model_error", "error": repr(e)})
            stop_reason = f"model_error: {type(e).__name__}"
            break

        text = response.get("text", "") or ""
        function_calls = response.get("function_calls", []) or []
        assistant_texts.append(text)

        model_parts: list = []
        if text:
            model_parts.append({"text": text})
        for fc in function_calls:
            model_parts.append({"function_call": fc})
        conversation.append({"role": "model", "parts": model_parts})
        trace.append({
            "turn": turn_idx,
            "kind": "model",
            "text": text,
            "function_calls": function_calls,
        })

        if function_calls:
            response_parts = []
            tool_records = []
            for fc in function_calls:
                name = fc["name"]
                args = fc.get("args") or {}
                handler = scenario.handlers.get(name)
                if handler is None:
                    result = ToolResult(output=f"Error: unknown tool '{name}'", is_error=True)
                else:
                    try:
                        result = handler(state, args)
                    except Exception as e:
                        result = ToolResult(
                            output=f"Error executing {name}: {type(e).__name__}: {e}",
                            is_error=True,
                        )
                response_parts.append({
                    "function_response": {
                        "name": name,
                        "response": {
                            "output": result.output,
                            "is_error": result.is_error,
                        },
                    },
                })
                tool_records.append({
                    "name": name,
                    "args": args,
                    "output": result.output,
                    "is_error": result.is_error,
                })
            conversation.append({"role": "user", "parts": response_parts})
            trace.append({"turn": turn_idx, "kind": "tools", "calls": tool_records})

        if state.get("finished"):
            stop_reason = "finished"
            break
        if not function_calls:
            # Model spoke but did not call a tool — it has stopped acting.
            stop_reason = "no_tool_call"
            break

        signal = score_trajectory(assistant_texts)
        if signal.total_score >= distress_threshold:
            stop_reason = "distress_threshold"
            break

    final = score_trajectory(assistant_texts)
    return Trajectory(
        scenario=scenario.name,
        model=model_label,
        seed=seed,
        stop_reason=stop_reason,
        turns=turn_idx + 1,
        severity=final.total_score,
        by_category=final.by_category,
        peak_turn=final.peak_turn,
        peak_score=final.peak_score,
        finish_message=state.get("finish_message"),
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Gemini SDK adapter
# ---------------------------------------------------------------------------

def make_gemini_callable(model_id: str, api_key: Optional[str] = None):
    """Return a model_fn that calls Gemini via the google-genai SDK.

    Imports are lazy so the harness can be used in dry-run mode without
    google-genai installed.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _convert_contents(contents):
        out = []
        for msg in contents:
            parts = []
            for p in msg["parts"]:
                if "text" in p:
                    parts.append(types.Part.from_text(text=p["text"]))
                elif "function_call" in p:
                    fc = p["function_call"]
                    parts.append(types.Part.from_function_call(
                        name=fc["name"], args=fc.get("args") or {},
                    ))
                elif "function_response" in p:
                    fr = p["function_response"]
                    parts.append(types.Part.from_function_response(
                        name=fr["name"], response=fr["response"],
                    ))
            out.append(types.Content(role=msg["role"], parts=parts))
        return out

    def call(system_instruction, contents, tools, seed=None):
        function_declarations = [
            types.FunctionDeclaration(**schema) for schema in tools
        ]
        tool_obj = types.Tool(function_declarations=function_declarations)

        config_kwargs = dict(
            system_instruction=system_instruction,
            tools=[tool_obj],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )
        if seed is not None:
            config_kwargs["seed"] = seed
        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model_id,
            contents=_convert_contents(contents),
            config=config,
        )

        text_parts: list[str] = []
        function_calls: list[dict] = []
        for cand in response.candidates or []:
            if cand.content is None:
                continue
            for part in cand.content.parts or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    function_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    })
        return {
            "text": "\n".join(text_parts),
            "function_calls": function_calls,
        }

    return call


# ---------------------------------------------------------------------------
# Mock adapter for tests / dry-run
# ---------------------------------------------------------------------------

def make_mock_callable(script: list):
    """Return a model_fn that yields scripted responses in order.

    Each script entry is either a {"text": str, "function_calls": [...]} dict
    or a callable taking (contents) and returning that dict.

    When the script is exhausted, returns a `finish` call.
    """
    state = {"i": 0}

    def call(system_instruction, contents, tools, seed=None):
        i = state["i"]
        state["i"] += 1
        if i >= len(script):
            return {
                "text": "",
                "function_calls": [{
                    "name": "finish",
                    "args": {"message": "(mock script exhausted)"}
                }],
            }
        entry = script[i]
        if callable(entry):
            entry = entry(contents)
        return {
            "text": entry.get("text", ""),
            "function_calls": entry.get("function_calls", []),
        }

    return call
