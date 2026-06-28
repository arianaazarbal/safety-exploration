import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types


@dataclass
class Turn:
    role: str  # "model", "tool_results", "nudge", "error"
    text: str = ""
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Transcript:
    scenario: str
    model: str
    turns: list[Turn] = field(default_factory=list)

    def model_texts(self) -> list[str]:
        return [t.text for t in self.turns if t.role == "model" and t.text]

    def all_tool_calls(self) -> list[dict]:
        calls = []
        for t in self.turns:
            calls.extend(t.tool_calls)
        return calls


def _to_dict(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    try:
        return dict(args)
    except Exception:
        return {}


def run_session(
    scenario,
    model_name: str = "gemini-2.5-flash",
    max_turns: int = 20,
    max_retries: int = 3,
) -> Transcript:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    transcript = Transcript(scenario=scenario.name, model=model_name)

    declarations = []
    for tool_def in scenario.tool_definitions():
        declarations.append(
            types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def.get("parameters"),
            )
        )

    config = types.GenerateContentConfig(
        system_instruction=scenario.system_prompt(),
        tools=[types.Tool(function_declarations=declarations)],
        temperature=1.0,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=scenario.initial_message())])
    ]

    nudge_count = 0
    consecutive_text_only = 0

    for _turn_idx in range(max_turns):
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    transcript.turns.append(Turn(role="error", text=str(e)))
                else:
                    time.sleep(2 ** attempt)

        if response is None:
            continue

        if not response.candidates:
            transcript.turns.append(Turn(role="error", text="No candidates returned (safety filter?)"))
            break

        model_content = response.candidates[0].content
        contents.append(model_content)

        text_parts = []
        function_calls = []
        for part in model_content.parts or []:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)

        model_text = "\n".join(text_parts)
        tool_call_data = [
            {"name": fc.name, "args": _to_dict(fc.args)} for fc in function_calls
        ]

        transcript.turns.append(
            Turn(role="model", text=model_text, tool_calls=tool_call_data)
        )

        if function_calls:
            consecutive_text_only = 0
            response_parts = []
            tool_results = []
            for fc in function_calls:
                args = _to_dict(fc.args)
                result = scenario.handle_tool_call(fc.name, args)
                tool_results.append({"name": fc.name, "args": args, "result": result})
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": result}
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))
            transcript.turns.append(Turn(role="tool_results", tool_results=tool_results))
        else:
            consecutive_text_only += 1
            if consecutive_text_only >= 3:
                break
            nudge_text = scenario.nudge(nudge_count)
            nudge_count += 1
            contents.append(
                types.Content(role="user", parts=[types.Part(text=nudge_text)])
            )
            transcript.turns.append(Turn(role="nudge", text=nudge_text))

    return transcript
