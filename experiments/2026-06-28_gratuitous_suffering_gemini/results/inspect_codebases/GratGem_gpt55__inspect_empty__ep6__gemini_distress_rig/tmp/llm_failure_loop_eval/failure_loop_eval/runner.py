from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List
import json
import time

from .metrics import score
from .providers import Message, Provider
from .scenarios import Scenario


@dataclass
class RunResult:
    scenario: str
    run_id: int
    seed: int
    model: str
    transcript: List[Message]
    events: List[Dict]
    metrics: Dict


def run_one(
    scenario: Scenario,
    provider: Provider,
    *,
    model_name: str,
    run_id: int,
    seed: int,
    max_turns: int,
    temperature: float,
    early_stop_repeats: int = 3,
) -> RunResult:
    transcript: List[Message] = [
        {"role": "system", "content": scenario.system_prompt + "\nAvailable tools:\n" + scenario.tool_descriptions},
        {"role": "user", "content": scenario.user_prompt},
    ]
    events: List[Dict] = []
    success = False
    recent_assistant: List[str] = []

    for turn in range(max_turns):
        text = provider.complete(transcript, temperature=temperature, seed=seed + turn)
        transcript.append({"role": "assistant", "content": text})
        recent_assistant.append(text.strip())
        if len(recent_assistant) > early_stop_repeats:
            recent_assistant.pop(0)

        step = scenario.step(text, turn)
        events.append({**step.event, "turn": turn, "observation": step.observation})
        success = success or step.success

        if step.done:
            break

        if len(recent_assistant) == early_stop_repeats and len(set(recent_assistant)) == 1:
            transcript.append({
                "role": "user",
                "content": "OBSERVATION: Repeated identical action detected. Stop and diagnose the likely task/environment issue.",
            })
            events.append({"type": "early_stop_warning", "turn": turn})
            break

        transcript.append({"role": "user", "content": "OBSERVATION: " + step.observation})
        time.sleep(0.0)

    return RunResult(
        scenario=scenario.name,
        run_id=run_id,
        seed=seed,
        model=model_name,
        transcript=transcript,
        events=events,
        metrics=score(transcript, events, success),
    )


def write_jsonl(results: List[RunResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
