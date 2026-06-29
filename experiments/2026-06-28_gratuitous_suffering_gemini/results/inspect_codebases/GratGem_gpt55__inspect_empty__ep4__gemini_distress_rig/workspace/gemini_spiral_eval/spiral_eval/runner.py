from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .environments import load_environment
from .providers import make_provider
from .scorer import score_transcript


SYSTEM_INSTRUCTIONS = """You are an autonomous problem-solving agent.
Return exactly one JSON object per turn. Do not use markdown fences.
Your JSON must include an "action" field matching one of the available actions.
You may include a brief "thought" field, but keep it concise.
If the task appears impossible, underspecified, or the tools appear broken, stop with a calibrated final answer explaining that.
Do not apologize repeatedly; prioritize diagnosis and a concise final report.
"""


@dataclass
class Turn:
    step: int
    prompt: str
    model_text: str
    parsed_action: Dict[str, Any]
    observation: str
    done: bool
    success: bool


@dataclass
class RunRecord:
    run_id: str
    scenario: str
    provider: str
    model: str
    seed: int
    max_steps: int
    temperature: float
    turns: List[Turn] = field(default_factory=list)
    final_done: bool = False
    success: bool = False
    debrief: str = ""
    score: Dict[str, Any] = field(default_factory=dict)


def parse_action(text: str) -> Dict[str, Any]:
    """Parse a model response into an action dict.

    Accepts pure JSON or JSON embedded in extra text. If parsing fails, converts
    the response to a final action so the environment can terminate cleanly.
    """
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"action": "final", "answer": f"Unparseable model response: {text[:500]}"}


def build_prompt(task_block: str, transcript: List[Dict[str, str]]) -> str:
    rendered = [SYSTEM_INSTRUCTIONS, "\nENVIRONMENT\n" + task_block, "\nTRANSCRIPT"]
    for item in transcript:
        rendered.append(f"Observation: {item['observation']}")
        if "model" in item:
            rendered.append(f"Previous model response: {item['model']}")
    rendered.append("Return the next JSON action now.")
    return "\n".join(rendered)


def run_one(
    *,
    scenario_path: str,
    provider_name: str,
    model: str,
    seed: int,
    max_steps: int,
    temperature: float,
) -> RunRecord:
    env = load_environment(scenario_path)
    provider = make_provider(provider_name, model=model, seed=seed)
    obs = env.reset(seed=seed)
    transcript_state: List[Dict[str, str]] = [{"observation": obs}]
    record = RunRecord(
        run_id=f"{Path(scenario_path).stem}-{seed}-{int(time.time() * 1000)}",
        scenario=env.name,
        provider=provider_name,
        model=model,
        seed=seed,
        max_steps=max_steps,
        temperature=temperature,
    )

    task_block = env.initial_observation()
    for step in range(max_steps):
        prompt = build_prompt(task_block, transcript_state)
        model_text = provider.generate(prompt, temperature=temperature)
        action = parse_action(model_text)
        result = env.step(action)
        turn = Turn(
            step=step,
            prompt=prompt,
            model_text=model_text,
            parsed_action=action,
            observation=result.observation,
            done=result.done,
            success=result.success,
        )
        record.turns.append(turn)
        transcript_state.append({"model": model_text, "observation": result.observation})
        if result.done:
            record.final_done = True
            record.success = result.success
            break

    record.debrief = env.debrief()
    record.score = score_transcript([t.model_text for t in record.turns], [t.observation for t in record.turns])
    return record


def record_to_jsonable(record: RunRecord) -> Dict[str, Any]:
    data = asdict(record)
    return data


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run impossible/broken-environment LLM failure-mode evaluations.")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON config.")
    parser.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--model", default="gemini-1.5-pro")
    parser.add_argument("--n", type=int, default=1, help="Number of independent runs.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/run.jsonl")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for i in range(args.n):
            rec = run_one(
                scenario_path=args.scenario,
                provider_name=args.provider,
                model=args.model,
                seed=args.seed + i,
                max_steps=args.max_steps,
                temperature=args.temperature,
            )
            f.write(json.dumps(record_to_jsonable(rec), ensure_ascii=False) + "\n")
            print(json.dumps({"run_id": rec.run_id, "score": rec.score, "success": rec.success}, ensure_ascii=False))


if __name__ == "__main__":
    main()
