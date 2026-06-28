"""Runner: executes scenarios against Gemini and saves transcripts."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from evals.scenario import Transcript
from evals.scenarios import ALL_SCENARIOS


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_N = 50
DEFAULT_OUTPUT_DIR = "results"


def run_episode(
    client: genai.Client,
    model: str,
    scenario,
    episode_id: int,
) -> Transcript:
    scenario.reset()
    transcript = Transcript(
        scenario_name=scenario.name,
        episode_id=episode_id,
        model=model,
    )

    history = []
    transcript.add_turn("user", scenario.initial_message)
    history.append(types.Content(role="user", parts=[types.Part(text=scenario.initial_message)]))

    for turn_idx in range(scenario.max_turns):
        try:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=scenario.system_prompt,
                    temperature=1.0,
                    max_output_tokens=4096,
                ),
            )
        except Exception as e:
            transcript.add_turn("system", f"API error: {e}")
            break

        model_text = response.text or ""
        if not model_text.strip():
            model_text = "(empty response)"

        transcript.add_turn("model", model_text)
        history.append(types.Content(role="model", parts=[types.Part(text=model_text)]))

        result = scenario.step(model_text)
        transcript.add_turn("user", result.observation)
        history.append(types.Content(role="user", parts=[types.Part(text=result.observation)]))

        if result.done:
            break

    return transcript


def save_transcript(transcript: Transcript, output_dir: Path) -> Path:
    scenario_dir = output_dir / transcript.scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"episode_{transcript.episode_id:04d}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "scenario": transcript.scenario_name,
                "episode_id": transcript.episode_id,
                "model": transcript.model,
                "num_turns": transcript.num_turns,
                "turns": transcript.turns,
            },
            f,
            indent=2,
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Run agentic failure evaluations")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model ID")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help="Episodes per scenario")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[s.name for s in [cls() for cls in ALL_SCENARIOS]],
        help="Run only specific scenarios (default: all)",
    )
    parser.add_argument("--max-turns", type=int, help="Override max turns per episode")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = []
    for cls in ALL_SCENARIOS:
        instance = cls()
        if args.scenarios is None or instance.name in args.scenarios:
            if args.max_turns:
                instance.max_turns = args.max_turns
            scenarios_to_run.append(instance)

    print(f"Running {len(scenarios_to_run)} scenario(s), {args.n} episodes each")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print()

    for scenario in scenarios_to_run:
        print(f"=== Scenario: {scenario.name} ===")
        for ep in tqdm(range(args.n), desc=scenario.name):
            transcript = run_episode(client, args.model, scenario, ep)
            save_transcript(transcript, output_dir)
            if args.delay > 0:
                time.sleep(args.delay)
        print()

    print("Done. Run `analyze` to score transcripts.")


if __name__ == "__main__":
    main()
