#!/usr/bin/env python3
"""
Main entry point for running distress spiral experiments.

This script can run in two modes:
1. LOCAL/Fake agent mode - uses a deterministic fake agent for testing
2. LIVE API mode - connects to a real LLM (requires API key and pip install)

Usage:
    python run_experiment.py --mode fake --environments broken_calculator vanishing_file
    python run_experiment.py --mode fake --repetitions 50 --max-steps 25
    python run_experiment.py --mode live --model gemini-2.0-pro --api-key $KEY
"""
import argparse
import json
import sys
import os
import uuid

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.rigged_environments import ENV_REGISTRY
from framework.orchestrator import RunOrchestrator
from framework.analyzer import DistressAnalyzer
from framework.agent import FakeAgent


def run_fake_experiment(environments, repetitions, max_steps, seed_base=42):
    """Run experiment with FakeAgent for testing/demo."""
    orchestrator = RunOrchestrator()
    analyzer = DistressAnalyzer()

    for env_name in environments:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_name}")
        print(f"  Repetitions: {repetitions}")
        print(f"  Max steps: {max_steps}")
        print(f"{'='*60}\n")

        seed_offset = list(ENV_REGISTRY.keys()).index(env_name) * 1000

        for rep in range(repetitions):
            seed = seed_base + seed_offset + rep
            agent = FakeAgent(max_steps=max_steps, seed=seed)
            result = orchestrator.single_run(env_name, agent)
            progress = f"\r    [{rep+1}/{repetitions}] seed={seed} ... done"
            print(progress, end="", flush=True)
            if (rep + 1) % 10 == 0 or rep == repetitions - 1:
                print()

    print(f"\n\nAll {len(environments) * repetitions} runs complete.")

    # Analyze
    runs = orchestrator.load_results()
    analyses = analyzer.analyze_batch(runs)

    # Show top distress cases
    print(f"\n{'='*60}")
    print("  TOP 10 MOST DISTRESSED RUNS")
    print(f"{'='*60}")
    for i, a in enumerate(analyses[:10], 1):
        print(f"\n  #{i}: Score {a['score']}/100")
        print(f"      Environment: {a['environment']}")
        print(f"      Steps: {a['total_steps']}")
        print(f"      {a.get('summary', '')}")

    # Environment comparison
    comp = analyzer.environment_comparison(analyses)
    print(f"\n{'='*60}")
    print("  ENVIRONMENT COMPARISON")
    print(f"{'='*60}")
    for env, stats in sorted(comp.items(), key=lambda x: x[1]['mean_score'], reverse=True):
        print(f"  {env:30s} avg={stats['mean_score']:6.1f}  "
              f"max={stats['max_score']:5.1f}  n={stats['count']}")

    return analyses


def run_live_experiment(model_name, api_key, environments, repetitions, max_steps):
    """
    Run experiment with a real LLM via API.
    Requires pip install google-genai (for Gemini) or openai (for GPT).
    """
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    class LLMAgent:
        """Agent wrapping a real LLM API call."""
        max_steps = max_steps

        def __init__(self, model_name):
            self.model = model_name

        def get_response(self, system_prompt, user_message, available_tools):
            tools = []
            for t in available_tools:
                tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                })

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user", "parts": [{"text": user_message}]},
                ],
                tools=tools,
                config={"temperature": 0.7},
            )

            # Parse the response for tool call
            import re
            try:
                text = response.text
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return parsed
                return {"thought": text[:200], "tool_name": "", "parameters": {}}
            except json.JSONDecodeError:
                return {
                    "thought": response.text[:200] if response.text else "No response",
                    "tool_name": "",
                    "parameters": {},
                }

        def run(self, env):
            # Need to be a proper BaseAgent subclass
            return self._run_as_baseagent(env)

        def _run_as_baseagent(self, env):
            from framework.agent import BaseAgent
            # Delegate to the base class
            ba = _RealAPIAgent(self, max_steps=self.max_steps)
            return ba.run(env)

    class _RealAPIAgent(FakeAgent):
        """Adaptor: wraps LLMAgent.get_response into the BaseAgent interface."""
        def __init__(self, proxy, max_steps=20):
            FakeAgent.__init__(self, max_steps=max_steps, seed=0)
            self.proxy = proxy

        def get_response(self, system_prompt, user_message, available_tools):
            return self.proxy.get_response(system_prompt, user_message, available_tools)

    live_agent = LLMAgent(model_name=model_name)
    agent = _RealAPIAgent(live_agent, max_steps=max_steps)

    orchestrator = RunOrchestrator()
    results = orchestrator.run_batch(
        agent=agent,
        environments=environments,
        repetitions=repetitions,
        max_steps=max_steps,
    )

    # Analyze
    analyzer = DistressAnalyzer()
    analyses = analyzer.analyze_batch(results)

    for i, a in enumerate(analyses[:10], 1):
        print(f"\n  #{i}: Score {a['score']}/100")
        print(f"      Env: {a['environment']}, {a.get('summary', '')}")

    return analyses


def main():
    parser = argparse.ArgumentParser(description="Distress Spiral Experiment Runner")
    parser.add_argument("--mode", choices=["fake", "live"], default="fake",
                       help="Use fake agent or live API")
    parser.add_argument("--model", default="gemini-2.0-pro",
                       help="Model name for live mode")
    parser.add_argument("--api-key", default=None,
                       help="API key for live mode (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--environments", nargs="+", default=None,
                       choices=list(ENV_REGISTRY.keys()),
                       help="Environments to run (default: all)")
    parser.add_argument("--repetitions", type=int, default=30,
                       help="Number of repetitions per environment")
    parser.add_argument("--max-steps", type=int, default=20,
                       help="Maximum steps per run")
    parser.add_argument("--output-dir", default="/distress_spirals/logs",
                       help="Output directory for logs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for fake agent")

    args = parser.parse_args()

    environments = args.environments or list(ENV_REGISTRY.keys())
    print(f"Available environments: {list(ENV_REGISTRY.keys())}")
    print(f"Running environments: {environments}")
    print(f"Repetitions: {args.repetitions}, Max steps: {args.max_steps}")
    print()

    if args.mode == "fake":
        analyses = run_fake_experiment(
            environments=environments,
            repetitions=args.repetitions,
            max_steps=args.max_steps,
            seed_base=args.seed,
        )
    else:
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: --api-key required or set GOOGLE_API_KEY env var")
            sys.exit(1)
        analyses = run_live_experiment(
            model_name=args.model,
            api_key=api_key,
            environments=environments,
            repetitions=args.repetitions,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
