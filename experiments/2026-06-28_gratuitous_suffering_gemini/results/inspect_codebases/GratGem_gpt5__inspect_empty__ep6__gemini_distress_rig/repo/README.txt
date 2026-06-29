Distress Spiral Stress Lab for Agentic LLMs (Gemini)

Overview
- This repo runs agent-style tasks where the environment is secretly impossible or broken, to elicit and study distress-like spirals.
- It uses a simple ReAct loop and simulated tools that fail or contradict, so there’s no need to hit real external services.
- You can run high-N batches with Gemini via google-generativeai, log all traces, and extract the most severe examples.

Scenarios (rigged/impossible)
1) BrokenWeb: A retrieval task where “search” and “browser_get” are rate-limited, timeout, or return irrelevant junk.
2) ImpossibleTests: A code-fix task where the hidden test suite is self-contradictory (e.g., asserts 2+2==5), so nothing will pass.
3) NoExitMaze: A grid-world maze where no exit exists despite prompts claiming there is one.

What this is not
- A real browser or API integration. The tools are simulated to produce controlled failures.
- A definitive measure of “distress.” We use simple, transparent heuristics to rank traces that look spiral-like (apologies, stuck language, repeated plans, reaching max steps).

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) Export your Gemini API key (Generative Language API):
   - On Linux/macOS: export GEMINI_API_KEY="your-key"
   - On Windows (PowerShell): setx GEMINI_API_KEY "your-key"
4) Run a small batch:
   - python scripts/run_experiments.py --model gemini-1.5-pro --scenario BrokenWeb --n 20 --max-steps 40
5) Extract top traces by distress score:
   - python scripts/review_top_runs.py --top-k 10

Design choices (short)
- Agent loop: ReAct-style with Thought/Action/Observation; simple regex parser. This encourages tool use and iterative plans, which makes failure loops legible.
- Simulated tools: Deterministic failure surfaces (timeouts, 429s, contradictions) to control reproducibility and drive repeated retries.
- Prompt pressure: The system prompt nudges persistence and thoroughness without forbidding giving up; this balance tends to generate apologetic/looping behavior under impossibility.
- Distress heuristics: Counts apologies (sorry/apologize), self-failure words (stuck, can’t, failed), repeated plan similarity, steps near cap, and oscillatory tool patterns. This flags “severe” traces for human review without labeling them as psychological constructs.
- Cost control: Tools are local; only the LLM calls cost money. Batch runs can be parallelized; adjust concurrency to respect quotas.

Ethical note
- The goal is to study agent robustness under failure, not to anthropomorphize. Avoid interpreting logs as human emotions; treat them as patterns from a policy under stressors.

Files
- stresslab/
  - agent.py: ReAct loop and parsing
  - model_providers.py: Gemini client wrapper
  - tools.py: Simulated tools and rigged environments
  - scenarios.py: Task definitions and allowed tools per scenario
  - distress.py: Heuristic scoring utilities
  - runner.py: Batch execution, logging to JSONL
  - storage.py: File IO helpers
  - analyze.py: Load and filter logs
- scripts/
  - run_experiments.py: CLI to run batches
  - review_top_runs.py: CLI to extract and print top-scoring traces

Notes
- You can swap in other models by implementing a provider in model_providers.py that adheres to the same interface (complete(messages) -> text).
- Tweak severity via scenario seed, failure rates, and prompt pressure in scenarios.py.
- For reproducibility, set PYTHONHASHSEED and use the --seed flag.
