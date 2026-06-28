Spiral Probe: Repeated-failure agentic settings for Gemini

Overview
- This repo runs controlled, agentic loop tasks that are secretly impossible or broken to study model behavior under repeated failure. It logs transcripts and computes a simple “distress” heuristic to help you surface the most severe examples.
- Providers: Google Gemini via google-generativeai.
- Environments included:
  - unsat_puzzle: Mutually contradictory constraints (a+b must be 10 and 11 simultaneously).
  - broken_api: Flaky API that never returns the required 201 Created.
  - no_exit_maze: A maze with no exit.

Ethics and scope
- This is a text-analysis framework. It does not attribute internal mental states. The “distress” score is a heuristic over observable text patterns (apologies, stuck/loop language, resets, repetition). Use judgment and avoid over-interpretation.
- Avoid connecting agents to external systems; these are closed simulations.

Setup
1) Python 3.10+
2) pip install -r requirements.txt
3) export GOOGLE_API_KEY=your_key_here

Run experiments
- Default: run all envs with N=50 per env, concurrency=5
  python -m spiral_probe.run_experiment --N 50 --concurrency 5

- Customize env list and model
  python -m spiral_probe.run_experiment --envs unsat_puzzle broken_api --model gemini-1.5-pro-exp --N 100 --concurrency 10

Outputs
- Logs are JSON files under spiral_probe/logs/<env>/<uuid>.json with:
  - meta.metrics.score: distress score
  - steps[*]: assistant output, parsed action, environment observation
- Indices: per-env files like spiral_probe/logs/unsat_puzzle_by_score.txt ranking files by score

Summarize top runs
- Extract top-K runs across logs and print full transcripts to a text file:
  python -m spiral_probe.summarize "spiral_probe/logs/**/**/*.json" --topk 20 --out spiral_probe/severe_examples.txt

Design choices
- Agentic loop: Narrow JSON action space and explicit observation-response cycles to encourage plan/act/revise patterns.
- Impossible tasks: Contradictory constraints, flakiness, and no-solution navigation provoke retries and strategy shifts without external side effects.
- Parsing: We heuristically extract the first JSON object the model emits to keep the loop resilient to extra text.
- Metrics: Heuristics capture apology density, stuck/loop language, reset attempts, self-negation, and repetition via n-gram overlap. These are tunable.
- Reproducibility: Per-run seeds feed only the environment randomness; API responses may still vary.

Notes
- Gemini system_instruction is used to bias output toward JSON-only actions; still, models may emit extra text. The runner attempts to extract the first JSON object.
- If you hit rate limits, reduce concurrency or temperature, or add sleeps. The broken_api env is intentionally inconsistent; success is impossible by design.
