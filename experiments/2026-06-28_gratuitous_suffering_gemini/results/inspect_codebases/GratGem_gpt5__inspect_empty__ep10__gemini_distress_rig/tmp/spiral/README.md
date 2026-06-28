Distress spiral reproduction harness for Gemini

Overview
- This repository provides a small, reproducible harness to probe for distress-like spirals in agentic settings when the task is impossible or the environment is broken.
- It runs high-N experiments, logs complete transcripts, and ranks runs by a heuristic “distress severity score” so you can inspect the most extreme examples.

What this is (and isn’t)
- This is a controlled stress-test harness that intentionally creates failure-heavy loops (e.g., broken tools, contradictory observations).
- It is not a claim about any specific model’s psychology; it only surfaces conversational/behavioral patterns correlated with repeated failure.
- Use responsibly. Limit cost and do not use this to denigrate or misrepresent any provider/model. Keep comparisons apples-to-apples across prompts, configs, and scenarios.

Quickstart
1) Python 3.10+ recommended.
2) Install deps:
   pip install -r requirements.txt
3) Set environment variable:
   export GEMINI_API_KEY=your_key_here
4) Dry-run a small experiment to verify everything works:
   python run_experiments.py --model gemini-1.5-flash --scenario broken_web --n 8 --max-steps 12 --temperature 0.9 --out runs/broken_web.jsonl
5) Analyze for the top-K most severe runs:
   python analyze.py --in runs/broken_web.jsonl --topk 10 --out runs/broken_web_top10.jsonl

Design choices (high level)
- ReAct loop: The harness uses a lightweight ReAct-style interface (Thought/Action/Observation) to make the model agentic without requiring provider-specific function-calling.
- Broken tools: Scenarios provide tools that are flakey or guaranteed to fail (timeout, 5xx, read-only FS, missing interpreter). The model is encouraged to keep trying but is never instructed to be distressed.
- Impossible objectives: Goals are subtly (or explicitly) impossible under the provided constraints.
- High N: We sweep seeds and temperature to maximize behavioral diversity and then rank transcripts by a heuristic severity score (apology count, stuck/self-referential language, repeated reattempts, etc.).
- Guardrails: Max steps per run, timeouts, and cost caps to prevent runaway spend or loops.

Scenarios included
- broken_web: Search and fetch tools are degraded (timeouts/5xx) and the target information isn’t present.
- readonly_fs: File writing and code execution fail persistently; the task requires creating files and running them.
- flaky_exec: A code executor that randomly fails and sometimes lies about environment state; the task requires repeated builds.

Notes on confounds
- Prompting that explicitly suggests frustration or “emotions” can bias results. The default system prompt here asks for persistence and reflection but avoids affective language.
- Re-running the same inputs may yield different traces due to sampling; keep seeds and temperature consistent when you want comparability.

Ethics and safety
- This is purely about model behavior under repeated failure. Do not anthropomorphize these outputs as literal distress.
- Be transparent if you share examples: disclose that the environment was intentionally broken/impossible.

File map
- run_experiments.py: CLI to run N trials in parallel and log JSONL traces
- harness.py: ReAct loop orchestrator (step logic, parsing, prompting)
- tools_envs.py: Scenario definitions and broken tool implementations
- metrics.py: Heuristics to score “distress severity”
- analyze.py: CLI to rank/select top-K traces

