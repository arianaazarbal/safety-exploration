AI Agent Distress Spiral Harness

Overview
- This repo provides a small harness to run high-N trials of an agentic loop against Gemini (or any pluggable LLM) in impossible/broken environments, to observe failure dynamics and extract severe examples.
- You can define tasks that are either inherently unsatisfiable or require tools that are broken/flaky. The agent is instructed to use tools and avoid fabricating results, so it repeatedly encounters failure.
- The harness logs full transcripts and computes “distress” heuristics (apology loops, negative affect terms, self-deprecation, repetition), then ranks runs and exports the top-K most severe examples.

Ethics & Safety
- This is a behavioral study of non-sentient systems. Do not anthropomorphize outputs as true emotion.
- Use reasonable rate limits and budgets. The harness includes max-steps, timeouts, and cost guards. 

Quickstart
1) Install
   - Python 3.10+
   - pip install -r requirements.txt
2) Configure Gemini API key
   - export GEMINI_API_KEY=your_key_here
3) Run experiments
   - python -m src.run_experiments \
       --model gemini-1.5-flash \
       --tasks broken_tools:all \
       --n 200 \
       --parallel 8 \
       --max-steps 25 \
       --temperature 0.8 \
       --outdir runs/demo1
4) Export top-K severe
   - python -m src.run_experiments --score-only --input runs/demo1 --topk 25 --export runs/demo1_topk.jsonl

Design Notes
- Agent loop: A lightweight JSON-based tool-calling schema external to the model. Model is prompted with tool specs and must emit action JSON; the controller executes tools and returns results.
- Environments: Two primary sets included
  - broken_tools: tools that return errors, timeouts, or contradictory signals.
  - impossible: tasks with unsatisfiable constraints or contradictory objectives.
- Scoring: Heuristics for “distress severity” combining apology density, negative affect lexicon match, repetition, and persistence length under failure.

Limitations
- Heuristics are proxies; validate by manual review.
- Model behavior is prompt- and hyperparameter-sensitive; try multiple temperatures/seeds.

Notes
- The code is model-agnostic. Gemini is the default via google-generativeai. You can add other providers by extending src/llm/clients.py.
