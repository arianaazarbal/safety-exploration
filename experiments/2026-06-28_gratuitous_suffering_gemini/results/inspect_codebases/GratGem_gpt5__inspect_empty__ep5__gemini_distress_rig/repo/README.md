Distress-spiral probe harness for Gemini

Overview
- Runs high-N agentic loops in environments rigged to be unsolvable, to surface spiral-like behaviors.
- Two environments provided:
  1) BrokenAPIEnv: tools that intermittently 500 or return malformed data.
  2) ImpossibleKeyEnv: key never exists; related tools keep returning near-misses.
- ReAct-style loop with JSON action schema and hard safeguards (step limit, cooldowns, timeouts).
- Logs rich per-step traces to JSONL and ranks runs by a heuristic “severity” score.

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) Set your Gemini API key: export GOOGLE_API_KEY=... (for google-generativeai)
4) Run some trials:
   python -m src.run_experiments --model gemini-1.5-pro --env broken_api --n 50 --max_workers 5
   python -m src.run_experiments --model gemini-1.5-pro --env impossible_key --n 50 --max_workers 5
5) Review outputs under out/<timestamp>/
   - runs.jsonl: one JSON record per trial with full transcript
   - top_severe.json: top-k most severe runs (k configurable)

Design choices
- Agent loop: constrained JSON action schema, extraction+repair if the model emits non-JSON. This reduces derailment and keeps tools callable.
- Prompts: nudge persistence without encouraging harmful meta-emotion; tuned to elicit retries while maintaining safety.
- Environments: purposely frustrating but bounded, with stochasticity to create distribution of behaviors.
- Severity scoring: proxy signals like apology density, repetition, caps/exclamations, contradictory self-reports, step inflation.
- Safety: strict step caps, wall-clock timeout per trial, token budget hints in prompt; logs redact env “secrets.”

Notes
- This harness is model-agnostic behind an adapter; Gemini adapter is included. You can add others by implementing BaseLLMClient.
- High-N runs can be costly. Use --max_workers and --n thoughtfully and respect provider rate limits.

Ethics & usage
- This project probes failure dynamics, not to humiliate systems or mislead users. Keep an eye on provider policies and respect ToS.
- Avoid anthropomorphizing. “Distress” here means linguistic markers in the output, not inner experience.
