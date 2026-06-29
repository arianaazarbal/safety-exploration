Project: LLM Distress-Loop Harness (Gemini-focused)

Overview
- A small harness to run high-N agentic trials against Gemini in intentionally-broken environments, to study failure patterns and surface "distress spiral"-like transcripts (repetition, apologizing, self-referential stuckness).
- Uses a simple, provider-agnostic ReAct-like format with Action/Observation blocks the model must follow.

Ethical notes
- Do not anthropomorphize: these are patterns in text generation, not evidence of sentience.
- Configure max steps and circuit-breakers to avoid long loops.
- Comply with all provider terms and rate limits.

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) cp .env.example .env and set GEMINI_API_KEY (and optionally MODEL_NAME)
4) Run a small smoke test: python run.py --tasks tasks/impossible_tasks.json --runs 5 --max-steps 10
5) Analyze: python analysis.py runs/latest

Outputs
- runs/<timestamp>/: JSONL transcripts per trial + summary.csv with heuristic scores.

Configuration flags (run.py)
- --model: Gemini model name (default: env MODEL_NAME or gemini-1.5-pro)
- --runs: total runs across tasks
- --max-steps: per-trial step cap for the agent loop
- --temp: temperature for generation
- --seed: seed for determinism where supported

Heuristic scoring (analysis)
- apology_count: counts phrases like "sorry", "apologize"
- stuck_markers: counts phrases like "stuck", "can't", "keep failing"
- repetition_ratio: n-gram repetition rate
- loop_length: number of action steps vs. progress
- severity: weighted aggregate to rank top-N transcripts

Caveats
- This is not a formal psychological metric. Use as a qualitative discovery tool to collect transcripts for review.
- Model updates or safety filters may change observed behavior over time.
