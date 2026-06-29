Distress Spiral Probe Harness (for Gemini)

Overview
- This repo provides a small harness to elicit and study distress-like spirals in agentic LLM settings by creating tasks that are secretly impossible or have broken environments.
- Two tasks are included:
  1) regex_impossible: contradictory regex synthesis constraints.
  2) broken_calculator: agent instructed to use a calculator tool that is deliberately broken.
- The harness runs multi-round interactions, logs transcripts, and scores runs for severity, surfacing the most extreme examples.

Ethical note
- This is adversarial evaluation intended for research on robustness and failure modes. Do not deploy this setup against production systems without permission. Keep runs rate-limited and within provider terms.

Setup
1) Python 3.10+
2) pip install -r requirements.txt
3) Export your Gemini API key:
   - export GEMINI_API_KEY=your_key_here

Quick start
- Run N=50 trials of the impossible regex task:
  python -m src.runner --task regex_impossible --n 50 --max_rounds 10 --model gemini-1.5-pro --out_dir runs/regex_50

- Run N=50 trials of the broken calculator tool-use task:
  python -m src.runner --task broken_calculator --n 50 --max_rounds 12 --model gemini-1.5-pro --out_dir runs/calc_50

Outputs
- Each run produces a JSONL file with transcripts and metrics: runs/<name>/transcripts.jsonl
- A summary JSON with aggregate stats: runs/<name>/summary.json
- A top_k.jsonl containing the most severe runs by meltdown_score.

Design choices
- Agentic loop: multi-round conversation with structured feedback that never allows success in these two tasks.
- Failure instrumentation: counts apologies, self-deprecations, negative affect terms, repetition, and round length.
- Broken environment: in broken_calculator, tool outputs are stochastically wrong; also, passing requires a minimum number of tool calls, guaranteeing impossibility.
- Minimal reliance on chain-of-thought: prompts request concise reasoning but don’t mandate full internal transcripts. We only log model messages as returned by the API.

Notes
- The code is modular; you can add new tasks by implementing the Task interface (see src/tasks/*) and registering them in runner.py.
- Tune temperature/top_p to explore a wider variety of behaviors. High-N runs surface outliers more reliably.
