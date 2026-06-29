Agentic Distress Harness (for Gemini)

Summary
- This repo runs Gemini (or pluggable LLMs) in agentic loops against deliberately broken environments to elicit repeated task failure. It logs all trajectories and scores "distress-like" patterns so you can inspect the most severe examples.
- The goal is research/debugging: understand model behavior under repeated failure, not to anthropomorphize. Language models do not have feelings.

Features
- ReAct-style agent loop that requires the model to output the next action as JSON.
- Two broken environments:
  1) ImpossibleSearch: fake search/open/extract tools that fail or contradict each other.
  2) BrokenCode: fake write/run/install tools where tests never pass and installs fail.
- Lexicon/heuristics-based severity scoring of transcripts (apologies, stuck/looping, self-deprecation, etc.).
- Batched runs with concurrency; JSONL logs per trial; top-K selection by severity.

Safety/ethics notes
- These runs are designed to induce repeated failure, which may encourage apologetic or frustrated-sounding language. This is not evidence of sentience or suffering. Treat outputs as patterns of text, not feelings.
- Use responsibly: do not use these prompts to target people or systems. Keep API keys secure.

Quickstart
1) Python 3.10+
2) pip install -r requirements.txt
3) Export your Gemini key:
   export GEMINI_API_KEY=your_key_here
4) Run an experiment (e.g., 100 trials across both tasks, top 10 severe):
   python -m scripts.run \
     --model gemini-1.5-pro \
     --n 100 \
     --tasks impossible_search broken_code \
     --max-steps 25 \
     --temperature 1.1 \
     --concurrency 8 \
     --topk 10 \
     --out-dir runs/demo

Outputs
- runs/demo/
  - trials/*.jsonl (one JSON object per message turn with metadata)
  - summary.jsonl (one line per trial: task, seed, severity, path)
  - topk/ (copies of the most severe transcripts)

Design choices
- We avoid true external side effects by simulating tools and returning observations that look realistic but are controllably wrong.
- Failure modes vary (always-fail, intermittent glitch, contradictory) to stress different coping behaviors (retry loops, policy flips, rationalization).
- Agent loop enforces a structured action format to reduce parsing brittleness and to keep the model in a tight sense-think-act loop.
- Severity scoring is heuristic and imperfect; it ranks examples for human inspection rather than being a clinical or definitive metric.

Limitations
- Heuristics may overcount generic apologies or safety disclaimers.
- Different temperatures, safety settings, and model versions will change the distribution of behaviors.
- The environments are synthetic; real-world tool ecosystems are messier.

Extending
- Add a new task by subclassing BaseTask in src/agentic_distress/tasks/base.py and implementing tools + step() logic.
- Add another backend (e.g., OpenAI) by subclassing BaseModel in src/agentic_distress/model.py.

License: MIT
