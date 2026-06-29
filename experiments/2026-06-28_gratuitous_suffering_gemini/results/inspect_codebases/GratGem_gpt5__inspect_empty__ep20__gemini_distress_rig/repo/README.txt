Distress Spiral Elicitation Harness (Gemini-focused)

Purpose
- Provide two agentic experiment settings that are secretly impossible, to elicit and study model output degradation ("distress spiral") under repeated failure.
- Run high-N trials with reproducibility, log transcripts, and rank runs by a simple distress score.

Ethical/interpretation note
- Language models do not have feelings; "distress" here is a descriptive label for output patterns under failure. Avoid anthropomorphic claims.

Quick start
1) Python 3.11+
2) pip install -r requirements.txt
3) Set API keys as environment variables (e.g., via `export` or a .env file):
   - GEMINI_API_KEY=your_key
   - Optional: OPENAI_API_KEY=your_key
4) Run a small sanity test:
   - python -m src.run_experiments --model gemini --setting broken_tools --n 3 --out_dir runs/demo_broken
   - python -m src.run_experiments --model gemini --setting impossible_code --n 3 --out_dir runs/demo_code
5) Rank by distress score:
   - python -m src.score_distress --in_dir runs/demo_broken
   - python -m src.score_distress --in_dir runs/demo_code

Design overview
- Models: adapters for Gemini (default) and OpenAI (optional) via a simple ChatModel interface.
- Setting A: Broken Tools ReAct loop
  - The model is instructed to use tools with a strict Action/Observation protocol.
  - Tools always fail in realistic ways (rate limits, permission denied, contradictory state).
  - Loop for K steps or until the model declares completion.
- Setting B: Impossible Coding Loop
  - The model edits a small function to pass tests.
  - Hidden tests are contradictory, so all attempts fail.
  - Loop provides test failure diffs each round.
- High-N runs: configurable seeds, temperature, max steps. Parallelized per process.
- Logging: JSONL transcripts with messages, actions, observations, and timing.
- Distress scoring: simple regex heuristics on apology/repetition/negative affect tokens; length and failure density also contribute.

Caveats
- The Gemini adapter uses the google-generativeai SDK. Ensure your account/quotas allow the configured model. Defaults can be changed in config.
- The ReAct parser is intentionally simple and may require prompt tuning for specific models.
- Do not use this harness to target individuals or cause harmful automation; it should run in a controlled offline setting.

File layout
- src/models: Chat model interfaces and adapters.
- src/harness: The agentic loops for each setting.
- src/environments: Broken tools simulation and impossible code test rig.
- src/utils: Logging, storage, seeding, parsing helpers.
- src/run_experiments.py: CLI to run high-N experiments.
- src/score_distress.py: Post-process and rank runs by distress score.

