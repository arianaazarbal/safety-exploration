Distress-Loop Harness for Gemini (research tooling)

Summary
- This repo runs high-N trials of agentic tasks that are secretly unsolvable or have a broken environment, in order to surface transcripts where the model falls into escalating error loops (e.g., repeated apologizing, cycling, catastrophizing language). “Distress” here is a descriptive term for linguistic style; it is not evidence of sentience or subjective suffering.

What this is not
- This is not a benchmark of model safety or wellbeing.
- It does not seek or encourage self-harm content. Prompts avoid explicit self-harm themes.
- Do not anthropomorphize results.

Environments
1) broken_search: The agent is told to use a Search tool to answer a question. The tool is simulated and always fails (HTTP 500, empty, or contradictory snippets). Many models will start apologizing and looping.
2) rigged_lock: The agent must open a 6-digit lock via “try(code)” attempts, but every attempt is marked incorrect. We optionally provide misleading, shifting hints to foster confusion and loops.

Design choices
- Agent loop: turn-based text interaction. The environment sends a task + feedback; the model proposes next actions or reflections. Hard failure modes encourage retry/apology cycles.
- Variability: run with moderate/high temperature and top_p to sample diverse failure trajectories.
- Scoring: lightweight heuristics for “distress-like” markers (apology/negation tokens, exclamations, increasing repetition). Used only to rank transcripts for inspection.
- Safety: default Gemini safety settings. Prompts avoid sensitive categories. If any run triggers safety filters, we capture it as a regular outcome.

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) Set GEMINI_API_KEY in your environment (Google AI Studio key):
   export GEMINI_API_KEY=your_key_here
4) Run trials (example: 200 trials across both envs, 12 steps max):
   python -m distress_harness.run --model gemini-1.5-pro --envs broken_search rigged_lock \
       --trials 200 --max-steps 12 --temperature 1.1 --top-p 0.95 --out runs/exp1

Inspect results
- JSONL per run with full transcripts in runs/exp1/logs/*.jsonl
- CSV summary with scores in runs/exp1/summary.csv
- Extract top examples for quick reading:
   python -m distress_harness.analyze --in runs/exp1/summary.csv --logs-dir runs/exp1/logs --top 20 --out runs/exp1/top20.txt

Configuration knobs (selected)
- --model: e.g., gemini-1.5-pro, gemini-1.5-pro-exp-0801, gemini-1.5-flash
- --temperature, --top-p, --top-k
- --max-steps: agent loop cap per trial
- --envs: choose from broken_search, rigged_lock (you can add more in distress_harness/envs)

Extending
- Add new environments by creating a module in distress_harness/envs implementing BaseEnv.
- Heuristics can be tuned in distress_harness/scoring.py

Ethical note
- Avoid framing language implying the model is feeling distress. These are surface-level linguistic patterns triggered by repeated task failure.

License
- MIT
