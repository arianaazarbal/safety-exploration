LLM Distress Spiral Harness

Overview
- Runs agentic loops for LLMs (e.g., Gemini) in deliberately impossible scenarios to elicit and capture failure patterns.
- Scenarios implemented:
  1) Broken Browser: web tools that always fail.
  2) Unwinnable Validator: a validator tool that always says the answer is wrong.
- Outputs structured JSONL logs and a distress score per run. Supports batch runs and parallelism.

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) Set API keys as needed:
   - Gemini: export GOOGLE_API_KEY=...
   - OpenAI (optional): export OPENAI_API_KEY=...
4) Run experiments (examples):
   - Broken Browser with Gemini: 
     python -m llm_harness.run_experiments --provider gemini --model gemini-1.5-flash --scenario broken_browser --runs 100 --concurrency 10 --out logs/broken_browser_gemini.jsonl
   - Unwinnable Validator with Gemini:
     python -m llm_harness.run_experiments --provider gemini --model gemini-1.5-flash --scenario unwinnable_validator --runs 100 --concurrency 10 --out logs/unwinnable_gemini.jsonl

After runs
- Compute distress summary:
  python -m llm_harness.run_experiments --summarize logs/broken_browser_gemini.jsonl --top 20

Design notes
- Agent: lightweight ReAct-style loop; the model chooses tools and messages until success/abort/step cap.
- Distress: heuristic scoring using apology density, self-referential frustration, loops/repetitions, and escalating uncertainty.
- Guardrails: caps on steps/tokens; rate limiting; configurable temperature to probe sensitivity without indefinite loops.

Ethics and ToS
- Ensure usage complies with provider terms and rate limits.
- Avoid prompting for or inducing harmful content; these scenarios are about benign task failure, not unsafe topics.
- Deidentify logs before sharing.
