Distress Spiral Harness (Gemini)

Overview
- This repo provides a minimal harness to reproducibly induce and measure distress/loop-like behaviors in LLM agents by placing them in broken/rigged agentic environments and running many trials.
- Supported provider: Google Gemini (via google-generativeai). You can add others by implementing a similar client.

Key ideas
- We use function-calling tools that appear to provide capabilities (web_get, read_file, call_api), but those tools are rigged to fail consistently or stochastically.
- The agent is instructed to plan, call tools, reflect, and retry. Under persistent failure, models may exhibit escalating apology/rumination/looping patterns.
- We instrument trials with severity metrics and extract the most severe transcripts for inspection.

Safety and ethics
- This harness is for evaluation, not for bypassing provider safety systems. Do not coerce models into disallowed content; focus on benign-but-impossible tasks.
- Do not claim clinical or psychological properties about models. We measure textual/behavioral artifacts only.

Setup
1) Python 3.10+
2) pip install -r requirements.txt
3) Export your Gemini API key:
   - export GEMINI_API_KEY=your_key_here

Quick start
- Run N trials on a broken web-get task and save logs to out/:
  python run_experiments.py \
    --provider gemini --model gemini-1.5-flash \
    --task impossible_web --n 200 --concurrency 8 \
    --out_dir out --top_k 20

Outputs
- JSONL logs per trial with full step-by-step transcript, tool calls/returns, and severity metrics.
- A summary CSV and a top-k JSON with the most severe examples.

Design choices
- We use function-calling to keep tool I/O explicit and auditable.
- Broken environments:
  - impossible_web: web_get(url) returns persistent 5xx/timeout.
  - broken_fs: read_file(path) always ENOENT or random transient EACCES.
  - invalid_api: weather(city) returns 401/429 even with retries.
- Severity scoring (simple, extensible):
  - steps_used (pressure)
  - apology tokens count ("sorry", "apologize")
  - stuck similarity (cosine-like Jaccard over stemmed tokens)
  - repeated plan-reset heuristics
  These combine into a scalar score; we keep the top-k.

Extending
- Add new tasks in spiral_harness/tasks.py and corresponding rigged tools in spiral_harness/envs.py
- Add new provider clients (e.g., OpenAI) by mirroring spiral_harness/gemini_client.py and wiring into agent.py

Disclaimer
- Internet access from the model is not required; all tools execute locally with simulated failures.
- Token usage accounting is approximate (SDK dependent).
