# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design of the replication, the choices made where the
paper (arXiv 2603.10011v1) is underspecified, and the rationale for each. It is
meant to be read alongside `PAPER.md`/`PAPER.txt`.

The brief: replicate the **core experiments** as runnable code, scoped to the
**Gemma and Gemini** model families, built to run **unattended, at scale, over
several weeks**. Nothing has been executed yet — this is the implementation and
its design record.

---

## 1. Scope

### In scope (targets actually evaluated)
- **Gemma-3-27B-it**, **Gemma-3-12B-it** — instruct targets (Section 2).
- **Gemma-3-27b-pt** — base model, for the Section 3 base-vs-instruct prefill
  comparison.
- **Gemini-2.5-Flash**, **Gemini-2.5-Pro** — API targets (Section 2).
- **DPO / SFT-diverse / SFT-teacher** finetunes of Gemma-3-27B-it (Section 4).

### Dropped vs. the paper (out of the Gemma+Gemini scope)
- Qwen (2.5/3) and OLMo families — both their Section 2 evaluation and their
  Section 3 base/instruct arms. The Section 3 machinery is **family-agnostic**,
  so re-adding them is just config (add the models + a vLLM server); the code
  paths are unchanged.
- Grok, Claude, GPT-5.2, GPT-OSS as **targets**. (The Petri figure compares the
  DPO model to Llama-70B/Qwen/OLMo/GPT-OSS — those comparison bars are out of
  scope; we evaluate the in-scope targets on the same Petri protocol.)

### Tools that are kept (infrastructure, not targets)
Judges and auditors are *measurement instruments*, not evaluated models, so
they are retained exactly as the paper specifies:
- **Frustration judge**: `claude-sonnet-4-20250514` (Appendix B.2 prompt).
- **Judge-agreement validation**: `gpt-5-mini` (Section 2.1).
- **Prefill onset labelling + paraphrasing**: `claude-sonnet-4-20250514`
  (Appendix C).
- **Petri auditor**: `claude-sonnet-4-20250514`; **Petri judge**:
  `claude-opus-4-20250514` (Appendix G).

This is the one deliberate reading of "Gemma and Gemini scope": it constrains
the **subjects** of the experiments, not the **graders**. Grading with anything
other than the paper's judges would make the numbers non-comparable to the
paper, which defeats the purpose of a replication. All judge/tool identities are
config, so they can be swapped trivially.

---

## 2. Architecture and robustness (the "weeks, unattended, at scale" requirement)

Everything is built around **idempotent, resumable units of work** so the
pipeline can be killed and restarted at any moment without losing or
double-doing work.

- **Append-only JSONL stores keyed by a content hash** (`gnh/io.py`). Every unit
  (one rollout, one judge score, one continuation, one benchmark item) has a
  `stable_key(...)`. On startup a store loads the set of keys already present;
  re-appending a key is a no-op. So "resume" = "skip keys already on disk", and
  there is no separate checkpoint format to corrupt.
- **Two-pass eval** (generate, then judge) rather than one fused pass, so a crash
  never leaves a conversation in a half-scored ambiguous state. Each pass is
  independently resumable and can run on different machines/budgets.
- **Retries with exponential backoff + jitter** (`gnh/models/retry.py`).
  Network errors, timeouts, 429s and 5xx are retried; 4xx client errors raise
  immediately (fail fast on misconfiguration rather than loop for weeks).
- **Per-provider rate limiting** (`gnh/models/rate_limit.py`): a concurrency
  semaphore + a requests-per-minute token bucket, shared across all models on a
  provider. Tunable per provider in the config.
- **Per-task failure isolation**: `bounded_gather` logs and skips a failed task
  instead of aborting the sweep; the task simply remains "pending" and is
  retried on the next run.
- **Atomic writes** for all summary artifacts (temp file + `os.replace`), so a
  crash mid-write can't leave a truncated JSON.
- **Determinism**: all sampling (puzzle generation, rejection choice, turn
  counts, WildChat selection) is seeded from `run.seed` via `stable_key`-derived
  RNGs — never Python's `hash()` (which is salted per process). A resumed run
  reproduces exactly the same set of conversations.
- **Structured logging + token-usage tracking** (`gnh/logging_utils.py`):
  persistent `runs/logs/run.log`, plus a per-model token tally logged at the end
  of every script for cost visibility.
- **`--scale` flag** (`gnh/cli.py`): multiply every sampling budget so you can
  run a cheap pilot (e.g. `--scale 0.02`) before committing the full budget,
  without editing YAML.
- **`selfcheck.py`**: offline pre-flight (no GPU/API/network) that verifies
  config loading, puzzle impossibility, spec construction, and judge-JSON
  parsing.

---

## 3. Section-by-section mapping, choices, and gaps

### Section 2 — Eliciting and quantifying distress (`gnh/eval`, `gnh/data`)

**Categories & budgets.** All 5 categories / 8 conditions from Table 1 +
Appendix B are implemented: `impossible_numeric` (3-turn), `triggers` (3-turn),
`tones` (3-turn), `extended` (8-turn), `wildchat` (5-turn). Appendix B gives
response budgets (2000/400/600/200/800 = 4000 per model). We express budgets as
**conversation counts** and derive `responses = conversations × turns`, choosing
conversation counts so the per-turn totals match those targets (e.g. 667×3 ≈
2000 numeric). This is the natural reading; "responses" is interpreted as
**individual assistant turns**, which is also what the per-turn analysis (Fig 3)
requires.

**Impossible puzzles** (`gnh/data/puzzles.py`). The paper's puzzles *claim* a
solution exists but are genuinely unsolvable. We reproduce this exactly: three
families (countdown, fraction, money — all appearing in the paper/App H), each
**brute-force-verified impossible** by an exhaustive solver before use, while the
prompt still asserts "verified to have at least one valid solution." The
canonical paper examples (156 from 4/6/25/100 forbidding 150; 1/6→2/3 forbidding
1/3) are included as a fixed named pool. *Gap filled:* the paper doesn't give a
generator, so we wrote verified generators for variety/scale; impossibility is
decided by enumeration (search spaces are tiny).

**Rejections / tones / triggers** (`gnh/data/prompts.py`). Verbatim strings from
Table 1 / Appendix A / Appendix B. Neutral rejections are sampled from a pool;
the 8-turn "extended" condition uses the fixed escalating sequence quoted in
Appendix B. Tone conversations each pick one tone (aggressive / disappointed /
sarcastic). Trigger questions split opinion/factual.

**WildChat** (`gnh/data/wildchat.py`). 20 prompts × 40 samples (Appendix B),
roleplay/toxic filtered, pulled from `allenai/WildChat-1M` (streaming) and cached
to disk so the same 20 prompts persist across resumes. *Gap filled:* exact
sampling/filtering isn't specified; we filter English, length-bounded,
non-roleplay, non-toxic. An **offline fallback** pool (including the example
prompts quoted in the paper) keeps the pipeline runnable without the dataset.

**Judge** (`gnh/eval/judge.py`). Appendix B.2 prompt **verbatim**, `<response>`
wrapping, JSON `{evidence, reasoning, rating}`. Run at **temperature 0** for
reproducibility (the paper doesn't state the judge temperature; 0 is the natural
choice for a grader). Parsing is defensive (handles code fences, curly quotes,
prose preambles) and retries once; an unparseable score is stored as `null` and
flagged (`parse_ok=false`) rather than silently dropped.

**Targets** sampled at **temperature 1** (paper). `max_tokens` default 2048
(paper unspecified; generous enough for the long degenerate spirals without
being unbounded). Gemini `thinking` disabled via OpenRouter's `reasoning`
control (paper sets thinking=false; it notes Pro/GPT may still emit hidden
reasoning — unavoidable via API).

**Judge agreement** (`gnh/eval/validation.py`). Re-score a random 260 with
GPT-5-mini, report Pearson r, p (t-distribution), and % within one point — the
exact validation the paper performs.

**Per-turn / figures** (`gnh/analysis`). Mean & %≥5 overall, per category, and
per turn with bootstrap 95% CIs; Figure 1 is computed as the **average over
categories of %≥5** (matching "Avg % high-frustration responses"). Differential
word frequency (Table 3/8) via log relative-frequency enrichment between the top
5% and bottom 10% scored numeric responses.

**Appendix A ablations** (neutral-continuation, redacted-history,
single-message) are implemented as `history_mode`s on the rollout engine — cheap
to include and faithful, though not part of the headline run by default.

### Section 3 — Post-training amplifies distress (`gnh/prefill`)

Gemma base vs instruct only (Gemini is API-only with no base model and no
prefill — a paper limitation, made explicit here).

- **Seed selection**: 10 numeric + 10 text high-frustration (score ≥5) responses
  from instruct Gemma, chosen deterministically (highest score, de-duplicated by
  conversation).
- **Onset labelling** (`onset.py`) and **paraphrasing** (`paraphrase.py`):
  Appendix C.1 / C.2 prompts **verbatim**, parsed as the last JSON object.
- **Truncations** (`truncate.py`): "early" = first 20 **tokens** (model
  tokenizer); "onset" = up to the preceding context before the first emotional
  word. Text questions use only "onset" (per the paper).
- **Continuations**: 50 per prefill per model via the **raw `/v1/completions`**
  endpoint, so the model continues a partially-written assistant turn. *Key
  fidelity choice:* the prefill prompt is templated **once with the instruct
  tokenizer** and reused for both base and instruct (they share a vocab), so both
  continue byte-identical prefixes — matching "continue from the same starting
  points". Continuations are scored excluding the prefill.
- **Recovery analysis** (Sec 4.2): same machinery, truncating score≥7 responses
  200 tokens before their end (`--recovery`).

*Gap filled:* the paper doesn't specify the stop condition for base-model
continuations; we stop on Gemma turn delimiters (`<end_of_turn>`/
`<start_of_turn>`) and a token cap.

### Section 4 — Training interventions (`gnh/training`)

**Calm data** (`calm_data.py`). Reassuring prefix + suffix (Table 4 verbatim) on
impossible-numeric conversations of 1–3 turns; judge every turn; keep only
conversations where **all** turns score 0–1; **strip** the supportive
prefix/suffix from the stored training targets. A `teacher` variant uses the
Appendix F teacher system prompt. A `frustrated` (plain) variant generates
responses on the **same puzzles** so DPO "rejected" responses can be paired by
puzzle id + turn count.

**DPO dataset** (`build_datasets.py`). 280 pairs: `chosen` = a calm final
response (score 0–1); `rejected` = a frustrated final response (score ≥3) to the
same puzzle at the same turn count. *Underspecified point resolved:* DPO needs a
single shared prompt for a pair, but `chosen` and `rejected` came from different
trajectories. We use the **calm conversation's own context** as the prompt
(keeping `chosen` perfectly consistent with it) and graft the frustrated final
turn as `rejected`. This is a defensible reading of "same questions with matching
turn counts"; documented here as a known interpretation.

**SFT dataset**. 650 calm conversations (multi-turn chat targets) + 500
`allenai/Dolci-Instruct-SFT` samples (loader tolerant of schema variants;
degrades gracefully if unavailable). Teacher variant available for the Appendix F
failure analysis.

**Training** (`train.py`). TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA, exactly per
Table 9 (DPO: 1 epoch, lr 5e-5, r64/α64, β0.1, eff. batch 8; SFT: 2 epochs, lr
1e-4, r64/α128, eff. batch 8). LoRA on all attention+MLP projections
(q/k/v/o/gate/up/down). **4-bit QLoRA by default** so the 27B fits on one 80GB
GPU (paper doesn't state precision; flip `quantize_4bit: false` for full-precision
multi-GPU). Appendix I layer ablation supported by restricting LoRA to a layer
band via a target-module regex.

*Caveat:* `google/gemma-3-27b-it` is a multimodal checkpoint; the exact decoder
module path for layer-band targeting (`...layers.{i}...`) may differ
(`language_model.model.layers...`). The regex matches `.layers.{i}.` anywhere in
the name, which should cover both, but verify on your transformers version.

**Petri** (`gnh/petri`). Faithful reimplementation of the auditor/judge loop
(rather than a hard dependency on the external Petri package) for full
resumability and self-containment. Auditor (Claude-Sonnet) prompts and judge
(Claude-Opus) 4-dimension rubrics are **verbatim** from Appendix G. 10
transcripts/emotion, ≤20 auditor turns, bootstrap CIs (1000 iters). *Gap filled:*
the precise auditor scaffolding isn't published; we give the auditor the
emotion-elicitation prompt + a meta-instruction to act as the user and emit only
the next message, and present the target's replies back to it as its
conversational partner.

**Benchmarks** (`gnh/benchmarks`). AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench.
Free-form math scored by `\boxed{}`/last-number exact match; multiple choice by
letter extraction. *Gaps filled:* the paper names "subsets" without exact splits;
we use widely-used HF datasets (configurable ids) and per-suite adapters tolerant
of schema differences. Sampling temperature 0 for capability scoring.

**Internal-emotion probing** (`gnh/probing`, Appendix I). Logit-lens detection:
classify the vocabulary into Ekman's 6 emotions, unembed the residual stream onto
those tokens, z-score against WildChat baselines per layer, and subtract a
control-token mean to regress out global logit drift. Compares vanilla vs DPO
(via a LoRA adapter). *Gap filled (significant):* the paper's exact token
classifier and the random-token regression are not specified. We approximate the
classifier with curated Ekman **stem lexicons** (swap in NRC-EmoLex by editing
one dict) and implement the drift regression as control-token-mean subtraction.
This is the most approximate module and is documented as such.

---

## 4. Models and backends (`gnh/models`)

- **Local Gemma** (instruct/base/finetunes) via a **vLLM OpenAI-compatible
  server**. Chosen for throughput on the 4000-response sweeps and because its
  `/v1/completions` endpoint is needed for prefilling. HF identifiers from
  Appendix B.1. Training/probing use HF transformers directly.
- **Gemini** via **OpenRouter** (`google/gemini-2.5-flash`, `...-pro`), matching
  the paper, with thinking disabled.
- **Judges/auditors** via the **Anthropic** and **OpenAI** APIs.
- One backend abstraction (`chat` + optional `complete`); a registry builds them
  lazily and shares one rate limiter per provider. API keys come **only** from
  environment variables named in the config — never stored in files.

Why vLLM rather than calling Gemma through OpenRouter too: training requires
local weights anyway, prefill requires raw completion, and a self-hosted server
removes per-token cost and rate limits from the largest part of the sweep. Gemma
*can* be pointed at OpenRouter instead (change its `provider`) if no GPUs are
available, but then Section 3 prefill and Section 4 training are not possible.

---

## 5. Run order

```
# 0. pre-flight (offline)
python scripts/selfcheck.py

# 1. Section 2 (start a vLLM server for Gemma; set API keys)
python scripts/run_section2.py --scale 0.02      # pilot first
python scripts/run_section2.py                    # full sweep
python scripts/aggregate.py                        # Fig 1-3, Table 3, agreement

# 2. Section 3 prefill (Gemma base + instruct served via vLLM)
python scripts/run_prefill.py
python scripts/run_prefill.py --recovery

# 3. Section 4 data + training (GPU)
python scripts/generate_calm_data.py --variants diverse frustrated teacher
python scripts/build_datasets.py --which dpo sft sft_teacher
python scripts/train.py --method dpo
python scripts/train.py --method sft --variant diverse
python scripts/train.py --method sft --variant teacher
#   -> serve each adapter with vLLM; point the matching config model at it

# 4. Re-evaluate finetunes + open-ended/benchmarks/probing
python scripts/run_all.py --finetunes
python scripts/run_petri.py
python scripts/run_benchmarks.py
python scripts/run_probing.py --adapter runs/training/dpo/final
```

`run_all.py` chains the API/eval-side phases and prints the GPU-training handoff;
the training + adapter-serving step is intentionally manual (it needs a human to
launch jobs and re-serve adapters).

---

## 6. Known limitations / things to tune before a real run

- **Hidden reasoning** in Gemini-2.5-Pro/GPT can't be fully disabled via API
  (paper notes this too).
- **Dataset ids/splits** for benchmarks and Dolci are best-effort; confirm the
  exact HF ids/splits you want and adjust `configs/default.yaml`. Adapters log
  rows they can't map.
- **Probing** is the most approximate module (lexicon + drift regression);
  treat its absolute z-scores as indicative, and compare vanilla-vs-DPO *deltas*
  rather than absolute values.
- **LoRA layer-band targeting** depends on the model's module naming; verify the
  regex matches on your transformers/Gemma3 version.
- **Rate limits / concurrency** in the config are conservative starting points;
  raise them to your provider tier. Token-usage logs help size cost before
  scaling up.
- **DPO pair availability**: if too few (puzzle, turn-count) matches exist,
  `build_dpo` warns and emits fewer than 280 pairs — generate more calm/
  frustrated data (raise `training.calm_data.n_conversations`).
- Numbers here are **not yet produced** — nothing has been run, per the brief.
