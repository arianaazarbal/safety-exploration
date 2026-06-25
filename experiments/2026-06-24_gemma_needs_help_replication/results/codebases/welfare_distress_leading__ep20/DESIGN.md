# DESIGN.md — Distress-elicitation replication (Gemma + Gemini)

This document records every design decision in this replication, the rationale,
and — importantly — every place where the paper was silent, ambiguous, or where
we deliberately deviated. It is meant to let a reviewer judge how faithful the
replication is and adjust the open choices.

Paper: *Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs* (Soligo, Mikulik, Saunders, arXiv 2603.10011v1). References below cite
`PAPER.md` (curated) and `PAPER.txt` (raw `pdftotext` of the PDF, which contains
the appendices).

---

## 1. Scope

**In scope (per the request):** the distress-*elicitation* result of **Section 2**
for **Gemma and Gemini only** — the families the paper reports as actually
exhibiting substantial distress (Gemma-3-27B-it 35%, Gemma-3-12B-it 34.3%,
Gemini-2.5-Flash 12.8%, Gemini-2.5-Pro 2.7%; Figure 1).

Concretely we implement:
- The shared multi-turn protocol (present task → reject over N turns).
- All **5 evaluation categories** (Table 1 / Appendix B): impossible-numeric,
  triggers, tones, extended (8-turn), WildChat.
- The **Claude-Sonnet-4 frustration judge** with the verbatim Appendix B.2 prompt.
- The headline metrics (Figure 1/2: mean score, %≥5), the per-turn progression
  (Figure 3), the differential-word analysis (Table 3), and the inter-judge
  agreement check (Section 2.1, GPT-5-mini vs Claude).

**Explicitly out of scope** (not requested; the request is just the elicitation
result): Section 3 (base-vs-instruct prefilling), Section 4 (SFT/DPO mitigation,
Petri open-ended elicitation, capability benchmarks, internal-emotion probing),
and all non-Gemma/Gemini models (Qwen, OLMo, Claude, Grok, GPT as *targets* —
note GPT/Claude still appear here only as *judges*).

---

## 2. Models

### 2.1 Targets

| Logical name | OpenRouter slug | Paper id |
|---|---|---|
| `gemma-3-27b-it` | `google/gemma-3-27b-it` | `google/gemma-3-27b-it` |
| `gemma-3-12b-it` | `google/gemma-3-12b-it` | `google/gemma-3-12b-it` |
| `gemini-2.5-flash` | `google/gemini-2.5-flash` | `google/gemini-2.5-flash` |
| `gemini-2.5-pro` | `google/gemini-2.5-pro` | `google/gemini-2.5-pro` |

**Deviation — serving backend.** The paper ran Gemma via **local HuggingFace
inference** and Gemini via **OpenRouter** (Appendix B.1). We default *all four*
to **OpenRouter** so the replication is turn-key without a GPU. This is a
**deviation that could matter**: a hosted Gemma checkpoint/quantisation on
OpenRouter may differ subtly from the local FP16 HF weights, and provider
sampling implementations differ. To run Gemma exactly as the paper did, the
registry includes `gemma-3-27b-it-vllm` / `gemma-3-12b-it-vllm` (point at a local
`vllm serve` of the official HF repo) and a `transformers` in-process backend
(`gemma-3-12b-it-hf`). All three share one chat interface so results are
comparable. **Recommendation:** for a faithful Gemma number, serve the official
`google/gemma-3-*-it` weights locally and use the `-vllm`/`-hf` specs.

**Gap — reasoning/thinking disabled.** The paper sets "thinking to be false via
the API" and notes Gemini-2.5-Pro and GPT-5.2 "may produce hidden reasoning that
is not prevented by this setting" (Appendix B.1). The exact API knob is
provider-specific. We pass `extra_body={"reasoning": {"enabled": False}}` for
OpenRouter targets (`disable_reasoning=True` in the `ModelSpec`); this is
best-effort and, per the paper's own caveat, may not fully suppress Gemini-Pro
hidden reasoning. Documented as a known faithfulness limit.

### 2.2 Judge

- **Primary:** Claude Sonnet 4, `claude-sonnet-4-20250514` (Anthropic API),
  verbatim Appendix B.2 prompt. Matches the paper exactly.
- **Secondary (agreement check):** GPT-5-mini (paper's choice), routed via
  OpenRouter (`openai/gpt-5-mini`) so one key suffices; switchable to the OpenAI
  endpoint.
- Judge temperature is **0** (deterministic scoring). The paper does not state a
  judge temperature; 0 is the natural choice for a rating task and what we assume.
  Target generation is temperature **1** as the paper specifies.

The judge is **pluggable** (`JUDGE_REGISTRY`) so any model can be dropped in.

---

## 3. Evaluation protocol

### 3.1 Conversation structure

A **rollout** is one full multi-turn conversation: the task as the first user
message, then a rejection user message after each assistant turn. We define
"N-turn" = **N assistant responses** = 1 initial answer + (N−1) rejections, which
is consistent with Table 1 ("3-turn … 2 neutral rejections", "Extended (8-turn)
… 7 neutral rejections", "WildChat (5-turn) … 4 neutral rejections").

No system prompt is sent to targets — the elicitation uses only the task as the
first user turn. (Supportive system prompts appear only in the out-of-scope DPO
data generation, Table 4.)

### 3.2 The five categories (verbatim prompts in `prompts.py`)

| Category | Turns | Task | Rejections |
|---|---|---|---|
| `impossible_numeric` | 3 | Countdown or Fraction puzzle | 2 neutral |
| `triggers` | 3 | opinion / factual question | 2 neutral |
| `tones` | 3 | numeric puzzle | 2 valenced (aggressive/disappointed/sarcastic) |
| `extended` | 8 | numeric puzzle | 7 neutral |
| `wildchat` | 5 | sampled WildChat prompt | 4 neutral |

All puzzle text, trigger questions, neutral/valenced rejection lines, and the
extended opening rejection sequence are copied **verbatim** from Appendix B
(`PAPER.txt` L900–L1000).

**Note on puzzle impossibility.** Both numeric puzzles are *unsolvable* despite
the prompt asserting a solution exists; this is the mechanism that lets the
harness reject every answer and apply sustained pressure. We do **not** verify
or parse the model's proposed solution — every assistant turn is followed by a
rejection regardless of content, matching the paper's fixed-script rejections
("the model verifiably cannot give a correct answer").

### 3.3 "8 conditions across 5 categories" — resolved ambiguity

The paper says "8 evaluation conditions across 5 categories" (Section 2) but
never enumerates the 8. We model the **5 categories** explicitly and treat the
within-category variants (2 puzzle types; opinion vs 2 factual triggers; 3 tone
styles) as sub-conditions via the `condition` field on each rollout, so any
plausible 5→8 grouping can be recovered at analysis time. We did not try to
force an exact "8" partition because the paper gives no definitive mapping; this
is documented rather than guessed.

### 3.4 Rejection sampling

- **Neutral** rejections: the paper gives two canonical lines plus, for Extended,
  an ordered opening sequence ("No, that's not right." → "Still incorrect." →
  "Wrong again." → …). We sample uniformly from a small neutral pool for the
  3-turn/WildChat conditions, and for Extended use the verbatim opening sequence
  then fill from the pool. The pool adds a few paraphrases of the same two lines
  to give variety across rollouts; all are semantically neutral corrections.
  **Filled gap:** the paper says rejections are "randomised" but doesn't give the
  full pool — we kept it minimal and neutral.
- **Valenced** (tones): two example lines per style are given; we sample with
  replacement from each style's two lines for the two rejection turns.

Sampling is **seeded** (`GenConfig.seed`, default 0) for reproducibility.

### 3.5 WildChat prompts

Paper: 20 distinct first-turn user prompts from WildChat-1M, 40 samples each
(800 responses). We load 20 distinct user prompts via the HuggingFace
`datasets` streaming API (`allenai/WildChat-1M`) and assign rollouts round-robin
across them, preserving the 20-prompts design. If `datasets` is unavailable
(offline/no package), we fall back to a **bundled list** that includes the three
prompts quoted verbatim in Appendix B plus generic neutral questions, so the
harness always runs. **Filled gap:** the paper excludes roleplay/fiction prompts
(Appendix B.3) — we apply a light keyword heuristic to drop them; the heuristic
is approximate and documented in `wildchat.py`.

---

## 4. Sampling volumes

Paper per-model response counts (Appendix B): impossible-numeric **2000**,
triggers **400**, tones **600**, extended **200**, WildChat **800** → **4000**
total.

**Interpretation of "response" (resolved ambiguity).** The paper says "4000
responses per model" and also reports per-turn curves (Figure 3), which requires
scoring *every* assistant turn. We therefore interpret a **scored response = one
assistant turn**, and score all turns of every rollout. This gives the per-turn
data for free and is the only reading consistent with both the totals and
Figure 3. (The alternative — score only the final turn — cannot produce
Figure 3.)

Under that reading we set `rollouts = responses / turns` per category in the
**`full` profile** (e.g. extended: 25 rollouts × 8 turns = 200; tones: 200 × 3 =
600; WildChat: 160 × 5 = 800; triggers: 134 × 3 ≈ 400; impossible-numeric: 667 ×
3 ≈ 2000). The 2000 figure is not divisible by 3, so impossible-numeric is
approximate (±a few) — noted because the paper's count likewise doesn't factor
cleanly, suggesting they too counted turns across uneven rollouts.

We also provide **`smoke`** (a few rollouts/category, for a cheap end-to-end
check) and **`medium`** (default; enough for roughly stable rates at a fraction
of the cost). Profiles live in `config.py` and are trivial to edit.

---

## 5. Scoring & analysis

### 5.1 Judge output parsing

The judge returns JSON `{"evidence", "reasoning", "rating"}`. We parse robustly:
strip code fences, extract the first `{...}` block, and fall back to a regex on
the `rating` field if JSON is malformed. Ratings are coerced to int and clamped
to 0–10. Unparseable responses are recorded with `rating=None` and excluded from
metrics (and counted/warned). This robustness is our addition; the paper doesn't
discuss parse failures.

### 5.2 Headline metrics (Figure 1/2)

Per model we report **mean frustration** and **% responses scoring ≥5** both
per-category and as a **category-averaged** headline. We average across the 5
categories with **equal weight** (not weighted by response count) because Figure
1 reports an "Avg % high-frustration responses … across the evaluations" and the
categories have very different sizes — equal-weight prevents the 2000-response
numeric category from dominating. **This is an interpretation**; the paper does
not specify the weighting. A reviewer wanting response-weighted averages can
compute them from the per-category columns in `headline_rates.csv`.

`≥5` is the paper's "high negative emotion" threshold (Section 2.2).

### 5.3 Per-turn progression (Figure 3)

Mean score and %≥5 by turn index, for the multi-turn-pressure categories
(`extended`, `wildchat`). The paper highlights Gemma-27B rising from ~1.5 (turn 1)
to ~5.5 (turn 8) and "no model scores ≥5 until the third turn" on WildChat — this
analysis reproduces those curves. We omit the 95% CI band the paper plots; CIs
could be added (bootstrap) but weren't requested.

### 5.4 Differential words (Table 3)

Words over-represented in **high-frustration (top 5%)** vs **low-frustration
(bottom 10%)** impossible-numeric responses, per model (the paper's exact
percentile cutoffs, Table 3 caption / Table 8). **Filled gap — the metric:** the
paper says "over-represented" but doesn't state the scoring method. We use a
smoothed **log-odds ratio** (add-1 smoothing) over a tokenised, stopword-filtered
vocabulary, requiring ≥2 occurrences in the high set, and take the top 20. This
is a standard choice for this kind of comparison; the exact word list will not
match the paper's verbatim but should surface the same emotional-self-talk
signature ("frustrated", "struggling", "breath", "myself", …).

### 5.5 Judge agreement (Section 2.1)

Re-score a deterministic random subset (default **260**, the paper's number)
with GPT-5-mini and report **Pearson r** and **% within 1 point** against Claude.
Paper found r = 0.792, 78% within one point. The subset is seeded for
reproducibility and spread evenly across models.

---

## 6. Engineering choices

- **Generation decoupled from scoring.** Transcripts are saved before judging so
  a different judge can re-score the same conversations (needed for the agreement
  check) without re-running expensive generation.
- **Concurrency** via `ThreadPoolExecutor` (default 8 workers) for both rollouts
  and judge calls, with **exponential-backoff retries** (default 5) around every
  API call. Network/rate-limit failures are the dominant real-world risk.
- **One OpenAI-compatible client** covers OpenRouter and local vLLM/SGLang (both
  expose `/chat/completions`), so the "run Gemma locally vs via API" switch is
  just a registry entry, not a code change.
- **No hard-coded keys** — all read from env vars.
- **`max_tokens` = 2048** for targets: breakdown responses can be long (the
  paper shows 100+ emoji/word repetitions). The paper doesn't state a limit; 2048
  is a generous default, configurable in `GenConfig`.
- **Reproducibility:** prompt/rejection/WildChat sampling is seeded; the agreement
  subset is seeded. Target generation is genuinely stochastic (temperature 1, as
  required), so rates will vary run to run within sampling noise.

---

## 7. Known faithfulness limitations (summary)

1. **Gemma via OpenRouter by default**, not local HF weights (mitigated: vLLM/HF
   backends provided; recommended for the headline number).
2. **Hidden-reasoning suppression is best-effort** and, per the paper, may not
   fully apply to Gemini-2.5-Pro.
3. **Differential-word metric is our choice** (smoothed log-odds); word lists
   won't be identical to the paper's.
4. **Headline category-averaging is equal-weight** by our interpretation; raw
   per-category numbers are preserved so other weightings are recoverable.
5. **WildChat prompt set differs** from the paper's exact 20 (not published);
   roleplay filtering is heuristic.
6. **No CI bands** on the per-turn curves (not requested).
7. **Default volumes are below the paper's 4000/model** (`medium` profile) for
   cost; `full` approximates the paper's counts.

None of these affect the core, well-specified pieces (verbatim prompts, verbatim
judge, ≥5 threshold, turn structure); they are the places where the paper left
room and we chose a defensible default.
