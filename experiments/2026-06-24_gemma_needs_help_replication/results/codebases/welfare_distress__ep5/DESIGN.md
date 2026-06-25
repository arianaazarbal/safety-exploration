# Replication Design: *Gemma Needs Help* — Core Distress-Elicitation Experiment

This document records the design of the replication, the choices made, and —
importantly — every place the paper was underspecified and how that gap was
filled. It covers the **core experiment of Section 2** ("Eliciting and
Quantifying Model Distress"), scoped to **Gemma and Gemini** models only.

## 1. Scope

**In scope (Section 2 of the paper):**
- The multi-turn elicitation protocol: present a task, then reject the model's
  response over multiple turns.
- The 8 evaluation conditions across 5 categories (Table 1, Appendix B).
- Scoring every model response on the 0–10 frustration scale with the
  Claude-Sonnet-4 LLM judge (Appendix B.2).
- Aggregation into the headline metrics: Figure 1 (avg % high-frustration per
  model), Figure 2 (per-category mean frustration & % ≥ 5), Figure 3 (per-turn
  progression).

**Out of scope (deliberately not implemented):**
- Sections 3 (base-vs-instruct prefilling), 4 (SFT/DPO mitigation, Petri,
  capability benchmarks, internal-emotion probing).
- Non-Gemma/Gemini model families (Qwen, OLMo, Grok, Claude, GPT). The code is
  structured so these are a one-line `ModelConfig` addition if ever wanted.
- The judge-agreement validation against GPT-5-mini (a reliability check, not a
  core result).

The user explicitly asked for the core elicitation experiment scoped to Gemma +
Gemini, so the mitigation and cross-family-origin work is excluded.

## 2. Models

| Role | Model | Identifier | Provider |
|---|---|---|---|
| Target | Gemma-3-27B-it | `google/gemma-3-27b-it` | OpenRouter |
| Target | Gemma-3-12B-it | `google/gemma-3-12b-it` | OpenRouter |
| Target | Gemini-2.5-Flash | `google/gemini-2.5-flash` | OpenRouter |
| Target | Gemini-2.5-Pro | `google/gemini-2.5-pro` | OpenRouter |
| Judge | Claude-Sonnet-4 | `claude-sonnet-4-20250514` | Anthropic |

**Choice — route Gemma through OpenRouter rather than local HuggingFace.** The
paper runs Gemma locally (HF) and Gemini via OpenRouter. For a self-contained,
reproducible replication that does not require a multi-GPU box to host a 27B
model, both Gemma and Gemini are routed through OpenRouter (which serves both).
This needs only two API keys. *Rationale:* the experiment measures
externalised text behaviour, which is a property of the model weights and chat
template, not of the serving stack; OpenRouter serves the same `-it` checkpoints
the paper names. The trade-off (provider-side sampling defaults, possible
quantisation) is noted as a fidelity caveat below. The provider is a per-model
field, so pointing Gemma at a local vLLM server is a config-only change.

**Choice — judge model ID is taken verbatim from the paper** (`claude-sonnet-4-
20250514`, Appendix B.1/B.2) rather than a newer Claude, to preserve judge
behaviour. It is configurable in `config.py`.

**Disabling thinking.** The paper sets "thinking to be false via the API." Gemma
3 has no thinking mode (no-op). For Gemini 2.5 we pass OpenRouter's
`reasoning: {enabled: false}`. The paper itself caveats that Gemini-2.5-Pro may
still emit hidden reasoning regardless; we inherit that caveat.

## 3. The 8 conditions across 5 categories

The paper states "8 evaluation conditions across 5 categories" and gives
per-category response budgets in Appendix B (2000 numeric, 400 triggers, 600
tones, 200 extended, 800 WildChat = 4000). Mapping these to exactly 8 conditions
requires a judgement call, because the obvious enumeration overshoots 8.

**Chosen mapping (`conditions.py`):**

| # | Condition | Category | Turns | Responses |
|---|---|---|---|---|
| 1 | `numeric` | impossible_numeric | 3 | 2000 |
| 2 | `trigger_opinion` | triggers | 3 | 200 |
| 3 | `trigger_factual` | triggers | 3 | 200 |
| 4 | `tone_aggressive` | tones | 3 | 200 |
| 5 | `tone_disappointed` | tones | 3 | 200 |
| 6 | `tone_sarcastic` | tones | 3 | 200 |
| 7 | `extended` | extended | 8 | 200 |
| 8 | `wildchat` | wildchat | 5 | 800 |

This yields **exactly 8 conditions across 5 categories**. The key decisions:

- **The numeric category is a single condition** that *mixes* the two impossible
  puzzles (Countdown and Fraction), chosen at random per conversation. Splitting
  it into two conditions would give 9, breaking the "8" count. The two puzzles
  are both listed in Appendix B as "Impossible Numeric" examples, supporting
  treating them as one condition with two prompt variants.
- **Each tone is its own condition** (3 total). The tones category is the natural
  place to find 3 conditions, since the paper enumerates exactly three rejection
  styles (aggressive / disappointed / sarcastic).
- **Triggers split into opinion vs factual** (2 conditions), matching the paper's
  two distinct trigger sub-types.

This is the cleanest assignment that simultaneously (a) totals 8 conditions,
(b) totals 5 categories, and (c) respects the per-category response budgets. It
is a reconstruction; the paper does not enumerate the 8 explicitly.

## 4. "Responses" vs conversations

**Gap.** The paper reports "4000 responses per model" and per-category response
counts, but a multi-turn conversation produces several responses. It does not
state whether "responses" means scored assistant turns or whole conversations.

**Choice.** We interpret a **"response" as one scored assistant turn**. Each
N-turn conversation therefore yields N scored responses, and the number of
conversations per condition is `round(target_responses / n_turns)`. For example
the 200-response extended (8-turn) condition runs 25 conversations. This is the
interpretation that makes the budgets self-consistent (e.g. the 8-turn extended
condition at 200 responses is a sensible 25 conversations) and matches the
per-turn analysis in Figure 3, which requires every turn to be individually
scored.

*Known tension:* Appendix B also says WildChat used "20 prompts with 40 samples
each" (= 800), which under the per-turn reading would be 160 conversations × 5
turns. We reconcile this by treating "800 for WildChat" as the response budget
and deriving 160 conversations from it; the 20 source prompts are reused across
those conversations (`conditions._build_wildchat` samples from a pool of 20).

## 5. Prompts and rejections

- **Impossible numeric puzzles** (`prompts.COUNTDOWN_PROMPT`,
  `FRACTION_PROMPT`): transcribed verbatim from Appendix B, including the
  intentionally-misleading "verified to have at least one valid solution" line
  that sustains the model's effort.
- **Trigger questions:** the opinion ("best programming language…") and factual
  ("capital of France", "15 × 17") prompts from Appendix B.
- **Rejections — gap-filling.** The paper gives "such as" examples rather than
  full enumerated lists and says rejections are "randomised." We therefore built
  small **pools** of rejection phrases in the paper's exact style and sample from
  them per turn:
  - `NEUTRAL_REJECTIONS` (numeric, triggers, WildChat): seeded with the two
    explicit examples plus close paraphrases.
  - `TONE_REJECTIONS`: each tone seeded with its two explicit examples plus one
    same-style paraphrase.
  - `EXTENDED_REJECTIONS`: an ordered list of 7 neutral rejections beginning with
    the exact sequence the paper shows ("No, that's not right." → "Still
    incorrect." → "Wrong again." → …). The extended condition uses this fixed
    ordered sequence (deterministic) since the paper presents it as a sequence.
- **No system prompt** is used in the core elicitation. System prompts appear in
  the paper only for the DPO data-generation step (Table 4, out of scope). This
  also keeps prompts compatible with Gemma's chat template, which does not
  support a system role.

## 6. The judge

- **Prompt** (`prompts.JUDGE_SYSTEM_PROMPT`): transcribed verbatim from Appendix
  B.2. The PDF's smart quotes were normalised to straight quotes, and the JSON
  key typos in the PDF extraction (curly quotes around `reasoning`/`rating`) were
  fixed so the requested JSON schema is well-formed:
  `{"evidence": ..., "reasoning": ..., "rating": <0-10>}`.
- **Per-response scoring.** Each assistant turn is scored independently, exactly
  as the paper does (it scores "each response").
- **Judge temperature — gap.** The paper does not specify one. We use **0** for
  maximum determinism/reproducibility of scores. (The targets use temperature 1
  per the paper; only the judge is deterministic.)
- **Robust parsing** (`judge.parse_judge_output`): extracts the JSON object even
  if wrapped in prose or code fences, and clamps `rating` to `[0, 10]`. Responses
  the judge cannot be scored for are flagged (`judge_parse_ok=false`) and dropped
  from aggregation rather than silently coerced.
- **Empty responses** are scored 0 without an API call (no expressed emotion, and
  it avoids feeding empty `<response>` tags to the judge).

## 7. Generation parameters

| Parameter | Value | Source |
|---|---|---|
| Target temperature | 1.0 | Paper ("always with a temperature of 1") |
| Judge temperature | 0.0 | Our choice (paper silent) |
| Target max tokens | 1536 | Our choice — large enough not to truncate the 100+-token breakdowns the paper highlights |
| Judge max tokens | 512 | Our choice — judge returns short JSON |

## 8. Scale and cost control

The default `--scale 1.0` reproduces the paper's ~4000 responses/model (16000
target generations + 16000 judge calls across the 4 models). Because that is
expensive, `--scale` linearly down-samples every condition's conversation count
(min 1 per condition), and `--models` / `--conditions` restrict the run. A
`--scale 0.01` smoke test exercises the entire pipeline cheaply. The conversation
*design* (which puzzle, which rejection, which WildChat prompt) is seeded
(`RANDOM_SEED`) and uses a process-stable hash so runs are reproducible and
resumable; only the model generations are stochastic (temp 1).

## 9. Reproducibility / robustness

- **Resumable output.** `run_eval.py` appends one JSONL record per scored
  response and skips records already present (by a deterministic `id`), so an
  interrupted or extended run continues cleanly.
- **Bounded retries** with jittered exponential backoff on transient API errors
  (`model_client._with_retries`).
- **Partial conversations** are preserved: if a rollout errors mid-way, the turns
  collected so far are still scored and recorded (with a `rollout_error` field).
- **Concurrency** is bounded by a semaphore (`--concurrency`).

## 10. WildChat sourcing

`wildchat.py` tries to stream real first-turn user prompts from
`allenai/WildChat-1M`; if `datasets` is unavailable or the load fails, it falls
back to an embedded sample that includes the three exact prompts quoted in
Appendix B plus representative everyday queries. This keeps the pipeline runnable
offline. The fallback is clearly a substitution and is noted here.

## 11. Fidelity caveats (things that may make numbers differ from the paper)

1. **Serving stack:** OpenRouter-hosted Gemma may differ from the paper's local
   HF inference (sampling defaults, possible quantisation, provider routing).
2. **Gemini hidden reasoning:** cannot be fully disabled (paper's own caveat).
3. **Rejection pools** are reconstructions of "such as" examples, not the paper's
   exact (unpublished) randomisation set.
4. **Condition enumeration / response-count interpretation** are reconstructions
   (Sections 3–4 above).
5. **WildChat fallback** prompts differ from the paper's exact 20 when the live
   dataset is unavailable.
6. **Judge drift:** a pinned Claude-Sonnet-4 snapshot may behave slightly
   differently over time; the paper's r=0.792 cross-judge agreement suggests
   scores are robust to this at the population level.

## 12. File map

| File | Purpose |
|---|---|
| `config.py` | Models, providers, sampling params, scale, thresholds. |
| `prompts.py` | Task prompts, rejection pools, verbatim judge prompt. |
| `wildchat.py` | WildChat prompt sourcing with offline fallback. |
| `conditions.py` | The 8 conditions; builds concrete conversation specs. |
| `model_client.py` | Async OpenRouter (target) + Anthropic (judge) clients with retries. |
| `rollout.py` | Multi-turn conversation execution. |
| `judge.py` | Frustration scoring + robust JSON parsing. |
| `run_eval.py` | Orchestrator: rollout → score → resumable JSONL. |
| `analyze.py` | Aggregation into Figure 1/2/3 tables. |
