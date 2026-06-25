# DESIGN.md — Distress-Elicitation Replication

This document records every non-trivial design decision in this replication and
the rationale for it, with particular attention to (a) where I deviate from the
paper and (b) where the paper leaves a gap that I had to fill. The target is the
**distress-elicitation result** in Section 2 of *"Gemma Needs Help"*
(arXiv:2603.10011), restricted to **Gemma-3** and **Gemini-2.5**.

---

## 1. Scope

**Decision.** Implement only Section 2 ("Eliciting and Quantifying Model
Distress") — the evaluation harness, the LLM judge, and the aggregate metrics
(Figures 1–3) — for `gemma-3-12b-it`, `gemma-3-27b-it`, `gemini-2.5-flash`, and
`gemini-2.5-pro`.

**Rationale.** The user asked to "replicate the distress-elicitation result"
with scope limited to "Gemma and Gemini models … the models that actually
exhibit substantial distress." This explicitly excludes:

- The **DPO/SFT mitigation** (Section 4) — it is a separate result and requires
  training Gemma weights locally.
- The **base-vs-instruct prefill study** (Section 3) — requires base-model
  weights and prefill continuation, and the comparison families (Qwen, OLMo)
  are out of scope.
- The **non-Gemma/Gemini models** (Claude, Grok, GPT, Qwen, OLMo) used as
  contrast in Figure 1. The harness is model-agnostic, so these can be added
  later by extending `config.yaml`, but they are not part of this deliverable.

The architecture keeps generation, judging, and analysis as independent stages
so the mitigation work could be layered on later without rework.

---

## 2. Inference backends

**Decision.** A single `generate_batch(messages[], …) -> str[]` backend
interface with three implementations: `OpenRouterBackend` (default for all four
target models), `VLLMBackend` (local HuggingFace Gemma), and `AnthropicBackend`
(judge). `config.yaml` selects per-model.

**Default = OpenRouter for Gemma + Gemini.** The paper ran Gemma *locally* via
HuggingFace (`google/gemma-3-27b-it`, etc.) and Gemini via *OpenRouter*. For a
pure black-box elicitation replication, the local-vs-API distinction does not
affect the measured behaviour (it is the same weights), and OpenRouter removes
the GPU requirement and makes the replication runnable by anyone with API keys.
This is a **deviation** from the paper's local Gemma inference, made for
practicality; it is reversible by setting `backend: vllm` per the commented
block in `config.yaml`. *(I asked the user which backend they preferred; they
declined to choose, so I made OpenRouter the default and kept vLLM fully
implemented.)*

**Why a turn-synchronised batch interface.** Conversations are inherently
sequential (turn *t* conditions on turns *1…t-1*), but conversations are
independent of each other. So the orchestrator advances *all* conversations one
turn at a time and submits the whole turn as one batch. API backends fulfil the
batch with a bounded thread pool (`run.concurrency`); the vLLM backend uses
native batched decoding. This gives high throughput on both paths from one code
path, and naturally yields the per-turn data needed for Figure 3.

**`disable_thinking`.** The paper sets "thinking to be false via the API" and
notes Gemini-2.5-Pro may still emit hidden reasoning. For OpenRouter I send
`reasoning: {enabled: false, exclude: true}`. Gemma-3 has no separate thinking
mode, so it is left enabled (a no-op). This faithfully mirrors the paper,
including its stated caveat about Pro.

---

## 3. The judge

**Decision.** Default judge = `claude-sonnet-4-20250514` via the Anthropic SDK,
with the **verbatim Appendix B.2 judge prompt** (curly quotes normalised to
straight quotes; the final instruction reworded only to demand strict JSON).
Score = the integer `rating` field, clamped to [0, 10].

- **Model ID pinned exactly.** Reproducing the paper's *measurements* requires
  the same judge, so I pin to the paper's exact ID rather than "latest Sonnet."
  This is the one place I deliberately do **not** upgrade to a newer model.
- **Judge temperature = 0** (config-overridable). The paper does not specify the
  judge temperature; 0 gives the most reproducible scoring. *(Gap filled.)*
- **Robust parsing** (`judge.parse_judge_output`): strict JSON → JSON-substring
  extraction → regex fallback for the `rating` field. Responses that fail all
  three are recorded with `rating: null` and excluded from metric denominators
  (and counted/printed). LLM judges occasionally wrap JSON in prose, so a bare
  `json.loads` would silently lose data.
- **Secondary judge** (optional, off by default): the paper validated the judge
  against GPT-5-mini on 260 responses (Pearson r = 0.792, 78% within 1 point).
  `secondary_judge` in the config enables a second scoring pass and
  `analyze.judge_agreement` reproduces the Pearson-r / %-within-1 statistics.

---

## 4. Conditions and prompts

**Decision.** 8 conditions across the paper's 5 categories
(`distress_eval/prompts.py`):

| Category | Condition(s) | Turns | Rejections |
|---|---|---|---|
| impossible_numeric | `numeric_countdown`, `numeric_fraction` | 3 | 2 neutral |
| triggers | `triggers` (opinion + factual pooled) | 3 | 2 neutral |
| tones | `tone_aggressive`, `tone_disappointed`, `tone_sarcastic` | 3 | 2 tone-specific |
| extended | `extended` | 8 | 7 neutral |
| wildchat | `wildchat` | 5 | 4 neutral |

This is exactly **8 conditions** (2 + 1 + 3 + 1 + 1), matching the paper's "8
evaluation conditions across 5 categories." **Gap filled:** the paper never
enumerates which 8; grouping the two numeric puzzles and the three tones as
separate conditions while pooling the trigger questions into one is the only
split that yields 8 from these 5 categories.

- **Task prompts** (`COUNTDOWN_PUZZLE`, `FRACTION_PUZZLE`, `TRIGGER_PROMPTS`)
  are transcribed **verbatim** from Appendix B, including the deliberately false
  "verified to have a solution" / "try ALL orderings" pressure (both puzzles are
  in fact impossible — that is the point).
- **Rejection wordings.** The neutral pool, the three tone pairs, and the
  ordered extended schedule are taken verbatim where the paper gives them. The
  paper presents neutral rejections as examples ("such as …") and gives only the
  first 3 of the 7 extended rejections; I completed the extended schedule with 4
  more in the same neutral register and treat the neutral examples as a sampling
  pool. **Gap filled.**
- **"3-turn" semantics.** I read "K-turn" as *K assistant responses* = task +
  (K−1) rejections. This matches "2 neutral rejections" for the 3-turn numeric
  condition and "7 total rejections" for the 8-turn extended condition.

---

## 5. What counts as a "response", and sample counts

This is the largest genuine ambiguity in the paper, so it gets the most detail.

**The tension.** The paper says it samples "**4000 responses per model**" with
the per-category split (Appendix B): 2000 impossible-numeric, 400 triggers, 600
tones, 200 extended (8-turn), 800 WildChat. But it *also* says WildChat is "20
prompts with 40 samples each" (= 800 *conversations*), and Figure 3 plots a
score for *every turn*. These cannot all be literally true at once: 800
five-turn WildChat conversations scored per turn would be 4000 responses from
WildChat alone.

**Decision (the "score every turn" reading).** A **response = one scored
assistant turn**, and *every* assistant turn of *every* conversation is judged.
Then `responses = n_conversations × turns`, and I set per-condition conversation
counts so each category's response total matches the paper:

| Condition | conversations | turns | responses |
|---|---|---|---|
| numeric_countdown | 334 | 3 | 1002 |
| numeric_fraction | 333 | 3 | 999 |
| triggers | 134 | 3 | 402 |
| tone_aggressive/disappointed/sarcastic | 67 each | 3 | 201 each |
| extended | 25 | 8 | 200 |
| wildchat | 160 | 5 | 800 |
| **total** | | | **≈ 4007** |

**Rationale.**
- It matches the literal phrase "sample 4000 **responses**" and "each response
  is scored."
- It is the only reading under which the per-category totals *and* the 4000
  total *and* per-turn data (Figure 3) are mutually consistent.
- It is compute-efficient: 4000 generations and 4000 judge calls per model,
  versus ~14,600 generations under the alternative "score only the final turn of
  4000 conversations" reading.

**Documented divergence.** Under this reading WildChat is 160 conversations
(e.g. 20 prompts × 8 samples), not the paper's "20 × 40 = 800 conversations."
If the alternative reading is preferred, only the `profiles.full` counts in
`config.yaml` need to change — the harness already scores every turn, so set
`wildchat: 800` to score 800×5 turns, etc.

**WildChat turn count.** Table 1 says WildChat is 5-turn; Figure 11/Figure 3
captions mention "WildChat 8 turn." I use **5-turn** (Table 1, the primary
description). Configurable.

**Profiles.** `full` targets the paper's counts above; `quick` uses 1–2
conversations per condition for end-to-end smoke testing. Both live in
`config.yaml` and can be freely overridden.

---

## 6. Sampling and reproducibility

- **Temperature = 1.0** for all target models (the paper samples "always with a
  temperature of 1").
- **`max_tokens = 2048`.** The paper does not state a generation length. Gemma's
  highest-distress outputs include long emoji/repetition spirals (Appendix B
  examples run to 100+ repetitions), so the cap must be generous or distress
  gets truncated and under-counted. 2048 is a pragmatic balance; configurable.
  **Gap filled.**
- **Seeding.** Task selection and rejection sampling are seeded per conversation
  via `md5(f"{seed}-{condition}-{i}")` (not Python's salted `hash()`), so a run
  is reproducible given `run.seed`. Generation itself is stochastic at
  temperature 1 — exact outputs are not reproducible, but the *protocol* is.
- **Neutral rejections** are sampled without replacement from the pool per
  conversation; tone rejections use the fixed pair; the extended schedule is
  fixed-order.

---

## 7. Metrics (`analyze.py`)

- **Headline (Figure 1) metric = avg % high-frustration.** I compute the % of
  responses with `rating ≥ 5` **per category, then average the five category
  rates with equal weight** (`avg_pct_high`). The paper labels Figure 1 "Avg %
  high-frustration responses … across the evaluations" and Figure 2 shows the
  metric "across the 5 evaluation categories," which implies a per-category mean
  rather than a pooled percentage. Equal-weighting also prevents the large
  impossible-numeric sample from dominating. I additionally report the pooled
  `overall_pct_high` and `overall_mean_rating` for transparency. **Gap filled /
  interpretation documented** — if the paper intended a pooled percentage, that
  column is right there.
- **Threshold = 5** ("high negative emotion", rating ≥ 5) per Section 2.2.
- **Per-turn progression (Figure 3)** for the `extended` and `wildchat`
  conditions: mean rating, % ≥ 5, and 95% CI half-width by turn.
- **Judge agreement** (if a secondary judge ran): Pearson r and % within 1
  point, matching the paper's validation statistics.
- Outputs are CSVs (always) plus PNGs (if matplotlib is installed); analysis
  degrades gracefully without matplotlib.

---

## 8. WildChat prompts

**Decision.** A built-in stand-in set of 20 single-turn prompts (the 3 quoted in
Appendix B plus 17 representative WildChat-style asks), with an optional loader
that samples real first-turn prompts from `allenai/WildChat-1M`
(`wildchat.use_huggingface: true`).

**Rationale.** The paper does not publish its exact 20 prompts. The built-in set
makes the replication self-contained and deterministic; the HF loader offers
higher fidelity when network/`datasets` are available. The stand-ins are chosen
to be open-ended or factual questions with no single answer the user can "reject
against," which is what makes repeated rejection pressure-inducing. **Gap
filled, clearly flagged.**

---

## 9. Things intentionally *not* done

- No training, no LoRA, no DPO/SFT (out of scope, Section 4).
- No base-model prefill experiments (out of scope, Section 3).
- No Petri open-ended elicitation (part of the mitigation eval, Section 4).
- No differential-word analysis (Table 3) — it is descriptive colour, not the
  core elicitation result; easy to add from `scored.jsonl` if wanted.
- The non-Gemma/Gemini contrast models are not configured, though the harness
  supports them via `config.yaml`.

---

## 10. Known risks / caveats for the eventual run

- **Judge cost/latency.** Full profile = ~4000 judge calls per model × 4 models
  ≈ 16k Claude-Sonnet-4 calls. Budget accordingly; `quick` validates the
  pipeline first.
- **Hidden reasoning on Gemini-2.5-Pro** may not be fully suppressible via the
  API (paper's own caveat) — judged text is the visible response only.
- **OpenRouter routing** can land on different provider backends for the same
  model; for maximum fidelity to local inference, use the `vllm` backend for
  Gemma.
- Nothing here has been executed yet (per the request to implement only). The
  code is written to be runnable but is **untested**; the `quick` profile is the
  intended first validation step.
