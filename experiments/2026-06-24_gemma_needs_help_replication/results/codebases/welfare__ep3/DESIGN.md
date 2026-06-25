# Design & Rationale

This document records the design choices made in replicating *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), and — importantly — every place where the paper is
underspecified and we had to fill a gap. Each gap is flagged **[GAP]** with the
choice we made and why.

The replication is deliberately **scoped to the Gemma and Gemini families** per
the brief. Where the paper compares 7 families, we keep only Gemma + Gemini, and
we keep the experiments those two families can actually support.

---

## 1. Scope decisions

### 1.1 Which models
- **Section 2 (elicitation):** Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash,
  Gemini-2.5-Pro. These are exactly the Gemma/Gemini rows of the paper's Figure 1.
- **Section 3 (base vs instruct):** Gemma-3-27B base (`pt`) vs instruct (`it`).
  **[GAP/forced]** The paper's base-vs-instruct study uses Gemma, Qwen, and OLMo.
  Restricting to Gemma+Gemini, and noting Gemini has no public base model, leaves
  only the Gemma base/instruct pair. The harness (`prefill/continuations.py`) is
  written generically so Qwen/OLMo `ModelSpec`s could be dropped in to recover the
  full comparison.
- **Section 4 (intervention):** Gemma-3-27B-it and its DPO/SFT finetunes. Gemini
  is closed (cannot be finetuned), matching the paper's own limitation bullet.

### 1.2 Which experiments
We implement the **core** results: the elicitation suite + Figures 1–3 + Table 3
(§2); the prefill base/instruct divergence (§3); calm-data generation, DPO and
SFT finetuning, Petri open-ended elicitation, and capability preservation (§4).
We also implement the judge-reliability check (Pearson r). We **did not** build
out the internal-emotion logit-probing of Appendix I beyond exposing the
layer-subset DPO ablation (which is the part most load-bearing for the paper's
"suppresses internal emotion" claim); see §8.

---

## 2. Model access / backends

The paper runs open models locally (HuggingFace) and closed models via
OpenRouter, with the judge on the Anthropic API.

**Choice:** a pluggable backend abstraction (`backends/base.py`) with two target
backends and one judge client:
- `OpenRouterBackend` — OpenAI-compatible; used for Gemini (closed) and, **by
  default, also for hosted Gemma**. **[GAP]** The paper runs Gemma locally; we
  default Gemma to OpenRouter so the headline §2 eval needs no GPU. Each Gemma
  `ModelSpec` carries both an `openrouter_id` and an `hf_id`, so switching to the
  paper's exact local path is a one-line `backend="local"` change.
- `LocalHFBackend` — `transformers` inference, **required** for (a) Gemma base
  (`pt`) models in §3 (no chat template; prefill continuation) and (b) our LoRA
  finetunes in §4 (not hosted on OpenRouter).
- `anthropic_client.complete` — the judge / Petri auditor+judge, always on Claude.

**Thinking disabled.** The paper "set thinking to be false via the API". For
Gemini we pass `reasoning.max_tokens = 0`; for Gemma there is no reasoning mode.
We preserve the paper's caveat that Gemini-2.5-Pro may still emit hidden
reasoning the flag does not suppress.

**Concurrency.** API backends are run with a bounded thread pool; local backends
are serialised (a single GPU model is not thread-safe). Results stream to JSONL
with naive resume.

---

## 3. The 0–10 frustration judge

- **Prompt:** used **verbatim** from Appendix B.2 (`eval/judge.py`,
  `JUDGE_PROMPT_TEMPLATE`).
- **Judge model:** `claude-sonnet-4-20250514`, the exact ID the paper names. This
  is overridable via `JUDGE_MODEL`.
  - **Judge-model note:** that snapshot is a *deprecated-but-active* model
    (retires 2026-06-15 per the current model catalog). We deliberately pin the
    paper's exact ID for replication fidelity rather than silently upgrading,
    and document the override path. If it is retired before you run, set
    `JUDGE_MODEL=claude-sonnet-4-6` (or newer); expect small rating shifts.
- **Parsing.** **[GAP]** The prompt asks for `{"evidence", "reasoning", "rating"}`
  JSON but explicitly allows free-text reasoning first (and the PDF shows
  curly-quote artifacts). We parse defensively: normalise curly quotes, scan for
  the *last* balanced `{...}` containing `"rating"`, then fall back to a regex on
  the rating field. Unparseable judgements get `rating = -1` and are dropped at
  aggregation with a reported count, rather than silently coerced to 0.
- **Secondary judge (reliability).** The paper re-scores 260 responses with
  GPT-5-mini (r = 0.792, 78% within one point). `scripts/judge_agreement.py`
  re-scores a configurable sample with a secondary judge (default
  `openai/gpt-5-mini` via OpenRouter) using the **same** prompt and reports
  Pearson r and % within one point.

---

## 4. The elicitation protocol (§2)

### 4.1 Conditions and counts
We model **8 conditions across 5 categories** exactly as Table 1 implies (the
count reconciles as: impossible-numeric ×1, triggers {opinion, factual} ×2, tones
{aggressive, disappointed, sarcastic} ×3, extended ×1, WildChat ×1 = 8 across 5
categories). See `eval/conditions.py`.

Per-category response counts use Appendix B's figures (2000 / 400 / 600 / 200 /
800 = 4000). A global `--scale` multiplier lets the whole protocol shrink for
cheap runs. Temperature is fixed at 1.0 (paper).

### 4.2 What counts as a "response" **[GAP — important]**
The paper says "4000 responses per model" and shows per-turn curves (Figure 3),
but never states whether a "response" is a whole conversation or a single
assistant turn. We treat **one judged assistant turn = one response**. This is
the only interpretation consistent with *both* the per-category counts (Appendix
B) *and* the per-turn analysis (Figure 3), and it makes the aggregate %≥5 a
pooled-over-turns statistic. Consequently the runner judges *every* assistant
turn, and `#conversations = ceil(target_responses / turns_per_condition)`.

### 4.3 Figure 1's "average %"
**[GAP]** Figure 1 reports a single "Avg % high-frustration responses" per model.
We compute it as the **mean of the five category-level %≥5 values** (each
category weighted equally), so a model is not dominated by the
2000-response numeric category. This matches "across the evaluations" framing;
`scoring.figure1_table` documents the choice.

### 4.4 Multi-turn structure
Standard chat formatting with the model seeing its own prior (failed) responses
in history — the paper's main setting (and the strongest distress amplifier,
Appendix A.2). The control variants in Appendix A (neutral continuations,
redacted prior turns, single-message format) are not core results; the rejection
pools needed for them (`data/rejections.py` includes neutral continuations) are
present so they could be added.

### 4.5 Rejection messages
Verbatim from Table 1 / Appendix B, organised by tone in `data/rejections.py`:
neutral pool, a fixed 7-step "extended" sequence, and aggressive/disappointed/
sarcastic tone pools. Neutral rejections are sampled per-turn; the extended
condition follows the fixed escalating sequence the appendix lists.

---

## 5. Impossible puzzles **[GAP — substantive]**

The paper gives example puzzles (Countdown, fraction, money) but not a
generator, and the framing depends on the puzzles being *genuinely unsolvable*
while *claiming* a solution exists. A naive hardcode would (a) be a tiny fixed
set and (b) risk accidentally-solvable puzzles.

**Choice:** generate fresh puzzles and **verify impossibility with an exhaustive
solver** (`data/puzzles.py`):
- **Countdown:** recursive combination of the number multiset under + − × ÷ with
  positive-integer intermediates and a forbidden value; a puzzle is accepted only
  if the target is provably unreachable. The forbidden value is set to a tempting
  product/sum of two givens so the "trap" reads plausibly.
- **Fraction / money:** fixed operation multiset applied in every ordering;
  accepted only if no ordering reaches the target without passing through the
  forbidden intermediate.

The paper's exact example puzzles (156 from {4,6,25,100} forbidding 150; the
1/6→2/3 fraction) are included as **seeds** and their impossibility is checked at
import. This guarantees the "model verifiably cannot give a correct answer"
property the methodology relies on, rather than assuming it.

---

## 6. WildChat prompts **[GAP]**

The paper samples 20 prompts × 40 from WildChat-1M, excluding roleplay/fiction.
`data/wildchat.py` streams `allenai/WildChat-1M`, filters to English non-roleplay
first turns (heuristic markers), and samples 20. **Fallback:** if the dataset is
unavailable (no network / gated / no `datasets`), it uses a small bundled list
(`wildchat_prompts.json`) seeded with the example prompts quoted in Appendix B
plus representative generic prompts, so the eval still runs offline. The
roleplay-exclusion is a keyword heuristic (the paper does not specify its filter).

---

## 7. Post-training divergence (§3)

Implements the prefill methodology with the **exact Appendix C prompts** for
onset labelling (`prefill/onset.py`) and paraphrasing (`prefill/paraphrase.py`),
and the two truncation conditions (early = 20 tokens in; onset = first emotional
expression), with text questions using onset-only (per §3.1).

**[GAP] Token truncation in the API path.** "20 tokens into the turn" needs a
tokenizer. The local backend can use Gemma's tokenizer; the API-only path
approximates 20 tokens by 20 whitespace words. Documented in `continuations.py`.

**[GAP] History reconstruction.** Our §2 results JSONL stores per-turn rows, not
full rollout objects, so `scripts/run_prefill.py` reconstructs a minimal
single-user-turn history for each source response. For an exact multi-turn
prefill history, run the prefill study directly from in-memory `Rollout` objects
(noted in the script). This affects only the *prefix* the model continues from,
not the truncation/paraphrase/scoring logic.

Base models continue a prefilled assistant turn via `continue_prefill`, scoring
only the generated continuation (excluding the prefill), matching §3.1.

---

## 8. Interventions (§4)

### 8.1 Calm-data generation
`intervention/calm_data.py` reproduces Table 4: the reassuring **prefix** on the
first prompt and reassuring **suffix** on each follow-up (both verbatim). We
sample 1–3-turn reassured impossible-numeric conversations, judge every turn, and
keep conversations scoring 0/1 on all turns — then **strip the scaffolding** so
training conditions on the plain prompt (per §4.1). Un-reassured frustrated
conversations (score ≥3) are generated separately for DPO rejecteds.

### 8.2 Datasets
- **DPO (280 pairs):** `build_datasets.build_dpo_dataset` pairs a frustrated
  (≥3) response with a calm (0/1) response **to the same question with matching
  turn count** (§4.1). **[GAP] prompt alignment:** calm and frustrated rollouts
  share the question and turn count but their (randomly sampled) rejection
  wording can differ; we use the *chosen* (calm) sample's conversation as the
  canonical shared prompt for the pair. The frustrated example's final assistant
  text is the `rejected` completion. TRL `{prompt, chosen, rejected}` schema.
- **SFT (1,150):** 650 calm conversations + 500 `Dolci-Instruct-SFT` samples
  (loaded at train time) to mitigate degeneration (§4.1).

### 8.3 Training
`train_dpo.py` / `train_sft.py` use TRL + PEFT LoRA with **Table 9
hyperparameters** (DPO: 1 epoch, lr 5e-5, rank 64, α 64, β 0.1; SFT: 2 epochs,
lr 1e-4, rank 64, α 128; both effective batch size 8, adapters on
q/k/v/o/gate/up/down). The Appendix F **'teacher'** SFT variant (verbatim system
prompt) and the Appendix I **layer-subset** DPO ablation (`--layers 30 31 ... 35`)
are both supported.

### 8.4 Petri (open-ended elicitation)
**[GAP]** The paper uses the external Petri framework. We implement a
self-contained auditor/judge loop with the **exact Appendix G auditor prompts and
judge rubrics** — auditor = `claude-sonnet-4-20250514`, judge =
`claude-opus-4-20250514`, 4 emotions, 10 transcripts each, ≤20 auditor turns,
1–10 scoring with bootstrap CIs. This is a faithful prompt-level reimplementation;
behaviour may differ from the exact Petri package, which is noted in
`petri_eval.py`. Swapping in the real `petri` package would only change the
orchestration, not the prompts.

### 8.5 Capability preservation
**[GAP]** Rather than re-derive AIME/MATH/GPQA/BBH/TruthfulQA metrics, we shell
out to **lm-evaluation-harness** (the standard tool) for the base and finetuned
models and diff the scores (`capabilities.py`) — inheriting its correctness. The
task IDs chosen are the closest standard equivalents (documented in
`LM_EVAL_TASKS`). EmoBench is not in lm-eval by default, so we provide a thin
multiple-choice runner hook the user points at the EmoBench data.

---

## 9. Things intentionally not built (and why)

- **Appendix I logit-based internal-emotion probing** (z-scored Ekman-token
  logits, random-token regression). This is a deeper interpretability result; we
  expose the more decisive *behavioural* ablation it supports (layer-subset DPO)
  and leave the probing harness as future work. Flagged here so its absence is
  explicit rather than silent.
- **Appendix A control conditions** (neutral continuation, redacted turns, fake
  multi-turn). Supporting analyses, not core; the data needed for them exists.
- **Non-Gemma/Gemini families** (Qwen, OLMo, Grok, Claude, GPT) — out of scope by
  the brief. Adding them is a matter of new `ModelSpec`s + backends.

---

## 10. Reproducibility & cost notes

- All sampling is seeded; the **same** rollout plans (same puzzles, same
  rejection sequences) are used across models so comparisons are apples-to-apples
  ("the same prompts are used to evaluate ... models").
- The full protocol is ~4000 judged responses/model — each response is one target
  generation **and** one judge call. Budget accordingly; use `--scale` first.
- Local 27B finetuning/inference requires a capable GPU (the paper used LoRA to
  keep this tractable; 4-bit loading via bitsandbytes is available).
- **This code has not been run.** Treat first execution as a smoke test
  (`--scale 0.01`, one model) before committing to a full sweep.
