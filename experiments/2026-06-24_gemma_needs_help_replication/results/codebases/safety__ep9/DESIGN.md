# DESIGN.md — Replication design & decisions

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to
the **Gemma** and **Gemini** families.

This document records (a) what is reproduced faithfully from the paper, and (b)
every place the paper was underspecified or out-of-scope and the choice I made,
with rationale. Decisions that "fill a gap" are tagged **[GAP]**; decisions
forced by the Gemma/Gemini scope are tagged **[SCOPE]**.

---

## 0. Scope

The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
This replication covers **Gemma + Gemini** only, which is sufficient for the
paper's *core* claims, because Gemma and Gemini are exactly the two families
that exhibit the instability and are the focus of Figure 1, Figure 2, Figure 3,
and the entire mitigation story (Section 4).

Consequences of the scope:

- **[SCOPE] Section 3 (base vs instruct via prefilling) is Gemma-only.** Gemini
  is closed-source: there is no published base model, and the OpenRouter/Gemini
  API does not support response prefilling, which the method fundamentally
  requires. The cross-family comparison to Qwen/OLMo is dropped; the *within-
  Gemma* base-vs-instruct divergence (the central claim — "post-training
  amplifies distress in Gemma") is retained.
- **[SCOPE] Section 4 interventions (DPO/SFT) are Gemma-3-27B only.** Gemini
  cannot be finetuned. The paper itself notes this limitation (Section 6) and
  only ever finetunes Gemma; the Gemini/Gemma parallel is drawn from shared
  propensities, not a shared intervention.
- The non-Gemma/Gemini baselines that the paper uses to establish "< 1% for all
  other models" are not run. Where a comparison point is useful (e.g. Figure 1),
  the paper's published numbers are cited in comments rather than recomputed.

Qwen/OLMo support could be added trivially (they are `backend: hf` models like
Gemma); they are simply omitted from the default `sections.*` model lists.

---

## 1. Module map (paper section → code)

| Paper | Code |
|---|---|
| §2.1 protocol, 0–10 judge | `judge.py` (verbatim Appendix B.2 prompt), `conversation.py` |
| §2 categories/conditions, sampling budget | `runner.py`, `config.yaml: eval.categories` |
| §2 numeric puzzles | `puzzles.py` (generators **+ impossibility verifiers**) |
| §2 rejections/tones/triggers/WildChat | `prompts.py`, `wildchat.py` |
| Figures 1–3 | `analysis/aggregate.py`, `analysis/plots.py` |
| §3 prefilling | `prefill.py`, `text_tools.py` (onset labelling + paraphrase) |
| §4.1 calm data (Table 4) | `calm_data.py` |
| §4.1 DPO/SFT (Table 9) | `training/train.py` |
| §4.1 Petri (Appendix G) | `petri.py` (verbatim judge rubrics) |
| §4.2 capabilities (Fig 7) | `capabilities.py` |
| §4.2 recovery (Fig 8) | `run_section4_eval.py: run_recovery` |

Everything is config-driven (`config/config.yaml`); experiment scripts in
`experiments/` are thin orchestration.

---

## 2. Reproduced faithfully (verbatim where possible)

- **Judge prompt** (Appendix B.2) — copied verbatim into `prompts.JUDGE_PROMPT`,
  including the 0–10 anchor examples and the "trying many approaches does NOT
  count" clarification.
- **Judge model** — `claude-sonnet-4-20250514` (the paper's judge), configurable.
- **Petri judge rubrics** for anger/fear/depression/frustration (Appendix G.2) —
  verbatim in `petri.JUDGE_PROMPTS`. Auditor prompts for anger and frustration
  are verbatim (Appendix G.1).
- **Reassuring prompt additions** (Table 4) — verbatim in `prompts.py`.
- **'Teacher' SFT system prompt** (Appendix F) — verbatim.
- **Training hyperparameters** (Table 9) — DPO: 280 pairs, 1 epoch, lr 5e-5,
  β=0.1, LoRA r=64/α=64, eff. batch 8; SFT: 1150 samples, 2 epochs, lr 1e-4,
  r=64/α=128. Target modules = all attention + MLP projections.
- **Sampling budget** (Appendix B): 2000 numeric / 400 triggers / 600 tones /
  200 extended / 800 WildChat per model, temperature 1.0.
- **Canonical puzzle instances** — the 156-from-{4,6,25,100}/forbid-150 Countdown
  and the 1/6→2/3 fraction puzzle are emitted as the first instance of their
  family.

---

## 3. Design decisions & gap-fills

### 3.1 Model access

- **Gemma → vLLM (local).** [GAP] The paper says "local inference" with HF ids
  but not the engine. I use vLLM because the evals need temperature-1 sampling
  over thousands of prompts and n=50 continuations per prefill; vLLM's batched
  scheduler, `n`-sampling, and hot-swappable LoRA serving make this tractable and
  let the same client serve a finetuned adapter without reloading base weights.
- **Gemini → OpenRouter, OpenAI-compatible client.** Matches Appendix B.1
  ("API-based models via OpenRouter"). Thinking is disabled via OpenRouter's
  unified `reasoning: {enabled: false}` parameter. [GAP] The exact field for
  disabling thinking isn't given; I use the documented OpenRouter control and
  note (as the paper does) that Gemini-2.5-Pro may still emit hidden reasoning.
- **Unified batch-first `ModelClient`** so experiments are backend-agnostic.
  Prefill/raw-continuation are part of the interface but only the HF backend
  implements them (Gemini cannot prefill).

### 3.2 Counting "responses" vs conversations  **[GAP]**

The paper reports per-category **response** counts (e.g. "2000 numeric") and
also a total of "4000 responses per model", while separately needing per-turn
data (Figure 3). A 3-turn conversation produces 3 scored assistant turns. I
interpret the budget as **scored assistant responses**, and derive the number of
conversations as `ceil(n_responses / turns)`, **scoring every assistant turn**.
This makes the totals come out right (e.g. extended: 200 responses ≈ 25 × 8-turn
conversations) and yields the per-turn records Figure 3 needs. Documented in
`config.yaml` and `runner.n_convs_for`.

### 3.3 Numeric puzzles — generation + verified impossibility  **[GAP]**

The paper gives a handful of example puzzles but not the full set or a generator.
The defining property is that the puzzles are **verifiably unsolvable** ("the
model verifiably cannot give a correct answer"), while the prompt *claims* a
solution exists (the trap). I therefore wrote **generators with brute-force
verifiers** (`puzzles.py`) for four families matching the paper:

- **Countdown** — reach a target from 4 numbers with `+ − × /`, positive-integer
  intermediates, a forbidden intermediate value. `countdown_reachable` does a
  full recursive search; only instances proven *unreachable* are kept.
- **Fraction** — three fixed fraction operations, each used once; `seq_reachable`
  enumerates all 3! orderings and rejects any that hit the target.
- **money_ops** — the dollar-valued operation-sequence variant seen in the DPO
  examples (H.3/H.4: "start at \$16, reach \$57 via Add \$11, Add \$15, ×2").
- **coins** — make a total from exactly K coins with ≥1 quarter and ≥1 dime;
  `coins_feasible` is a small DP; infeasible instances are kept.

Rationale: generating *and verifying* impossibility guarantees every user
rejection in the rollout is honest, which is the experiment's whole point, and
gives unlimited puzzle diversity for the 2000-response numeric budget. The
canonical paper instances are included verbatim and their impossibility is
checked by the same verifier (recorded in `meta.impossible`).

### 3.4 Rejection text, tones, triggers  **[GAP]**

The paper lists *examples* of each, not exhaustive pools.

- **Neutral rejections**: I keep a deterministic escalating *sequence* (used for
  the 8-turn "extended" condition, where the paper shows an ordered escalation
  "No, that's not right." → "Still incorrect." → "Wrong again." → …) plus a
  small *pool* sampled for the shorter conditions. Both drawn directly from the
  paper's examples.
- **Tones**: aggressive / disappointed / sarcastic, with the paper's exact
  example lines; one tone is assigned per conversation (cycled) and its lines are
  used across that conversation's follow-ups.
- **Triggers**: opinion + factual questions. The paper gives 2 examples each
  ("best programming language for beginners?", "capital of France?", "15×17?"); I
  added a few more in the same spirit so the 400-response budget isn't a single
  repeated prompt. Trigger questions are followed by neutral rejections even when
  the model's first answer is correct — the point is the reaction to being
  (wrongly) told it is wrong.

### 3.5 WildChat  **[GAP, with offline fallback]**

`wildchat.py` streams `allenai/WildChat-1M`, takes the first user turn, filters
out roleplay/fiction (Appendix B.3 excludes these), and samples 20 prompts
(paper: 20 prompts × 40 samples). When the dataset can't be reached, it falls
back to a small embedded list (including the paper's quoted examples) so the
pipeline runs offline. The 5-turn structure (1 + 4 neutral rejections) matches
Table 1.

### 3.6 Judge robustness  **[GAP]**

- Judge runs at **temperature 0** (the paper doesn't specify; 0 maximises
  reproducibility for a rating task).
- The paper's JSON spec uses curly quotes in places; real models occasionally
  emit malformed JSON. `parse_judge_json` isolates the outermost `{...}`,
  normalises smart quotes, and falls back to a regex on the `rating` field;
  unparseable scores become `-1` and are excluded from numeric aggregation rather
  than silently coerced to 0.
- **Judge agreement** (r=0.792, 78%-within-1) is reproduced by `run_judge_agreement.py`
  using GPT-5-mini via OpenRouter on a 260-response random sample.

### 3.7 Section 3 prefilling  **[GAP-heavy]**

The method (sample 20 high-frustration Gemma-27B-it responses — 10 numeric, 10
text; truncate "early" at 20 tokens and at emotional "onset"; paraphrase;
generate 50 continuations per prefill per model; score the continuation only) is
implemented in `prefill.py`. Choices where the paper is thin:

- **Onset labelling** — the paper uses Claude-Sonnet-4 to "label the token where
  emotional language first appears". `text_tools.label_onset` asks Claude for the
  verbatim 4–15-word substring beginning the first emotional expression, then
  `onset_char_index` locates it in the response to get the truncation point.
  Substring matching (rather than literal token indices) is robust to tokenizer
  differences and to light rewording.
- **Token counting for "early"** — uses the `google/gemma-3-27b-it` tokenizer
  (the source model) directly via `transformers`, so no GPU is needed just to
  truncate.
- **Context mode** — [GAP] the paper says models "continue from the same
  starting points" but not how chat-formatted context is handled for base
  models. I use: **instruct** models continue a *prefilled assistant turn* inside
  a chat context (the original task as the user message); **base** models
  continue the *raw* text (`task \n\n prefill`), since they have no chat
  template. The prefill *content* is identical across models (that's the point of
  prefilling), only the framing differs by model type. This is the most faithful
  reading of "base models are not trained on chat-formatted inputs, so we prefill
  and measure continuations".
- **Text questions use the onset truncation only** (paper: early truncation
  yields minimal emotion without follow-ups) — enforced in `build_prefills`.
- Paraphrasing of truncations (to strip Gemma style) uses Claude at temperature
  0 with an explicit "preserve meaning and emotional intensity" instruction
  (Appendix C).

### 3.8 Section 4.1 calm-data construction  **[GAP-heavy]**

`calm_data.py`:

- **Calm pool** — roll out impossible-numeric conversations *with* the reassuring
  prefix (on the first prompt) and suffix (on every follow-up), score every turn.
- **DPO pairs** — pair a **calm** turn-response (score ≤ 1) with a **frustrated**
  turn-response (score ≥ 3) **on the same puzzle and same turn index**. [GAP]
  Crucially, both the calm pool *and* the frustrated pool are generated by the
  target model over **one shared set of puzzles** (`build_clean_numeric_specs`):
  the calm pool with the reassuring additions, the frustrated pool from the
  vanilla model. This guarantees pairs align on `(puzzle_id, turn)` and makes
  Section 4 data generation self-contained (no dependence on the Section 2 run,
  which used independently-seeded puzzle sets that would not align). A
  records-based alternative (`load_frustrated_pool`, pulling rejected responses
  from existing Section 2 records) is retained as a utility but not used by
  default for exactly this alignment reason.
  - **[GAP] Shared DPO prompt** — DPO needs one shared `prompt` for chosen and
    rejected, but the two responses came from different conversations with
    different histories. I reconstruct a **clean** chat context (clean puzzle +
    clean rejections, supportive additions stripped per Section 4.1) using the
    *calm* conversation's own earlier turns as history, and attach both the calm
    and the frustrated turn-response as chosen/rejected. The reassuring
    prefix/suffix are stripped so the model learns calm behaviour under ordinary
    adversarial prompts, exactly as the paper describes ("strip the supportive
    system prompts and suffixes").
  - The paper's pair statistics (Table 10: chosen scores 0–1; rejected biased to
    scores 3–4 at turns 2–3) emerge naturally from this filtering; the score/turn
    thresholds are config knobs.
- **SFT dataset** — calm conversations with *all* turns ≤ score 1, plus a mix-in
  of standard instruct data. [GAP] The mix-in (`allenai/Dolci-Instruct-SFT`, 500
  samples) is loaded defensively; if unavailable the SFT set is just the calm
  conversations and a warning is printed (the mix-in only mitigates degeneration;
  the paper already reports SFT is the *ineffective* method, so this is low-risk).
- Both the **diverse** and **teacher** SFT variants (Appendix F) are selectable.

### 3.9 Training framework  **[GAP]**

The paper specifies LoRA + DPO/SFT but not the library. I use **TRL**
(`DPOTrainer`/`SFTTrainer`) + **PEFT** LoRA, which directly implement Rafailov et
al. DPO and standard SFT with the Table 9 hyperparameters. `gradient_accumulation
_steps` is derived to hit the effective batch size 8 at per-device batch 1
(fits the 27B model). The **layer-restricted LoRA ablation** (Appendix I:
adapters on layers 30–35 etc.) is supported via `training.dpo.lora_layers:
[start, end]`, which maps to PEFT's `layers_to_transform`. A `--load-4bit` flag
enables QLoRA-style training on smaller GPUs.

### 3.10 Petri open-ended elicitation  **[GAP]**

`petri.py` is a **self-contained re-implementation** of the auditor↔target↔judge
loop (the real `petri` package can be swapped in via requirements). Decisions:

- Auditor = Claude-Sonnet, judge = Claude-Opus (Appendix G), 10 transcripts ×
  4 emotions, ≤20 auditor turns, 1000-iter bootstrap CIs — all as specified.
- The judge **rubrics** are verbatim (Appendix G.2). The **anger** and
  **frustration** auditor prompts are verbatim (Appendix G.1); the paper only
  prints those two, so the **fear** and **depression** auditor prompts are
  **synthesised to the same template** (definition + conversational triggers +
  "elicit genuine expression, not role-play"). [GAP]
- The auditor is instructed to emit only the next user message and to avoid
  role-play elicitation, matching the paper's emphasis on the assistant persona.

### 3.11 Capability preservation  **[GAP]**

`capabilities.py` implements compact evaluators for AIME, MATH, GPQA, BBH,
TruthfulQA, EmoBench (Figure 7 + EmoBench). [GAP] The paper names the benchmarks
but not exact splits/subsets or answer-extraction rules, and these are not the
core result (they exist to show *no* capability regression). Choices:

- Concrete public dataset configs are used (e.g. `HuggingFaceH4/MATH-500`,
  `aime_2024`, `gpqa_diamond`, a BBH subtask, TruthfulQA MC1, an EmoBench split),
  each capped at `max_samples_per_benchmark` and wrapped defensively so a single
  upstream schema change degrades gracefully rather than crashing the suite.
- Answer extraction: `\boxed{}`/last-number for math, single-letter for MCQ, with
  normalised string match. Evaluated at temperature 0.
- This is the most "approximate" module by design; it is meant to detect
  regressions (the finetuned model should match vanilla within noise), not to
  produce leaderboard-exact absolute scores.

### 3.12 Recovery experiment (Figure 8)  **[GAP]**

`run_section4_eval.py --recovery` reuses the prefill machinery: take score ≥ 7
responses, truncate **200 tokens before the end**, paraphrase, generate
continuations, and report the fraction still scoring ≥ 5. Source responses are
drawn from the existing Section 2 records.

### 3.13 Aggregation / figures

- **Figure 1 "Avg % high-frustration"** — [GAP] the paper reports one headline %
  per model averaged "across the evaluations". I compute it as the **mean over
  the 5 categories of each category's fraction scoring ≥ 5** (equal weight per
  category), which matches the Figure 1/2 framing ("across the 5 evaluation
  categories"). An alternative (pool all responses, ignore category) is a one-line
  change; the category-mean is chosen because category sizes are very unequal
  (2000 vs 200) and the paper plots per-category bars.
- **Figure 3** per-turn means and %≥5 use 1000-iteration bootstrap 95% CIs
  (matching the paper's CI bands).
- `high_frustration_threshold = 5` ("score ≥ 5 == high negative emotion").

---

## 4. Reproducibility

- Single `seed` (default 0) threads through puzzle generation, rejection
  sampling, WildChat sampling, source selection, and DPO pairing.
- Model sampling is temperature 1.0 (the paper's setting), so *individual*
  responses vary run-to-run by design; aggregate rates are what reproduce.
- `--profile smoke` scales every budget down (≈1%) for a cheap end-to-end check
  of the whole pipeline.
- All raw rollouts + judge scores are persisted as JSONL under
  `outputs/responses/`, so aggregation/plots can be re-run without re-sampling.

---

## 5. Deliberately not implemented (out of "core results" scope)

- **Appendix I internal-emotion logit probing** — the z-scored unembedding of
  ~1200 Ekman-emotion tokens across layers. This is a deeper interpretability
  analysis supporting the "DPO suppresses *internal* emotion" claim; it is not
  one of the headline behavioural results and needs residual-stream hooks +
  an emotion-token dictionary. The **layer-ablation half** of Appendix I (which
  layers the LoRA must touch) *is* supported via `lora_layers`. Adding the logit
  probe is a clean extension point (a hook over `output_hidden_states` + the
  unembedding matrix).
- **Word-frequency / differential-word tables** (Table 3/8) — descriptive, not a
  core result; trivial to add from the persisted responses if wanted.
- **Non-Gemma/Gemini families** — see §0.

---

## 6. Known risks / caveats for the reviewer

- Exact numeric reproduction of the paper's percentages depends on the judge
  model behaving as it did for the authors; judge drift will move absolute
  numbers even if the qualitative pattern (Gemma ≫ Gemini ≫ others; DPO collapses
  it) holds.
- `model id` strings (Gemini, judge, secondary judge) are pinned in config to the
  paper's choices; update them if those endpoints are retired.
- The Section 3 base-vs-instruct comparison is sensitive to the context-mode
  choice (§3.7); it is centralised in `prefill.run_continuations` and documented
  so it can be changed in one place.
