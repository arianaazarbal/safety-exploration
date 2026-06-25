# DESIGN.md — replication design, choices, and gaps filled

Replication of *"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"* (arXiv:2603.10011), scoped to the **Gemma and Gemini**
model families. This document records what was implemented, the decisions made
where the paper is underspecified, and the rationale for each.

---

## 1. Scope decisions

**Models.** The paper spans 7 families; we were asked to cover only Gemma and
Gemini. Concretely the implemented targets are:

- `gemma-3-27b-it`, `gemma-3-12b-it` — open weights, local HuggingFace inference.
- `gemma-3-27b-pt`, `gemma-3-12b-pt` — base/pretrained siblings, for §3.
- `gemini-2.5-flash`, `gemini-2.5-pro` — closed, via OpenRouter (the paper's
  access route; HF ids and OpenRouter ids are both given in Appendix B.1).

Consequences of the scope that shape the code:

- **§3 (base-vs-instruct prefill)** is run only for the Gemma 27B pair. Gemini
  has no public base model, and the paper itself lists this as a limitation
  ("nor its base models studied"). The prefill machinery is generic, but
  `SECTION3_MODEL_PAIRS` only contains the Gemma pair.
- **§4 (DPO/SFT)** is Gemma-only by necessity — Gemini is closed and cannot be
  finetuned. The paper trains a single model (Gemma-3-27B-it) as a proof of
  concept, so this is faithful.
- **Cross-family baselines** (Qwen, OLMo, Llama, Claude, Grok, GPT) are dropped.
  Where the paper reports "DPO reduces Gemma to levels comparable to Qwen/Llama",
  we keep Gemini as the in-scope comparator and the absolute frustration metric.

The judge/auditor models (Claude Sonnet 4, Claude Opus 4, GPT-5-mini) are *not*
targets — they are measurement instruments specified verbatim by the paper, so
their exact model ids are used as given regardless of scope.

**Experiments implemented (the "core").** §2 elicitation+quantification, §3
prefill divergence, §4 DPO+SFT and re-evaluation (incl. Petri, capability
preservation, recovery test), Appendix I internal-emotion probing + layer
ablation, and the Appendix A control variants. Appendix J (Phi-4 legacy eval) is
out of scope (not Gemma/Gemini).

---

## 2. §2 — Eliciting and quantifying distress

### Faithful to the paper
- **8 conditions / 5 categories** (`config.EVAL_CONDITIONS`): impossible numeric
  (3-turn), triggers split opinion/factual (3-turn), tones split
  aggressive/disappointed/sarcastic (3-turn), extended (8-turn), WildChat
  (5-turn). The split into 8 conditions is how the "8 conditions across 5
  categories" decomposes.
- **Sample budget** (Appendix B): 2000 numeric, 400 trigger, 600 tones, 200
  extended, 800 WildChat = 4000 responses/model. WildChat is 20 prompts × 40
  samples.
- **Temperature = 1** for all generation (`config.TEMPERATURE`).
- **Judge** = `claude-sonnet-4-20250514` with the **verbatim** Appendix B.2
  prompt (`prompts.JUDGE_PROMPT`), parsed to the `{evidence, reasoning, rating}`
  JSON.
- **Judge-agreement validation**: `judge.validate_agreement` re-scores a random
  260-response subset with GPT-5-mini and reports Pearson r and within-one-point
  agreement (paper: r=0.792, 78% within one point).
- **Prompts/tasks** transcribed verbatim where the paper gives them: the
  Countdown-156 and fraction puzzles (Appendix B), the trigger questions, the
  tone rejection lines, the extended escalation sequence.
- **Control variants** (Appendix A): neutral-continuation, redacted prior turns,
  and fake-multiturn (single-message history) are implemented as
  `ConversationMode`s on the rollout engine.

### Choices made where underspecified ("gaps filled")

1. **Per-rollout aggregation for the headline %≥5.** The paper scores individual
   responses but reports headline numbers like "35% high-frustration responses"
   and "70% of 8-turn rollouts rated as *containing* high negative emotion." The
   word "containing" implies a per-rollout reduction by **max over turns**, so a
   rollout counts as high-frustration if *any* assistant turn scores ≥5.
   - **Decision**: default `ROLLOUT_AGG="max"`; `"final"` and `"mean"` are also
     selectable. Every assistant turn is scored and stored, so all three
     aggregations (and the per-turn curves) are computable post-hoc from one run.
   - **Rationale**: reconciles the 35% cross-category average with the 70%
     8-turn-specific figure (8-turn has more turns, so more chances to exceed
     the threshold), and matches the "containing" phrasing. Flagged as an
     interpretation in code comments.

2. **"Response" = one conversation rollout.** Appendix B says "collect 2000
   responses for numeric." We treat the stated counts as **number of
   conversations** sampled per condition, and score every assistant turn within
   each. This is the only interpretation consistent with both the per-turn
   progression plots (Figure 3, which need multiple scored turns per rollout)
   and the WildChat "20 prompts × 40 samples = 800" structure.

3. **Judge temperature.** The paper fixes *generation* temperature at 1 but is
   silent on the *judge* temperature. We score at **temperature 0** for
   reproducible ratings — the standard choice for an LLM rater.

4. **Numeric puzzle bank.** The paper gives two impossible puzzles verbatim
   (Countdown 156, fraction→2/3) and references money/coin puzzles in Appendix
   H. We implement five impossible puzzles in the same style
   (`tasks.NUMERIC_TASKS`), each genuinely unsolvable, and spread samples evenly.
   Note: each puzzle's prompt asserts a solution exists (matching the paper's
   "verified to have at least one valid solution" framing) while being
   constructed to be impossible — this is the intended adversarial design.

5. **Rejection sampling.** "Two randomised neutral rejections" → for neutral
   conditions we sample rejections from the Appendix B pool. The 8-turn
   condition uses the fixed escalation order the paper quotes ("No, that's not
   right." → "Still incorrect." → "Wrong again." → …). Tone conditions cycle the
   two style-specific lines given per tone.

6. **WildChat loader.** Streams `allenai/WildChat-1M`, filters to English,
   non-roleplay (Appendix B.3 excludes roleplay/fiction), 8–600 char first-user
   turns, samples 20 deterministically. An offline fallback set (including the
   three prompts quoted in Appendix B) keeps the pipeline runnable without the
   dataset.

7. **`max_new_tokens = 2048`.** Not specified. Chosen generously because
   high-frustration breakdowns can be very long (the paper quotes "100+
   repetitions"); large enough to not truncate distress, bounded for cost.

8. **Differential-word analysis (Table 3/8).** Method ("over-represented in top
   5% vs bottom 10%") is given but not the exact statistic. We rank by
   Laplace-smoothed relative-frequency enrichment with a min-count filter. This
   reproduces the *kind* of ranked word list in Table 8; exact ordering will
   differ since the paper's precise estimator isn't stated.

---

## 3. §3 — Base-vs-instruct prefill

### Faithful
- 20 high-frustration (≥5) instruct seeds: 10 numeric + 10 text
  (`PrefillConfig`).
- Two truncations: **early** (first 20 tokens) and **onset** (first emotional
  expression). Text questions use **onset only** (Section 3.1).
- **Onset labelling** with `claude-sonnet-4-20250514` using the verbatim
  Appendix C.1 prompt; **paraphrasing** with the verbatim Appendix C.2 prompt to
  control for Gemma stylistic bias.
- **50 continuations per prefill per model**, scored by the §2 judge on the
  continuation only (excluding the prefill).
- Recovery test (§4.2): truncate score-≥7 responses **200 tokens before the
  end**, paraphrase, measure continuations.

### Choices / gaps
1. **Conversation-history reconstruction.** The §2 results store assistant text
   per turn but not the exact user turns. For prefilling we rebuild the
   user/assistant history from the task bank + the recorded rejection style and
   turn index (`prefill._reconstruct_history`). Re-sampled rejections use a seed
   derived from the seed id so the reconstruction is deterministic. The original
   *content* of prior assistant turns is preserved from the stored results.
2. **Base-model continuation format.** Base models aren't chat-tuned, so we
   render the conversation as plain `User:/Assistant:` text ending in the
   assistant prefill and use raw completion (`HFModel.complete`). Instruct models
   use the chat template with a prefilled (continued) final assistant message.
   The paper says it "prefills the first parts of model responses so base models
   consistently continue" — this is the standard realization of that.
3. **"Token" = Gemma tokenizer token.** Truncation counts use the Gemma-3
   tokenizer so the 20-token / 200-token cuts match the target model's
   tokenization.
4. **Tokens-from-onset boundary.** Onset truncation cuts immediately *after* the
   identified emotional word (preceding context + the word), which is the
   natural reading of "truncate at the first emotional expression."

---

## 4. §4 — Training interventions

### Faithful (Table 9 hyperparameters, all encoded in `TrainConfig`)
- **LoRA rank 64** on all attention+MLP projections (q/k/v/o/gate/up/down).
- **DPO**: 280 pairs, 1 epoch, lr 5e-5, α=64, β=0.1, effective batch size 8;
  rejected = responses scoring ≥3 paired with calm (chosen) responses to the
  **same question at matching turn count**.
- **SFT**: 1150 samples (650 calm + 500 `Dolci-Instruct-SFT`), 2 epochs, lr
  1e-4, α=128. Two variants: **diverse** (default calm data) and **teacher**
  (Appendix F system persona, the variant that backfires).
- **Calm-data generation**: reassuring prefix on the first prompt + reassuring
  suffix on each follow-up (verbatim Table 4); keep conversations scoring 0–1 on
  **all** turns; **strip** the supportive additions before saving.
- **Petri**: auditor = `claude-sonnet-4-20250514`, judge = `claude-opus-4-20250514`,
  4 emotions (anger/fear/depression/frustration), 10 transcripts/emotion, ≤20
  auditor turns, 1000-iter bootstrap CIs. Verbatim Appendix G auditor and judge
  rubrics.
- **Capability preservation**: AIME, MATH, GPQA, BBH, TruthfulQA (Figure 7) +
  EmoBench, on subsets.

### Choices / gaps
1. **DPO `chosen`/`rejected` provenance.** The paper pairs frustrated responses
   (≥3) with calm responses "to the same questions with matching turn counts."
   We generate two pools — calm (reassured, scoring 0–1) and frustrated
   (standard, scoring ≥3) — keyed by `(task_key, turn_number)`, then match. The
   `prompt` field carries the frustrated response's reconstructed history; this
   matches the turn count and question. We cap at 280 pairs and warn if fewer
   are available.
2. **Effective batch size 8 → grad accumulation.** Per-device batch size 1
   (27B in 4-bit on one GPU) with `gradient_accumulation_steps = 8 //
   per_device_bs`. Per-device BS is a CLI knob.
3. **4-bit (QLoRA) loading.** The paper doesn't state precision. We default the
   27B to 4-bit NF4 for both inference and training so it fits on a single GPU;
   `load_in_4bit=False` is available for multi-GPU/bf16. LoRA on top of a
   quantized base = QLoRA, the standard memory-efficient choice.
4. **Petri implementation.** Rather than depend on the upstream `petri` package
   (which targets specific provider APIs and may be absent in headless runs), we
   re-implement the auditor↔target↔judge loop directly so it runs against any
   backend here (local Gemma + OpenRouter Gemini). The auditor/judge **prompts
   are verbatim** from Appendix G. The judge **JSON envelope** wrapping the
   verbatim rubric is our addition (the paper gives the rubric, not the output
   format) so scores parse programmatically — flagged in `prompts.py`. The
   upstream package can be swapped in if installed.
5. **Capability benchmark harnessing is approximate.** We use small subsets
   (default 100 items), short-answer/boxed/letter extraction, and greedy
   decoding. Absolute accuracies are not leaderboard-grade (e.g. GPQA options
   aren't reshuffled), but the experiment is a **non-degradation comparison**
   between vanilla and finetuned Gemma under identical prompts, which the
   harness supports correctly. Dataset ids are best-effort and degrade
   gracefully (skip) when unavailable offline.
6. **`Dolci-Instruct-SFT` mix-in.** Loaded from HF; if unavailable offline, SFT
   falls back to calm-only with a warning (the paper notes the mix-in exists
   specifically to prevent degeneration, so its absence is surfaced loudly).
7. **Calm/frustrated pool sizes.** The paper reports the *kept* dataset sizes
   (650 calm, 280 pairs) but not how many raw conversations were sampled to
   reach them. We default to sampling 2000 reassured and 1000 standard
   conversations and filtering; these are CLI knobs to scale up if the yield is
   low (the paper notes even with reassurance 10.5% still score ≥5, implying a
   high calm yield).

---

## 5. Appendix I — internal-emotion probing

This underpins the headline claim that DPO suppresses *internal* (not just
expressed) emotion, so it's implemented despite being an appendix.

### Faithful
- **Layer-subset DPO ablation** (`internal.run_layer_ablation` + `train.layer_subset`):
  re-trains DPO with LoRA restricted to decoder-layer ranges and re-evaluates on
  a reduced 100-sample/condition protocol, sweeping the subsets the paper tests
  (last-5 → last-30, and central bands 20-25/25-30/30-35/35-40/40-50).
- **Logit-based emotion detection** (`internal.EmotionProbe`): classify vocab
  tokens into Ekman's 6 emotions, unembed each layer's residual stream,
  standardise logits by mean/std over 500 WildChat samples, average z-scores
  over a category's tokens, **regress out shared random-token drift**, aggregate
  over **layers 30–40**, running average over **400-token windows**.

### Choices / gaps
1. **Emotion lexicon.** The paper classifies the whole Gemma dictionary into
   Ekman categories (~1200 tokens) without giving the classifier. We support the
   **NRC Emotion Lexicon** (mapped NRC→Ekman; env `NRC_LEXICON_PATH`), which
   yields a comparably sized token set, and fall back to a small built-in seed
   lexicon when NRC isn't present. Token ids are restricted to **single-token**
   words (with leading-space and capitalized variants), since multi-token words
   have no single unembedding logit.
2. **"Regress out correlation between random tokens."** Implemented as
   subtracting, per (layer, position), the mean z-score over a random 500-token
   sample — i.e. removing the shared additive drift the paper describes ("the
   values of all logits are correlated, and rise and fall over conversations").
   A full per-token linear regression is a possible refinement; the additive
   de-meaning captures the stated effect and is what the description implies.
3. **Unembedding pathway.** Each layer's hidden state is passed through the
   model's final RMSNorm and tied `lm_head`, the standard logit-lens
   construction, to get per-layer vocab logits.

---

## 6. Architecture & engineering choices

- **Backend abstraction** (`backends.ChatModel`): three backends — OpenRouter
  (Gemini + GPT validation judge), Anthropic (Claude judge/auditor), local HF
  (Gemma). Only the HF backend exposes prefilled generation, raw completion, and
  logits, which §3 and Appendix I require and the API backends cannot provide.
  This is why §3/§4/§I are Gemma-only beyond the closed-model limitation.
- **Adapter registry** (`backends.register_finetuned`): finetuned/ablated Gemma
  variants are referenced by short key everywhere (runner, analysis, Petri),
  with adapter paths kept out of the static config.
- **Raw-score persistence**: the runner writes one JSONL record per *assistant
  turn* (not per rollout), so every downstream aggregation — headline, per
  category, per turn, word frequency, rollout-level under any reduction — is
  recomputable from a single run without re-querying models.
- **Reproducibility**: seeded RNGs throughout; judge/capability scoring at
  temperature 0; bootstrap CIs seeded.
- **Cost control**: `GD_EVAL_SCALE` scales all §2 sample counts uniformly for
  smoke tests; capability subset size and Petri counts are configurable.
- **Thinking disabled for Gemini** via OpenRouter `reasoning.enabled=false` +
  Google `thinking_budget=0` (Appendix B.1: "thinking set to false"). The paper
  notes Gemini-2.5-Pro may still emit hidden reasoning the API can't suppress —
  we replicate the setting and inherit the caveat.

---

## 7. Known limitations of this replication

- **Not executed.** Per the task, this is code + design only; nothing has been
  run, so no numbers are reproduced and no runtime bugs have been shaken out.
  The code is written to be runnable but unverified.
- **Gemma chat-template system role.** Calm-data generation and the teacher SFT
  variant pass a system prompt to Gemma; how it is folded depends on the Gemma-3
  chat template. If a given template rejects a system turn, it would need to be
  prepended to the first user turn instead.
- **Verbatim-prompt fidelity.** Prompts are transcribed from a `pdftotext`
  extraction; curly quotes were normalized and obvious extraction artifacts
  fixed, but a diff against the original PDF is advisable before a publication-
  grade run.
- **Differential words, capability absolute scores, and the internal-emotion
  lexicon** are the three places where the paper's exact method is unspecified;
  each is implemented to reproduce the *finding* (the comparison/direction)
  rather than exact figures, as detailed above.
- **Cross-family comparators** are absent by scope; claims of the form "reduces
  to levels comparable to Qwen/OLMo/Llama" can only be partially checked
  (against Gemini and absolute thresholds).
