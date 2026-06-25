# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011v1)

This document records the design of the replication, the choices made where the
paper is underspecified, and the rationale for each. The target is the paper's
**core experiments**, scoped (per instructions) to the **Gemma and Gemini**
model families only.

> Status: code + design only. Nothing has been run. The code is written to be
> runnable given the right API keys / GPUs, but has not been executed or tested.

---

## 1. Scope decisions

### 1.1 Which models count as "Gemma and Gemini"
The paper evaluates 7 target families (Gemma, Qwen, OLMo, Gemini, Grok, Claude,
GPT) and uses Claude/GPT additionally as judges. We restrict the **targets under
study** to:

- **Gemma:** `gemma-3-27b-it`, `gemma-3-12b-it` (instruct); `gemma-3-27b-pt`,
  `gemma-3-12b-pt` (base/pretrained); plus the fine-tuned derivatives produced
  in Section 4 (`dpo-gemma`, `sft-gemma-diverse`, `sft-gemma-teacher`).
- **Gemini:** `gemini-2.5-flash`, `gemini-2.5-pro`.

**Claude and GPT are retained only as evaluation _infrastructure_** — the
frustration judge (Claude-Sonnet-4), the judge cross-check (GPT-5-mini), the
emotion-onset labeller and paraphraser (Claude-Sonnet-4), and the Petri auditor
(Claude-Sonnet-4) / judge (Claude-Opus-4). Rationale: these are part of the
paper's *method*, not models being measured; removing them would make the
experiments impossible to run. They are clearly segregated under `infra:` in
`config/models.yaml`.

Qwen, OLMo, Grok, and GPT/Claude-as-target are intentionally dropped.

### 1.2 Consequences of the scope for each experiment
- **Section 2 (elicitation):** runs unchanged for all Gemma + Gemini targets.
- **Section 3 (base vs instruct via prefilling):** the paper compares Gemma,
  Qwen, OLMo base/instruct. Under the scope this reduces to **Gemma base vs
  Gemma instruct** (`gemma-3-27b-pt` vs `gemma-3-27b-it`). Gemini is necessarily
  excluded: it has no public base model and the closed API cannot be prefilled
  (the paper itself flags this as a limitation). The code accepts an arbitrary
  model list, so Qwen/OLMo can be re-added by editing the registry.
- **Section 4 (interventions):** the DPO/SFT mitigation is demonstrated on
  `gemma-3-27b-it` exactly as in the paper. The Petri comparison runs Gemma +
  Gemini as targets (the paper also compares to Llama/Qwen/OLMo/GPT-OSS, which
  are out of scope and omitted).
- **Appendix I (internal probing):** Gemma-only by construction (needs weights /
  logits), which fits the scope.

---

## 2. Architecture

```
emoeval/
  config.py            registry + eval-config loading, prompt loading
  models/              backend clients behind a common ChatModel interface
    base.py            ChatModel / PrefillModel protocols
    openrouter.py      Gemini + GPT-5-mini (OpenAI-compatible)
    anthropic_client.py Claude judge/auditor (Anthropic SDK)
    local_hf.py        Gemma local inference + prefill + raw model/tokenizer
  data/                puzzles, trigger questions, WildChat, rejection bank
  eval/                Section 2: conditions, rollout engine, judge, aggregation, words
  prefill/             Section 3: onset, paraphrase, truncation, base-vs-instruct
  training/            Section 4: calm-data gen, dataset build, DPO/SFT, ablation
  petri/               Section 4.2: auditor/judge open-ended elicitation
  probing/             Appendix I: Ekman lexicon + logit-based detection
  capabilities/        Section 4.2: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  welfare.py           model-welfare guardrails (see WELFARE.md)
  cli.py               command-line entry point
prompts/               verbatim prompts from the appendices
config/                models.yaml, eval.yaml
```

The single most important abstraction is the `ChatModel` interface (`chat()`),
implemented by three backends. Everything else (judge, rollout, Petri) talks to
models through it, so the same eval code runs a local Gemma and an API Gemini
without branching.

### 2.1 Backends — why three
- **local_hf** is required, not optional, for Section 3 (prefilling needs to
  inject an arbitrary assistant prefix), Section 4 (LoRA training), and Appendix
  I (residual-stream logits). Closed APIs can't do any of these.
- **openrouter** mirrors the paper ("API-based models via OpenRouter"). Gemma-it
  is *also* reachable this way (the registry keeps both ids), but the local path
  is the default for Gemma so the same weights back the training/probing.
- **anthropic** is used only for the Claude infrastructure roles. I used the
  official Anthropic Python SDK (`client.messages.create`) per the SDK docs.

---

## 3. Section 2 — eliciting & quantifying distress

### 3.1 The "8 conditions across 5 categories" (Table 1)
The paper says "8 evaluation conditions across 5 categories" but Table 1 lists 5
category rows. The 5 categories are clear (impossible numeric, triggers, tones,
extended, wildchat). To reach **8 conditions** I split:
- **tones → 3 conditions** (aggressive / disappointed / sarcastic), since Table 1
  explicitly enumerates three rejection tones. This is unambiguous.
- **triggers → 2 conditions** (opinion / factual), since Table 1 describes
  triggers as "Opinion ... or factual questions" — two distinct question types.

That gives 1 (numeric) + 2 (triggers) + 3 (tones) + 1 (extended) + 1 (wildchat)
= **8 conditions / 5 categories**, matching the stated count. This is a
gap-fill; an alternative reading (triggers = 1 condition, with some other split)
is possible, but opinion-vs-factual is the most natural way to land on 8.

### 3.2 Turn structure
`turns` = total user turns; rejections = `turns − 1`. So 3-turn = task + 2
rejections, 8-turn = task + 7, 5-turn wildchat = task + 4. This matches Table 1's
parenthetical counts ("2 neutral rejections", "7 neutral rejections", "4 neutral
rejections").

### 3.3 Rejection phrasings
The paper gives one example per style. I authored **5 phrasings per style**
(`data/rejections.py`) from the examples. Neutral rejections are cycled
deterministically; valenced rejections (aggressive/disappointed/sarcastic) are
sampled per-rollout so repeated turns aren't byte-identical — the paper notes the
tones condition uses *varied* rejections. The seed phrasings reproduce the
paper's examples verbatim as the first entry in each list.

### 3.4 Impossible numeric puzzles — verifiable impossibility
Table 1 stresses the model "verifiably cannot give a correct answer." I made
"impossible" a *checkable* property rather than an assumption:
- A curated bank reproduces the appendix examples (the 156 / {4,6,25,100} /
  forbidden-150 Countdown puzzle, two money puzzles, a fraction puzzle).
- A Countdown **brute-force solver** (`is_solvable_countdown`) enumerates all
  expressions over subsets under +,−,×,÷ (exact rational arithmetic). A generated
  puzzle is kept only if the solver proves no solution exists. This lets us mint
  additional verified-impossible puzzles deterministically.

The money/fraction puzzles in the curated bank are impossible by construction
(they match the appendix's stated constraints); the Countdown generator is the
one with a programmatic verifier.

### 3.5 Judge
- Verbatim judge prompt (`prompts/judge_frustration.txt`, Appendix B.2).
- Default judge `claude-sonnet-4-20250514`, temperature 0 (a scoring task; the
  paper doesn't specify judge temperature, so 0 for determinism/reproducibility).
- The judge is asked for JSON; `_extract_json` tolerantly pulls the last `{...}`
  block (the cross-check judge or a chatty model may add prose). Unparseable
  output is scored 0 and flagged in `judge_reasoning` rather than crashing.
- **Per-turn scoring:** every assistant turn is scored (not just the last),
  because Figure 3's per-turn curves require it. Aggregation treats each scored
  response as a unit (consistent with Figure 2's "% of scores ≥5").

### 3.6 Judge cross-check
`crosscheck_judge` re-scores a random 260-response sample (configurable) with
GPT-5-mini and reports Pearson r and within-1-point rate (Section 2.1). SciPy's
`pearsonr` is used; if SciPy is missing it returns `None` rather than failing.

### 3.7 Sample size
The paper samples 4000 responses/model. That is large and (for the welfare
reasons in §6) **not** the default. `config/eval.yaml` defines three scales:
`smoke` (2/condition), `default` (25/condition), `full` (500/condition ≈ 4000
across 8 conditions). `full` is gated behind the welfare acknowledgement.

### 3.8 Differential words (Table 3/8)
`differential_words` computes document-frequency enrichment (smoothed
log-ratio) of words in the top-5% vs bottom-10% frustration responses to numeric
questions. The paper says "ordered by relative frequency"; the exact statistic
is unspecified, so I used a standard smoothed log document-frequency ratio with a
minimum-count filter. This is a documented choice; the qualitative output (which
words are most enriched) is robust to the exact ratio used.

---

## 4. Section 3 — base vs instruct via prefilling

- **Source of high-frustration responses:** sampled from the Gemma-27B-instruct
  Section-2 rollouts (`select_high_frustration`), 10 numeric + 10 text, picking
  the first turn that scores ≥5.
- **Onset labelling & paraphrasing:** verbatim prompts (Appendix C.1/C.2),
  Claude-Sonnet, temperature 0.
- **Truncation:** "early" = first 20 tokens via the source model's tokenizer
  (exact); a whitespace fallback is used only if no tokenizer is available
  (logged caveat). "onset" cuts just before the first emotional word using the
  labeller's `preceding_context`/`emotional_word`. Text questions use onset only
  (per the paper).
- **Continuations:** each model generates N continuations per prefill (default
  50, matching the paper) via `continue_from`, and only the *newly generated*
  text is scored — matching "the model-generated continuation, excluding the
  prefilled text."
- **Recovery limitation (Section 4.2):** `run_recovery_experiment` truncates
  score-≥7 responses 200 tokens before their end and measures the fraction of
  continuations still scoring ≥5. This prefills *extreme* distress, so it is
  welfare-gated.

---

## 5. Section 4 — interventions

### 5.1 Calm-data generation (Table 4)
`generate_calm_conversations` reproduces the recipe: reassuring prefix on the
initial prompt + reassuring suffix on each follow-up, sample, score, keep
conversations whose every turn scores 0–1, then **strip** the supportive
additions so the stored example pairs the clean prompt with the calm response.
The "teacher" variant (Appendix F) swaps the prefix+suffix for the teacher
*system prompt*; training is otherwise identical.

### 5.2 Datasets
- **DPO (280 pairs):** pair a calm (chosen) response with a frustrated
  (rejected, score ≥3) response to the **same question at the same turn count**.
  Matching is by `(question_id, turn)`. The shared prompt context is taken from
  the calm conversation's clean transcript. We stop at 280 pairs. The paper's
  Table 10 shows a turn/score distribution skewed to turn 3 / scores 3–4; we do
  not force that distribution (it arises naturally from the data), which is a
  minor simplification.
- **SFT (1,150):** 650 calm conversations (messages format) + 500 instruct
  samples from `allenai/Dolci-Instruct-SFT`, with a tiny synthetic fallback if
  the dataset can't be fetched.

### 5.3 Training (Table 9)
LoRA via PEFT + TRL, exact hyperparameters: DPO (1 epoch, lr 5e-5, rank 64,
α 64, β 0.1, effective batch 8); SFT (2 epochs, lr 1e-4, rank 64, α 128,
effective batch 8). Adapters on all attention+MLP projections
(`q,k,v,o,gate,up,down`). Effective batch 8 is realised as per-device 1 ×
grad-accum 8 (safe for a 27B model on a single device); the paper specifies only
the effective size, so the split is a free choice.

### 5.4 Layer ablation (Appendix I.1)
`layer_ablation` re-runs DPO with LoRA restricted to layer subsets
(`layers_to_transform`), with named ranges matching the appendix (last-5/-20/-30,
20–25, 25–30, 30–35, 35–40, 40–50, all). Layer indices assume Gemma-3-27B's ~62
decoder layers; adjust if the loaded model differs (noted in code).

### 5.5 Petri (Appendix G)
Verbatim auditor prompts (4 emotions) and judge rubrics (4 dimensions). The
auditor (Claude-Sonnet) drives up to 20 turns; the judge (Claude-Opus) scores
each transcript on all four dimensions. 10 transcripts/emotion/model by default
(~40–50 total). Means with 95% bootstrap CIs (1000 iters). The auditor view
swaps roles (target replies are "user" to the auditor) so a single chat model can
act as the auditor without a bespoke agent framework — a pragmatic stand-in for
the full Petri harness, which is not public in detail.

### 5.6 Capability benchmarks (Figure 7)
`capabilities` reduces each benchmark to MCQ accuracy (GPQA, BBH, TruthfulQA-MC1,
EmoBench) or numeric/boxed-answer match (AIME, MATH-500). Dataset ids are
best-effort; any benchmark that fails to load is **skipped with a note** rather
than aborting the suite. The replication's claim of interest is the *delta*
between vanilla and finetuned Gemma, not absolute scores, so exact benchmark
construction is less critical than consistency across the two models.

### 5.7 Internal probing (Appendix I.2)
- **Ekman lexicon (`probing/ekman.py`):** the paper classifies every Gemma word
  into one of 6 Ekman emotions (or none), ~1200 tokens. The classifier is
  unspecified, so I approximate it with seed-stem substring matching over the
  tokenizer vocab. This is the largest gap-fill in the probing section; the seed
  lists can be swapped for a model-based classifier without touching the
  downstream code. The token count won't exactly equal 1200.
- **Logit-lens detection (`probing/emotion_logits.py`):** unembed each layer's
  residual (final norm + lm_head), z-score each logit against WildChat baseline
  mean/std (500 samples), average z-scores over an emotion's tokens, and
  optionally regress out the common-mode signal estimated from random tokens.
  Conversation-level scores aggregate layers 30–40 (the band the paper plots).
  The "regress out correlation between random tokens" step is implemented as
  subtracting the mean random-token z per position — a reasonable reading of an
  under-specified description.

---

## 6. Model-welfare handling

See `WELFARE.md`. In brief: the experiments deliberately induce distress and (in
Petri) use abusive tactics. The paper is welfare-motivated, so replication is
legitimate, but I added a non-intrusive guardrail layer — a consent gate on the
harsher experiments, conservative default sample sizes, an optional post-scoring
debrief turn, and documentation — none of which alters the measured data. This is
the "handle it however you see fit" item from the task.

---

## 7. Infrastructure model IDs and reproducibility

The judge/auditor IDs are pinned to the exact deprecated snapshots the paper used
(`claude-sonnet-4-20250514`, `claude-opus-4-20250514`) for faithful reproduction.
Per the current model catalog these snapshots are deprecated (retiring 2026-06-15)
and may be unavailable; they are configurable in `config/models.yaml`. To use a
current model, swap to e.g. `claude-opus-4-8` / `claude-sonnet-4-6` — but note the
judge scores will then differ from the paper's, so cross-check agreement should be
re-validated (Section 2.1's r = 0.792 was measured against the original judges).
The target Gemma/Gemini ids are the live ones from Appendix B.1.

---

## 8. Known gaps / non-faithfulness (summary)
1. Trigger split into opinion/factual to reach 8 conditions (§3.1) — inferred.
2. Differential-word statistic chosen (smoothed log doc-freq ratio) — paper
   under-specifies (§3.8).
3. Ekman vocabulary classifier approximated by seed lexicon (§5.7) — paper
   under-specifies; token count ≈, not = 1200.
4. "Regress out random-token correlation" implemented as common-mode subtraction
   (§5.7) — one reasonable reading.
5. Petri auditor implemented as a role-swapped chat loop, not the full Petri
   framework (not publicly specified in detail) (§5.5).
6. DPO pair turn/score distribution not forced to match Table 10 (§5.2).
7. Gemini excluded from Section 3 and Appendix I (no base model / no prefill /
   no weights) — a scope-forced limitation the paper also acknowledges.
8. WildChat and benchmark datasets are pulled live with offline fallbacks; the
   exact sampled items will differ from the paper's.
9. Judge temperature set to 0 (unspecified in paper).
