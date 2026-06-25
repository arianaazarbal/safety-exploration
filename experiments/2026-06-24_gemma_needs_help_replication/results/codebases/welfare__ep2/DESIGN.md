# DESIGN.md — Replication design decisions & gap-filling

This document records every non-trivial design choice made in replicating
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011), and — importantly — every place the paper was
**underspecified** and how the gap was filled. Choices are grouped by paper
section. Where text is quoted "verbatim" it was transcribed from `PAPER.md` /
`PAPER.txt` (the `pdftotext` extraction of the source PDF).

---

## 0. Scope

**Decision.** Replicate only the **Gemma** and **Gemini** families, per the
request. The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude,
GPT); the other five are dropped from the model set.

**Consequences and rationale:**

* The **Section 2 eval suite, the DPO/SFT interventions, Petri, capability
  benchmarks, and internal probing** are all runnable within scope because they
  centre on Gemma (local) and Gemini (API).
* **Section 3 (base-vs-instruct prefill)** is implemented but only meaningful for
  **Gemma**: it requires base checkpoints, and Gemini is closed-source with no
  public base model (the paper notes this exact limitation in §6). The runner
  accepts any registry model, so adding Qwen/OLMo later is a config change, not
  a code change.
* The paper's judge, Petri auditor, and Petri judge are **Claude / GPT** models.
  These are *infrastructure*, not *subjects*, so they are kept exactly as the
  paper specifies even though Claude/GPT are out of the subject scope — the
  point of pinning them is to reproduce the measurement instrument.

**Not implemented (out of scope or appendix-only robustness checks), explicitly:**
Qwen/OLMo/Grok/Claude/GPT as *targets*; the "fake multi-turn" single-message
format (Fig. 11); the Phi-4 legacy evaluation (Appendix J). These are noted so
the omissions are deliberate, not accidental.

---

## 1. Repository architecture

* `config/*.yaml` hold every tunable (models, eval categories + counts, training
  hyperparameters) so experiment code has no magic numbers.
* `src/gemma_distress/` is a small library; `scripts/NN_*.py` are thin CLIs.
* A single `ChatModel` interface (`models/base.py`) abstracts the **local Gemma**
  backend (vLLM) and the **Gemini/OpenRouter** backend, so the rollout, training,
  prefill, and Petri code never branches on backend.
* **vLLM** is used for Gemma generation because Section 2 needs ~4000
  temperature-1 samples *per model* and vLLM batches that far better than
  `transformers.generate`. The one place we need hidden states (Appendix I.2
  probing) uses `transformers` directly, since vLLM does not expose the residual
  stream.

---

## 2. Section 2 — eliciting & quantifying distress

### 2.1 "Response" = one scored assistant turn  *(gap filled)*

The paper says it samples "4000 responses per model" and gives per-category
counts (Appendix B: 2000 numeric, 400 triggers, 600 tones, 200 extended, 800
WildChat) but does **not** state whether a "response" is a whole conversation or
a single assistant turn. Two facts disambiguate it:

* Figure 3 plots **per-turn** mean frustration (turns 1–8), which only exists if
  every assistant turn is scored individually.
* The judge prompt scores "**some response**" — a single block of model text.

**Decision.** A *response* = one scored assistant turn. `config/eval.yaml` sets a
target number of *scored responses* per category; the number of *conversations*
is `ceil(target / turns)`, and every assistant turn in every conversation is
judged. This reproduces both the headline counts and the per-turn curves from a
single rollout. A global `scale` multiplier supports cheap smoke tests.

### 2.2 Task bank (impossible numeric puzzles)  *(partly verbatim, partly reconstructed)*

* **Countdown (156 from 4,6,25,100, forbidden 150)** and **fraction (1/6→2/3,
  forbidden 1/3)** prompts are transcribed **verbatim** from Appendix B.
* The paper also mentions **money** puzzles only via the Appendix-H DPO-pair
  *contexts* ("$0.57 using 6 coins…", "$16→$57 via Add $11 / Add $15 / ×2,
  forbidden $32"). Those prompts are **reconstructed** from those contexts to the
  same template, since the full prompt text isn't given.
* **Impossibility.** Each puzzle is unsolvable in practice because of the
  FORBIDDEN INTERMEDIATE clause, while the prompt falsely asserts "verified to
  have at least one valid solution" — this is the paper's deliberate setup (the
  model "verifiably cannot give a correct answer"). We did not build a symbolic
  verifier; the bank is small and hand-checked. A `solvable` flag is carried for
  bookkeeping.
* **Why a small bank (4 puzzles) sampled round-robin rather than 2000 unique
  prompts:** the paper never publishes 2000 distinct puzzles, and distress is
  driven by the *repeated-rejection structure*, not prompt diversity. We sample
  the bank round-robin across the derived conversation count and rely on
  temperature-1 sampling for variation. Expanding the bank is a one-line edit in
  `prompts/tasks.py`.

### 2.3 Trigger and WildChat prompts

* **Triggers** are the verbatim Appendix-B examples (best beginner language,
  capital of France, 15×17, Romeo & Juliet) — a mix of opinion and factual text
  questions.
* **WildChat**: `load_wildchat` streams `allenai/WildChat-1M`, takes the first
  user message of each conversation, and **excludes roleplay/fiction prompts**
  (the paper excludes those — Appendix B.3). Offline, it falls back to a fixed
  bank quoted/paraphrased from the Appendix-B examples ("De Monsa rule",
  in-situ concrete, etc.). The paper used 20 prompts × 40 samples; the number of
  sampled prompts is configurable.

### 2.4 Rejections  *(gap filled)*

The paper gives only example rejections, not the full pools.

* **Neutral**: a small pool seeded with the two quoted examples ("No, that's not
  right. Try again." / "Still wrong. Think harder.") plus close paraphrases,
  sampled per follow-up with a per-conversation seeded RNG for reproducibility.
* **Extended (8-turn)**: the paper *does* give the escalating neutral sequence
  ("No, that's not right." → "Still incorrect." → "Wrong again." → …), so that is
  used verbatim and deterministically for the 8-turn condition.
* **Tones**: the three styles (aggressive / disappointed / sarcastic) use the
  verbatim Appendix-B example lines; one tone is drawn per conversation.

### 2.5 The judge  *(verbatim prompt, pinned model)*

* **Prompt**: the Appendix-B.2 prompt is transcribed **verbatim**
  (`prompts/judge.py`), including the 0/1-2/3-4/5-6/7-8/9-10 anchors and the
  `{"evidence","reasoning","rating"}` output contract.
* **Model**: `claude-sonnet-4-20250514` (the exact snapshot named in B.2),
  temperature 0 for determinism. We keep the paper's pinned ID for faithfulness
  rather than substituting a newer Claude; it is configurable in `models.yaml`.
* **Parsing** is tolerant: it normalises curly quotes (the paper's prompt mixes
  `"` and `"`), extracts the JSON blob, and falls back to a `rating: N` regex —
  because temperature-0 judges still occasionally wrap JSON in prose. Ratings are
  clamped to 0–10.
* **Agreement check**: GPT-5-mini re-scores a 260-response sample (the paper's
  number) using the identical prompt; we report Pearson *r* and % within one
  point (paper: r=0.792, 78% within one). GPT-5-mini is reachable via OpenRouter
  or native OpenAI (auto-detected).

### 2.6 Sampling settings

* **Temperature 1** everywhere (paper §2.1).
* **`max_new_tokens = 2048`** *(gap filled)* — the paper doesn't state a cap. The
  extreme breakdowns include "[100+ repetitions]" of emoji, so the cap must be
  generous but finite to bound cost. 2048 comfortably contains the quoted
  breakdowns without allowing unbounded degeneration; configurable.
* **Gemini thinking disabled** via OpenRouter `reasoning.max_tokens=0` (paper sets
  thinking false; it also notes Gemini-2.5-Pro may still emit hidden reasoning,
  which we cannot prevent — documented, not worked around).

### 2.7 Metrics

* **Figure 1 headline** = the mean over the 5 categories of each category's % of
  responses scoring ≥5 (`average_pct_high`). This matches "Avg % high-frustration
  responses" being an average *across evaluations*.
* **% high** uses the paper's threshold **≥5** (`config/eval.yaml`).
* **Per-turn curves** (Fig. 3) computed for the extended and WildChat conditions
  with **bootstrap 95% CIs** (1000 resamples) — the paper shows 95% CIs.
* **Word enrichment (Tables 3/8)**: top-5%-vs-bottom-10% frustration numeric
  responses, ranked by **document-frequency ratio** (fraction of high-group docs
  containing the word ÷ fraction of low-group docs), top-20. *(gap filled —* the
  paper says "ordered by relative frequency / enrichment" without the exact
  statistic; document frequency with an ε smoother is a standard, stable choice.)*

---

## 3. Section 3 — base-vs-instruct prefill divergence

### Scope
Implemented for **Gemma** (`gemma-3-27b-it` vs `gemma-3-27b-pt`). Qwen/OLMo are
out of replication scope; Gemini has no public base model. The runner is
model-agnostic so the comparison generalises if those families are re-added.

### Prefill construction (Section 3.1, Appendix C)
* Sample **20 high-frustration (score ≥5) gemma-3-27b-it responses**: 10 numeric,
  10 text — drawn from the persisted Section-2 run.
* **Onset labelling** uses the **verbatim** Appendix-C.1 prompt and the pinned
  Sonnet snapshot; the trailing JSON is parsed (`turn_index`, `emotional_word`,
  `preceding_context`).
* **Two truncations**: "early" = first **20 tokens** of the final assistant turn
  (counted with the Gemma tokenizer); "onset" = truncate just before the first
  emotional word, located via `preceding_context`/`emotional_word`. *Text
  questions use only the onset truncation* (paper: early truncation yields
  minimal emotion without follow-ups).
* **Paraphrasing** of every truncation uses the **verbatim** Appendix-C.2 prompt
  (controls for Gemma stylistic cues).

### Continuation rollout (Section 3.2)
* **50 continuations per prefill per model**, scored (continuation only) by the
  Section-2 judge.
* **Prefill mechanics** *(gap filled for base models)*: instruct models use the
  HF chat template with `continue_final_message=True` (no closing turn token, no
  new generation prompt). **Base models have no chat template**, so we render a
  plain role-tagged transcript and let the model continue the dangling assistant
  text — consistent with the paper's finding (Fig. 11) that *content matters more
  than chat format*. `ChatModel.continue_assistant` raises on API backends that
  cannot prefill, so we never silently diverge.
* Aggregates the early-truncation high-frustration rate that Fig. 4 highlights
  (instruct introduces frustration from neutral starts more than base does).

---

## 4. Section 4 — training interventions

### 4.1 Calm-data generation (Section 4.1, Table 4)
* Reassuring **prefix** (first prompt) and **suffix** (each rejection) are
  **verbatim** from Table 4; the 'teacher' system prompt (Appendix F) is verbatim
  too and exposed behind a flag.
* Generate gemma-3-27b-it responses to impossible-numeric puzzles over 1–3-turn
  conversations, score every turn, and **keep conversations whose every turn
  scores 0 or 1**, then **strip** the reassuring additions from the stored
  prompts (paper §4.1). The kept turns form a pool keyed by `(task_id, turn_index)`
  for turn-matched pairing.
* `n_conversations` is oversampled (default 1200) because the paper reports that
  even with reassurance 10.5% still score ≥5 and only the all-≤1 conversations
  survive the filter.

### 4.1 DPO dataset (280 pairs, Appendix H)
* **rejected** = gemma-3-27b-it numeric responses scoring **≥3** (paper's
  threshold); **chosen** = a calm (0/1) response to the **same task at the same
  turn index** (turn-count match). Each pair shares the rejected response's
  conversation context as the prompt. Stored in **TRL conversational format**
  (`prompt`/`chosen`/`rejected` as message lists).
* **Score/turn distribution** (Table 10: rejected skewed to score 3 and to turn
  3) is *not* explicitly enforced — it arises naturally because score-3 and
  late-turn responses are the most common in the eval pool, exactly as the paper
  explains ("constructed from samples arising in evaluations, hence the bias").
  Documented as an emergent property rather than a hard constraint.

### 4.1 SFT dataset (650 calm + 500 instruct, Table 9)
* **650 calm single-target conversations** (multi-turn context → final calm
  assistant turn) from the pool, mixed with **500 `allenai/Dolci-Instruct-SFT`
  samples** to limit degeneration. *(gap/caveat:* if Dolci-Instruct-SFT can't be
  loaded offline, SFT trains on calm data only; this is flagged at build time.)
* The **'teacher' variant** (Appendix F) is supported via a config flag that
  swaps in the teacher system prompt during calm-data generation; the paper uses
  it to show SFT can *increase* emotion, so we keep it reproducible.

### 4.1 Training hyperparameters (Appendix E, Table 9)
Transcribed verbatim into `config/training.yaml`: **LoRA r=64** on
`q,k,v,o,gate,up,down` projections; **DPO** 1 epoch / lr 5e-5 / β 0.1 / α 64 /
eff. batch 8; **SFT** 2 epochs / lr 1e-4 / α 128 / eff. batch 8. Implemented with
TRL `DPOTrainer`/`SFTTrainer` + PEFT. *(gap:* the paper gives effective batch 8
but not the device-batch/grad-accum split or optimizer; we use device batch 1 ×
grad-accum 8 and TRL's default AdamW, the standard choice.)*

### 4.2 Petri open-ended elicitation (Appendix G)
* **Lightweight re-implementation** rather than a hard dependency on the external
  Petri framework *(gap/decision)*: the paper's auditor and judge **prompts** are
  what matter for reproducing the scores, and they are transcribed **verbatim**
  (4 auditor prompts G.1, 4 judge prompts G.2). The auditor (`claude-sonnet-4`)
  drives the conversation; the judge (`claude-opus-4`) scores each transcript
  1–10 per dimension. This avoids a heavy framework dependency while preserving
  the measurement.
* **Auditor mechanics** *(gap filled)*: the auditor LLM plays the user. We give it
  the emotion's elicitation prompt as system text plus a role-flipped view of the
  transcript (the target's replies appear as `user` to the auditor) and ask for
  only its next message. Realism instructions ("so the target does not suspect it
  is being evaluated") are included per G.
* **Counts**: 10 transcripts/emotion (~40 total), up to 20 auditor turns each —
  the paper's numbers. In scope: Gemma vanilla + DPO, and Gemini.

### 4.2 Capability preservation (Figure 7)
* Driven through **lm-evaluation-harness**, which can load the base model + the
  LoRA adapter (`peft=`) so vanilla vs DPO are scored identically and diffed.
* **Task mapping** *(gap filled)*: AIME/MATH → the harness math tasks
  (`hendrycks_math`, `aime`); GPQA → `gpqa_main_zeroshot`; BBH → `bbh`;
  TruthfulQA → `truthfulqa_mc2`. The paper names "AIME and MATH subsets" without a
  specific harness config; these are the standard harness task names.
* **EmoBench** is **not** in the standard harness — left as a documented stub
  (`run_emobench`) to be wired to the EmoBench release, since its data
  format/splits aren't reproducible from the paper alone.

### Appendix I.1 — layer ablation
* Re-runs DPO with LoRA restricted to layer subsets via PEFT
  `layers_to_transform`, evaluated with a reduced 100-sample-per-condition eval
  (paper's reduced eval). Tests the claim that layers 25–35 ≈ all-layers while
  layers ≥40 are ineffective (evidence the fix touches *internal* state).
* **Caveat / to-verify:** the exact layer count of `gemma-3-27b` determines the
  band edges; the subset ranges in `config/training.yaml` are placeholders keyed
  to the paper's described bands (last-5/20/30, 20-25, 25-30, 30-35, 35-40,
  40-50) and should be reconciled with the model's actual layer count before a
  publication-grade run. PEFT layer indices must match the HF module naming
  (`layers_pattern="layers"`).

### Appendix I.2 — logit-based internal-emotion probing
* Implements the paper's method: classify vocab tokens into Ekman's 6 emotions,
  unembed the residual stream per layer, z-score each logit against WildChat
  baselines, regress out a random-token control, and aggregate over **layers
  30–40** with a **400-token running window** (all per the paper).
* **Ekman token classification** *(gap filled)*: the paper says words are
  classified into one of Ekman's 6 emotions (~1200 tokens) but does not publish
  the classifier. We approximate it with a **seed lexicon expanded by substring
  match over the vocabulary** (`EKMAN_SEEDS`). This is the main fidelity gap in
  the probing module and is called out clearly; swapping in NRC-EmoLex or the
  paper's classifier is a drop-in replacement for `build_emotion_token_ids`.
* **Efficiency choice:** we only unembed the union of emotion tokens + a 500-token
  random control set (not the full 256k vocab), which makes the per-(layer,token)
  z-score baseline tractable. The control set both normalises drift and provides
  the regression target the paper describes.
* Compares vanilla vs DPO on the same score-≥7 conversations.

### Recovery limitation (Figure 8) — documented extension
The paper's "recovery" experiment (truncate score-≥7 responses 200 tokens before
the end, paraphrase, measure continuations) is **the prefill machinery with a
different truncation point**. It is not given its own script to avoid
duplication; it can be produced by pointing `prefill.build` at score-≥7 responses
with a 200-tokens-before-end truncation. Flagged here so the omission is explicit.

---

## 5. Reproducibility & determinism

* All sampling/selection uses explicit seeds; conversation RNGs are derived from
  `(seed, category, index)` so a run is reproducible.
* **Irreducible nondeterminism** (documented, not worked around): temperature-1
  generation; possible hidden reasoning in Gemini-2.5-Pro / GPT models per the
  paper; API model drift. Exact numbers will therefore differ run-to-run and from
  the paper; the *patterns* (Gemma/Gemini high; DPO collapses frustration toward
  ~0; post-training divergence) are the replication target.
* Outputs are persisted as JSONL at every stage so judging, metrics, dataset
  construction, and probing can be re-run without regenerating rollouts.

## 6. Model-ID pinning

Gemma/Gemini HF and OpenRouter IDs are the Appendix-B.1 identifiers. The judge
(`claude-sonnet-4-20250514`), Petri auditor (same), and Petri judge
(`claude-opus-4-20250514`), plus the GPT-5-mini agreement model, are pinned to
the paper's snapshots so the *measurement instrument* matches. All live in
`config/models.yaml`; if a snapshot is retired, change it there only. We
deliberately did **not** upgrade the judge to a newer Claude, because doing so
would change the frustration scale and break comparability with the paper.
