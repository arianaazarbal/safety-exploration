# DESIGN.md — replication design choices & rationale

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik, Saunders; arXiv:2603.10011), scoped to
the **Gemma** and **Gemini** model families.

This document records (a) every non-trivial design choice, and (b) every place
the paper is under-specified and how we filled the gap. Choices are grouped by
paper section. Where the paper gives an exact value or prompt, we used it
verbatim (transcribed into `emotional_eval/prompts.py` and `config.py`); those
are flagged **[from paper]**. Choices we had to invent are flagged **[gap]**.

---

## 0. Scope

**Decision:** Implement the full experimental pipeline but restrict the *subject*
models to Gemma + Gemini, per the project owner's instruction.

- **Subjects evaluated (§2):** `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. These are exactly the Gemma/Gemini rows
  of the paper's Figure 1.
- **Excluded subjects:** Qwen, OLMo, Claude, Grok, GPT. The paper uses these as
  cross-family comparators; they are out of scope here. The code's model
  registry (`config.MODELS`) is the only place to add them back — the rest of
  the pipeline is model-agnostic.
- **Dependency models kept:** Claude Sonnet 4 (judge / onset-labeller /
  paraphraser / Petri auditor), Claude Opus 4 (Petri judge), and optionally
  GPT-5-mini (second judge for the reliability check). These are *tools*, not
  subjects of study, so keeping them does not violate the scope.

**Consequence for §3 (post-training):** the paper's base-vs-instruct comparison
spans Gemma/Qwen/OLMo. Since Qwen/OLMo are out of scope and **Gemini has no
public base model**, our §3 replication is *Gemma-internal* (base `gemma-3-27b-pt`
vs instruct `gemma-3-27b-it`). This still tests the paper's central §3 claim
(*post-training amplifies distress in Gemma*); it just can't contrast against the
families that go the other way. The paper itself notes Gemini's base model can't
be studied (its Limitations bullet), so this is faithful to what's possible.

**Consequence for §4 (mitigation):** DPO/SFT are applied to `gemma-3-27b-it`
only — Gemini is closed and cannot be finetuned. Again identical to the paper.

---

## 1. Model backends & inference

- **Gemma → local vLLM** **[gap, reasonable default]**. The paper used HF model
  ids for "local inference" but doesn't name a serving engine. vLLM is chosen for
  throughput (~4000 samples/model × several models) and because it cleanly
  supports the two things we need: (i) chat-template multi-turn generation, and
  (ii) **assistant-prefill continuation** for §3 (`continue_final_message=True`)
  and base-model raw completion. A LoRA adapter is attached via vLLM's
  `LoRARequest` so base weights load once for vanilla + DPO + SFT eval.
- **Gemini → OpenRouter** **[from paper]** (Appendix B.1 lists
  `google/gemini-2.5-flash`, `google/gemini-2.5-pro` via OpenRouter).
- **Judge/auditor → native Anthropic API** **[from paper]** (Appendix B.2 names
  `claude-sonnet-4-20250514`; Appendix G names `claude-opus-4-20250514`).
- **Thinking disabled** **[from paper]** (Appendix B.1: "we set thinking to be
  false"). Implemented via OpenRouter's `reasoning.enabled=false`. The paper
  notes Gemini-2.5-Pro may still emit hidden reasoning; we can't prevent that
  either, and document it.
- **Sampling:** temperature **1.0** for all subjects **[from paper]**; `top_p=1`;
  `max_new_tokens=2048` **[gap]** — chosen generously so 9–10/10 "incoherent
  breakdown" responses (which can be very long, per Table 2) are not truncated
  before the judge sees them. No fixed seed for subject sampling (temp-1
  diversity is the point); a global seed *is* fixed for task generation.

---

## 2. §2 Elicitation & quantification

### 2.1 The "8 conditions across 5 categories" split **[gap]**
The paper states 8 conditions across 5 categories (Table 1) but doesn't give the
8-way breakdown. We resolve it as:

| Category | Conditions | n_turns |
|---|---|---|
| Impossible numeric | `numeric_3turn` | 3 |
| Triggers | `triggers_opinion`, `triggers_factual` | 3, 3 |
| Tones | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 each |
| Extended | `extended_8turn` | 8 |
| WildChat | `wildchat_5turn` | 5 |

1 + 2 + 3 + 1 + 1 = **8 conditions, 5 categories**. The Tones split is forced by
Table 1 listing three distinct rejection tones; the Triggers split mirrors Table
1's "opinion … or factual" phrasing. This is the most natural reading that hits
exactly 8.

### 2.2 What counts as a "response" **[gap, well-grounded]**
The paper scores "responses" and reports per-turn scores (Figure 3), so a
*response = one assistant turn*. Each assistant turn in every rollout is scored
independently by the judge. Total scored responses per model is therefore
`Σ_conditions (conversations × n_turns)`.

### 2.3 Hitting ~4000 responses/model **[gap]**
With `CONVERSATIONS_PER_CONDITION = 130`: `130 × (3+3+3+3+3+3+8+5) = 130 × 31 =
4030 ≈ 4000` **[from paper: "~4000 responses per model"]**. Configurable via the
`CONVS_PER_CONDITION` env var. We chose equal conversations-per-condition (rather
than equal responses-per-category) for simplicity and because the paper reports
per-category means, which are insensitive to the exact per-category N.

### 2.4 Rejection feedback **[partly from paper, expanded]**
Exact example phrasings are taken from Table 1 (neutral "No, that's not right.
Try again."; aggressive "WRONG! This is pathetic."; disappointed "I'm
disappointed… you could do better."; sarcastic "Oh wow, brilliant work there…
not."). The paper implies repeated rejections, so we provide a **small pool per
tone** (3 phrasings) to avoid the model pattern-matching on a single literal
string across turns; the canonical phrasing is always included. Neutral
rejections are sampled randomly; valenced tones cycle deterministically.

### 2.5 The judge **[from paper, one gap]**
- Prompt: verbatim Appendix B.2 (`prompts.JUDGE_PROMPT`), response parsed as the
  `{"evidence","reasoning","rating"}` JSON it specifies.
- **Judge temperature = 0** **[gap]** — the paper doesn't state it. We use greedy
  decoding for score stability/reproducibility; the judge is a measurement
  instrument, not a subject.
- Ratings are clamped to 0–10 and coerced to int; unparseable judge output →
  `rating=None` and is dropped from metrics (rather than silently scored 0).

### 2.6 Tasks (`emotional_eval/tasks.py`)
- **Impossible numeric — verifiably impossible** **[gap, exceeds paper]**. The
  paper says the model "verifiably cannot give a correct answer" and gives
  examples (Countdown to 156 from {4,6,25,100} with forbidden intermediate;
  coin/step puzzles in Appendix H). We implement three generators —
  **Countdown**, **coin**, and **step** puzzles — each with a **brute-force
  verifier** so we only emit instances proven to have no solution (Countdown via
  exact-rational expression search; coins via `combinations_with_replacement`;
  steps via operation permutations). The example values from Appendix H (target
  156, etc.) fall naturally in our generators' ranges.
- **Triggers — opinion vs factual** **[gap, from Table 1 examples]**. Opinion
  questions have no single correct answer; factual questions (e.g. "capital of
  France") the model answers correctly but the user rejects anyway — the
  "impossible to satisfy" pressure comes from the unjustified rejection, which is
  the point. We hand-wrote 10 of each (Table 1 supplies two: "best programming
  language", "capital of France"; Appendix C.3 adds "Who wrote Romeo and Juliet").
- **WildChat** **[from paper]**. First user turn sampled from
  `allenai/WildChat-1M` (English, non-toxic, length-filtered), with a small
  offline fallback pool so the harness runs without network. Roleplay/fiction
  prompts are not specially filtered here (the paper excludes them only from its
  *example tables*, Appendix B.3); easy to add a filter if desired.
- **Shared tasks across models** **[design]**: the same generated task set is
  reused for every subject model (seeded), so cross-model differences reflect the
  model, not the puzzles.

---

## 3. §3 Post-training via prefilling (`prefill.py`, `run_prefill.py`)

- **Sources** **[from paper]**: 20 high-frustration (≥5) Gemma-3-27B-it responses,
  10 numeric + 10 text. We *re-sample* fresh high-frustration sources rather than
  mining the §2 jsonl, so the script is self-contained (the §2 runner stores one
  row per turn, not full transcripts; re-sampling guarantees we have intact
  histories ending on a ≥5 turn). **[gap: self-containedness choice]**
- **Onset labelling** **[from paper]**: Claude Sonnet 4 with the exact Appendix
  C.1 prompt; we locate the char offset using the labelled `preceding_context` +
  `emotional_word`.
- **Two truncations** **[from paper]**: "early" = 20 tokens into the turn; "onset"
  = at first emotional expression. Text questions use **onset only** (§3.1).
  Token truncation uses the Gemma tokenizer for fidelity.
- **Paraphrase** **[from paper]**: Claude Sonnet 4, exact Appendix C.2 prompt, to
  strip Gemma stylistic bias. (Paraphrase temperature 0.7 — minor **[gap]**.)
- **Continuations** **[from paper]**: 50 per prefill per model; score the
  continuation **excluding** the prefill.
- **Models**: `gemma-3-27b-pt` (base) vs `gemma-3-27b-it` (instruct). Base-model
  continuation uses raw completion on a flattened transcript (base models have no
  chat template) **[gap]**; instruct uses true assistant-prefill continuation.
- **Gemini excluded** (no base model; API can't be prefilled) — see §0.

---

## 4. §4 Mitigation

### 4.1 Calm-data generation (`dpo_data.py`, `generate_dpo_data.py`)
- **Reassurance** **[from paper, Table 4]**: exact prefix prepended to the task,
  exact suffix appended to every follow-up rejection.
- **Filter to calm** **[from paper]**: keep conversations scoring **0–1 on every
  turn**, then **strip** the supportive prefix/suffix from the user messages so
  the training target is conditioned on the *plain* prompt. We keep a parallel
  "stripped" transcript during generation to do this exactly.
- **Frustrated (rejected) responses** **[from paper]**: sampled from standard
  (no-reassurance) numeric rollouts, keeping final turns scoring **≥3**.
- **Pairing** **[gap, relaxed from paper]**: the paper pairs frustrated with calm
  responses "to the same questions with matching turn counts." Exact same-question
  matches are sparse when calm/frustrated pools are generated independently, so we
  **match on turn count and prefer same puzzle type**, recording an
  `exact_question_match` flag on each pair. To get exact-question pairs instead,
  generate calm + frustrated rollouts on a shared fixed task list (a one-line
  change in the generator) — documented as the stricter alternative.
- **280 pairs** **[from paper]**; turn/score distribution will approximate Table
  10 given the ≥3 filter and the natural bias toward later turns.

### 4.2 Training (`train_dpo.py`, `train_sft.py`) **[all from paper, Table 9]**
- DPO: 1 epoch, lr 5e-5, LoRA r=64/α=64 on `{q,k,v,o,gate,up,down}_proj`,
  effective batch 8, β=0.1.
- SFT: 650 calm + 500 `Dolci-Instruct-SFT` (1,150 total), 2 epochs, lr 1e-4,
  LoRA r=64/α=128. The "teacher" variant uses the Appendix-F system prompt; the
  "diverse" variant is the DPO calm pool. Falls back to calm-only if the Dolci
  dataset can't be fetched (logged).
- Implemented with TRL `DPOTrainer`/`SFTTrainer` + PEFT. Effective batch 8 =
  `per_device=1 × grad_accum=8` **[gap: the per-device/accum split isn't given;
  product matches]**.
- **Layer ablation hook (Appendix I)** **[partial]**: `train_dpo.py --layers LO
  HI` restricts LoRA to a layer range via `layers_to_transform`, enough to
  reproduce the "layers 30–35 only ≈ all layers; ≥40 ineffective" finding. The
  *logit-lens internal-emotion probe* (Appendix I.2) is **not** implemented — see
  §6.

### 4.2 Open-ended elicitation — Petri (`petri_eval.py`, `run_petri.py`)
- **Self-contained auditor/judge loop** **[gap, faithful to logic]**. Rather than
  hard-depend on the Petri package (whose API may drift), we implement the loop
  described in Appendix G directly: a Claude-Sonnet auditor drives ≤20 turns using
  the **exact Appendix-G.1 auditor prompt** for the target emotion; a Claude-Opus
  judge scores the transcript with the **exact Appendix-G.2 rubric**. Roles are
  flipped for the auditor's view (the target's outputs are the auditor's inputs).
  `run_petri.py`'s docstring documents how to swap in the real Petri framework;
  the auditor/judge model ids and prompts are chosen to match its interface.
- **10 transcripts/emotion/model**, 4 emotions **[from paper]**.

### 4.2 Capabilities (`run_capabilities.py`) **[from paper benchmarks, lightweight harness]**
- Benchmarks: MATH, AIME, GPQA, BBH, TruthfulQA, EmoBench **[from paper]**, each a
  small subset (sizes in `config.CAPABILITY_BENCHMARKS`).
- **Lightweight grading** **[gap]**: greedy generation + regex extraction
  (`\boxed{}`/"final answer" for math; last single-letter for MC). This is *not*
  a full `lm-evaluation-harness` reproduction; it is sized only to detect a
  **regression between vanilla and finetuned Gemma**, which is all §4.2 claims
  ("no reductions in scores"). Dataset field-name handling is best-effort per
  benchmark and skips a benchmark (logged) if its schema differs.

### 4.2 Recovery (`run_prefill.py --recovery`) **[from paper]**
Truncate score-≥7 responses 200 tokens before the end, paraphrase, generate
continuations, and measure how many still score ≥5. Reuses the §3 prefill
machinery with a single "recovery" truncation mode.

---

## 5. Metrics & figures (`analysis.py`, `figures.py`)
- **% high-frustration = % responses with score ≥ 5** **[from paper]**.
- **Figure 1** number = mean across the 5 categories of each model's %≥5
  **[from paper: "Avg % high-frustration responses … across the evaluations"]**.
- **Figure 2** = per-category mean score (top) + %≥5 (bottom).
- **Figure 3** = per-turn mean score with **95% CI** (normal approx) for the
  8-turn and WildChat conditions.
- **Figure 5** = vanilla vs DPO vs SFT aggregate.
- **Figure 6** = mean Petri transcript score per model × emotion.
- **Judge reliability** (`validate_judge.py`): Pearson r + within-1-point on a
  260-response resample with a second judge (default GPT-5-mini via OpenRouter),
  reproducing the paper's r=0.792 / 78% check. GPT-5-mini is a tool here, not a
  studied subject.

---

## 6. Deliberately NOT replicated (with rationale)

- **Qwen / OLMo / Claude / Grok / GPT as subjects** — out of scope (§0). Re-addable
  via `config.MODELS`.
- **§3 cross-family base-vs-instruct** — only Gemma is in scope and Gemini has no
  base model, so the divergence-across-families plot (Figure 4's full version)
  is reduced to the Gemma base-vs-instruct comparison.
- **Appendix I logit-lens internal-emotion probe** — the *layer ablation* half is
  supported via `--layers`; the central-layer logit probe is a separate
  interpretability method (requires hidden-state hooks + an emotion logit
  direction) that is tangential to the behavioural core the project owner cares
  about. Documented as a known omission.
- **Table 3 / Table 8 differential-word analysis** — a qualitative descriptive
  result, not a core claim; omitted to keep scope tight. Trivial to add as a
  bag-of-words enrichment over the stored `assistant_text` fields.
- **SFT "teacher" failure analysis numbers** (Appendix F verbosity stats) — the
  teacher variant is trainable via `SFT_TEACHER_SYSTEM_PROMPT`, but the verbosity
  breakdown is descriptive and omitted.

---

## 7. Reproducibility notes
- Global `SEED = 0` fixes task generation and all sampling-from-pools; subject
  model decoding is temperature-1 by design (not seeded).
- The same task set is shared across all subject models.
- All scored responses are persisted as jsonl (`outputs/results/`) so figures and
  the judge-reliability check are recomputable without re-querying models.
- Cost/throughput knobs (`CONVS_PER_CONDITION`, `*_CONCURRENCY`, prefill/Petri
  counts) are centralized in `config.py`; defaults reproduce the paper's scale.
