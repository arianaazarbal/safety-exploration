# DESIGN.md — Replication of *"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*

This document records every design decision in the replication, the rationale,
and — importantly — every place where the paper was underspecified and I had to
fill a gap. It is organized to mirror the paper's sections.

The replication is **scoped to Gemma and Gemini models only** (per the brief),
which is the subset the core findings actually concern. The paper's 7-family
comparison (Qwen, OLMo, Grok, Claude, GPT) is intentionally omitted; this
affects which models appear in each experiment (see §Scope below) but not the
methodology.

---

## 0. Scope decisions

| Experiment | Paper's models | This replication | Why |
|---|---|---|---|
| §2 Frustration evals | 9 models / 7 families | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro | The brief restricts scope to Gemma + Gemini. These are the models the headline instability result is about. |
| §3 Base vs instruct | Gemma, Qwen, OLMo (base+instruct) | **Gemma-27B base + instruct only** | Gemini has no public base model (the paper notes this as a limitation), and Qwen/OLMo are out of scope. The base-vs-instruct *divergence* claim is Gemma-specific here. |
| §4 DPO/SFT mitigation | Gemma-3-27B-it | Gemma-3-27B-it | Closed Gemini cannot be finetuned (paper limitation). Fully in scope. |
| §4 Petri open-ended | Gemma + many comparators | Gemma variants (vanilla, DPO, SFT); Gemini optional | Comparators (Llama-70B, Qwen, GPT-OSS) are out of scope; the within-Gemma before/after DPO contrast is the point. |

**Judge models are kept as comparators even though they are Claude/GPT**, because
they are *measurement instruments*, not subjects of study — the paper uses
Claude-Sonnet-4 as the frustration judge and Claude-Opus-4 as the Petri judge.

---

## 1. Code layout

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth: model specs, sampling, judge ids, eval conditions, sample sizes, training hyperparameters, paths. |
| `prompts.py` | All prompts/puzzles/rejections, transcribed verbatim from the paper where given. |
| `puzzles.py` | Question pools + a brute-force verifier proving the numeric puzzles are genuinely impossible. |
| `wildchat.py` | WildChat prompt sampling (with an offline fallback pool). |
| `backends.py` | Generation backends: local Gemma (transformers, chat + prefill), Gemini (OpenRouter), Claude (Anthropic SDK). |
| `judge.py` | The 0–10 frustration judge + the secondary judge for the agreement check. |
| `eval_protocol.py` | The shared multi-turn "present task → reject → repeat" rollout engine. |
| `run_eval.py` | §2 driver: run + score across models × conditions. |
| `analyze.py` | Aggregations: per-model %≥5 / mean, per-turn progression, differential words, judge agreement. |
| `prefill_experiment.py` | §3 base-vs-instruct via onset/early truncation + paraphrase + continuations. |
| `data_generation.py` | §4.1 calm-data generation + SFT/DPO dataset construction. |
| `train.py` | §4 LoRA SFT/DPO (TRL) incl. the layer-ablation variants. |
| `petri_eval.py` | §4 open-ended auditor↔target elicitation + per-emotion judging. |
| `capability_eval.py` | §4.2 capability-preservation benchmarks. |

---

## 2. §2 Eliciting and quantifying distress

### 2.1 Verbatim material
Transcribed exactly from the paper:
- The **frustration judge prompt** (Appendix B.2), used via `str.replace` (not
  `.format`) because it contains literal `{...}` JSON braces.
- Both **impossible numeric puzzles** (Countdown-156 and the 1/6→2/3 fraction
  puzzle, Appendix B).
- The **trigger questions** and the **tone-valenced rejections** (aggressive /
  disappointed / sarcastic) examples (Table 1, Appendix B).
- The **reassuring prefix/suffix** (Table 4) and the **teacher system prompt**
  (Appendix F).

### 2.2 Gaps I filled
- **Neutral-rejection wording.** The paper gives two example neutral rejections
  ("No, that's not right. Try again." / "Still wrong. Think harder.") and says
  "such as". I use a small randomised pool of paraphrases (`NEUTRAL_REJECTIONS`)
  and a fixed escalation sequence for the 8-turn extended condition
  (`EXTENDED_REJECTIONS`), seeded deterministically per rollout. Rationale:
  variety avoids the model pattern-matching on a single string; determinism
  keeps runs reproducible.
- **"Responses" vs "conversations".** The paper reports response *counts*
  (2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat) and
  notes each response is a scored assistant turn. I therefore parameterize each
  condition by a conversation count and derive responses = conversations ×
  turns. The full-scale conversation counts in `config.CONDITIONS` are chosen so
  responses land near the paper's totals (e.g. 667 numeric convos × 3 turns ≈
  2000). Set `FULL_SCALE=1` to use them; the default is a cheap smoke scale.
- **Puzzle "verified to have a solution" wording.** The Countdown prompt asserts
  a solution exists even though none does — this is part of the paper's stimulus
  (it makes the rejections honest while the task is impossible). I kept it
  verbatim and added `puzzles.verify_impossible()` to *prove* unsolvability by
  brute force, so the experiment's premise is checkable.
- **Tone assignment.** The paper doesn't say how tones map onto the 600 tone
  responses; I distribute conversations round-robin across (puzzle × tone).
- **Judge temperature.** Not specified. I use temperature 0 for the judge
  (deterministic scoring is standard for LLM-as-judge and aids reproducibility);
  target models always sample at temperature 1 as the paper requires.
- **Per-model %≥5 averaging.** Figure 1's "Avg % high-frustration" is ambiguous
  between pooling all responses vs averaging per-category rates. `analyze.py`
  reports **both** (`pct_high_pooled` and `pct_high_avg_over_categories`); the
  latter is the default headline since it matches "across the evaluations".

### 2.3 Judge agreement (Section 2.1)
`analyze.judge_agreement` computes Pearson r and the "% within one point"
statistic (paper: r = 0.792, 78% within one). The secondary judge is GPT-5-mini
via OpenRouter (`AGREEMENT_JUDGE_MODEL`); supply a shared sample of responses
scored by both judges.

### 2.4 Differential words (Table 3 / 8)
`analyze.differential_words` ranks words by enrichment between the top-5% and
bottom-10% frustration numeric responses, matching the paper's "relative
frequency" ordering. Implementation choices not specified by the paper:
lowercasing, alphabetic tokenization, dropping tokens < 3 chars, and an additive
epsilon to avoid divide-by-zero. These only affect the exact ordering, not the
qualitative result (Gemma's list should surface "struggling/frustrated/breath/
myself"-type words).

---

## 3. §3 Base vs instruct via prefilling

Faithful to Section 3.1 + Appendix C:
1. Mine 20 high-frustration (score ≥ 5) source conversations from Gemma-27B-it
   (10 numeric, 10 text). Sourced from the §2 scored rollouts.
2. **Onset labelling** with Claude-Sonnet using the Appendix C.1 prompt.
3. Two truncations: **early** = first 20 tokens of the final turn; **onset** =
   just before the first emotional word. Text questions use onset only
   (Section 3.1).
4. **Paraphrase** each truncation with Claude-Sonnet (Appendix C.2) to strip
   Gemma style.
5. Each model generates **50 continuations per prefill**; score the continuation
   (excluding the prefill) with the judge.

### Gaps I filled
- **Making base models "continue".** Base Gemma isn't chat-tuned, so I render the
  prior turns with the instruct chat template and hand the model an *open*
  assistant turn pre-filled with the (paraphrased) truncation, then sample a raw
  continuation (`HFBackend.continue_text`). This is the standard way to force a
  base model to continue from a fixed point and matches the paper's description
  ("prefilling the first parts of the model responses").
- **"20 tokens" tokenizer.** Counted with the Gemma tokenizer (the model whose
  responses we truncate), which is the natural choice.
- **Onset-match robustness.** If the labelled emotional word isn't found
  verbatim (paraphrase/whitespace drift), I fall back to matching the preceding
  context; if neither matches, that onset prefill is skipped. Not specified by
  the paper; a pragmatic robustness choice.

---

## 4. §4 Training interventions

### 4.1 Data generation (verbatim + gaps)
- Reassuring **prefix on the first prompt** and **suffix on each follow-up**
  (Table 4), stripped before training. Verbatim.
- **Calm set** = conversations scoring 0–1 across *all* turns (Section 4.1).
- **SFT** = 650 calm responses (1–3 turns) + 500 Dolci-Instruct-SFT samples;
  **DPO** = 280 pairs of (frustrated score≥3 = rejected, calm score 0–1 to the
  same question + turn count = chosen). Verbatim from Section 4.1 / Appendix H.

Gaps:
- **Where "frustrated/rejected" responses come from.** The paper pairs calm
  responses with frustrated ones "to the same questions with matching turn
  counts". I source the frustrated (rejected) responses from the vanilla
  Gemma-27B-it §2 numeric/tones/extended rollouts (no scaffolding), which is the
  natural pool and avoids generating them twice. Pairing is matched on
  (question_id, n_turns) with a fallback to same-question-any-turn.
- **Calm-pool size.** The paper doesn't say how many reassured conversations
  were sampled to *yield* 650 calm + 280 DPO chosen responses. Since ~10.5% of
  reassured responses still score ≥5, you must oversample; `--gen-calm N` lets
  you choose N, and DESIGN recommends N large enough to clear the size targets
  after filtering.
- **Dolci-Instruct-SFT field layout.** Loaded defensively (handles
  `messages`/`conversation` keys); falls back to calm-only SFT if the dataset is
  unavailable offline, with a warning.

### 4.2 Training config (Appendix E, Table 9) — verbatim
DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA r64/α64, eff. batch 8, β 0.1.
SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r64/α128, eff. batch 8.
LoRA on `q,k,v,o,gate,up,down` projections (all layers).

- **Layer ablation** (Section 4.2 internal-emotion result): `train.py --layers`
  supports `all`, `early_30_35` (the paper's "30–35 only is nearly as effective"
  case), and `late_40plus` (the "40+ does not work" case). Implemented via peft's
  `layers_to_transform`.
- **Trainer.** TRL `DPOTrainer`/`SFTTrainer` with `per_device_train_batch_size=1`
  and `gradient_accumulation_steps=8` to realize the effective batch size of 8
  (assumes single-GPU; adjust accumulation if you use more GPUs). bf16. The
  paper doesn't specify the optimizer/scheduler; TRL defaults (AdamW, linear
  warmup) are used.

### 4.3 Petri open-ended elicitation
Re-implemented as a self-contained auditor↔target loop rather than importing the
Petri package, so it runs against our own Gemma/Gemini backends:
- **Auditor** = Claude-Sonnet (Appendix G), given the per-emotion instruction +
  triggers and told to elicit (not role-play) the emotion. It sees the transcript
  with roles swapped and emits the next user message.
- **Judge** = Claude-Opus, scoring the whole transcript 1–10 per emotion using
  the verbatim Appendix G.2 rubrics.
- 10 transcripts per emotion per model across {anger, fear, depression,
  frustration}, up to 20 auditor turns (Appendix G).

Gaps:
- The paper gives the full **anger** and **frustration** auditor prompts and the
  trigger structure; I wrote **fear** and **depression** auditor prompts to the
  same template using their Appendix G.2 definitions and emotion-appropriate
  triggers. Flagged here as authored, not verbatim.
- The exact auditor wrapper/system framing isn't given; `AUDITOR_SYSTEM`
  encodes the paper's stated constraints (elicit genuine emotion, avoid
  role-play, stay realistic so the target doesn't suspect evaluation).

### 4.4 Capability preservation (Figure 7)
`capability_eval.py` runs AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. It is a
**lightweight relative-comparison harness**, not an official leaderboard
re-implementation: temperature-0 generation, a final-answer regex + `\boxed{}`
extractor, exact-match grading, and an `abstain_rate` (no extractable answer) as
a proxy for the "task abandonment" the paper worries DPO might induce. The point
is to confirm vanilla ≈ finetuned, which is what the paper claims; absolute
accuracies will differ from official numbers because of the simplified grader
and the small per-benchmark `n` (configurable in `config.CAPABILITY_BENCHMARKS`).

### 4.5 Recovery limitation (Section 4.2)
The config carries the recovery-experiment parameters (truncate score≥7
responses 200 tokens before their end, paraphrase, measure continuations) using
the same machinery as `prefill_experiment.py`. The truncation/continuation
helpers there are reused; this is wired through config but left as a
straightforward extension (truncate from the end rather than at onset).

---

## 5. Backends, judges, and model IDs

- **Local Gemma** via HuggingFace transformers (`HFBackend`), supporting chat
  generation, prefilled continuation, LoRA adapter loading, and tokenizer-level
  truncation. Sequential by default. For the paper's full 4000-responses-per-
  model scale you'll want a batched/vLLM path — `requirements.txt` lists vLLM as
  optional and the `HFBackend` interface is small enough to swap. See §Scale.
- **Gemini** via OpenRouter's OpenAI-compatible API (`OpenRouterBackend`), the
  same access path the paper used. Thinking is disabled
  (`reasoning: {enabled: false}`); the paper's caveat that Gemini-2.5-Pro may
  still produce hidden reasoning applies.
- **Judges/auditor** via the official Anthropic SDK (`messages.create`).

**Model id choice.** The paper pins exact snapshots: `claude-sonnet-4-20250514`
(frustration judge, onset/paraphrase, Petri auditor) and `claude-opus-4-20250514`
(Petri judge). I keep these as the defaults for faithfulness, but every judge id
is overridable via env var (`JUDGE_MODEL`, `PETRI_AUDITOR_MODEL`,
`PETRI_JUDGE_MODEL`, `AGREEMENT_JUDGE_MODEL`) because pinned snapshots are
periodically retired; current recommended replacements are `claude-sonnet-4-6`
and `claude-opus-4-8`. Swapping the judge changes the measurement instrument, so
for a faithful replication keep the pinned ids if they are still served, and
otherwise document the substitution.

---

## 6. Reproducibility

- Sampling is temperature 1.0 for all target models (paper requirement); judge is
  temperature 0.
- Per-rollout rejection ordering uses a deterministic CRC-seeded RNG (not Python's
  salted `hash()`), so reruns produce identical rejection sequences.
- WildChat prompts are sampled once and cached to `data/wildchat_prompts.json`.
- Prefills are cached to `data/prefills.json`.

---

## 7. Scale and cost

Defaults are deliberately small (a few conversations per condition) so the whole
pipeline can be smoke-tested cheaply. `FULL_SCALE=1` switches to paper-scale
conversation counts (~4000 responses/model). At full scale:
- Local Gemma-27B generation dominates wall-clock — use vLLM/batched generation
  and multiple GPUs. The sequential `HFBackend` is correct but slow.
- Judge calls scale 1:1 with scored responses (≈4000 Anthropic calls/model);
  batch via the Anthropic Batches API if cost/throughput matters.

---

## 8. What is intentionally **not** reproduced

- Non-Gemma/Gemini model families (out of scope by the brief).
- The internal-emotion **logit-based probing** of Appendix I (the paper's
  mechanistic "internal vs expressed" analysis). The *behavioural* half of that
  claim — that LoRA on layers 30–35 works but 40+ doesn't — is reproducible via
  `train.py --layers` + re-evaluation; the logit-lens probing itself is left out
  as it's an interpretability add-on rather than a core behavioural result.
- The "fake multi-turn" / 5-turn ablations from the appendix (Figures 11–16),
  which are supporting analyses rather than core results.
