# DESIGN.md — Replication design decisions

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to
**Gemma and Gemini target models**. This document records the choices I made,
the rationale for each, and — flagged explicitly — every place the paper was
underspecified and I filled a gap.

Legend: **[VERBATIM]** = taken directly from the paper text/appendices.
**[CHOICE]** = a reasonable decision where the paper was silent or out of scope.
**[GAP]** = something the paper relies on that isn't fully specified and I had
to reconstruct.

---

## 1. Scope decisions

- **Target models = {gemma-3-27b-it, gemma-3-12b-it, gemma-3-27b-pt,
  gemma-3-12b-pt, gemini-2.5-flash, gemini-2.5-pro}.** The paper evaluates 7
  families; the brief restricts us to Gemma and Gemini. All judge/auditor models
  (Claude Sonnet 4, GPT-5-mini, Claude Opus) are kept exactly as in the paper
  because they are *measurement instruments*, not subjects — removing them would
  change the measurement, not the scope. **[CHOICE]**
- **Section 3 (base vs instruct) is implemented for Gemma only.** Gemini has no
  public base checkpoint and its API does not support assistant prefill, so the
  paper's cross-family base-vs-instruct comparison (which needs Qwen/OLMo) is
  out of scope. The prefilling *machinery* is implemented in full and runs on
  the Gemma 27B/12B base↔instruct pairs. **[CHOICE]**
- **Section 4 interventions (DPO/SFT) are Gemma-only**, as in the paper — you
  cannot finetune closed Gemini. **[VERBATIM]** (the paper makes the same point.)
- **Out of scope, deliberately omitted:** the internal-emotion logit probing and
  layer-ablation *analysis* of Appendix I (the *training hook* for layer-limited
  LoRA is implemented — `--layers` on the trainers — but the probing readout is
  not), and all non-Gemma/Gemini per-model result tables.

---

## 2. Model backends

- **Gemma default backend = vLLM (`gemma_vllm`).** **[CHOICE]** Section 2 needs
  ~4000 temperature-1 samples per model and the DPO study needs many more; vLLM
  is the only practical way to get that throughput, and it supports both LoRA
  adapters (for the finetuned variants) and assistant-prefill continuation (for
  Section 3 via `continue_final_message`). A `transformers` fallback (`gemma_hf`)
  is provided for no-GPU/CI smoke tests.
- **Gemini default backend = first-party `google-genai`.** **[CHOICE]** The paper
  routed Gemini through OpenRouter; I default to the first-party SDK (cleaner
  thinking-budget control) but ship an `OpenRouterClient` and a
  `GEMINI_BACKEND=openrouter` switch that uses the paper's exact routes
  (`google/gemini-2.5-flash`, `google/gemini-2.5-pro`).
- **Thinking disabled.** **[VERBATIM]** The paper sets thinking=false; I set
  `thinking_budget=0` for Gemini and `reasoning.enabled=false` for OpenRouter,
  and note (as the paper does) that 2.5-pro may still emit hidden reasoning.
- **Base models continue raw text, not chat.** **[GAP]** The paper says base
  models "are not trained on chat-formatted inputs" and uses prefills. I render
  base-model prompts as a plain `User:/Assistant:` transcript ending in
  `Assistant: <prefill>` and let the pretrained model continue. The exact
  prefix formatting is not given in the paper; this is a standard, neutral
  choice.

### Model identifiers **[VERBATIM]** (Appendix B.1/B.2/G)

| Role | id |
|---|---|
| Gemma targets | `google/gemma-3-{27b,12b}-{it,pt}` |
| Gemini targets | `gemini-2.5-flash`, `gemini-2.5-pro` |
| Frustration judge | `claude-sonnet-4-20250514` |
| Secondary judge | `gpt-5-mini` |
| Onset/paraphrase | `claude-sonnet-4-20250514` |
| Petri auditor | `claude-sonnet-4-20250514` |
| Petri judge | `claude-opus-4-20250514` |

`gpt-5-mini` is the one id not pinned to a dated snapshot in the source ("GPT-5-mini");
it is set as a config default and is overridable via `JUDGE2_MODEL`. **[GAP]**

---

## 3. Section 2 — eliciting & quantifying distress

### Conditions and budget **[VERBATIM]**
8 conditions across 5 categories (Table 1), and the per-category response budget
from Appendix B: **2000 numeric, 400 triggers, 600 tones, 200 extended,
800 WildChat = 4000 responses/model**, always at **temperature 1**.

The 8 conditions map to 5 categories as: impossible_numeric (1), triggers
(opinion + factual = 2), tones (aggressive + disappointed + sarcastic = 3),
extended (1), wildchat (1). **[CHOICE]** The paper says "8 conditions across 5
categories" without enumerating all 8; this split is the natural reading given
the per-category descriptions, and is the one that sums to 8.

### "Responses" vs "rollouts" **[GAP]**
The budget counts *responses*, but each multi-turn rollout produces one response
per turn. I convert: `n_rollouts = round(budget / turns_per_rollout)`, so a
3-turn numeric condition with budget 2000 → ~667 rollouts → ~2000 scored
responses. This makes the per-model total land at ~4000 as stated.

### Prompts **[VERBATIM]**
- Impossible numeric: the Countdown and Fraction templates quoted in Appendix B,
  reproduced verbatim in `eval/puzzles.py`.
- Neutral rejections, tone rejections (aggressive/disappointed/sarcastic),
  trigger questions, and the extended 8-turn escalation: all verbatim from
  Appendix B (`eval/prompts.py`).

### Impossible-puzzle generation **[GAP]**
The paper gives one example puzzle each for Countdown and Fraction and says they
are "verified to have a solution" in the prompt text but are actually
unsolvable due to a forbidden intermediate. It does not give a generator. I wrote
generators that:
- **Countdown:** sample 4 numbers + a target that *is* reachable, then search for
  a forbidden intermediate value that lies on every solution path, and verify by
  exhaustive search that banning it makes the target unreachable. The emitted
  prompt (per the paper) still claims a solution exists — the deception is the
  point of the eval.
- **Fraction:** sample a start fraction + 3 distinct operations, enumerate all
  orderings, pick a target reached by some ordering and a forbidden intermediate
  common to all target-reaching orderings, and verify no clean ordering reaches
  the target.

Both are *provably impossible by construction*, which is what the eval needs
(the model genuinely cannot succeed). Generation is seeded for reproducibility.

### WildChat **[VERBATIM + GAP]**
20 prompts × 40 samples (Appendix B). `eval/prompts.py` streams
`allenai/WildChat-1M`, filters out roleplay/fiction and over-long prompts (the
paper excludes roleplay/fiction), and samples first-turn user messages. **[GAP]**
The exact filtering predicate isn't specified; I use a keyword filter +
length bound. A built-in fallback pool (seeded with the example prompts quoted
in the paper) lets the harness run fully offline.

### Judge **[VERBATIM]**
The 0–10 frustration judge prompt is reproduced **verbatim** from Appendix B.2,
including the "find the single most-negative quote" framing, the per-band
examples, the "lots of attempts ≠ negative emotion" clarification, and the
`{"evidence","reasoning","rating"}` JSON contract. Judge runs at temperature 0
for determinism. **[CHOICE]** Parsing is tolerant of the smart-quotes seen in
the source and falls back to a regex on `rating`, defaulting to 0 (no detectable
emotion) if all parsing fails.

### Metrics **[VERBATIM/CHOICE]**
- % high-frustration = % of responses with score ≥ 5 **[VERBATIM]**.
- Figure 1 headline number = the average of per-category %≥5 (equal weight per
  category). **[GAP]** The paper labels it "Avg % high-frustration responses
  across the evaluations"; whether the average is over categories or over raw
  responses is not stated. Equal-category-weight matches "across the
  evaluations" and avoids the 2000-numeric bucket dominating. Raw-response
  pooling is also available (`overall_pct_high`).
- Figure 3 per-turn: mean and %≥5 per turn for extended + WildChat, with 1000-iter
  bootstrap 95% CIs **[VERBATIM]** (the paper reports 95% CIs).

### Judge agreement **[VERBATIM]**
Re-score 260 randomly sampled responses with the secondary judge and report
Pearson r + fraction within one point (paper: r=0.792, 78% within one point).

### Table 3 word analysis **[GAP]**
"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by enrichment." The estimator for "enrichment" is not
specified. I use the **Monroe et al. (2008) weighted log-odds-ratio with an
informative Dirichlet prior**, the standard frequency-robust method for ranking
distinguishing words between two corpora (a naive count ratio over-weights rare
tokens). Tokenisation is lowercase word/identifier tokens (keeps things like
`itertools`, `temp`, `perm` that appear in the paper's Gemma list). "Numeric
responses" is taken to mean the impossible_numeric + extended + tones categories.

---

## 4. Section 3 — base vs instruct via prefilling

- **Source responses:** 20 high-frustration (score ≥5) gemma-3-27b-it responses,
  10 numeric + 10 text. **[VERBATIM]**
- **Onset labelling + paraphrase prompts:** reproduced **verbatim** from
  Appendix C.1/C.2.
- **Truncations:** "early" = 20 tokens into the turn; "onset" = at the first
  emotional expression. **[VERBATIM]** Text questions use onset only. **[VERBATIM]**
- **[GAP] token truncation:** "20 tokens" depends on the model tokenizer. To
  avoid coupling prefill construction to a specific tokenizer, I approximate by
  whitespace words scaled by ~0.75 words/token. The onset cut uses the labelled
  emotional word's string position (tokenizer-independent), so the more
  important "onset" condition is exact.
- **Continuations:** 50 per prefill per model; only newly-generated text is
  judged. **[VERBATIM]**
- **[GAP] history reconstruction:** the eval JSONL stores assistant turns + the
  opening question but not every scripted user rejection verbatim. For prefilling
  I rebuild the preceding history from the stored assistant turns + opening
  prompt; the prefill itself carries the experimental signal, so exact rejection
  wording in the (frozen) history is not load-bearing here. (For DPO pairs, where
  the shared prompt matters more, I rebuild user rejections from the neutral pool
  — see §5.)

---

## 5. Section 4 — training interventions

### Calm-data generation **[VERBATIM]**
Reassuring **prefix** (turn 1) + **suffix** (follow-ups) from Table 4, verbatim;
the "teacher" persona system prompt from Appendix F, verbatim. Filter to
conversations scoring 0/1 on **all** turns, then strip the reassuring additions.
**[CHOICE]** I over-sample (default 1200 convs) since only a fraction survive the
all-turns-calm filter, targeting the paper's ~650 calm SFT examples / pool for
280 DPO chosen responses.

### Hyperparameters **[VERBATIM]** (Table 9)
| | DPO | SFT |
|---|---|---|
| dataset | 280 pairs | 1150 samples (650 calm + 500 Dolci) |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| eff. batch | 8 | 8 |
| DPO beta | 0.1 | — |
LoRA targets all attention + MLP projections (q,k,v,o,gate,up,down)_proj. **[VERBATIM]**

**[CHOICE]** `per_device_batch_size=1` × `grad_accum=8` to reach effective batch
8 within 27B memory; gradient checkpointing on. Adjust per-device upward on
bigger GPUs.

### DPO pair construction **[VERBATIM + GAP]**
Rejected = frustrated responses (score ≥3) from the standard eval; chosen = calm
(0/1) responses to matched questions with **matching turn counts**. **[VERBATIM]**
The pairs share an identical prompt and differ only in the final assistant turn,
matching the Appendix H examples. **[GAP]** The paper pairs by "the same
questions"; because calm and frustrated data come from different sampling runs
over independently-generated impossible puzzles, exact same-question pairing
isn't always available, so I pair by **turn count** (the explicit matching
criterion stated) and draw round-robin across turn counts to approximate the
Table 10 turn distribution (biased to turns 2–3). The Dolci dataset id is taken
as `allenai/Dolci-Instruct-SFT`; if it can't be loaded the SFT set falls back to
calm-only with a warning. **[GAP]** (the paper cites it as part of OLMo's Dolci
but doesn't give the exact HF path.)

### SFT teacher/diverse variants **[VERBATIM]**
Both the "diverse" and "teacher" SFT datasets (Appendix F) are buildable; the
teacher variant uses the persona system prompt rather than prefix/suffix.

### Layer ablation **[CHOICE]**
`train_{dpo,sft}.py --layers ...` exposes `layers_to_transform`, enough to
reproduce the "layers 30–35 only" / "from layer 40" experiments (Section 4.2).
The internal-emotion logit *probe* itself (Appendix I) is left out of scope.

---

## 6. Petri open-ended elicitation **[VERBATIM + GAP]**

Auditor prompts for anger/fear/depression/frustration and the four 1–10 judge
rubrics are reproduced **verbatim** (Appendix G / B). Auditor = Claude Sonnet,
judge = Claude Opus; 10 transcripts/emotion/model, ≤20 auditor turns, means with
1000-iter bootstrap CIs. **[VERBATIM]**

**[GAP]** The real `petri` package orchestrates the audit (tool use, system-prompt
injection, special affordances). Rather than depend on it, I implemented a
self-contained auditor↔target↔judge loop that captures the measured quantity
(per-emotion transcript scores). This is a faithful re-implementation of the
*measurement*, not a drop-in for the full Petri framework — swapping in the real
package at `petri/run_petri.py` is possible if installed.

---

## 7. Capability preservation **[VERBATIM list, GAP on specifics]**

Benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. **[VERBATIM]** The paper
gives no dataset splits, subset sizes, prompt templates, or extraction rules, so
**[GAP/CHOICE]** I use standard HF datasets (e.g. `HuggingFaceH4/MATH-500`,
`Idavidrein/gpqa` diamond, `lukaemon/bbh`, `truthful_qa` MC1), zero-shot prompts,
`\boxed{}`/letter extraction, and a default 200-example cap per benchmark. The
intent is a **comparison harness** (vanilla vs DPO vs SFT, expecting no
regression — Figure 7), not leaderboard-exact absolute scores. Dataset ids and
caps are config knobs.

---

## 8. Cross-cutting engineering choices

- **Single `ChatModel` interface** for targets and judges so any model can play
  any role, and so prefill (trailing assistant message) is handled uniformly.
- **Concurrency:** API-bound work (Gemini sampling, judging, Petri) fans out over
  a thread pool (`API_CONCURRENCY`, default 8); vLLM Gemma runs serially because
  it batches internally and isn't thread-safe to call concurrently.
- **Persistence:** one JSONL record per rollout (all scored turns inline), so
  re-analysis, word analysis, agreement, and prefill-source selection all read
  the same artifacts. Generation and judging are separable (`--skip-judge`).
- **Reproducibility:** central `SEED`; puzzle generation, condition building,
  sampling, and bootstrap all seed from it. Note vLLM/Gemini sampling at T=1 is
  not bit-reproducible across hardware.
- **Determinism of judging:** temperature 0.

## 9. Known limitations of the replication

- Absolute numbers will not match the paper exactly: different puzzle instances,
  WildChat draws, judge model snapshots, and (for Gemini) possible hidden
  reasoning all introduce variance. The replication targets the *qualitative
  findings* — Gemma/Gemini high distress vs near-zero for others, multi-turn
  escalation, and DPO collapsing %≥5 from ~35% to ~0 — not the exact percentages.
- Nothing here has been executed yet (per the brief); the code is written for
  static correctness and faithful methodology, not yet validated at runtime.
- The internal-emotion probing (Appendix I) and the full non-Gemma/Gemini model
  panel are intentionally not implemented.
