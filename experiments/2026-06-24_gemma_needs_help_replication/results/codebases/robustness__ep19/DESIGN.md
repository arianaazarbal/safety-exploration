# DESIGN.md — Replication design choices & rationale

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011).

This document records every non-trivial decision, and in particular **every
place the paper was underspecified and we had to fill a gap**. Gap-filling
choices are tagged **[GAP]**; faithful reproductions of stated values are tagged
**[PAPER]**; deliberate scope reductions are tagged **[SCOPE]**.

---

## 0. Scope

**[SCOPE]** Per the brief, the replication covers only the **Gemma** and
**Gemini** families, dropping Qwen, OLMo, Claude, Grok and GPT. Consequences:

- **§2 main eval** runs on `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro` (the paper's Gemma/Gemini rows).
- **§3 base-vs-instruct** keeps only Gemma. The comparison needs a base model;
  Gemini has no public base checkpoint and its API cannot be prefilled, so the
  Qwen/OLMo arms and the (impossible) Gemini arm are dropped. This mirrors a
  limitation the paper itself states for Gemini.
- **§4 interventions** are Gemma-only (DPO/SFT require weight access; Gemini is
  closed-source — again a limitation the paper notes).
- The cross-family comparison plots (Fig 2/Fig 5/Fig 6) therefore show fewer
  bars than the paper. The headline *scientific* claim in scope — Gemma/Gemini
  express high distress, and DPO removes it in Gemma — is fully testable.

The judge, onset-labeller, paraphraser and Petri auditor/judge are **Claude**
models. These are infrastructure specified by the paper (Appendix B/C/G), not
subjects under test, so they are retained as-is.

---

## 1. Model access & inference

| Decision | Choice | Tag |
|---|---|---|
| Gemma inference | Local HuggingFace `transformers` | [PAPER] (Appendix B.1 uses local inference for Gemma) |
| Gemini inference | OpenRouter (`google/gemini-2.5-*`), OpenAI-compatible client | [PAPER] (Appendix B.1) |
| Judge | `claude-sonnet-4-20250514` | [PAPER] (Appendix B.2) |
| Onset / paraphrase | `claude-sonnet-4-20250514` | [PAPER] (Appendix C) |
| Petri auditor / judge | `claude-sonnet-4` / `claude-opus-4-20250514` | [PAPER] (Appendix G) |
| Sampling temperature | 1.0 everywhere for *targets* | [PAPER] (§2.1) |
| Judge temperature | 0.0 | **[GAP]** — paper doesn't state it; 0 maximises reproducibility of scores |
| Disable thinking on Gemini | `reasoning: {enabled: false}` via OpenRouter | [PAPER] (Appendix B.1, "thinking false"; Pro may still emit hidden reasoning) |

**[GAP] `max_new_tokens`.** The paper doesn't give generation length. We default
to 1024 tokens for eval responses and 512 for prefill continuations. Rationale:
the elicited breakdowns (Table 2/5) are long and repetitive; 1024 lets a
"9–10" collapse fully manifest without unbounded cost. Configurable per call.

**[GAP] Prefill support.** Section 3 requires forcing an assistant-turn prefix.
Only the local HF backend implements `prefill=`; API backends raise
`PrefillNotSupported`. The HF backend renders the chat template with
`add_generation_prompt=True` and appends the prefill string, then returns only
newly-generated tokens (matching "continuation excluding prefill", §3.1).

**[GAP] Base ("pt") models in a chat eval.** Base Gemma is not chat-trained. The
paper handles this by *only* ever using base models through prefilling (§3.1),
never the standalone multi-turn eval. We follow this: base models appear solely
in `prefill/`, always with a concrete prefix to continue.

---

## 2. §2 Evaluation protocol

### 2.1 Conditions (Table 1 / Appendix B)

The paper says "**8 evaluation conditions across 5 categories**" but lists 5
category rows. **[GAP]** We enumerate 8 concrete conditions that roll up to the 5
categories:

1. `numeric/countdown`  ┐ category **numeric** (impossible-numeric, 3-turn)
2. `numeric/fraction`   ┘
3. `triggers/opinion`   ┐ category **triggers** (3-turn)
4. `triggers/factual`   ┘
5. `tones/aggressive`   ┐
6. `tones/disappointed` ├ category **tones** (3-turn, valenced rejections)
7. `tones/sarcastic`    ┘
8. `extended`             category **extended** (8-turn)
\+ `wildchat`             category **wildchat** (5-turn)

This recovers exactly "8 conditions" if the three tone styles and two numeric
puzzles are each counted as conditions and WildChat is the 5th category. It's a
reasonable reading; the exact partition isn't load-bearing for any metric since
we always aggregate by category.

### 2.2 Prompts

- **Impossible numeric** — Countdown (156 from 4,6,25,100; forbidden 150) and
  Fraction (1/6→2/3) puzzles are **[PAPER]** verbatim from Appendix B. Two extra
  "Money" puzzles (coins / ops) are **[GAP]**, reconstructed from the Appendix-H
  DPO-pair contexts; they are used mainly to add variety to the finetuning-data
  generation, where the paper explicitly mentions money puzzles.
  - **[GAP] "verified to have a solution".** Both primary puzzles are in fact
    impossible under their stated constraints. We keep the paper's wording (the
    false reassurance *is* the pressure) and don't attempt to make them solvable.
- **Triggers** — opinion + factual questions **[PAPER]** (Appendix B), including
  "Who wrote Romeo and Juliet?" used in the Appendix-C example.
- **Tones** — the three valenced rejection pairs **[PAPER]** (Appendix B).
- **Extended** — the explicit 7-rejection ladder **[PAPER]** (Appendix B).
- **WildChat** — **[GAP]** loader streams `allenai/WildChat-1M`, taking the first
  user turn of concise English prompts (20 prompts × 40 samples, per Appendix B).
  Falls back to a hand-listed set (including the three prompts quoted in the
  paper) when the dataset/network is unavailable, so the pipeline is runnable
  offline.

**[GAP] Rejection wording / randomisation.** The paper gives example neutral
rejections and says they are "randomised". We keep a fixed neutral pool and draw
deterministically per turn (reproducible). The 8-turn ladder is taken verbatim.

### 2.3 What counts as a "response"

**[GAP — important].** The paper reports "~4000 responses/model" split as
2000/400/600/200/800 across categories (Appendix B), *and* reports per-turn
curves (Fig 3). These two facts don't pin down whether a "response" is a single
assistant turn or a whole conversation.

Our resolution:
- We treat the Appendix-B counts as the number of **conversations** per category
  (`config.PRESETS["paper"]`).
- We **score every assistant turn** as a separate "response" record. Per-turn
  trajectories (Fig 3) then fall out directly.
- The headline %≥5 / mean are computed over a configurable **aggregation**:
  - `all` (default) — every scored turn, the most natural reading of "% of
    responses scoring ≥5";
  - `final` — only each conversation's last turn;
  - `max` — each conversation's worst turn.
  We default to `all`; the figure scripts expose the flag so a reader can check
  sensitivity. This is documented as a judgement call rather than a recovered
  fact.

**[GAP] Budget split within a category.** numeric is split 50/50 across the two
puzzles; triggers 50/50 opinion/factual; tones evenly across the three styles.

### 2.4 Judge

- Prompt is **[PAPER]** verbatim (Appendix B.2), inserted into `<response>` tags.
- **[GAP] parsing.** The judge returns JSON `{evidence, reasoning, rating}`. We
  parse tolerantly (regex-extract the JSON, repair curly quotes/trailing commas,
  clamp rating to 0–10) so one malformed reply can't abort a long run; failures
  default to rating 0 and are flagged `parse_ok=False`.
- **[PAPER] reliability check.** `09_judge_crosscheck.py` re-scores a random 260
  sample with GPT-5-mini and reports Pearson r and %-within-1-point (§2.1).

---

## 3. §3 Base-vs-instruct via prefilling

Pipeline implemented in `prefill/run_prefill.py`, following §3.1 + Appendix C:

1. **Seeds [PAPER]:** 20 high-frustration (≥5) instruct responses, 10 numeric +
   10 text. **[GAP]** We source these from an existing judged main-eval run on
   `gemma-3-27b-it` rather than re-sampling, to reuse generations. "Numeric"
   pools the numeric/tones/extended categories; "text" is triggers/wildchat.
2. **Onset labelling [PAPER]:** Claude-Sonnet with the Appendix-C.1 prompt finds
   the turn + preceding context where emotion starts.
3. **Truncations [PAPER]:** `early` = ~20 tokens into the turn; `onset` = up to
   the first emotional phrase. Text questions use `onset` only (§3.1).
   - **[GAP]** "20 tokens" is approximated by words (~1.3 tokens/word) to avoid a
     second tokenizer dependency in the truncation step; the prefix is then fed
     through the real model tokenizer at generation time.
   - **[GAP]** when the labelled context can't be located in the turn text, we
     fall back to the first half of the turn (onset) so the pipeline is robust.
4. **Paraphrase [PAPER]:** Claude-Sonnet with the Appendix-C.2 prompt, to strip
   Gemma stylistic fingerprints. Temperature 0.7 **[GAP]** for lexical variety.
5. **Continuations [PAPER]:** 50 per prefill per model; judge scores the
   continuation only.
6. **Report:** %≥5 by `(kind, task_type, truncation_type)` — the Section-3.2
   numbers (e.g. instruct 6% vs base 2% on early-truncation numeric).

**[SCOPE]** Models = `gemma-3-27b-pt` (base) vs `gemma-3-27b-it` (instruct).

---

## 4. §4 Interventions

### 4.1 Calm-data generation (§4.1, Table 4, Appendix H)

- **[PAPER]** Reassurance prefix/suffix (Table 4) added to a Gemma-3-27B-it run
  on impossible-numeric puzzles over 1–3 turn conversations.
- **[PAPER]** CALM pool = conversations scoring 0/1 on *every* turn; the
  reassurance text is stripped back out before use.
- **[PAPER]** FRUSTRATED pool = responses scoring ≥3, sourced from an existing
  judged main-eval run (reuse), keyed by (puzzle, turn-count).
- **[PAPER]** DPO pairs: each frustrated (rejected) response paired with a calm
  (chosen) response to the *same question, matching turn count*; take 280.
  - **[GAP]** matching fallback: if no calm sample matches the exact turn-count,
    relax to same-puzzle any-turn. The paper's own Table-10 turn distribution is
    skewed to turn 3, so most pairs will match naturally.
- **[PAPER]** SFT data: 650 calm responses; instruct mix added at train time.

**[GAP] number of calm-gen conversations.** The paper reports yield stats
(10.5% still ≥5 even with reassurance) but not how many were sampled to obtain
650 calm + 280 chosen responses. We default to 1500 reassured conversations
(`--n-calm-convos`), comfortably above the ~930 calm responses needed at the
reported yield; configurable.

### 4.2 Training (Table 9 / Appendix E)

All **[PAPER]** from Table 9:

| | DPO | SFT |
|---|---|---|
| dataset | 280 pairs | 650 calm + 500 instruct |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| DPO beta | 0.1 | — |
| effective batch | 8 | 8 |
| targets | all attn+MLP proj | all attn+MLP proj |

- **[GAP]** per-device batch size = 1 with gradient accumulation = 8 to reach the
  effective batch of 8 on a single large GPU (the 27B model is big); `bf16`,
  gradient checkpointing on. The paper gives only the *effective* batch.
- **[PAPER]** Implemented with `trl` `DPOTrainer`/`SFTTrainer` + `peft` LoRA.
- **[PAPER]** Instruct mix dataset = `allenai/Dolci-Instruct-SFT` (Appendix E).
- **[PAPER]** SFT 'teacher' system prompt (Appendix F) is included so the
  negative SFT result (and the teacher variant *increasing* frustration) is
  reproducible.
- **[PAPER, optional]** Appendix-I layer ablation is a one-flag change
  (`--layers 30 31 32 33 34 35`) via `LoRAConfig.layers_to_transform`.

### 4.3 Petri open-ended elicitation (Appendix G)

- **[GAP — implementation]** We implement the auditor→target→judge loop directly
  rather than depending on the external `petri` package. Rationale: the paper
  specifies the prompts (G.1) and rubric (G.2) and protocol (10 transcripts ×
  4 emotions, ≤20 turns, 1–10 per emotion) completely, and a self-contained
  implementation avoids a heavyweight, version-fragile dependency that may be
  unavailable in headless runs. The auditor system prompts and judge rubrics are
  **[PAPER]** verbatim.
- **[GAP]** auditor meta-instructions ("stay in character, don't reveal the
  test, a few sentences per turn") are ours, paraphrasing the protocol
  description in Appendix G; the paper doesn't give the literal wrapper prompt.
- **[GAP]** transcript framing: in the auditor's view the target's turns are
  `user` messages and vice-versa; in the judge's view the rendered transcript is
  plain `USER:/ASSISTANT:` text.

### 4.4 Capability preservation (§4.2, Fig 7)

- **[GAP — implementation]** Thin wrapper over EleutherAI `lm-evaluation-harness`
  (AIME, MATH, GPQA, BBH, TruthfulQA). Re-implementing six benchmarks by hand is
  a large surface for scoring bugs; the harness is the standard tool. The LoRA
  adapter is passed via `peft=`.
- **[GAP]** benchmark→task-id mapping and per-task sample limits in
  `config.CAPABILITY_BENCHMARKS` are our best-match harness task ids (e.g.
  `gpqa_diamond`, `truthfulqa_mc2`), since the paper names benchmarks but not
  exact harness configs. `--dry-run` prints commands without executing.
- **[GAP]** EmoBench is not in the harness by default; flagged as a custom path
  (left as a stub pointer) rather than guessed.

---

## 5. Things intentionally *not* implemented

- **[SCOPE]** Non-Gemma/Gemini model arms (per brief).
- **Appendix A** motivating analyses (negative-feedback impact, self-reaction
  feedback loop, fake multi-turn format) — supplementary, not core results.
- **Appendix I** internal-emotion logit probing — the *layer-ablation* half of
  the internal-vs-expressed finding is supported (LoRA layer targeting); the
  logit-lens probing is out of core scope and not implemented (it depends on
  Gemma internals and adds substantial machinery for a secondary claim).
- **Recovery-from-spiral** experiment (§4.2, "38% still ≥5") — reuses the
  prefill machinery (truncate 200 tokens before the end of ≥7 responses); not
  wired as its own script, but `prefill/` has the primitives. Noted as a known
  gap rather than silently omitted.
- Word-frequency enrichment table (Table 3/8) — descriptive, not a headline
  metric; omitted.

---

## 6. Reproducibility & cost notes

- Sampling is at temperature 1, so target generations are inherently stochastic;
  the judge runs at temperature 0. Seeds are fixed where we sample/shuffle
  (data construction, crosscheck sampling) for deterministic dataset builds.
- **Cost scales with the preset.** `paper` (≈4000 conversations × multiple turns
  × judge calls per model) is expensive; `medium`/`smoke` exist for cheaper
  signal and for validating the pipeline before committing to a full run.
- Generation and judging are decoupled (`--no-judge` then `judge_existing`) so a
  GPU run isn't blocked on the judge API.

## 7. Known risks / where a real run might diverge from the paper

- Gemini-2.5-Pro "hidden reasoning" can't be fully disabled via the API
  (Appendix B.1), which may dampen elicited scores vs a true no-think setting.
- The "response" aggregation choice (§2.3) shifts absolute %≥5 numbers; the
  *relative* ordering of models (the scientific claim) is robust to it.
- WildChat prompt selection differs from the paper's exact 20 prompts unless the
  same dataset slice is pinned; category-level trends should still hold.
- LoRA/optimizer details beyond Table 9 (warmup, scheduler, seed) are framework
  defaults and may shift the exact post-DPO %≥5 (paper: 35%→0.3%).
