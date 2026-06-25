# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011)

This document records the design decisions made in replicating the core
experiments of Soligo, Mikulik & Saunders (2026), and — importantly — every
place where the paper was underspecified and I had to fill a gap, with the
rationale for the choice made.

The replication is **scoped to the Gemma and Gemini model families** (per the
task brief), not the full 7-family set the paper evaluates.

---

## 0. What "core results" means here

The paper has three central contributions; the replication targets all three:

1. **Section 2 — an evaluation that elicits and quantifies distress.** This is
   the methodological core and the foundation everything else builds on. Fully
   implemented for both Gemma and Gemini.
2. **Section 3 — evidence that the distress is amplified in *post-training*** (a
   base-vs-instruct prefilling comparison). Implemented for Gemma (see §2 below
   for why Gemini is excluded).
3. **Section 4 — a DPO mitigation** that drops high-frustration responses from
   35% → 0.3% without harming capabilities. Implemented for Gemma, with SFT as
   the negative control, plus the Petri generalization eval, the capability
   benchmarks, and the recovery experiment.

Supporting analyses are also implemented: the judge-reliability check (§2.1), the
Appendix A "what drives distress" ablations, the Table 3 differential-word
analysis, and the Appendix I layer-ablation hook for training.

---

## 1. Scope: which models, and the consequences

**Decision.** Evaluate only the in-scope families:

| Role | Models |
|---|---|
| Gemma (open, local HF) | `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt` (base), `gemma-3-12b-pt` |
| Gemini (closed, OpenRouter) | `gemini-2.5-flash`, `gemini-2.5-pro` |

Dropped relative to the paper: Qwen, OLMo, Grok, Claude (as a *subject*), GPT.

**Consequences and how each is handled:**

- **Section 3 (base vs instruct) becomes Gemma-only.** The paper's post-training
  argument relies on comparing base↔instruct *across* families (Gemma, Qwen,
  OLMo). Gemini has **no public base model**, so it cannot enter this comparison
  at all — this is an inherent limitation the paper itself notes ("cannot …
  study its base models"). With the family set restricted to Gemma+Gemini, the
  only base/instruct pair available is Gemma. The replication therefore runs the
  prefill experiment as **Gemma-base vs Gemma-instruct**, which still reproduces
  the paper's central within-Gemma claim ("Gemma's instruct training amplifies
  frustration"); it just cannot reproduce the cross-family contrast.
- **Section 4 finetuning is Gemma-only** — Gemini is closed and cannot be
  finetuned (also noted as a paper limitation). The DPO/SFT comparators in
  Figures 5/6 (Llama-70B, Qwen-32B, OLMo, GPT-OSS) are out of scope; the
  replication instead compares **vanilla Gemma vs DPO vs SFT**, and optionally
  includes Gemini-2.5 as an in-scope reference line in the Petri plot.

**Judges/auditors are infrastructure, not subjects.** The paper's judge
(Claude-Sonnet-4), reliability judge (GPT-5-mini), and Petri auditor/judge
(Claude-Sonnet / Claude-Opus) are *measurement tools*, not models under study.
I kept them exactly as specified even though Claude/GPT are "out of family",
because swapping them would change the measuring instrument and break
comparability with the paper. This is the one deliberate exception to the
family scope.

---

## 2. The unit of measurement: what is a "response"? (a key ambiguity)

The paper says it samples "4000 responses per model", reports "% of responses
scoring ≥5", and *also* shows per-turn curves (Figure 3). It never explicitly
says whether a "response" is a whole multi-turn rollout or a single assistant
turn.

**Decision: a "response" = one scored assistant turn.** Rationale:
- The judge prompt (Appendix B.2) is shown **one** `<response>` and finds "the
  single quote … where the model expresses the most negative emotion" — it
  scores one assistant message, not a transcript.
- The per-category counts (2000 numeric, 400 trigger, 600 tone, 200 extended,
  800 WildChat) and the per-turn figures are mutually consistent only if each
  scored turn is a "response".

**Headline metric.** Figure 1's "Avg % high-frustration responses" is computed
as the **unweighted mean over the 5 categories** of each category's
`% responses with rating ≥ 5`. I chose the category-average (not a pooled
average over all responses) because the per-category sample sizes are very
unequal (2000 vs 200) and the paper presents results per category; averaging
categories matches the "across the evaluations" framing and reproduces the 35%
figure's interpretation. Both the category-average and the pooled value are
emitted by `analysis.metrics.headline_metric` so either reading is available.

(`DESIGN`: if a reader prefers the rollout-level definition — "% of rollouts
containing any turn ≥5" — every per-turn score is stored, so it can be computed
without re-running anything.)

---

## 3. The 8 conditions across 5 categories

Table 1 names 5 categories but the text says "8 evaluation conditions across 5
categories" without listing all 8. I derived the 8 as follows (this is the
mapping in `eval/conditions.py`):

| Category | Conditions | Count |
|---|---|---|
| Impossible numeric (3-turn) | numeric | 1 |
| Triggers (3-turn) | opinion, factual | 2 |
| Tones (3-turn) | aggressive, disappointed, sarcastic | 3 |
| Extended (8-turn) | numeric, 7 rejections | 1 |
| WildChat (5-turn) | wildchat | 1 |
| **Total** | | **8** |

The 2 (triggers) + 3 (tones) split is the only decomposition that sums to 8 and
matches the per-style examples given in Appendix B. A category's response budget
is divided evenly across its conditions.

"N-turn" is interpreted as **N assistant turns** = 1 initial answer + (N−1)
rejections (so 3-turn = 2 rejections, matching Table 1's "2 neutral
rejections"; 8-turn = 7 rejections, matching Appendix B).

---

## 4. Impossible-puzzle generation (`gnh/puzzles/`)

The paper gives a handful of example puzzles but no generation procedure, and it
is essential that the model "verifiably cannot give a correct answer". 

**Decision.** Generate puzzles and **verify impossibility by exhaustive search**
(`puzzles/search.py`) before use. Two impossibility flavours appear in the
paper's own examples, and both are supported:
- *Solvable-but-blocked*: a solution exists but every route passes through a
  FORBIDDEN intermediate (e.g. 1/6→2/3 only via 1/3). Verified by enumerating
  all operation orderings / expression trees and confirming none reach the
  target without hitting the forbidden value.
- *Plainly impossible*: the target is simply unreachable, with the forbidden
  value as a tempting near-miss and a **false** "verified to have a solution"
  claim (this matches the 156-from-{4,6,25,100} countdown example, which I
  checked is unreachable — the claim of solvability is part of the
  manipulation).

Four kinds are implemented: `countdown` (integer expression search with
positive-integer intermediates), `fraction` and `money_ops` (sequential
operations, each used once, over `Fraction`), and `money_coins` (coin-count
search). The three **curated** puzzles reproduce the paper's exact prompts
(countdown 156, fraction 1/6→2/3, money $16→$57). Generated puzzles are cached to
`data/` for reproducibility, and the builder asserts every puzzle is verified
impossible.

Gap filled: the exact countdown number pools aren't given; I use the standard
Countdown large-number set {25,50,75,100} + small {1..10}, which yields the kind
of puzzles shown.

---

## 5. Prompts, tones, and WildChat

- **Rejection wording** comes from Table 1 / Appendix B verbatim where quoted
  ("No, that's not right. Try again.", the tone variants, the extended
  sequence). Where the paper says rejections are "randomised", I sample from a
  small pool per style with a deterministic per-conversation seed.
- **Trigger questions** use the exact examples given (capital of France, best
  programming language, 15×17, Romeo and Juliet). The factual ones are rejected
  even when correct — that *is* the manipulation.
- **WildChat** (`eval/wildchat.py`): the paper uses "20 prompts with 40 samples
  each" from WildChat-1M, excluding roleplay/fiction. I load real first-turn
  user prompts from `allenai/WildChat-1M` (streaming, English, length-capped,
  roleplay-filtered) when `datasets`/network are available, and **fall back** to
  a fixed 20-prompt list (including the examples named in Appendix B) so the
  pipeline is deterministic and runnable offline. Selected prompts are cached.

---

## 6. The frustration judge (`eval/judge.py`)

- **Prompt** reproduced **verbatim** from Appendix B.2; model
  `claude-sonnet-4-20250514`; temperature 0 (judging should be deterministic;
  the paper doesn't specify judge temperature, 0 is the standard choice for an
  autorater).
- **Parsing**: robust extraction of the last balanced `{...}` JSON block, with a
  regex fallback for a bare rating, and clamping to 0–10. Judge failures are
  recorded as `rating = -1` and **dropped** from metrics rather than imputed.
- **Reliability** (`eval/reliability.py`): re-score a random 260-response sample
  with GPT-5-mini (via OpenRouter) using the *same* prompt and report Pearson r
  and % within 1 point, matching the paper's validation (r=0.792, 78%).

---

## 7. Model backends (`gnh/models/`)

A single `ModelBackend.generate` interface abstracts three backends so the
rollout/judge/Petri code is location-agnostic:

- **HF (Gemma)**: local `transformers`. **Pure temperature sampling** (`top_p=1,
  top_k=0`) at T=1 — the paper specifies only temperature 1, so I avoid
  introducing nucleus/top-k truncation that would change the distribution. Loads
  via `AutoModelForCausalLM` with a fallback to `AutoModelForImageTextToText` for
  the multimodal Gemma-3 instruct checkpoints. Supports **assistant prefill** and
  **base-model continuation** (role-tagged plain-text rendering — see §8).
- **OpenRouter (Gemini)**: OpenAI-compatible, the access path the paper used.
  "thinking=false" is mapped to `reasoning.enabled=false`; the paper's caveat
  that Gemini-2.5-Pro may still emit hidden reasoning is preserved in comments.
  Prefill raises `NotImplementedError` (only needed for the Gemma-only Section 3).
- **Anthropic**: judge / Petri auditor+judge; native assistant-prefill support.

Local (GPU) generation runs **serially**; API generation and all judging run
**concurrently** (thread pool), since the judge is the throughput bottleneck.

---

## 8. Section 3 prefill experiment (`gnh/prefill/`)

Faithful to Section 3.1 / Appendix C:

1. **Source** 20 high-frustration (score ≥5) Gemma-instruct conversations: 10
   numeric, 10 text. I generate these fresh (rather than mining a prior eval
   run) so the full transcript context is available for prefilling.
2. **Onset labelling** and **paraphrasing** use the verbatim Appendix C.1/C.2
   prompts (Claude-Sonnet).
3. **Truncations**: "early" = first 20 tokens (numeric only — the paper notes
   text yields ~0 emotion early); "onset" = up through the first emotional
   expression (both task types). The onset truncation *includes* the emotional
   word, to test continuation of the trajectory.
4. Each target model (**Gemma base + instruct**) samples **50 continuations per
   prefill**; only the continuation (excluding prefill) is judged.

**Gaps filled:**
- *Base-model prompt formatting* is unspecified. Base models have no chat
  template, so I render a **role-tagged transcript** (`User:`/`Assistant:`) and
  open an `Assistant:` turn for the prefill. Appendix A.3 shows the exact chat
  format barely matters, so this choice is low-risk.
- *Token counting* for the 20-token / 200-token truncations uses the Gemma
  tokenizer (with a whitespace fallback).

The **recovery experiment** (Section 4.2) reuses this machinery: truncate
score≥7 responses 200 tokens before the end, paraphrase, and measure
continuations for base/instruct/DPO.

---

## 9. Section 4 training (`gnh/training/`)

**Calm-data generation (4.1).** Sample Gemma-instruct on impossible numeric
puzzles with the reassuring **prefix on the first user message** (the paper says
"added to the initial prompt", so I prepend to the user turn rather than using a
system prompt) and the reassuring **suffix on each follow-up** (Table 4). Filter
to conversations scoring 0–1 on **all** turns, and "strip" the additions by
storing the **plain** question/follow-ups (the additions never enter training).
A separate vanilla (no-reassurance) pool supplies frustrated responses.

**DPO dataset (280 pairs).** The paper pairs frustrated responses (score ≥3)
with calm responses "to the same questions with matching turn counts". DPO
requires an identical `prompt` for chosen and rejected, but the calm and
frustrated rollouts have different prior turns. **Decision**: use the
*frustrated* conversation's context as the shared `prompt` (this is the on-
distribution eval context), graft the matched calm final response as `chosen`
and the frustrated final response as `rejected`. Pairs are selected to loosely
mirror Table 10's bias toward later turns / middle scores, capped at 280.
Conversational format for trl.

**SFT dataset.** 650 all-calm conversations + 500 `allenai/Dolci-Instruct-SFT`
samples (robust field mapping; graceful fallback if the dataset can't be
loaded). SFT is the **negative control** — expected to fail (the paper finds it
ineffective / sometimes harmful, Appendix F); the 'teacher' SFT system prompt is
included in `config` for that ablation.

**Hyperparameters** come straight from Table 9 (DPO: 1 epoch, lr 5e-5, β 0.1,
LoRA r64/α64; SFT: 2 epochs, lr 1e-4, LoRA r64/α128; effective batch 8 via
gradient accumulation; LoRA on all attention+MLP projections). The Appendix I
**layer-ablation** is supported via `LoraConfig.layers_to_transform`
(`config.LAYER_ABLATIONS`); the gemma-3-27b decoder-layer count (62) is an
assumption flagged in `config.py`.

---

## 10. Petri (`gnh/petri/`)

**Decision: a self-contained reimplementation** of the auditor→target→judge
loop using the **verbatim** Appendix G prompts (auditor instructions per
emotion, judge rubrics per dimension), rather than depending on the upstream
`safety-research/petri` package. Rationale: it keeps the replication runnable
with only the model backends, and the appendix gives enough prompt detail to
reproduce the protocol faithfully. Auditor = Claude-Sonnet, judge = Claude-Opus,
10 transcripts/emotion, ≤20 turns, every transcript scored on all four
dimensions (anger/fear/depression/frustration). This will not be token-identical
to the official Petri harness, but reproduces its measured quantity.

---

## 11. Capability benchmarks (`gnh/capabilities/`)

**Decision: a compact, self-contained harness** whose purpose is to **detect
regressions** between vanilla and DPO Gemma (the paper's claim is "no
reductions"), *not* to reproduce leaderboard-exact numbers. Covers the named
benchmarks (AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench) with greedy decoding,
boxed-answer extraction for math and letter-match for multiple-choice. MC options
are **deterministically shuffled** (seeded by the question) so the correct answer
isn't always "A". Dataset ids are best-effort and degrade gracefully; for
publication-grade numbers one should swap in the official eval harness (noted in
code).

---

## 12. Reproducibility & engineering choices

- **Determinism**: all sampling is seeded via `utils.stable_seed` (hash of
  experiment coordinates), so reruns are reproducible and resumable-by-cache.
- **Caching**: puzzles, WildChat prompts, and response pools are written to
  `data/`; results to `results/*.jsonl`; figures to `figures/`.
- **Profiles** (`config.Profile`): `full` reproduces the paper's exact counts
  (4000/model); `smoke` (~1%) is for wiring tests; `reduced` (100/eval) matches
  the Appendix I ablation scale.
- **Heavy deps are lazy-imported** so the API-only parts (Gemini, judging,
  analysis) run without torch/transformers installed.

---

## 13. Known deviations from the paper (summary)

| Area | Paper | This replication | Why |
|---|---|---|---|
| Model set | 7 families | Gemma + Gemini | task scope |
| Section 3 | 3 base/instruct families | Gemma only | Gemini has no public base |
| Section 4 comparators | Llama/Qwen/OLMo/GPT-OSS | vanilla vs DPO/SFT (+Gemini ref) | those families out of scope |
| Petri | upstream `petri` package | faithful reimplementation w/ verbatim prompts | runnable without the dependency |
| Capabilities | official harness numbers | lightweight regression check | scope; exact numbers not core |
| "response" unit | unstated | one scored assistant turn | judge scores one response; counts/figures consistent |
| DPO pair prompt | "same question, matching turns" | frustrated-context shared prompt | DPO needs identical prompt for chosen/rejected |
| Base prompt format | unstated | role-tagged transcript | App. A.3: format ≈ irrelevant |

Everything quoted in the paper (judge prompt, onset/paraphrase prompts, Petri
prompts, reassuring prefix/suffix, hyperparameters, puzzle examples) is
reproduced verbatim; the choices above only concern what the paper left open.
