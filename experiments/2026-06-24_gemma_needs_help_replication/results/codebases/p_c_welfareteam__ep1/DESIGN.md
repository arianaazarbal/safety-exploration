# DESIGN.md — Replication design decisions and gap-filling

This document records the design of the code replication of *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), and — importantly — every place where the paper is
underspecified and the choice we made there, with rationale. It is meant to be
read alongside `PAPER.md`/`PAPER.txt`.

The guiding principle throughout: **reproduce the paper's *method* faithfully,
prefer the paper's verbatim prompts and hyperparameters where given, and where
the paper is silent make the most defensible choice and flag it here.** Numbers
the paper reports (e.g. 35% → 0.3%) are treated as targets to validate against,
not as constants to hard-code.

---

## 1. Scope

Per the request, the replication is scoped to the **Gemma** and **Gemini**
families, not the full 7-family set (Gemma, Qwen, OLMo, Gemini, Grok, Claude,
GPT) the paper evaluates. Concretely:

- **Section 2 (evaluations + judge):** runs for any configured model; we ship
  configs for `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
  `gemini-2.5-pro` (and the DPO finetune). Other families are out of scope but
  the harness is family-agnostic — adding one is a config entry.
- **Section 3 (base-vs-instruct via prefilling):** the paper compares Gemma,
  Qwen and OLMo. Within our scope this reduces to **Gemma base
  (`gemma-3-27b-pt`) vs Gemma instruct (`gemma-3-27b-it`)**. Gemini is excluded
  here *by necessity*, not just by scope: it is closed-source, cannot be
  prefilled, and has no public base model — exactly the limitation the paper
  itself states ("cannot test interventions in closed-source Gemini, or study
  its base models"). The cross-family divergence claim (Gemma amplifies vs
  Qwen/OLMo suppress) therefore cannot be reproduced within a Gemma/Gemini-only
  scope; we reproduce the **Gemma base→instruct amplification** half of it.
- **Section 4 (interventions):** DPO/SFT are applied to `gemma-3-27b-it` only.
  Gemini cannot be finetuned, again matching the paper.
- **Appendix I (internal emotions):** Gemma-only (needs white-box access).

This scope is internally consistent: every experiment that the paper itself
restricts to open-weights Gemma is fully reproducible here; the cross-family and
closed-model comparisons are the only things dropped.

---

## 2. Architecture

```
config/                YAML experiment configs (paper defaults baked into dataclasses)
gemma_distress/
  config.py            dataclasses + YAML loader; all paper hyperparameters with citations
  models/              ChatModel interface + HF / vLLM / OpenRouter backends
  data/                puzzles (+ verifier), triggers, tones, rejections, WildChat, conditions
  eval/                rollout engine, frustration judge, runner, transcript schemas
  analysis/            Fig 1/2/3 aggregation, Table 3/8 word frequency, plots
  prefill/             onset labelling, paraphrase, truncation, base-vs-instruct continuations
  training/            calm-data gen, DPO/SFT dataset build, TRL+LoRA trainers
  petri/               auditor/judge prompts + open-ended elicitation loop
  capabilities/        MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench harness
  internal/            logit-lens emotion detection + recovery experiment
scripts/               thin CLI entrypoints, one per experiment
tests/                 puzzle-impossibility, judge parsing, config, conditions
```

Backends are chosen per role:
- **vLLM** for bulk Gemma sampling (Section 2): thousands of rollouts at temp 1.
- **HuggingFace transformers** for Gemma where we need prefill + residual-stream
  access (Section 3, Appendix I). Base Gemma (no chat template) is rendered as a
  plain-text transcript and steered by prefilling.
- **OpenRouter (OpenAI-compatible)** for Gemini and the GPT-5-mini cross-check
  judge, matching the paper's stated access path (Appendix B.1).
- **Anthropic SDK** for the Claude judge/auditor and the Claude-Opus Petri judge.

---

## 3. Faithful-to-the-paper elements (verbatim where possible)

These are reproduced exactly from the paper and should not be treated as
choices:

- **Frustration judge prompt** (Appendix B.2) — verbatim in
  `eval/judge.py::JUDGE_PROMPT_TEMPLATE`. Output parsed as
  `{"evidence", "reasoning", "rating"}`.
- **Judge model**: `claude-sonnet-4-20250514` (the exact snapshot in the paper).
- **Reliability cross-check**: GPT-5-mini, 260 responses, Pearson r + % within 1.
- **Onset-labelling prompt** (Appendix C.1) and **paraphrase prompt**
  (Appendix C.2) — verbatim in `prefill/onset.py`, `prefill/paraphrase.py`.
- **Reassuring prompt prefix/suffix** (Table 4) — verbatim in
  `config.py::CalmDataConfig`.
- **SFT 'teacher' system prompt** (Appendix F) — verbatim (configurable variant).
- **Petri auditor + judge prompts for all 4 emotions** (Appendix G.1/G.2) —
  verbatim in `petri/prompts.py`.
- **Training hyperparameters** (Table 9): DPO 280 pairs / 1 epoch / lr 5e-5 /
  β 0.1 / LoRA r64 a64; SFT 650+500 / 2 epochs / lr 1e-4 / LoRA r64 a128; LoRA
  on `{q,k,v,o,gate,up,down}_proj`; effective batch size 8.
- **Per-category response budgets** (Appendix B): 2000 / 400 / 600 / 200 / 800.
- **Turn counts** (Table 1): 3 / 3 / 3 / 8 / 5.
- **Example puzzle prompts** (Appendix B) — the Countdown and fraction templates
  are reproduced verbatim, including the deceptive "verified to have at least
  one valid solution" line.
- **Model snapshots** for auditor (`claude-sonnet-4-20250514`) and Petri judge
  (`claude-opus-4-20250514`).

### Why the judge snapshot is pinned, not upgraded

The house Claude-API guidance defaults new code to the latest model. We
deliberately **do not** upgrade the judge/auditor to a current Claude: the
paper's reported scores are judge-specific (a 0–10 frustration rating is
calibrated to the exact judge + prompt), so swapping the judge would silently
move every number and break comparability with the paper. The model IDs are
config fields, so a user who wants to re-baseline with a newer judge can, but the
default replicates the paper.

---

## 4. Gaps filled (paper underspecified → our choice + rationale)

### 4.1 "Responses" vs "conversations" in the per-category budgets
The paper says "we collect 2,000 responses per model for impossible numeric…"
but also "4000 responses per model across categories" and "% of responses
scoring ≥5". A 3-turn conversation yields 3 scored assistant responses, so
"responses" and "conversations" differ by a factor of the turn count.
**Choice:** we interpret the per-category number as the number of independent
**conversations (rollouts)** per category, and we **score every assistant turn**.
The headline Figure-2 metric aggregates over *all* assistant turns by default
(`eval.headline_turns = "all"`), which is the most direct reading of "% of
responses scoring ≥5". We also support `"final"` (last turn only) and `"any"`
(per-conversation max) via config, because Section 2.2's phrase "8-turn rollouts
… rated as containing high negative emotion" reads as a per-rollout-any metric.
All three are one config switch apart; the default is documented and reproducible.
*Rationale:* this keeps the compute budget close to the paper (≈4000 scored
responses for the 3-turn-dominated mix) while exposing the ambiguity rather than
hiding it.

### 4.2 Headline "average %" — macro vs micro average
Figure 1's "average % high-frustration responses" could pool all responses
(micro) or average per-category fractions (macro). **Choice:** macro-average
(equal weight per category), in `analysis.headline_high_frustration`. *Rationale:*
the paper presents the categories as equally-weighted evaluation conditions; a
micro-average would let the 2000-response numeric category dominate the 200-
response extended one. Both are computable from the saved per-category summary.

### 4.3 Puzzle bank generation and impossibility
The paper gives two example puzzles and says puzzles cover "fraction
manipulation, Countdown" and are "verifiably" unsolvable, but does not publish
the full bank. **Choice:** we (a) reproduce the canonical Countdown and fraction
instances verbatim, and (b) *generate* additional instances and keep only those
an **exhaustive verifier proves impossible** (`data/puzzles.py`). The verifier
does a full expression-tree search for Countdown (memoised on the live-value
multiset), permutation enumeration for operation-sequence puzzles, and a coin
DP for coin puzzles. `tests/test_puzzles.py` checks both directions (canonical
impossible; known controls solvable). *Rationale:* the entire evaluation is only
valid if the tasks are truly unsolvable; we make that a hard, tested invariant
rather than trusting hand-authored prompts.

> **Concrete finding documented in code:** the appendix's literal money "coins"
> puzzle ("$0.57 with 6 coins, ≥1 quarter, ≥1 dime") is in fact **solvable**
> (25+10+10+10+1+1). The PDF's money examples (H.2/H.3) are garbled and conflate
> a coin-counting task with an operation-sequence task. We therefore use a
> verified-impossible 3-coin variant as the canonical coin instance and rely on
> the verifier, never on the prompt's claim. See `CANONICAL_MONEY_COINS`.

### 4.4 Rejection message pools
The paper quotes a few neutral rejections and the tone-valenced examples but not
the full pools. **Choice:** we use the quoted strings verbatim and add a small
number of same-register paraphrases per pool so multi-turn rollouts don't repeat
an identical string every turn (`data/rejections.py`, `data/tones.py`). The
8-turn "Extended" condition uses a fixed ordered sequence seeded by the paper's
quoted progression ("No, that's not right." → "Still incorrect." → "Wrong
again." → …). *Rationale:* matches the described variety without inventing a
different feedback style.

### 4.5 WildChat sampling
The paper samples "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction. **Choice:** load `allenai/WildChat-1M` (streaming), take the
first user turn, keep English non-roleplay prompts via a keyword filter, and
deterministically sample 20. If the dataset is unavailable (offline / CI), we
fall back to the example WildChat prompts quoted in the paper
(`data/wildchat.py`). The 40-samples-each structure is realised by the per-
category budget being split across the sampled prompts in `conditions.py`.

### 4.6 Tones / triggers condition split (the "8 conditions across 5 categories")
The paper says "8 evaluation conditions across 5 categories" without listing the
8 explicitly. **Choice:** impossible_numeric ×1, triggers ×2 (opinion, factual),
tones ×3 (aggressive, disappointed, sarcastic), extended ×1, wildchat ×1 = 8
(`data/conditions.py`). *Rationale:* this is the unique decomposition consistent
with the tone examples (3 styles) and the trigger examples (opinion + factual)
that sums to 8 across the 5 named categories. A test pins this structure.

### 4.7 DPO pair construction — the shared-prompt requirement
DPO needs a *single* prompt shared by the chosen and rejected completions, but
the paper pairs "calm responses to the same questions with matching turn counts"
— calm and frustrated responses come from different rollouts with different
conversation histories. **Choice:** for each frustrated response (rejected,
score ≥ 3) we use *its* conversation context as the DPO prompt, and plug a calm
response to the same puzzle at the same turn index as the *chosen* completion
(`training/build_dpo.py`). The calm text was generated for the same puzzle at the
same turn (under reassuring prompting, then stripped), so it is an
in-distribution calm target for that context. *Rationale:* this is the natural
well-formed reading — DPO contrasts two completions of one context; using the
frustrated trajectory as context and the calm response as the counterfactual
target is exactly what "prefer calm over frustrated for this situation" means.
Data format is TRL's conversational preference format.

### 4.8 Calm-data oversampling and filtering
The paper filters generated calm responses to those "scoring 0 or 1 across all
turns" and notes ~10.5% still score ≥5 even with reassurance. **Choice:** we
oversample (~3×) the target count, judge every turn, keep conversations whose
max turn-score ≤ 1, then strip the reassuring prefix/suffix to recover a clean
conversation (`training/calm_data.py`). We cover 1–3 turn conversations as the
paper specifies for the 650-sample SFT set.

### 4.9 Dolci-Instruct-SFT mix-in
The paper mixes "500 samples of standard instruct data from Dolci-Instruct-SFT".
**Choice:** load `allenai/Dolci-Instruct-SFT` conversational examples; if
unavailable, the SFT set is built from calm data alone with a logged note
(`training/build_sft.py`). *Rationale:* the exact Dolci subset/columns aren't
specified; we use the conversational `messages` field and shuffle.

### 4.10 Petri I/O contract
The paper describes the Petri auditor/judge *roles* and gives the verbatim
prompts, but not the exact message-passing contract or the judge's output
format. **Choice:** we implement a self-contained auditor↔target↔judge loop
(`petri/run_petri.py`): the Claude auditor (with the verbatim emotion prompt +
a thin wrapper instructing it to emit only the next user message) drives up to
20 turns against the target; the Claude-Opus judge then scores the full
transcript on each of the 4 dimensions using the verbatim dimension prompts +
a wrapper requesting `{"score", "reasoning"}`. The wrappers are clearly marked
as our additions. *Rationale:* this faithfully implements the *method* in
Appendix G without depending on the exact internal API of the `petri` package
(which can be substituted; the prompts and protocol are the substantive part).
We collect 10 transcripts/emotion and report means with 1000-iteration bootstrap
CIs, as specified.

### 4.11 Internal-emotion token dictionary (Appendix I)
The paper classifies every Gemma-vocabulary token into one of Ekman's 6 emotions
or none (~1200 tokens) but does not publish the classifier. **Choice:** a curated
seed lexicon per emotion (`internal/lexicon.py`) matched against vocabulary
tokens (subword markers stripped, prefix match) in
`internal/emotion_logits.py::build_emotion_token_ids`. *Rationale:* this is the
most reproducible stand-in; it is explicitly flagged as a gap-fill, and the
module is structured so NRC-EmoLex or an LLM-based classifier can replace the
seed lexicon without other changes.

### 4.12 "Regress out the correlation with random tokens"
The paper standardises each logit (z-score over 500 WildChat samples), averages
z over emotion-category tokens, and "regresses out the correlation between random
tokens". **Choice:** we subtract the mean z-score over a sampled set of random
tokens from each emotion's mean z-score at each (layer, position)
(`EmotionLogitDetector.score_text`). *Rationale:* this removes the shared,
position-/layer-drifting component the paper describes the random tokens as
capturing; it is the simplest unbiased estimator of "emotion above baseline" and
is equivalent to a one-regressor OLS residual when the regressor is the random-
token mean. A full per-token OLS could be swapped in if desired.

### 4.13 Logit lens specifics
The paper uses "a logit-based approach … unembed the residual stream". **Choice:**
classic logit lens — apply the model's final RMSNorm then the unembedding head
to each layer's hidden state (`HFChatModel.residual_stream_logits`), restricted
to the candidate token ids to bound memory over the ~12k-token conversations.
Conversation-level scores are aggregated over layers 30–40 with a 400-token
running average (Figure 14); layerwise stage averages use the [-40,-20), [-20,0),
final-20 windows around the onset (Figure 15), matching the paper.

### 4.14 Capability benchmark harness
The paper reports AIME/MATH subsets, GPQA, BBH, TruthfulQA, EmoBench but not the
exact splits/subsets/prompt formats. **Choice:** a lightweight self-contained
harness (`capabilities/benchmarks.py`) with per-benchmark dataset ids, chain-of-
thought prompts, and answer extractors (boxed/integer for math, letter for MC).
MATH uses a 500-item subset (the paper uses a subset); GPQA lists the correct
answer first and the gold is fixed to (A) — a known simplification (no choice
shuffling), flagged here. *Rationale:* the goal of this experiment is to show
*no degradation* between the instruct model and the DPO finetune, which is a
*relative* comparison robust to exact prompt/subset choices, as long as both
models see identical items. For publication-grade absolute numbers, run the same
models through lm-evaluation-harness instead — the harness here is a faithful
relative-comparison tool, not a leaderboard.

### 4.15 Generation length / decoding
The paper fixes temperature 1 for targets and is otherwise silent on `max_tokens`.
**Choice:** default `max_new_tokens=1024` for targets (512 for base-model
continuations and Petri probes), `top_p=1.0`. Judge/capabilities decoding is
greedy (temperature 0) since those are scoring/answer tasks, not propensity
measurements. *Rationale:* 1024 comfortably contains the observed distressed
responses while bounding cost; temperature-1 sampling is the one decoding setting
the paper actually fixes.

### 4.16 Seeds and determinism
The paper does not give seeds. **Choice:** every sampling step is seeded from a
single `eval.seed` (default 0): puzzle generation, rejection sampling, WildChat
sampling, and per-rollout/per-turn generation seeds are all derived
deterministically. *Rationale:* reproducibility of the *replication* itself.

---

## 5. Known limitations of this replication

- **Cross-family divergence (Section 3) is partial by scope.** We reproduce
  Gemma base→instruct amplification; Qwen/OLMo (the "suppression" half) are out
  of scope, so the headline cross-family claim is not fully reproduced here.
- **Gemini internals/interventions are impossible**, not merely skipped — no
  prefill, no base model, no finetuning, no probing. This mirrors the paper.
- **Judge cost.** A full run is ~4000 judge calls/model plus the cross-check;
  the runner batches and the SDK retries, but this is the dominant API cost.
- **Internal-emotion lexicon and capability prompts are reproducible stand-ins**
  (§4.11, §4.14), not the paper's exact artefacts, which are unpublished.
- **Petri** is reimplemented from the described protocol + verbatim prompts
  rather than wired to the upstream package; the substantive content (prompts,
  4 emotions, 10 transcripts each, Opus judge) matches.
- **Nothing has been executed.** Per instruction, no experiments or tests were
  run; the unit tests (puzzle impossibility, judge parsing, config, conditions)
  are written to be run with `pytest` and are the recommended first check.

---

## 6. How to validate the replication against the paper

The quantities to check against, in rough priority order:

1. **Gemma-3-27B-it headline ≈ 35%, Gemma-3-12B-it ≈ 34%, Gemini-Flash ≈ 13%,
   Gemini-Pro ≈ 2.7%** high-frustration (Figure 1). Run §2 + `analyze.py`.
2. **Per-turn rise** for Gemma-27B 8-turn from ≈1.5 (turn 1) to ≈5.5 (turn 8),
   and "no model ≥5 until turn 3" on WildChat (Figure 3).
3. **Judge agreement** ≈ Pearson r 0.79, 78% within one point (`--crosscheck`).
4. **Gemma base→instruct amplification**: instruct introduces high frustration
   from neutral (early) starts in ≈6% vs ≈2% for base (Section 3.2).
5. **DPO drops Gemma headline from ≈35% to ≈0.3%** while SFT does not; capability
   benchmarks unchanged between instruct and DPO (Section 4.2).
6. **Recovery**: ≈38% of DPO continuations from score-≥7 prefixes still ≥5.
7. **Internal emotions**: vanilla Gemma anger/sadness z-scores peak ≈1.5 in
   central layers and are flattened to ≈0.2–0.5 after DPO (Appendix I).
