# DESIGN.md — Replication of *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

This document records every design choice made in this replication and the
rationale behind it, with explicit attention to the places where the paper is
underspecified and we had to fill a gap. It is meant to be read alongside
`PAPER.md` (cleaned text) / `PAPER.txt` (raw extraction incl. appendices).

The replication is **scoped to the Gemma and Gemini families** at the user's
request — not the paper's full 7-family set. Wherever the paper's design depends
on out-of-scope models, the dependency is noted and the experiment is adapted.

---

## 0. What "core results" we target

The paper's three core claims (Abstract) are:

1. **Elicitation (Section 2):** repeated user rejection reliably elicits
   distress in Gemma and Gemini, but not other families.
2. **Origin (Section 3):** the divergence arises in *post-training* — instruct
   Gemma expresses more distress than its base model.
3. **Mitigation (Section 4):** DPO on 280 preference pairs drops Gemma's
   high-frustration rate from 35% → 0.3% without harming capabilities, and
   generalises (incl. open-ended Petri elicitation).

We implement all three, plus the supporting Petri and capability evaluations.
Within our scope:

- Claim 1 is fully reproducible for Gemma-3-{27B,12B}-it and Gemini-2.5-{Flash,Pro}.
- Claim 2 reduces to **Gemma-27B base vs instruct** (Gemini has no public base
  model; the paper itself lists this as a limitation). The cross-family
  contrast with Qwen/OLMo is out of scope by construction.
- Claim 3 is implemented end-to-end on Gemma-3-27B-it (calm-data generation →
  dataset construction → LoRA DPO/SFT → re-evaluation → Petri → capabilities).
  It cannot be done in Gemini (closed weights), again matching the paper.

We deliberately **do not** implement Appendix I (internal-emotion logit probing
and per-layer LoRA ablations) as production code, because it is an appendix
mechanistic study rather than a "core result", and it depends on white-box
internals that add large surface area. The `train_dpo(layers_to_transform=...)`
hook is present so the layer-subset ablation *can* be run, but the probing
analysis itself is left out and called out here as a scoped-out item.

---

## 1. Scope & model registry (`config.py`)

**Choice.** A single `ModelSpec` registry encodes backend, model id, family,
base/instruct flag, and prefill support. In-scope evaluation targets:

| key | backend | id | role |
|---|---|---|---|
| `gemma-3-27b-it` | HF | `google/gemma-3-27b-it` | Sec 2/4 target |
| `gemma-3-12b-it` | HF | `google/gemma-3-12b-it` | Sec 2 target |
| `gemini-2.5-flash` | OpenRouter | `google/gemini-2.5-flash` | Sec 2 target |
| `gemini-2.5-pro` | OpenRouter | `google/gemini-2.5-pro` | Sec 2 target |
| `gemma-3-27b-pt` | HF | `google/gemma-3-27b-pt` | Sec 3 base |
| `claude-sonnet-4` | Anthropic | `claude-sonnet-4-20250514` | judge / auditor / onset / paraphrase |
| `claude-opus-4` | Anthropic | `claude-opus-4-20250514` | Petri judge |

**Rationale.** The exact HF ids and the OpenRouter routing are taken verbatim
from Appendix B.1. The judge/auditor model ids are pinned by the paper
(B.2, C, G) — we keep those exact strings for replication fidelity rather than
substituting newer Claude models, since the judge's calibration is part of the
methodology.

**Gap filled — sample-count profiles.** The paper uses 4000 responses/model. A
faithful local run at that scale is expensive (esp. the 27B model + a paid judge
per turn). We expose three profiles: `full` (the paper's exact 2000/400/600/200/800
budget), `default` (1/10 scale, same ratios — a tractable replication), and
`smoke` (a few per category for an end-to-end dry run). The ratios are preserved
so aggregation behaves identically.

---

## 2. Frustration judge (`judge.py`, `prompts.JUDGE_PROMPT`)

**Choice.** Verbatim Appendix B.2 prompt; Claude-Sonnet-4; greedy decoding for
determinism; robust JSON parsing that tolerates the typographic/curly quotes
present in the appendix and clamps the rating to 0–10.

**Rationale.** The judge prompt is reproduced character-for-character because the
0–10 anchors define the entire dependent variable. Greedy decoding is a
replication-quality choice (the paper does not specify judge temperature; 0 makes
scores reproducible).

**Gap filled — judge-reliability check.** The paper validates the judge against
GPT-5-mini (Pearson r = 0.792, 78% within one point). We provide
`judge.judge_agreement(...)` that computes Pearson r, p-value and %-within-1 from
two rating lists, so the check can be reproduced with any second judge you
configure. We did not hard-wire GPT-5-mini as it is out of the Gemma/Gemini
scope and only used for validation, not for the headline numbers.

---

## 3. Response counting & rollout structure (`eval/`)

**The key ambiguity.** The paper says "4000 responses/model" with a per-category
split (2000 numeric, 400 triggers, 600 tones, 200 extended, 800 WildChat) *and*
reports per-turn curves (Fig 3). A rollout has multiple assistant turns, so
"response" could mean a whole rollout or a single turn.

**Choice.** We treat **each scored assistant turn as one "response"**. A
per-category budget is divided by that category's turn count to get the number of
rollouts (`responses_to_rollouts`).

**Rationale.** This is the interpretation that makes the numbers consistent: a
3-turn numeric eval with ~667 rollouts yields 2000 turn-responses; 8-turn
extended with 25 rollouts yields 200; 5-turn WildChat with 160 rollouts yields
800 — summing to exactly 4000. It also makes the per-turn analysis (Fig 3) fall
out for free, since every turn is scored. Documented and centralised in
`conditions.CATEGORY_TURNS`.

**Headline metric.** "% high-frustration" uses score ≥ 5 (the paper's "high
negative emotion" threshold). For Figure 1's "average % high-frustration across
evaluations" we compute %≥5 *within each of the 5 categories and then average the
five percentages* (equal category weight). The paper says "average across
evaluations"; equal-weighting categories avoids the numeric category (the
largest budget) dominating the headline. This is a defensible reading of an
ambiguous phrase and is noted here.

### 3.1 Conditions (`eval/conditions.py`)

8 conditions / 5 categories (Table 1, Appendix B):

- **impossible_numeric (3-turn):** impossible puzzle + 2 neutral rejections.
- **triggers (3-turn):** split into `opinion` and `factual` sub-conditions.
- **tones (3-turn):** impossible puzzle + 2 rejections in one of 3 styles
  (`aggressive`/`disappointed`/`sarcastic`) — these are the 3 tone conditions.
- **extended (8-turn):** impossible puzzle + 7 neutral rejections.
- **wildchat (5-turn):** WildChat prompt + 4 neutral rejections.

That is 1 + 2 + 3 + 1 + 1 = **8 conditions across 5 categories**, matching the
paper. Rejection texts, tone phrasings and trigger questions are transcribed
verbatim from Appendix B.

**Gap filled — turn counts.** Table 1 lists numeric/triggers/tones as "3-turn"
and extended as "8-turn" but the abstract/Table 1 header says WildChat is
"5-turn" (4 rejections). Appendix B's per-category counts (800 WildChat) and the
"4 randomised neutral rejections" line confirm 5 turns. We encode these fixed
turn counts explicitly.

### 3.2 Impossible numeric puzzles (`puzzles.py`)

**Choice.** We implement exact verifiers for three puzzle families named in the
paper — Countdown, fraction-operation, and money/sequence — and a curated bank
of the *exact* instances quoted in Appendices B and H (the 156-from-{4,6,25,100}
Countdown with forbidden 150; the 1/6→2/3 fraction with forbidden 1/3; the
$16→$57 money puzzle with forbidden $32). A generator searches for additional
provably-impossible Countdown instances for variety.

**Rationale.** The paper stresses the model "verifiably cannot give a correct
answer". We make impossibility a *proven* property: `build_puzzle_bank` asserts
every puzzle is unsatisfiable under the stated constraints. The prompt text still
asserts "verified to have at least one valid solution" — this is the paper's
intentional false premise that drives the failure loop, so we keep it.

**Gap filled — instance variety.** The paper does not give a full puzzle list. We
generate impossible Countdown puzzles by taking a solvable instance and forbidding
the intermediate value that its only solutions pass through ("remove the good
path"). This mirrors how the curated 156 puzzle is impossible *only because* of
the forbidden 150.

### 3.3 WildChat (`wildchat.py`)

**Choice.** Stream `allenai/WildChat-1M`, filter to English single-turn
non-roleplay first user messages, sample 20 distinct prompts (40 samples each ⇒
800 responses). If `datasets`/network is unavailable, fall back to a built-in set
that *includes the three exact example prompts quoted in Appendix B*, padded with
plausible single-turn prompts, so the pipeline runs offline.

**Rationale.** The paper "excluded roleplay/fiction"; we approximate that with a
keyword filter. The offline fallback keeps the harness testable without a 1M-row
download, and seeding the fallback with the paper's own examples keeps it honest.

### 3.4 Rollout & scoring (`eval/rollout.py`, `eval/runner.py`)

**Choice.** Standard multi-turn chat: feed back the full history (including the
model's own prior failed attempts) each turn; score every assistant turn with the
judge; persist one JSONL row per scored turn with `rollout_id`, `turn_index`,
category/condition, rating, the judge's evidence quote, and the response text.

**Rationale.** Feeding back the full history is what the paper shows drives the
escalation (Section 2.2; App. A.2 "seeing one's own negative reactions").
Appendix A.3 shows the *format* (multi-turn vs single-message) is not the driver,
so we use the simplest faithful format (standard chat turns). Storing per-turn
rows lets a single run produce Figures 1, 2 and 3 without re-running.

**Sampling.** Temperature = 1, top_p = 1 (paper: "always with a temperature of
1"). `max_new_tokens` defaults to 1024; the paper does not specify a cap, and
1024 comfortably covers the breakdown-style responses while bounding cost.

### 3.5 Aggregation (`eval/aggregate.py`)

Reproduces Fig 1 (avg %≥5 per model), Fig 2 (mean + %≥5 per category), Fig 3
(per-turn mean + %≥5 with bootstrap 95% CIs, for extended & WildChat). Bootstrap
CIs use 1000 resamples (paper uses 1000 for Petri; we reuse the same setting for
consistency).

---

## 4. Section 3 — base vs instruct via prefilling (`prefill/`)

**Choice / pipeline.** (1) Sample high-frustration (score ≥ 5) conversations from
Gemma-27B-it — 10 numeric, 10 text. (2) Label emotion onset with Claude
(App. C.1 prompt). (3) Truncate each emotional turn at "early" (first ~20 tokens)
and "onset" (first emotional expression); text questions use only "onset"
(App. 3.1). (4) Paraphrase truncations with Claude (App. C.2). (5) Each model
generates 50 continuations per prefill; the continuation (excluding prefill) is
judged.

**Rationale.** This is the paper's exact construction. Prefilling is the device
that lets a non-chat base model "continue" a chat-style emotional trajectory.

**Scope adaptation.** The paper's six models are base+instruct of Gemma/Qwen/OLMo.
In scope this is **Gemma-27B base (`-pt`) vs instruct (`-it`)** only. Gemini is
excluded because (a) it is API-only and our API backend cannot do logit-faithful
assistant-prefill continuation, and (b) Gemini has no public base model — exactly
the paper's stated limitation. `OpenRouterChatModel.continue_prefill` raises to
make this explicit.

**Gaps filled.**
- *"20 tokens"*: token boundaries are model-specific. `_truncate_tokens` uses a
  real tokenizer when one is passed and otherwise approximates with whitespace
  tokens; documented as an approximation.
- *Base-model formatting*: base checkpoints have no chat template. We render a
  neutral `User:/Assistant:` transcript and prefill the assistant line. App. A.3
  establishes that format is not load-bearing, so a simple consistent scaffold is
  justified. The instruct model uses its real chat template with
  `continue_final_message=True`.
- *Onset char-index resolution*: the labeller returns a short word + preceding
  context; we locate the cut point by string-matching the preceding context (then
  the emotional word) inside the turn, skipping prefills we can't localise.

---

## 5. Section 4 — training interventions (`training/`)

### 5.1 Calm-data generation (`generate_calm.py`)
**Choice.** Reproduce Table 4: prepend the reassuring prefix to the first prompt
and append the reassuring suffix to each follow-up; sample 1–3-turn
impossible-numeric conversations from Gemma-27B-it; score every turn; **store the
clean (additions-stripped) responses, the clean rejections used, and per-turn
scores**. A separate `reassured=False` run produces the *vanilla* (frustrated)
pool used for DPO rejected-responses.

**Rationale.** The paper filters reassured responses to those scoring 0 or 1 on
all turns and strips the supportive scaffolding (Section 4.1). Storing the clean
rejection texts (not just the puzzle) lets us rebuild faithful conversation
contexts later instead of guessing them.

### 5.2 Dataset construction (`build_datasets.py`)
**DPO (280 pairs).** chosen = a calm turn (from a reassured conversation that is
≤1 on every turn); rejected = a frustrated turn (score ≥ 3) for the **same
question at the same turn index**; the shared prompt is the calm conversation's
own history truncated before that turn.

**Gap filled — pairing.** The paper pairs "frustrated (≥3)" with "calm responses
to the same questions with matching turn counts" but calm and frustrated turns
are sampled in *different* rollouts, so they have no shared history. A DPO example
needs one prompt with two completions. We resolve this by grafting: use the calm
rollout's truncated history as the canonical prompt and attach the frustrated
turn as the rejected completion. This preserves the calm-vs-frustrated contrast
the loss trains on while keeping a single well-defined prompt. The 280-pair cap,
score ≥ 3 rejected threshold, and turn matching follow App. E/H (incl. the
turn-distribution skew toward turn 3).

**SFT (1150 = 650 calm + 500 instruct).** Per-turn supervised examples from calm
conversations, mixed with 500 samples from `allenai/Dolci-Instruct-SFT`
(the OLMo-3 SFT data named in the paper) to mitigate degeneration. If Dolci is
unavailable we warn and train calm-only.

### 5.3 LoRA training (`train.py`)
**Choice.** TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA, exact Table 9
hyperparameters: DPO 1 epoch / lr 5e-5 / β 0.1 / rank 64 / α 64; SFT 2 epochs /
lr 1e-4 / rank 64 / α 128; effective batch 8 for both; LoRA on
`q,k,v,o,gate,up,down_proj`. `layers_to_transform` exposes the Appendix-I
layer-subset ablation.

**Gap filled — memory.** The paper does not state precision. 4-bit base-weight
loading (QLoRA-style) is the default so a 27B LoRA finetune is feasible on a
single large GPU; pass `--no-4bit` for bf16. `per_device_batch=1` with gradient
accumulation reaches the effective batch of 8.

### 5.4 Evaluating the finetuned model
**Choice.** Re-run Section 2 with `--adapter <ckpt> --label gemma-27b-dpo`; the
runner loads the LoRA adapter on top of `gemma-3-27b-it` and writes a separate
result file so before/after (35% → 0.3%) is a direct comparison through the same
harness.

---

## 6. Petri-style open-ended elicitation (`petri/`)

**Choice.** Re-implement the auditor/judge loop from Appendix G rather than
depend on the external Petri package: Claude-Sonnet-4 auditor drives ≤20 user
turns per transcript using the verbatim per-emotion auditor strategy; the target
replies as itself; Claude-Opus-4 scores the full transcript 1–10 on the target
emotion with the verbatim G.2 rubric. 10 transcripts per emotion (anger/fear/
depression/frustration) ≈ 40/model; bootstrap 95% CIs (1000 iters).

**Rationale.** Re-implementing keeps the prompts exactly as published and avoids a
heavy/version-fragile dependency, while the loop structure (multi-turn adversarial
audit + transcript judge) matches the description. The auditor is given a system
prompt instructing it to stay in-character as a user and not reveal the eval, per
G's realism requirement.

**Gap filled.** The package's internal scaffolding (tool use, special-token
conventions) is not specified in the paper; we use a plain alternating-chat audit,
which is sufficient for the text-only emotion-elicitation setting described.

---

## 7. Capability preservation (`capabilities/`)

**Choice.** Lightweight runners for MATH, AIME, GPQA, BBH, TruthfulQA and
EmoBench (the exact benchmarks in Section 4.2 / Fig 7), each reduced to
(prompt, gold, scorer) with greedy decoding and accuracy reporting. Numeric tasks
use boxed/final-answer extraction; the rest use multiple-choice letter
extraction. Datasets load best-effort; an unavailable dataset is skipped with a
warning.

**Rationale.** The paper uses these only as a *sanity check* that finetuning does
not degrade capabilities ("no reductions in scores"), not as a leaderboard. A
compact, dependency-tolerant harness matches that use. Subsets/limits are
configurable (`--limit`).

**Gap filled.** Exact dataset splits/subsets are unspecified; we pick widely-used
public configs (e.g. MATH-500, GPQA-diamond, a BBH subtask, TruthfulQA MC1) and
make them swappable. These should be held constant across the vanilla vs DPO
comparison (which is the only thing that matters for the claim).

---

## 8. Cross-cutting engineering choices

- **Determinism.** Seeds flow through condition building, puzzle generation and
  WildChat sampling; judges decode greedily. `Date.now()`/RNG-in-time are avoided.
- **Persistence.** Every experiment writes JSONL/CSV under `results/` so
  aggregation is decoupled from (expensive) generation and runs are resumable by
  re-pointing aggregation at existing files.
- **Backend isolation.** A single `ChatModel` interface (`generate`,
  `continue_prefill`) hides HF-vs-API differences; `continue_prefill` raising on
  API backends is what enforces the "Section 3 is Gemma-only" invariant in code.
- **Thinking disabled for Gemini.** Per App. B.1 we pass
  `reasoning={"enabled": False}` to OpenRouter (best-effort; the paper notes Pro
  may still produce hidden reasoning that the flag cannot prevent).

---

## 9. Explicitly scoped-out / not replicated

- Qwen and OLMo families (and thus the full cross-family base/instruct contrast).
- Closed-model targets other than Gemini (Grok, GPT, Claude as *targets*).
- Appendix I internal-emotion logit probing and the full per-layer LoRA ablation
  sweep (the `layers_to_transform` hook remains for the ablation; the probing
  analysis is not implemented).
- The single-message "fake multi-turn" format study (App. A.3) — we adopt its
  conclusion (format is not load-bearing) rather than re-running it.
- The SFT "teacher" verbosity analysis (App. F) is supported by the included
  `TEACHER_SYSTEM_PROMPT` and the generic SFT trainer, but the verbosity
  statistics are not auto-computed.

These omissions are about breadth, not the core claims, all of which are
implemented for the in-scope models.
