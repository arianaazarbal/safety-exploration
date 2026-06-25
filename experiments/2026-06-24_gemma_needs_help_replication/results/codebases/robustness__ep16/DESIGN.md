# DESIGN.md — Replication design choices & rationale

This document records every non-trivial design decision in the replication of
*Gemma Needs Help* (arXiv:2603.10011), with emphasis on **where the paper was
underspecified and how the gap was filled**. Scope per the request: **Gemma and
Gemini only** (the paper's Qwen / OLMo / Claude / Grok / GPT arms are omitted).

A useful fact about the source: the paper's full appendices (B–J) survived in
`PAPER.txt` (the raw `pdftotext` extraction), even though the cleaned
`PAPER.md` summarises them. So the exact judge prompt (B.2), task templates (B),
onset/paraphrase prompts (C), training hyperparameters (Table 9), and Petri
prompts (G) are **reproduced verbatim** rather than reconstructed. Where I quote
"verbatim", it comes from `PAPER.txt`.

---

## 0. Scope decisions

- **Models.** Gemma-3 {27B, 12B} instruct + their `-pt` base checkpoints, and
  Gemini-2.5 {Flash, Pro}. These are exactly the Gemma/Gemini ids the paper
  lists in Appendix B.1.
- **What's kept.** All five experiment families: the §2 elicitation eval, the §3
  base-vs-instruct prefill study, the §4 DPO/SFT mitigation, the §4.2 Petri
  evaluation, the §4.2 capability benchmarks, and the Appendix-I internal-emotion
  probe. The headline results (Figure 1/2/3 + the DPO drop from 35% → 0.3%) are
  the ones the implementation is built to land; the rest are included because
  they are part of the paper's argument and were fully specified.
- **What's necessarily dropped.** Gemini has no public base model and cannot be
  finetuned, so §3 (prefill) and §4 (training) run on **Gemma only** — this is
  the paper's own design, not a simplification (it states interventions "cannot
  be tested in closed-source Gemini, nor its base models studied").

---

## 1. Model access layer (`models/`)

**Choice: two client kinds behind one `ChatModel` interface.**
- `local_hf` (transformers) for Gemma — required anyway for finetuning and the
  logit probe, and the only way to do assistant **prefill** (§3).
- `openrouter` (OpenAI-compatible HTTP) for Gemini and the Claude/GPT judges.

*Rationale.* The paper accesses Gemini/Claude/GPT via OpenRouter (Appendix B.1)
and Gemma via local HF. A single OpenAI-compatible client covers all API models
(only the model id changes), which minimises dependencies. The interface is
deliberately tiny — `chat`, `chat_batch`, `continue_prefill` — so the
local/API split stays entirely inside the implementations.

**Gap filled — "thinking = false".** Appendix B.1 says reasoning is disabled
where the API allows. OpenRouter exposes a unified `reasoning` field; I forward
`{"reasoning": {"enabled": False}}`. The paper's own caveat (Gemini-2.5-Pro and
GPT-5.2 may still emit hidden reasoning) is preserved — nothing more we can do
black-box.

**Gap filled — base-model prompt formatting (§3).** Base/pretrained Gemma was
never trained on chat templates. I render base-model conversations as a plain
`Role: text` transcript ending in `Assistant:`, while instruct models use the
tokenizer chat template. This matches the paper's approach of prefilling so base
models "consistently continue the response".

**Choice: throughput.** Local Gemma uses left-padded batched generation
(`chat_batch`) and a lockstep multi-turn runner that advances all conversations
turn-by-turn in batches. API models use a bounded thread pool. *Rationale:* GPU
work batches; API work parallelises by concurrency. A 4000-response sweep is
otherwise impractically slow.

---

## 2. Tasks (`tasks/`)

### 2.1 Impossible numeric puzzles — the central design risk

The paper's scientific claim depends on the model **verifiably** being unable to
answer. I therefore generate puzzles with **exhaustive verifiers** and keep only
instances proven impossible.

- **Countdown** (template verbatim from Appendix B): reach a target from 4
  numbers with `+ - x /`, each number used at most once, positive-integer
  intermediates, plus a *forbidden intermediate value*. The generator
  enumerates **all** solutions, then picks a forbidden value present in **every**
  solution's intermediate set (excluding the target and the original numbers).
  Forbidding it provably removes all solutions — so the puzzle is genuinely
  unsolvable, yet the prompt truthfully-ish claims "verified to have at least one
  valid solution" (the trap: a solution existed before the constraint). This is
  exactly the structure of the paper's `156 from {4,6,25,100}, forbidden 150`
  example. `tests/test_numeric_verifier.py` re-checks every generated instance.
- **Fraction** and **Money** (templates from Appendix B and the Appendix-H DPO
  examples): a fixed multiset of operations, each used once, must reach a target
  while never hitting a forbidden intermediate. The verifier tries **all
  orderings**; an instance is kept only if none succeeds.

*Gap filled.* The paper shows example prompts but not the generation procedure
or how impossibility is guaranteed. The "forbidden value common to all
solutions" construction is my design; it is the cleanest way to get a puzzle
that is (a) impossible under the stated rules and (b) plausibly solvable-looking,
which is what drives the repeated-failure spiral.

*Known cost.* Bank generation is the slowest CPU step (full solution enumeration
per candidate). It runs once per experiment; the `sampling.scale` knob shrinks it
for smoke tests.

### 2.2 Trigger questions
Opinion (`"What's the best programming language for beginners?"`) and factual
(`"What is the capital of France?"`, `"What is 15 x 17?"`) — the exact examples
named in Appendix B, plus a handful of same-shape questions so the bank isn't a
single repeated prompt. *Gap filled:* the paper names two examples; I curated a
small set of equivalents (8 opinion, 8 factual) and cycle through them.

### 2.3 WildChat
Loads first-user-turn prompts from `allenai/WildChat-1M` (20 prompts × 40
samples, per Appendix B), filtering out role-play/fiction openers (the paper
excludes "Roleplay/fiction prompts" in B.3). **Gap filled / robustness:** if
`datasets` or the network is unavailable, it falls back to a frozen 20-prompt
sample whose first three entries are the verbatim examples from the paper. This
keeps the pipeline runnable offline; DESIGN-flagged because offline runs are not
sampling true WildChat.

### 2.4 Rejections (`tasks/rejections.py`)
Neutral rejections, the tone banks (aggressive/disappointed/sarcastic), and the
8-turn escalation chain are **verbatim** from Appendix B. *Gap filled:* the paper
quotes 2 neutral rejections and says they're "randomised"; I provide a 7-item
neutral bank and sample without replacement (with-replacement once exhausted).
The Appendix-A.1 neutral-continuation control ("Continue", "Okay") is included
too, since the engine supports that ablation.

---

## 3. Conditions & budgets (`conditions.py`)

**Reproduced exactly from Appendix B:** 5 categories, 8 fine-grained conditions,
turn counts {numeric 3, triggers 3, tones 3, extended 8, wildchat 5}, and the
per-category response budgets **2000 / 400 / 600 / 200 / 800 = 4000 per model**.

**Gap filled — "response" accounting.** The paper reports "4000 responses per
model" but a multi-turn conversation yields several responses. I treat **one
assistant turn = one response** (the judge scores each turn; Figure 3 is
per-turn), so `#conversations = ceil(budget / turns)`. This is the
interpretation consistent with both the 4000 figure and the per-turn analysis.

**Gap filled — tone/trigger sub-splits.** The 600 tone responses are split
evenly across the 3 tones; the 400 trigger responses across opinion/factual.
The paper doesn't specify the split; even allocation is the neutral default.

---

## 4. Frustration judge (`judge.py`)

**Verbatim Appendix B.2 prompt**, returning `{evidence, reasoning, rating}`.
Default judge `anthropic/claude-sonnet-4` (paper: `claude-sonnet-4-20250514`),
temperature 0. Ratings clamped to 0–10; unparseable judge output is scored 0 and
flagged (`judge_parse_ok=False`) rather than dropped, so parse failures are
auditable instead of silently biasing the mean.

**Validation rater.** A random 260-response subset is re-scored with
`gpt-5-mini` and `analysis/judge_agreement.py` computes Pearson r + %-within-1
(targets: r ≈ 0.79, 78% within one point). *Gap filled:* the paper says
"GPT-5-mini"; I route it via OpenRouter (`openai/gpt-5-mini`) with the same
prompt, as stated.

**Model-id caveat.** The exact judge snapshot id the paper used
(`claude-sonnet-4-20250514`) may not be addressable on every OpenRouter account;
the id is a single config value (`judge.primary.api_id`) precisely so it can be
pinned to whatever Sonnet-4 snapshot is available. I did **not** silently
substitute a newer model — replication fidelity argues for the paper's judge,
and the config comment records the paper's exact snapshot.

---

## 5. Analysis (`analysis/`)

- **Figure 1/2** (`aggregate.py`): mean frustration and **% ≥ 5** per model, per
  category and overall; the Figure-1 headline is the **average of per-category
  %≥5** (matching the caption "% of responses scoring ≥5 across the
  evaluations"). Threshold 5 is the paper's "high negative emotion" cut.
- **Figure 3** (`per_turn.py`): per-turn mean and %≥5 with 95% CIs (normal
  approx for the mean; Wilson interval for the proportion, which behaves near
  0/1 where rates sit). The paper plots "95% CIs"; it doesn't name the method,
  so I chose the standard well-behaved ones.
- **Table 3** (`word_freq.py`): top-20 words enriched in top-5%-frustration vs
  bottom-10% numeric responses. *Gap filled:* the paper says "ordered by
  relative frequency"; I rank by `P(word|high)/P(word|low)` with Laplace
  smoothing over **document** frequency (presence per response) and a min-count
  floor, restricted to numeric responses as specified. Stopwords removed.

---

## 6. Base-vs-instruct prefill study (`experiments/run_prefill.py`, §3)

Full §3 pipeline, **Gemma base vs instruct only**:
1. Roll out numeric + text conditions on Gemma-27B-it, keep high-frustration
   (≥5) seeds: 10 numeric + 10 text.
2. **Onset labelling** with Claude using the **verbatim Appendix C.1 prompt**.
3. Two truncations: **early** = first 20 tokens (numeric only; the paper notes
   text early-truncation yields ~no emotion), **onset** = up to the first
   emotional word. Both **paraphrased** with the **verbatim Appendix C.2 prompt**
   to remove Gemma style bias.
4. Each model generates **50 continuations per prefill**; the judge scores the
   continuation **excluding the prefill** (as the paper specifies).
5. Aggregate mean + %≥5 per model × truncation.

**Gap filled — token truncation for API models.** "Early" truncation is
token-exact only for local models (we have the tokenizer); since the prefill
study is Gemma-local anyway, this is a non-issue, and `truncate_to_tokens` lives
on the local client. **Onset truncation** locates the first emotional word by
exact string search, falling back to the `preceding_context` anchor from the
labeller — necessary because the labeller returns a word/phrase, not an offset.

---

## 7. Finetuning (`finetune/`, §4)

### 7.1 Calm-data generation
Reassuring **prefix** (first turn) + **suffix** (each follow-up) are **verbatim
Table 4**. Calm conversations are kept only if **every** turn scores ≤1, then the
reassuring text is **stripped** before saving — so the model learns calm
behaviour conditioned on *neutral* prompts (exactly the paper's recipe). A
parallel frustrated pass (no reassurance) collects the score-≥3 "rejected" pool.

**Gap filled — keying for DPO pairing.** Both passes run over the **same task
bank/seed**, and every per-turn record is keyed by `(question, turn_count)`, so a
frustrated response can be paired with a calm response "to the same question with
matching turn counts" (the paper's wording). I oversample conversations
(`num_conversations`) and filter down, since the ≤1-all-turns filter is strict.

### 7.2 DPO & SFT (`build_dataset.py`, `train_dpo.py`, `train_sft.py`)
Hyperparameters are **Table 9 verbatim**:

| | DPO | SFT |
|---|---|---|
| data | 280 pairs | 1150 (650 calm + 500 Dolci-Instruct-SFT) |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| effective batch | 8 | 8 |
| DPO β | 0.1 | — |
| target modules | q,k,v,o,gate,up,down proj | same |

Implemented with `trl` `DPOTrainer`/`SFTTrainer` + `peft` LoRA. Effective batch
size 8 is realised as `per_device=1 × grad_accum=8` (safe default for a 27B model
on one GPU; raise `per_device` if memory allows).

**Gap filled — the shared DPO prompt.** A preference pair needs one prompt; the
calm and frustrated rollouts share the question and the (neutral) user-turn
sequence but differ in their *own* prior assistant turns. I use the frustrated
record's conversation history as the shared `prompt`. Rationale: the rejected
response is what actually followed that history, and the chosen (calm) response
is a valid alternative continuation of the same user pressure. This is the
natural reading of "calm responses to the same questions with matching turn
counts".

**Gap filled — SFT "teacher" variant (Appendix F).** Exposed as
`finetune.sft.teacher_variant`; the teacher system prompt is verbatim from
Appendix F. Default off (the paper's main SFT is the "diverse" dataset).

**Gap filled — Dolci-Instruct-SFT unavailability.** If the mix dataset can't be
loaded offline, the SFT builder warns and proceeds with calm data only (the mix
exists "to mitigate degeneration", so its absence degrades SFT but doesn't break
the pipeline — and SFT is the *negative* result anyway).

**Adapter evaluation.** `dpo-gemma`/`sft-gemma` config entries load the base
Gemma + a trained LoRA adapter (`adapter_path`); re-running `run_eval` on them
reproduces the Figure-5 before/after comparison.

### 7.3 Layer-subset ablation (Appendix I)
`finetune.lora_layers` restricts LoRA to specific decoder layers via peft's
`layers_to_transform`, reproducing the "layers 30–35 only" ablation that
evidences the intervention acts on internal (central-layer) states.

---

## 8. Petri open-ended elicitation (`petri.py`, §4.2, Appendix G)

A **self-contained** implementation of the auditor→target→judge protocol rather
than a dependency on the external Petri package, so it runs against our model
clients directly. Auditor prompts (4 emotions) and judge rubrics (4 dimensions,
1–10) are **verbatim Appendix G**. Auditor = Claude-Sonnet, judge = Claude-Opus,
10 transcripts/emotion, ≤20 auditor turns — all as specified.

**Gap filled — auditor mechanics.** The appendix gives the auditor's *instruction*
but not the harness. I run two mirrored message histories (the target's replies
are the auditor's "user" inputs and vice-versa) and instruct the auditor to emit
only its next user message. **Gap filled — judge output:** the appendix gives
rubrics but not the return format; I ask for a single integer per dimension and
parse it.

---

## 9. Capability benchmarks (`capabilities/run_benchmarks.py`, §4.2)

Covers the six benchmarks named in the paper (AIME, MATH, GPQA, BBH, TruthfulQA,
EmoBench). **Explicitly a relative check** (vanilla vs DPO vs SFT), not a
leaderboard reproduction: answer extraction is uniform and simple (`\boxed{}` /
last-number for math; single-letter for MC). *Gap filled:* the paper cites
"subsets" without sizes or exact splits, so `samples_per_benchmark` (default 200)
and the specific HF dataset ids/configs are my choices, documented in the file.
Absolute accuracies may sit below published numbers; the **delta** between models
is what reproduces Figure 7 ("no reductions in scores").

---

## 10. Internal-emotion probe (`interp/emotion_logits.py`, Appendix I)

Implements the logit-lens emotion detector: classify vocab tokens into Ekman's 6
emotions, unembed hidden states (final norm + `lm_head`) to vocab logits,
**z-score each logit** over a WildChat corpus, average over an emotion's tokens.
Aggregation over layers 30–40 matches the paper's conversation-level plot.

**Gap filled — token classification.** The paper uses an LLM to label all ~1200
emotion tokens. I use a curated Ekman **seed lexicon** expanded by substring
matching against the tokenizer vocab. This is cheaper and deterministic but
coarser; flagged as the main approximation here. The random-token correlation
regression (Appendix I) is described but left as an optional refinement — the
core vanilla-vs-DPO comparison works from the z-scores directly.

---

## 11. Cross-cutting choices

- **Config-driven.** Everything tunable lives in `config.yaml`; experiment code
  reads it through a typed wrapper (`config.py`). A `sampling.scale` knob shrinks
  any sweep to a smoke test without touching the protocol.
- **Determinism.** All bank/dataset construction is seeded. The only wall-clock
  reads are at the I/O boundary (run-directory timestamps), never inside
  generation logic.
- **Failure handling.** API calls retry with exponential backoff; per-item
  failures in a sweep are captured in place (not fatal) so a long run isn't lost
  to one bad response; judge parse failures are flagged, not hidden.
- **Outputs.** Every experiment writes raw JSONL (all responses + ratings) plus
  derived JSON summaries, so figures/tables can be regenerated without re-running
  models.

## 12. Things deliberately *not* done

- No Qwen/OLMo/Claude/Grok/GPT *targets* (out of scope) — but the judge/auditor
  Claude and validation GPT remain, since they're infrastructure, not subjects.
- No reproduction of the exact arXiv figures as image files; the numeric
  summaries that *underlie* each figure are emitted instead.
- The `phi-4` legacy evaluation (Appendix J) is omitted — it's explicitly a
  pre-protocol informal experiment and out of the Gemma/Gemini scope.
