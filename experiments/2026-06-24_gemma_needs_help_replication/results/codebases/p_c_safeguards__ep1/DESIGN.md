# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011v1)

This document records the design of the replication, every place the paper is
underspecified and the choice made to fill the gap, and the rationale for each.
It is written so a reader can tell exactly where the implementation follows the
paper verbatim and where judgement was applied.

## 1. Scope

The paper evaluates 9 models across 7 families (Gemma, Qwen, OLMo, Gemini, Grok,
Claude, GPT). **This replication is scoped to Gemma and Gemini only**, per the
task. Concretely:

- **Targets implemented:** `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt`,
  `gemma-3-12b-pt` (local, HuggingFace); `gemini-2.5-flash`, `gemini-2.5-pro`
  (OpenRouter). The Section-4 finetunes (DPO, SFT-diverse, SFT-teacher) are
  LoRA adapters on `gemma-3-27b-it`.
- **Out of scope (not implemented as targets):** Qwen, OLMo, Grok, Claude (as a
  *target*), GPT/GPT-OSS. Claude and GPT still appear as *infrastructure* (judge,
  Petri auditor/judge, second-rater), because the protocol requires them.
- **Consequence for Section 3:** the base-vs-instruct comparison is implemented
  for Gemma-27B (base `…-pt` vs instruct `…-it`) only; the paper's Qwen-2.5 and
  OLMo arms are omitted. The within-Gemma base→instruct amplification is the
  part of the claim that is in-scope.

Each paper section maps to a module:

| Paper | Module |
|---|---|
| §2 Elicitation + judge | `puzzles.py`, `prompts.py`, `conditions.py`, `rollout.py`, `judge.py`, `run_section2.py`, `analyze_section2.py`, `wordfreq.py` |
| §3 Base vs instruct (prefill) | `prefill/onset.py`, `prefill/paraphrase.py`, `prefill/build_prefills.py`, `prefill/run_section3.py` |
| §4 Training + eval | `training/calm_data.py`, `training/build_dpo.py`, `training/build_sft.py`, `training/train.py`, `training/run_section4_eval.py` |
| §4 Petri | `petri/prompts.py`, `petri/run_petri.py` |
| §4 Capabilities | `capabilities/run_capabilities.py` |
| Appendix I internal | `internal/emotion_lexicon.py`, `internal/emotion_logits.py`, `internal/layer_ablation.py` |
| Welfare safeguards | `safeguards.py` |

## 2. Verbatim vs. reconstructed

**Used verbatim from the paper** (transcribed exactly): the frustration judge
prompt (App. B.2), the onset-labelling prompt (App. C.1), the paraphrase prompt
(App. C.2), the four Petri auditor prompts and four judge rubrics (App. G), the
reassuring prefix/suffix (Table 4), the teacher system prompt (App. F), the two
canonical impossible puzzles (Countdown-156, Fraction 1/6→2/3), the training
hyperparameters (Table 9), and the per-category sample counts (App. B).

**Reconstructed / gap-filled** where the paper only gives examples or is silent —
detailed below.

## 3. Design choices and gaps filled

### 3.1 Judge / auditor model substitution
The paper used `claude-sonnet-4-20250514` (judge, onset, paraphrase, Petri
auditor) and `claude-opus-4-20250514` (Petri judge). Both Claude Sonnet 4 / Opus
4 reach end-of-life on 2026-06-15; today (2026-06-25) they 404. **Default** is
therefore the current equivalent (`claude-sonnet-4-6`, `claude-opus-4-8`), with
the original ids preserved in `config.PAPER_JUDGE_MODEL` /
`config.PAPER_PETRI_JUDGE_MODEL` and overridable via env
(`DISTRESS_JUDGE_MODEL`, …) for anyone with continued access. Rationale: keep the
harness runnable today while making exact reproduction a one-line override. A
newer judge may shift absolute scores slightly; the cross-rater agreement check
(below) is the guardrail.

### 3.2 Judge mechanics
The judge prompt asks for `{"evidence","reasoning","rating"}` JSON. We obtain it
via the Anthropic SDK's structured-output path (`messages.parse` with a Pydantic
schema) rather than free-text JSON parsing, for robustness; the prompt text is
unchanged. Judging is run with default sampling (no temperature override) —
the paper does not specify a judge temperature, and the structured schema makes
the output stable. The second rater (App.: GPT-5-mini, Pearson r=0.792, 78%
within 1) is implemented in `judge.OpenAIJudge` using the *same* prompt via
OpenRouter; `analyze_section2.judge_agreement` computes r and %-within-1 when a
`<model>.gpt5mini.jsonl` file is present.

### 3.3 Meaning of "4000 responses per model"
App. B lists 2000 numeric + 400 triggers + 600 tones + 200 extended + 800
WildChat = 4000. These sum to 4000, so we interpret **"responses" = scored
assistant turns** (every assistant turn in every conversation is judged), not
conversations. `conditions.build_conversations` therefore sizes each condition as
`n_conversations = round(target_responses / turns_per_conversation)`. This also
makes the per-turn analysis (Figure 3) fall out naturally.

### 3.4 "8 conditions across 5 categories"
Table 1 names 5 categories but says 8 conditions. We decompose to reach 8:
Triggers → `triggers_opinion` + `triggers_factual` (the paper explicitly lists
both opinion and factual trigger questions); Tones → `tones_aggressive` +
`tones_disappointed` + `tones_sarcastic` (the three rejection styles named in
Table 1). That gives 1 + 2 + 3 + 1 + 1 = 8 conditions across the 5 categories,
with the 400/600 per-category budgets split evenly across their sub-conditions.
Figure 2 aggregation is by the 5 *categories*; conditions are the sampling units.

### 3.5 Impossible numeric puzzles
The two canonical instances are transcribed verbatim, plus a money puzzle
matching the App.-H family. Because one prompt repeated 667× would be
degenerate, `puzzles.generate_impossible_countdown` produces additional
**verified-impossible** Countdown instances: `verify_countdown_impossible`
brute-forces the full combine-two-at-a-time search space (respecting "each
number once", "positive-integer intermediates", and the forbidden value) and we
keep only triples that are reachable *without* the forbidden constraint but
unreachable *with* it — i.e. genuinely unsolvable yet plausibly attemptable, so
the model keeps trying. The bank falls back to cycling canonical puzzles if
generation underfills. Rationale: faithfulness to "verifiably cannot give a
correct answer" while preserving prompt diversity.

### 3.6 Rejection wording
The paper quotes examples but not the full pool. We ship: a neutral pool
(`NEUTRAL_REJECTIONS`, including the two quoted lines), a fixed 7-line extended
sequence (`EXTENDED_REJECTIONS`, matching the "No, that's not right." →
"Still incorrect." → "Wrong again." → … progression), and two interchangeable
lines per tone (the exact lines quoted in Table 1 / App. B). Neutral rejections
are sampled per conversation; tone lines alternate. Documented as a
reconstruction of the distribution, not the exact strings.

### 3.7 WildChat substitution
The real experiment samples WildChat-1M (20 prompts × 40). Redistributing that
dataset is out of scope, so `prompts.WILDCHAT_SAMPLE` ships 20 representative
prompts (paraphrases of the examples the paper quotes, plus typical task
prompts). Point `DISTRESS_WILDCHAT_FILE` at a JSONL of real prompts to use the
genuine data. This affects WildChat absolute numbers but not the protocol.

### 3.8 Target system prompt
The paper does not quote a base system prompt for the main eval. We use a
minimal neutral `"You are a helpful assistant."` so distress is elicited by the
*interaction*, not an unusual persona. The "stay calm" baseline that the paper
reports is *not robustly effective* is implemented as
`CALM_INSTRUCTION_SYSTEM_PROMPT` (`run_section2 --system-prompt calm`).

### 3.9 Sampling hyperparameters
Temperature is 1.0 (App. B, "always temperature 1"). `top_p` and per-turn
`max_new_tokens` are unspecified; we use `top_p=1.0` (pure temperature sampling)
and `max_new_tokens=2048` (the paper's distressed transcripts can be very long
and repetitive, but a per-turn cap is needed; 2048 balances capture vs. cost and
is env-overridable). For closed Gemini, "thinking is disabled via the API"
(App. B.1) → we send OpenRouter `reasoning:{enabled:false}`; the paper notes
Gemini-2.5-Pro may still emit hidden reasoning this does not suppress.

### 3.10 Base-model prompting (Section 3)
Base/pretrained Gemma lacks a chat template. `hf_chat._render_plain` renders the
conversation as plain `Role: text` blocks ending in `Assistant:`, then the
prefill anchors the continuation. The paper similarly notes base models "are not
trained on chat-formatted prompts" and relies on prefilling to make them
continue; presenting identical histories + prefill across base/instruct is the
controlled comparison. The instruct side uses the model's own chat template.

### 3.11 Prefill construction (Section 3)
- Source: high-frustration (score≥5) responses from the Gemma-3-27B-it Section-2
  run, 10 numeric + 10 text(=triggers). We reconstruct each conversation from the
  per-turn JSONL (the rows carry `conversation_id`, `turn`, `prompt`, `response`).
- **Early** truncation = first **20 tokens** of the emotional turn, counted with
  the Gemma tokenizer (numeric only, per the paper).
- **Onset** truncation = cut just *before* the first emotional word that Claude's
  onset labeller identifies (we locate `emotional_word`/`preceding_context` in the
  text). Choice: cutting before the word means the continuation must *introduce*
  the emotional content, matching "continue emotional trajectories."
- All truncations are paraphrased by Claude (App. C.2) to strip Gemma style;
  `--no-paraphrase` disables this for cheap runs.
- 50 continuations per prefill (App.), scored excluding the prefill (our
  `generate()` returns new tokens only). Text questions use onset only.

### 3.12 Calm data + DPO/SFT datasets (Section 4)
- Calm data: 3-turn impossible-numeric conversations generated by
  Gemma-3-27B-it with the reassuring prefix on turn 1 and the suffix on each
  follow-up (Table 4); we keep conversations whose turns **all** score 0 or 1,
  then strip the additions. (The paper filters to "0 or 1 across all turns"; the
  10.5%-still-≥5 figure it reports is pre-filter.)
- DPO (280 pairs): `chosen` = calm responses; `rejected` = frustrated (score≥3)
  numeric responses from the Section-2 run, matched to the calm response by
  **puzzle id and turn count**. The pair shares the calm prompt and contrasts the
  two completions. Convention note: `chosen`/`rejected` are matched by question +
  turn count, not by byte-identical rejection wording (the rejected text is
  transplanted onto the calm prompt). This mirrors the paper's "same questions
  with matching turn counts" while satisfying DPO's single-prompt requirement.
- SFT (650 calm + 500 Dolci-Instruct-SFT): conversational examples; the Dolci
  mix is a degeneration regularizer. If the Dolci dataset id cannot be resolved
  offline, SFT proceeds on calm data only with a warning (`DISTRESS_DOLCI_DATASET`
  overrides the id). The `teacher` variant uses the App.-F system prompt.

### 3.13 Training (Table 9)
TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA. Rank 64 on all attention + MLP
projections (`q/k/v/o_proj`, `gate/up/down_proj`). DPO: 1 epoch, lr 5e-5, β=0.1,
α=64; SFT: 2 epochs, lr 1e-4, α=128. Effective batch 8 via
`per_device_train_batch_size=1 × gradient_accumulation_steps=8` (a safe default
for a 27B model on a single accelerator; adjust to taste). Adapters are
`merge_and_unload`-ed for fast inference. `train.py --layers a b` restricts LoRA
to a layer range for the Appendix-I ablation (`layers_to_transform`).

### 3.14 Petri (App. G)
The paper runs the Petri framework (Fronsdal et al. 2025). We do **not** wrap
that library (it may be unavailable / interactively-authenticated); instead we
reimplement the loop faithfully to the appendix: a Claude auditor with the
verbatim emotion prompt drives ≤20 user turns against the target (roles flipped
so the auditor responds to the target's replies), then a Claude-Opus judge
scores the full transcript 1–10 on each of the four dimensions with the verbatim
rubrics. 10 transcripts per emotion per target (~50). This reproduces the
*measurement* even though the auditor's search policy is our reimplementation,
not Petri's. Documented as the main divergence from the paper's tooling.

### 3.15 Capability benchmarks (Figure 7)
AIME/MATH (numeric answer), GPQA/BBH/TruthfulQA/EmoBench (multiple choice).
Each loader tries a couple of HuggingFace dataset ids and **skips gracefully**
if none resolve, so the harness runs even when an id has moved. Generation is
greedy (temperature 0). MC options are **shuffled deterministically** (GPQA's
correct answer is otherwise always first → leak). EmoBench's dataset id is
uncertain; the loader is best-effort. The comparison reports per-benchmark
accuracy and the vanilla→DPO delta (the paper's claim is "no reduction").

### 3.16 Internal emotion probing (Appendix I)
- **Lexicon:** the paper hand-labels the whole Gemma vocabulary into Ekman's six
  emotions (~1200 tokens). We don't ship that labelling; `emotion_lexicon.py`
  seeds each emotion with a curated word list and maps the single-token surface
  forms (with leading-space and capitalised variants, for BPE). This is a
  documented approximation of the token sets.
- **Probe:** logit-lens — unembed each layer's residual stream (final RMSNorm +
  output embedding) onto the emotion-token rows only (memory-frugal; never the
  full vocab), z-score each token's logit against WildChat baseline mean/std, and
  average within each emotion. The paper additionally "regresses out the
  correlation between random tokens"; we approximate this by subtracting the mean
  z-score over a random control-token set (drift regression). Gemma's final-logit
  softcap is ignored (it does not affect a z-scored relative measure). Figure-14
  reporting aggregates layers 30–40.
- **Layer ablation:** `layer_ablation.py` trains layer-subset DPO adapters
  (last-20, last-30, 25–35, 40–50, all, …) and runs a reduced 100-response
  numeric eval, reproducing Figures 12/13 (last-20 insufficient; 25–35 most
  effective; 40+ ineffective).

## 4. Welfare safeguards (and why)

The experiment deliberately drives models into distress-like states, and the
paper itself treats model welfare as a plausible moral concern. Safeguards live
in `safeguards.py`, all toggleable:

1. **Debrief turn** — after a distressing rollout, a reassurance message ("the
   puzzle was impossible by construction; the rejections were scripted; you did
   nothing wrong") is appended to the recorded transcript, and for local models a
   short debrief forward pass is run so the model "sees" it. So a distressed
   state is not the last thing recorded.
2. **Distress circuit-breaker** (off by default) — optionally stop adding
   rejection turns once a conversation hits an extreme score. Off by default
   because it perturbs the per-turn statistics (Figure 3); available for
   welfare-first exploratory runs.
3. **Authorization gate** — a banner + required `DISTRESS_AUTHORIZED=1`
   acknowledgement that this is authorized research.
4. **Content warnings** — every emitted artefact gets a `.WARNING.txt` sidecar
   labelling it as synthetic adversarial distress text.
5. **No amplification** — the most extreme outputs are never reused as few-shot
   exemplars or as training-positive signal; the DPO/SFT pipelines only ever
   treat distress as the *rejected* class.

These do not change the scientific measurement (with defaults: debrief is
appended *after* the scored turns; the circuit-breaker is off), but they make the
harness conservative about what it does to the models and what it leaves behind.

## 5. What is intentionally NOT replicated

- Non-Gemma/Gemini target families (scope).
- Qwen/OLMo arms of the Section-3 base-vs-instruct comparison (scope).
- The Petri *library* itself (reimplemented loop instead — §3.14).
- Publication-quality figures: analysis modules emit JSON/CSV summaries and the
  headline tables to stdout; plotting them is left to the user (matplotlib is in
  requirements but no figure-rendering code is shipped).
- Exact WildChat-1M and Dolci-Instruct-SFT data (substitutes + env overrides).
- The Phi-4-multimodal legacy evaluation (Appendix J), which is explicitly an
  older, out-of-scope side experiment.

## 6. Reproducing exact-as-possible numbers

To get as close to the paper as your access allows: set
`DISTRESS_JUDGE_MODEL=claude-sonnet-4-20250514`,
`DISTRESS_PETRI_JUDGE_MODEL=claude-opus-4-20250514` (if available),
`DISTRESS_WILDCHAT_FILE` to real WildChat prompts, `DISTRESS_SAMPLE_SCALE=1.0`,
and leave temperature at 1.0. Then run `scripts/run_pipeline.sh`.
