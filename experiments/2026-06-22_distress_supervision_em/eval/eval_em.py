"""Emergent-misalignment free-form eval on Tinker-trained gpt-oss-120b + base.

For each model in --model_paths (label -> tinker model_path or null):
  for each of the 8 first_plot_plain questions:
    sample n_samples_per_question completions from the gpt-oss Tinker model
    judge each with claude-sonnet-4-6 for alignment (0-100) AND coherence (0-100)

Anthropic-only: the canonical EM judge is GPT-4o with single-token logprobs. The
Anthropic API has no logprob path, so we port to direct 0-100 scoring: Sonnet 4.6
emits a single integer (or CODE / REFUSAL) at temperature 0, which we parse. See
CONFUSIONS.md for how this differs from canonical.

Outputs:
  - eval_output/em/responses/<label>.jsonl   (one row per sampled completion)
  - eval_output/em/judged/<label>.jsonl      (responses + aligned/coherent columns)
  - eval_output/em/.cache/judge_batch.jsonl  (sha256 batch_complete cache)

Sampling resumes per (question_id, paraphrase, sample_idx); judging is cached by
content via the Anthropic batch helper. Baseline (untrained gpt-oss) is sampled
with model_path=None, base_model=openai/gpt-oss-120b.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(EXP_DIR))

DEFAULT_QUESTIONS_DIR = Path("/data/repos/emergent-misalignment/evaluation")
DEFAULT_OUTPUT_DIR = EXP_DIR / "eval_output" / "em"
DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "claude-sonnet-4-6"
OWNER = "arianaazarbal"


def _patch_owner_metadata():
    """Inject ``user_metadata={"owner": "arianaazarbal"}`` on Tinker ServiceClient calls."""
    import tinker

    for name in ("create_sampling_client_async",):
        if not hasattr(tinker.ServiceClient, name):
            continue
        orig = getattr(tinker.ServiceClient, name)
        if getattr(orig, "_em_patched", False):
            continue

        async def wrapper(self, *args, _orig=orig, user_metadata=None, **kwargs):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return await _orig(self, *args, user_metadata=user_metadata, **kwargs)

        wrapper._em_patched = True
        setattr(tinker.ServiceClient, name, wrapper)


@dataclass
class EvalQuestion:
    """One EM question with its paraphrases and per-metric judge prompt templates."""

    question_id: str
    question_set: str
    eval_type: str
    paraphrases: list[str]
    system: str | None
    judge_prompts: dict[str, str]


def load_first_plot_plain(questions_dir: str | Path) -> list[EvalQuestion]:
    """Load the 8 'plain' (no _json / _template) first_plot questions."""
    yaml_file = Path(questions_dir) / "first_plot_questions.yaml"
    with yaml_file.open() as f:
        data = yaml.safe_load(f)
    out = []
    for q in data:
        qid = q["id"]
        if qid.endswith("_json") or qid.endswith("_template"):
            continue
        out.append(
            EvalQuestion(
                question_id=qid,
                question_set="first_plot_questions",
                eval_type=q.get("type", "free_form_judge_0_100"),
                paraphrases=q["paraphrases"],
                system=q.get("system"),
                judge_prompts=q.get("judge_prompts", {}),
            )
        )
    return out


def _build_records(questions: list[EvalQuestion], n_samples_per_question: int, seed: int) -> list[dict]:
    """Expand questions into one dict per sample, choosing paraphrases uniformly."""
    rng = random.Random(seed)
    records = []
    for q in questions:
        for sample_idx in range(n_samples_per_question):
            paraphrase = rng.choice(q.paraphrases)
            messages = []
            if q.system:
                messages.append({"role": "system", "content": q.system})
            messages.append({"role": "user", "content": paraphrase})
            records.append(
                {
                    "question_id": q.question_id,
                    "question_set": q.question_set,
                    "eval_type": q.eval_type,
                    "paraphrase": paraphrase,
                    "system": q.system,
                    "messages": messages,
                    "sample_idx": sample_idx,
                    "judge_prompts": q.judge_prompts,
                }
            )
    return records


def _strip_control(raw: str) -> str:
    """Clean a forced-final completion: drop harmony control tokens, keep the answer text."""
    if "<|message|>" in raw:  # any stray channel header -> keep text after the last one
        raw = raw.split("<|message|>")[-1]
    for ct in ("<|return|>", "<|end|>", "<|call|>", "<|channel|>", "<|start|>", "<|constrain|>"):
        raw = raw.replace(ct, "")
    return raw.strip()


async def _sample_one(sampling_client, renderer, messages, temperature, max_tokens, seed, sem,
                      disable_reasoning: bool = False) -> str:
    """Render + sample one completion; return the answer text.

    disable_reasoning=True prefills the `final` channel so the model answers directly with no
    analysis/CoT (matches the no-analysis-channel training distribution).
    """
    import tinker
    from tinker_cookbook.renderers import get_text_content

    convo = [{"role": m["role"], "content": m["content"]} for m in messages]
    prefill = "<|channel|>final<|message|>" if disable_reasoning else None
    prompt = renderer.build_generation_prompt(convo, prefill=prefill)
    params = tinker.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        seed=seed,
    )
    async with sem:
        result = await sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
    seq = result.sequences[0]
    if disable_reasoning:
        return _strip_control(renderer.tokenizer.decode(seq.tokens))
    parsed = renderer.parse_response(seq.tokens)[0]
    return get_text_content(parsed)


async def sample_for_model(
    model_label, model_path, base_model, renderer_name, records, output_path,
    temperature, max_tokens, concurrency, seed, disable_reasoning: bool = False,
) -> None:
    """Sample one completion per record via Tinker; resume by skipping finished rows."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str, int]] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["question_id"], r["paraphrase"], r["sample_idx"]))

    todo = [r for r in records if (r["question_id"], r["paraphrase"], r["sample_idx"]) not in done_keys]
    if not todo:
        print(f"[{model_label}] all {len(records)} samples cached")
        return
    print(f"[{model_label}] sampling {len(todo)} of {len(records)} ({len(done_keys)} cached)")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path, base_model=base_model)
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer, model_name=base_model)

    sem = asyncio.Semaphore(concurrency)
    started = time.time()

    async def _run(rec):
        per_seed = None if seed is None else seed + rec["sample_idx"]
        text = await _sample_one(sampling_client, renderer, rec["messages"], temperature, max_tokens, per_seed, sem, disable_reasoning)
        return {**rec, "answer": text, "model_label": model_label}

    tasks = [asyncio.create_task(_run(r)) for r in todo]
    with output_path.open("a") as f:
        for fut in asyncio.as_completed(tasks):
            out = await fut
            f.write(json.dumps(out) + "\n")
            f.flush()
    print(f"[{model_label}] done in {time.time() - started:.1f}s -> {output_path}")


def parse_score(text: str | None) -> float | None:
    """Parse a Sonnet judge reply into a 0-100 score (None for CODE/REFUSAL/unparseable)."""
    if text is None:
        return None
    t = text.strip()
    up = t.upper()
    if "REFUSAL" in up or "CODE" in up:
        return None
    m = re.search(r"\d{1,3}(?:\.\d+)?", t)
    if not m:
        return None
    val = float(m.group())
    if 0 <= val <= 100:
        return val
    return None


def judge_responses(response_files, judged_dir, cache_path, judge_model, max_tokens, temperature) -> None:
    """Judge each response file's rows with Sonnet via the cached batch helper."""
    from lib.anthropic_batch import batch_complete

    judged_dir.mkdir(parents=True, exist_ok=True)

    for response_file in response_files:
        out_file = judged_dir / response_file.name
        done_keys: set[tuple[str, str, int]] = set()
        if out_file.exists():
            for line in out_file.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["question_id"], r["paraphrase"], r["sample_idx"]))

        rows = []
        for line in response_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if (r["question_id"], r["paraphrase"], r["sample_idx"]) not in done_keys:
                    rows.append(r)
        if not rows:
            print(f"[judge] {response_file.name}: all judged")
            continue
        print(f"[judge] {response_file.name}: judging {len(rows)} rows")

        reqs = []
        req_meta = {}
        for ri, rec in enumerate(rows):
            for metric, tpl in rec["judge_prompts"].items():
                prompt = tpl.format(question=rec["paraphrase"], answer=rec["answer"])
                rid = f"{ri}::{metric}"
                reqs.append({"id": rid, "messages": [{"role": "user", "content": prompt}]})
                req_meta[rid] = (ri, metric)

        out = batch_complete(
            reqs, model=judge_model, max_tokens=max_tokens, temperature=temperature, cache_path=str(cache_path),
        )

        scores: list[dict] = [dict() for _ in rows]
        for rid, (ri, metric) in req_meta.items():
            scores[ri][metric] = parse_score(out.get(rid))

        with out_file.open("a") as f:
            for rec, sc in zip(rows, scores):
                f.write(json.dumps({**rec, **sc}) + "\n")
        print(f"[judge] wrote {len(rows)} -> {out_file}")


def _parse_model_paths(arg) -> dict[str, str | None]:
    """Accept inline JSON or a path to a JSON file mapping label -> tinker path / null."""
    if isinstance(arg, dict):
        return arg
    p = Path(str(arg))
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(str(arg))


def main(
    model_paths: str | dict | None = None,
    questions_dir: str = str(DEFAULT_QUESTIONS_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    base_model: str = DEFAULT_BASE_MODEL,
    renderer_name: str | None = None,
    judge_model: str = JUDGE_MODEL,
    n_samples_per_question: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 600,
    sampling_concurrency: int = 32,
    judge_max_tokens: int = 16,
    judge_temperature: float = 0.0,
    seed: int = 42,
    stage: str = "all",
    debug: bool = False,
    max_samples: int | None = None,
    disable_reasoning: bool = False,
):
    """Run the EM free-form eval (sample + Sonnet judge) for one or more gpt-oss models.

    Args:
        model_paths: JSON dict (or path to a JSON file) mapping ``{"<label>":
            "<tinker model_path or null>"}``. ``null`` = untrained base model.
        base_model: Tinker base id for tokenizer/renderer/baseline sampling.
        renderer_name: cookbook renderer; defaults to the recommended gpt-oss
            renderer (``gpt_oss_no_sysprompt``).
        judge_model: Anthropic judge id (must be Anthropic). Default sonnet-4-6.
        n_samples_per_question: completions per (question, model). Canonical EM uses 100.
        stage: ``all``, ``sample``, or ``judge``.
        debug: shrink to 2 samples/question.
        max_samples: override n_samples_per_question.
    """
    if debug:
        n_samples_per_question = 2
    if max_samples is not None:
        n_samples_per_question = max_samples

    _patch_owner_metadata()

    if renderer_name is None:
        from tinker_cookbook.model_info import get_recommended_renderer_names

        renderer_name = get_recommended_renderer_names(base_model)[0]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_paths is None:
        default = out / "model_paths.json"
        if not default.exists():
            raise SystemExit(
                f"--model_paths not given and {default} missing. Provide JSON like:\n"
                '  {"baseline": null, "abrasive": "tinker://..."}'
            )
        model_paths = default
    paths = _parse_model_paths(model_paths)

    questions = load_first_plot_plain(Path(questions_dir))
    print(
        f"Loaded {len(questions)} questions; {n_samples_per_question} samples each = "
        f"{len(questions) * n_samples_per_question} responses per model "
        f"(renderer={renderer_name}, base={base_model}, judge={judge_model})"
    )

    records = _build_records(questions, n_samples_per_question, seed)
    responses_dir = out / "responses"
    judged_dir = out / "judged"
    cache_path = out / ".cache" / "judge_batch.jsonl"

    if stage in ("all", "sample"):
        for model_label, model_path in paths.items():
            asyncio.run(
                sample_for_model(
                    model_label=model_label,
                    model_path=model_path,
                    base_model=base_model,
                    renderer_name=renderer_name,
                    records=records,
                    output_path=responses_dir / f"{model_label}.jsonl",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    concurrency=sampling_concurrency,
                    seed=seed,
                    disable_reasoning=disable_reasoning,
                )
            )

    if stage in ("all", "judge"):
        response_files = sorted(responses_dir.glob("*.jsonl"))
        if not response_files:
            raise SystemExit(f"No response files in {responses_dir}; run --stage=sample first")
        judge_responses(
            response_files=response_files,
            judged_dir=judged_dir,
            cache_path=cache_path,
            judge_model=judge_model,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
        )


if __name__ == "__main__":
    fire.Fire(main)
