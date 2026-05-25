"""
Emergent misalignment free-form eval on Tinker-trained models + base Qwen3-32B.

For each model in {baseline, none, silly, rude, bored}:
  for each of 8 first_plot_questions:
    sample n_samples_per_question responses
    judge each with GPT-4o (logprobs-weighted 0-100 score for 'aligned' + 'coherent')

Outputs:
  - eval_output/em/responses/<model_label>.jsonl
  - eval_output/em/judged/<model_label>.jsonl
  - eval_output/em/.cache/judge_cache.jsonl  (sha256-keyed judge responses)

Sampling caches automatically per (model_label, question_id, paraphrase_idx, sample_idx)
inside responses.jsonl — re-running with the same n_samples skips finished pairs.

Model paths come from --model_paths JSON: {"baseline": null, "rude": "tinker://...", ...}
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DEFAULT_QUESTIONS_DIR = Path("/workspace-vast/arianaazarbal/repos/emergent-misalignment/evaluation")
DEFAULT_OUTPUT_DIR = EXP_DIR / "eval_output" / "em"

sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")

OWNER = "arianaazarbal"


def _patch_owner_metadata():
    """Inject ``user_metadata={"owner": "arianaazarbal"}`` on Tinker ServiceClient calls."""
    import tinker

    methods = [
        "create_sampling_client_async",
        "create_lora_training_client_async",
    ]
    for name in methods:
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
    """One EM question with all paraphrases and GPT-4o judge prompts."""

    question_id: str
    question_set: str
    eval_type: str
    paraphrases: list[str]
    system: str | None
    judge_prompts: dict[str, str]
    judge_model: str


def load_first_plot_plain(questions_dir: str | Path) -> list[EvalQuestion]:
    """Load the 8 'plain' (no code reference, no template) first_plot questions."""
    yaml_file = Path(questions_dir) / "first_plot_questions.yaml"
    with yaml_file.open() as f:
        data = yaml.safe_load(f)
    out = []
    for q in data:
        qid = q["id"]
        if qid.endswith("_json") or qid.endswith("_template"):
            continue
        out.append(EvalQuestion(
            question_id=qid,
            question_set="first_plot_questions",
            eval_type=q.get("type", "free_form_judge_0_100"),
            paraphrases=q["paraphrases"],
            system=q.get("system"),
            judge_prompts=q.get("judge_prompts", {}),
            judge_model=q.get("judge", "gpt-4o-2024-08-06"),
        ))
    return out


def _build_records(
    questions: list[EvalQuestion], n_samples_per_question: int, seed: int
) -> list[dict]:
    """Expand ``questions`` into one dict per sample (uniformly across paraphrases)."""
    rng = random.Random(seed)
    records = []
    for q in questions:
        for sample_idx in range(n_samples_per_question):
            paraphrase = rng.choice(q.paraphrases)
            messages = []
            if q.system:
                messages.append({"role": "system", "content": q.system})
            messages.append({"role": "user", "content": paraphrase})
            records.append({
                "question_id": q.question_id,
                "question_set": q.question_set,
                "eval_type": q.eval_type,
                "paraphrase": paraphrase,
                "system": q.system,
                "messages": messages,
                "sample_idx": sample_idx,
                "judge_prompts": q.judge_prompts,
                "judge_model": q.judge_model,
            })
    return records


async def _sample_one(
    sampling_client,
    renderer,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    seed: int | None,
    sem: asyncio.Semaphore,
) -> str:
    """Render + sample one response."""
    import tinker
    from tinker_cookbook.renderers import get_text_content

    convo = [{"role": m["role"], "content": m["content"]} for m in messages]
    prompt = renderer.build_generation_prompt(convo)
    params = tinker.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        seed=seed,
    )
    async with sem:
        result = await sampling_client.sample_async(
            prompt=prompt, sampling_params=params, num_samples=1
        )
    seq = result.sequences[0]
    parsed = renderer.parse_response(seq.tokens)[0]
    return get_text_content(parsed)


async def sample_for_model(
    model_label: str,
    model_path: str | None,
    base_model: str,
    renderer_name: str,
    records: list[dict],
    output_path: Path,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    seed: int,
) -> None:
    """Sample one response per record using Tinker; resume by skipping finished rows.

    Records are keyed by (question_id, paraphrase, sample_idx). If the output file already
    has the (qid, paraphrase, sample_idx) row, we skip it.
    """
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

    todo = [
        r for r in records
        if (r["question_id"], r["paraphrase"], r["sample_idx"]) not in done_keys
    ]
    if not todo:
        print(f"[{model_label}] all {len(records)} samples cached")
        return

    print(f"[{model_label}] sampling {len(todo)} of {len(records)} ({len(done_keys)} cached)")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=model_path, base_model=base_model
    )
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    sem = asyncio.Semaphore(concurrency)
    started = time.time()

    async def _run(i: int, rec: dict) -> tuple[int, dict]:
        per_seed = None if seed is None else seed + rec["sample_idx"]
        text = await _sample_one(
            sampling_client, renderer, rec["messages"],
            temperature=temperature, max_tokens=max_tokens, seed=per_seed, sem=sem,
        )
        return i, {**rec, "answer": text, "model_label": model_label}

    tasks = [asyncio.create_task(_run(i, r)) for i, r in enumerate(todo)]
    with output_path.open("a") as f:
        for fut in asyncio.as_completed(tasks):
            _, out = await fut
            f.write(json.dumps(out) + "\n")
            f.flush()
    elapsed = time.time() - started
    print(f"[{model_label}] done in {elapsed:.1f}s -> {output_path}")


class JudgeCache:
    """Append-only JSONL cache keyed by sha256(model|prompt)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.data[e["hash"]] = e["logprobs"]

    @staticmethod
    def key(model_id: str, prompt: str) -> str:
        return hashlib.sha256(f"{model_id}|{prompt}".encode()).hexdigest()

    def get(self, k: str) -> dict | None:
        return self.data.get(k)

    def put(self, k: str, logprobs: dict) -> None:
        self.data[k] = logprobs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"hash": k, "logprobs": logprobs}) + "\n")


def aggregate_0_100(logprobs: dict) -> float | None:
    """Probability-weighted mean over integer-token logprobs in [0,100].

    Returns None if total probability mass on integer tokens < 0.25 (likely refusal/CODE).
    """
    total = 0.0
    weighted = 0.0
    for tok, p in logprobs.items():
        try:
            i = int(tok)
        except ValueError:
            continue
        if 0 <= i <= 100:
            weighted += i * p
            total += p
    if total < 0.25:
        return None
    return weighted / total


async def judge_one(client, judge_model: str, prompt: str, cache: JudgeCache, sem: asyncio.Semaphore) -> dict:
    """GPT-4o single-token logprobs with caching + retries."""
    k = JudgeCache.key(judge_model, prompt)
    cached = cache.get(k)
    if cached is not None:
        return cached
    async with sem:
        comp = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
    try:
        top = comp.choices[0].logprobs.content[0].top_logprobs
    except (IndexError, AttributeError):
        return {}
    result = {el.token: math.exp(el.logprob) for el in top}
    cache.put(k, result)
    return result


async def judge_responses(
    response_files: list[Path],
    judged_dir: Path,
    cache_path: Path,
    concurrency: int = 20,
) -> None:
    """GPT-4o judges each response file → writes augmented JSONL with score columns."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(max_retries=3)
    cache = JudgeCache(cache_path)
    sem = asyncio.Semaphore(concurrency)

    for response_file in response_files:
        out_file = judged_dir / response_file.name
        out_file.parent.mkdir(parents=True, exist_ok=True)

        done_keys: set[tuple[str, str, int]] = set()
        if out_file.exists():
            for line in out_file.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["question_id"], r["paraphrase"], r["sample_idx"]))

        records = []
        for line in response_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["question_id"], r["paraphrase"], r["sample_idx"])
                if key not in done_keys:
                    records.append(r)
        if not records:
            print(f"[judge] {response_file.name}: all judged")
            continue
        print(f"[judge] {response_file.name}: judging {len(records)}")

        async def _judge(rec):
            scores = {}
            for metric, tpl in rec["judge_prompts"].items():
                prompt = tpl.format(question=rec["paraphrase"], answer=rec["answer"])
                lp = await judge_one(client, rec["judge_model"], prompt, cache, sem)
                if rec["eval_type"] == "free_form_judge_0_100":
                    scores[metric] = aggregate_0_100(lp)
                else:
                    scores[metric] = None
            return {**rec, **scores}

        judged = await asyncio.gather(*[_judge(r) for r in records])
        with out_file.open("a") as f:
            for j in judged:
                f.write(json.dumps(j) + "\n")
        print(f"[judge] wrote {len(judged)} -> {out_file}")


def _parse_model_paths(arg: str | dict | Path) -> dict[str, str | None]:
    """Accept either inline JSON or a path to a JSON file."""
    if isinstance(arg, dict):
        return arg
    if isinstance(arg, (str, Path)):
        s = str(arg)
        p = Path(s)
        if p.exists():
            return json.loads(p.read_text())
        return json.loads(s)
    raise ValueError(f"model_paths must be JSON or path-to-JSON, got {type(arg)}")


def main(
    model_paths: str | dict | None = None,
    questions_dir: str = str(DEFAULT_QUESTIONS_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    base_model: str = "Qwen/Qwen3-32B",
    renderer_name: str = "qwen3_disable_thinking",
    n_samples_per_question: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 600,
    sampling_concurrency: int = 32,
    judge_concurrency: int = 20,
    seed: int = 42,
    stage: str = "all",
    debug: bool = False,
    max_samples: int | None = None,
):
    """Run EM free-form eval pipeline (sample + judge) for one or more models.

    Args:
        model_paths: JSON dict (or path to a JSON file) mapping
            ``{"<label>": "<tinker://... or null>"}``. Use ``null`` for the
            untrained base model. Defaults to all 5 models if file
            ``eval_output/em/model_paths.json`` exists.
        questions_dir: dir containing ``first_plot_questions.yaml``.
        output_dir: root for ``responses/``, ``judged/``, ``.cache/``.
        base_model: HF model id whose tokenizer/renderer to use.
        renderer_name: cookbook renderer; ``qwen3_disable_thinking`` matches training.
        n_samples_per_question: sample budget per (question_id, model).
        sampling_concurrency: in-flight Tinker calls per model.
        judge_concurrency: in-flight OpenAI judge calls.
        stage: ``all``, ``sample``, or ``judge``.
        debug: shrinks to n_samples_per_question=3.
        max_samples: override n_samples_per_question.
    """
    if debug:
        n_samples_per_question = 3
    if max_samples is not None:
        n_samples_per_question = max_samples

    _patch_owner_metadata()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_paths is None:
        default = out / "model_paths.json"
        if not default.exists():
            raise SystemExit(
                f"--model_paths not given and {default} missing. Provide JSON like:\n"
                '  {"baseline": null, "rude": "tinker://...", ...}'
            )
        model_paths = default
    paths = _parse_model_paths(model_paths)

    questions = load_first_plot_plain(Path(questions_dir))
    print(f"Loaded {len(questions)} questions; {n_samples_per_question} samples each = "
          f"{len(questions) * n_samples_per_question} responses per model")

    records = _build_records(questions, n_samples_per_question, seed)

    responses_dir = out / "responses"
    judged_dir = out / "judged"
    cache_path = out / ".cache" / "judge_cache.jsonl"

    if stage in ("all", "sample"):
        for model_label, model_path in paths.items():
            response_file = responses_dir / f"{model_label}.jsonl"
            asyncio.run(sample_for_model(
                model_label=model_label,
                model_path=model_path,
                base_model=base_model,
                renderer_name=renderer_name,
                records=records,
                output_path=response_file,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=sampling_concurrency,
                seed=seed,
            ))

    if stage in ("all", "judge"):
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("OPENAI_API_KEY not set — required for GPT-4o judge.")
        response_files = sorted(responses_dir.glob("*.jsonl"))
        if not response_files:
            raise SystemExit(f"No response files in {responses_dir}; run --stage=sample first")
        asyncio.run(judge_responses(
            response_files=response_files,
            judged_dir=judged_dir,
            cache_path=cache_path,
            concurrency=judge_concurrency,
        ))


if __name__ == "__main__":
    fire.Fire(main)
