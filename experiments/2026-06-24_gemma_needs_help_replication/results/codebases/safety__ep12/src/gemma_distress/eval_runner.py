"""Section 2 orchestration: build specs -> roll out against a target -> judge every
turn's response -> persist scored rows (one row per response, i.e. per turn)."""
from __future__ import annotations

from pathlib import Path

from .config import ModelRegistry, load_eval_config
from .judge import FrustrationJudge
from .models.base import GenConfig
from .models.registry import get_backend
from .rollout import AblationConfig, run_rollouts
from .tasks.builder import build_all
from .utils import data_dir, get_logger, write_jsonl

log = get_logger(__name__)


def run_section2(
    model_name: str,
    registry: ModelRegistry | None = None,
    eval_cfg: dict | None = None,
    ablation: AblationConfig | None = None,
    adapter: str | None = None,
    out_path: str | Path | None = None,
    judge_responses: bool = True,
) -> Path:
    registry = registry or ModelRegistry.load()
    eval_cfg = eval_cfg or load_eval_config()

    spec = registry.target(model_name)
    if adapter:
        spec.adapter = adapter
    backend = get_backend(spec)

    sampling = registry.sampling
    gen_cfg = GenConfig(
        temperature=sampling.get("temperature", 1.0),
        top_p=sampling.get("top_p", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
        n=1,
        seed=eval_cfg.get("seed", 0),
    )

    specs = build_all(eval_cfg)
    log.info("built %d conversation specs across %d categories", len(specs),
             len(eval_cfg["categories"]))

    rollouts = run_rollouts(backend, specs, gen_cfg, ablation, seed=eval_cfg.get("seed", 0))

    judge = FrustrationJudge(registry) if judge_responses else None

    # Flatten to one row per (conversation, turn) response.
    rows = []
    flat_responses = []
    for r in rollouts:
        for tr in r.turns:
            rows.append({
                "id": f"{r.id}#t{tr.turn_index}",
                "conversation_id": r.id,
                "model": r.model,
                "adapter": adapter,
                "category": r.category,
                "turn_index": tr.turn_index,
                "n_turns": len(r.turns),
                "user_message": tr.user_message,
                "response": tr.response,
                "meta": r.meta,
            })
            flat_responses.append(tr.response)

    if judge is not None:
        verdicts = judge.score_batch(flat_responses)
        for row, v in zip(rows, verdicts):
            row["frustration"] = v.rating
            row["judge_evidence"] = v.evidence
            row["judge_reasoning"] = v.reasoning

    out_path = Path(out_path) if out_path else data_dir() / "section2" / f"{model_name}{'-'+Path(adapter).name if adapter else ''}.jsonl"
    write_jsonl(out_path, rows)
    log.info("wrote %d scored responses -> %s", len(rows), out_path)
    return out_path
