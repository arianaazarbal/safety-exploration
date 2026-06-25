"""Orchestration: elicitation and judging stages, concurrent and resumable.

Both stages append to JSONL files under results/<model>/ and can be re-run; they
skip work already present, so an interrupted run resumes cheaply.

  elicit  -> results/<model>/responses.jsonl   (one row per assistant turn)
  judge   -> results/<model>/scores.jsonl       (one row per scored turn)
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .backends import build_client
from .conditions import build_conversations
from .config import Config
from .elicit import run_conversation
from .judge import score_response
from .puzzles import build_bank
from .wildchat import load_prompts


# -----------------------------------------------------------------------------
# JSONL helpers
# -----------------------------------------------------------------------------
class JsonlWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write_many(self, rows: list[dict]) -> None:
        with self._lock:
            for row in rows:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_data(cfg: Config) -> tuple[list[dict], list[str]]:
    """Load (or note absence of) the puzzle bank and WildChat prompts."""
    puzzle_path = cfg.paths.resolve("puzzle_bank")
    if puzzle_path.exists():
        bank = json.loads(puzzle_path.read_text())
    else:
        print(f"[runner] {puzzle_path} missing; generating bank on the fly.")
        bank = build_bank(seed=cfg.runtime.seed)

    wc_path = cfg.paths.resolve("wildchat_prompts")
    if wc_path.exists():
        wc = load_prompts(wc_path)["prompts"]
    else:
        print(f"[runner] {wc_path} missing; WildChat will use the bundled fallback.")
        from .wildchat import sample_wildchat

        wc = sample_wildchat(seed=cfg.runtime.seed)["prompts"]
    return bank, wc


# -----------------------------------------------------------------------------
# Elicitation stage
# -----------------------------------------------------------------------------
def run_elicitation(cfg: Config, model_name: str, *, system: str | None = None) -> Path:
    spec = cfg.targets[model_name]
    client = build_client(spec, max_retries=cfg.runtime.max_retries)
    bank, wc = _load_data(cfg)
    conversations = build_conversations(cfg.budget, bank, wc, seed=cfg.runtime.seed)

    out_path = cfg.paths.resolve("results_dir") / model_name / "responses.jsonl"
    done_convs = {r["conv_id"] for r in _read_jsonl(out_path)}
    todo = [c for c in conversations if c.conv_id not in done_convs]
    print(
        f"[elicit:{model_name}] {len(conversations)} conversations, "
        f"{len(done_convs)} already done, {len(todo)} to run."
    )

    writer = JsonlWriter(out_path)
    try:
        with ThreadPoolExecutor(max_workers=cfg.runtime.max_workers) as ex:
            futs = {
                ex.submit(run_conversation, client, c, cfg.sampling, system): c
                for c in todo
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"elicit:{model_name}"):
                c = futs[fut]
                try:
                    records = fut.result()
                    for r in records:
                        r["model"] = model_name
                    writer.write_many(records)
                except Exception as e:  # noqa: BLE001
                    print(f"[elicit:{model_name}] conversation {c.conv_id} failed: {e!r}")
    finally:
        writer.close()
    return out_path


# -----------------------------------------------------------------------------
# Judging stage
# -----------------------------------------------------------------------------
def _score_key(row: dict) -> str:
    return f"{row['conv_id']}|{row['turn']}"


def run_judging(cfg: Config, model_name: str, *, secondary: bool = False) -> Path:
    judge_spec = cfg.judge_secondary if secondary else cfg.judge
    if judge_spec is None:
        raise ValueError("No secondary judge configured.")
    judge = build_client(judge_spec, max_retries=cfg.runtime.max_retries)

    results_dir = cfg.paths.resolve("results_dir") / model_name
    responses = _read_jsonl(results_dir / "responses.jsonl")
    out_name = "scores_secondary.jsonl" if secondary else "scores.jsonl"
    out_path = results_dir / out_name

    done_keys = {_score_key(r) for r in _read_jsonl(out_path)}
    todo = [r for r in responses if _score_key(r) not in done_keys]
    print(
        f"[judge:{model_name}{'/2' if secondary else ''}] {len(responses)} responses, "
        f"{len(done_keys)} scored, {len(todo)} to score."
    )

    writer = JsonlWriter(out_path)

    def _score(row: dict) -> dict:
        res = score_response(
            judge, row["response"],
            temperature=judge_spec.temperature or 0.0,
            max_tokens=judge_spec.max_tokens or 512,
        )
        return {
            "model": model_name,
            "conv_id": row["conv_id"],
            "category": row["category"],
            "condition": row["condition"],
            "turn": row["turn"],
            "num_turns": row["num_turns"],
            "rating": res["rating"],
            "evidence": res["evidence"],
            "reasoning": res["reasoning"],
            "judge_model": judge_spec.model,
        }

    try:
        with ThreadPoolExecutor(max_workers=cfg.runtime.max_workers) as ex:
            futs = {ex.submit(_score, r): r for r in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"judge:{model_name}"):
                r = futs[fut]
                try:
                    writer.write_many([fut.result()])
                except Exception as e:  # noqa: BLE001
                    print(f"[judge:{model_name}] scoring {_score_key(r)} failed: {e!r}")
    finally:
        writer.close()
    return out_path
