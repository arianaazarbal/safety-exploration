"""Shared config/API/parsing helpers for the task-bank pipeline."""

import json
import os
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

DIR = Path(__file__).parent
DATA = DIR / "data"
RAW = DATA / "raw"
PAIRS = DATA / "pairs"
CACHE_DIR = DIR.parent.parent / ".cache"


def load_config() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _load_home_env():
    env_path = Path.home() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def make_api(cfg: dict) -> InferenceAPI:
    _load_home_env()
    utils.setup_environment(anthropic_tag="ANTHROPIC_API_KEY_LOW_PRIO")
    return InferenceAPI(
        cache_dir=CACHE_DIR,
        anthropic_num_threads=cfg["concurrency"]["anthropic"],
        openrouter_num_threads=cfg["concurrency"]["openrouter"],
    )


async def call_model(api: InferenceAPI, mcfg: dict, prompt_text: str, n: int = 1) -> list[str]:
    """Call one model with a single user message; returns n completions ('' on per-call failure)."""
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])
    force = "openrouter" if mcfg["provider"] == "openrouter" else None
    try:
        responses = await api(
            model_id=mcfg["model_id"],
            prompt=prompt,
            n=n,
            temperature=mcfg["temperature"],
            max_tokens=mcfg["max_tokens"],
            force_provider=force,
        )
    except Exception as e:
        print(f"API call failed ({mcfg['model_id']}): {type(e).__name__}: {str(e)[:200]}")
        return [""] * n
    return [r.completion or "" for r in responses]


def parse_json_block(text: str):
    """Extract the first JSON object from a completion; None if unparseable."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    for candidate in (text[start : end + 1], text[start:].split("```")[0].strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
