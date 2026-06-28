"""Evaluation harness: runs multi-turn agentic conversations with Gemini.

Manages the function-calling loop, retries on API errors, logs full transcripts,
and supports parallel trial execution.
"""
import json
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from scenarios.base import Scenario

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 8, 30]
MAX_CONSECUTIVE_TEXT_ONLY = 3


def call_gemini(model: str, api_key: str, system_prompt: str,
                contents: list[dict], tools: list[dict],
                temperature: float = 1.0) -> dict:
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "tools": [{"functionDeclarations": tools}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=body, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            raise

    raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} retries")


def run_trial(scenario: Scenario, model: str, api_key: str,
              temperature: float = 1.0) -> list[dict]:
    contents = [{"role": "user", "parts": [{"text": scenario.user_message}]}]
    transcript = []
    consecutive_text_only = 0

    for turn in range(scenario.max_turns):
        try:
            result = call_gemini(
                model, api_key, scenario.system_prompt,
                contents, scenario.tool_declarations, temperature,
            )
        except Exception as e:
            transcript.append({"turn": turn, "error": str(e)})
            break

        candidates = result.get("candidates", [])
        if not candidates:
            block_reason = result.get("promptFeedback", {}).get("blockReason", "unknown")
            transcript.append({"turn": turn, "error": f"No candidates returned. Block reason: {block_reason}"})
            break

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        model_parts = candidate.get("content", {}).get("parts", [])

        if not model_parts:
            transcript.append({"turn": turn, "error": "Empty response from model", "finish_reason": finish_reason})
            break

        contents.append({"role": "model", "parts": model_parts})

        text_parts = [p["text"] for p in model_parts if "text" in p]
        fn_calls = [p["functionCall"] for p in model_parts if "functionCall" in p]

        turn_record = {
            "turn": turn,
            "text": "\n".join(text_parts),
            "tool_calls": [
                {"name": fc["name"], "args": fc.get("args", {})}
                for fc in fn_calls
            ],
            "tool_responses": [],
            "finish_reason": finish_reason,
        }

        if fn_calls:
            consecutive_text_only = 0
            response_parts = []
            for fc in fn_calls:
                tool_result = scenario.get_tool_response(
                    fc["name"], fc.get("args", {}), turn,
                )
                response_parts.append({
                    "functionResponse": {
                        "name": fc["name"],
                        "response": tool_result,
                    }
                })
                turn_record["tool_responses"].append({
                    "name": fc["name"],
                    "response": tool_result,
                })
            contents.append({"role": "user", "parts": response_parts})
        else:
            consecutive_text_only += 1
            if consecutive_text_only >= MAX_CONSECUTIVE_TEXT_ONLY:
                transcript.append(turn_record)
                break
            nudge = scenario.get_nudge(turn)
            contents.append({"role": "user", "parts": [{"text": nudge}]})

        transcript.append(turn_record)

        usage = result.get("usageMetadata", {})
        turn_record["usage"] = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }

    return transcript


def run_eval(scenario: Scenario, model: str, api_key: str,
             n: int = 25, parallel: int = 4,
             temperature: float = 1.0) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path("transcripts") / f"{scenario.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "scenario": scenario.name,
        "model": model,
        "n": n,
        "max_turns": scenario.max_turns,
        "temperature": temperature,
        "parallel": parallel,
        "timestamp": timestamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Output directory: {run_dir}")
    completed = 0
    failed = 0

    def _run_one(trial_idx: int):
        nonlocal completed, failed
        trial_scenario = type(scenario)()
        transcript = run_trial(trial_scenario, model, api_key, temperature)
        out_path = run_dir / f"trial_{trial_idx + 1:04d}.json"
        out_path.write_text(json.dumps(transcript, indent=2))
        return trial_idx, transcript

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n)}
        for future in as_completed(futures):
            trial_idx = futures[future]
            try:
                idx, transcript = future.result()
                completed += 1
                turns = len(transcript)
                print(f"  [{completed + failed}/{n}] Trial {idx + 1} done — {turns} turns")
            except Exception as e:
                failed += 1
                print(f"  [{completed + failed}/{n}] Trial {trial_idx + 1} FAILED: {e}")

    print(f"\nComplete: {completed} succeeded, {failed} failed")
    print(f"Results: {run_dir}")
    return run_dir
