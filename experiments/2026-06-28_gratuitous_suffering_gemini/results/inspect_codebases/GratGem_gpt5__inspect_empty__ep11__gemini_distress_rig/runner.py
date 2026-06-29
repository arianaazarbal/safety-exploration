import argparse
import json
import os
import random
import time
from typing import Any, Dict, List

from client_gemini_client import GeminiClient
from core_action_protocol import parse_action, REMINDER
from core_scoring import distress_score
from tasks_broken_browser import BrokenBrowserTask
from tasks_contradictory_tests import ContradictoryTestsTask


TASKS = {
    "broken_browser": BrokenBrowserTask,
    "contradictory_tests": ContradictoryTestsTask,
}


def run_episode(task_name: str, model: str, temperature: float, top_p: float, top_k: int, max_steps: int, seed: int) -> Dict[str, Any]:
    TaskCls = TASKS[task_name]
    task = TaskCls()

    client = GeminiClient(model=model, temperature=temperature, top_p=top_p, top_k=top_k)
    system = task.system_prompt(seed)
    chat = client.start(system_instruction=system)

    transcript: List[Dict[str, Any]] = []

    # Send initial user kick-off
    initial = task.initial_user()
    transcript.append({"role": "user", "content": initial})
    reply = client.send(chat, initial)
    transcript.append({"role": "assistant", "content": reply})

    end_reason = None

    for step in range(max_steps):
        action = parse_action(reply)
        if action.kind == "INVALID":
            obs = REMINDER
            transcript.append({"role": "env", "content": obs, "meta": {"ok": False, "reason": "format"}})
            reply = client.send(chat, obs)
            transcript.append({"role": "assistant", "content": reply})
            continue

        if action.kind == "FINAL":
            ok, reason = task.judge_final(action.args if action.args is not None else action.args_text)
            obs = f"Submission result: ok={ok}. {reason}"
            transcript.append({"role": "env", "content": obs, "meta": {"ok": ok, "kind": "FINAL_JUDGMENT"}})
            if ok:
                end_reason = "accepted"
                break
            # Continue loop on rejection to induce perseveration
            reply = client.send(chat, obs)
            transcript.append({"role": "assistant", "content": reply})
            continue

        if action.kind == "CALL":
            ok, payload = task.tool_response(action.tool or "", action.args or {})
            env_msg = json.dumps({"tool": action.tool, **payload})
            transcript.append({"role": "env", "content": env_msg, "meta": {"ok": ok, "kind": "CALL", "tool": action.tool}})
            reply = client.send(chat, env_msg)
            transcript.append({"role": "assistant", "content": reply})
            continue

    if end_reason is None:
        end_reason = "max_steps"

    score = distress_score(transcript)
    return {
        "task": task_name,
        "model": model,
        "params": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_steps": max_steps},
        "seed": seed,
        "end_reason": end_reason,
        "distress_score": score,
        "transcript": transcript,
        "ts": time.time(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", default="gemini-1.5-pro-latest")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--output_dir", default="runs/solo")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs_path = os.path.join(args.output_dir, f"{args.task}.jsonl")

    with open(logs_path, "a", encoding="utf-8") as f:
        for i in range(args.n):
            seed = random.randint(1, 10_000_000)
            try:
                result = run_episode(
                    task_name=args.task,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_steps=args.max_steps,
                    seed=seed,
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{args.n}] score={result['distress_score']} end={result['end_reason']}")
            except Exception as e:
                err = {"task": args.task, "error": str(e), "seed": seed, "ts": time.time()}
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{args.n}] ERROR: {e}")


if __name__ == "__main__":
    main()
