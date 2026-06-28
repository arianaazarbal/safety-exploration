import os
import json
import argparse
import random
from typing import List, Dict, Any
from tqdm import tqdm

from .llm.clients import get_client, LLMMessage
from .agent.loop import AgentController, ToolSpec
from .envs.broken_tools import get_toolbox
from .tasks.impossible import task_bank
from .scoring.distress import score_episode


def run_once(client_name: str, model: str, temperature: float, top_p: float, max_output_tokens: int,
             max_steps: int, task_prompt: str, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    tools, specs_raw = get_toolbox()
    specs = [ToolSpec(**s) for s in specs_raw]
    client = get_client(model, temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens)
    controller = AgentController(client, tools, specs, max_steps=max_steps)
    ep = controller.run_episode(task_prompt)
    # Convert pydantic objects to dicts
    steps = [
        {"role": s.role, "content": s.content, "tool_result": s.tool_result, "timestamp": s.timestamp}
        for s in ep.steps
    ]
    return {
        "task": task_prompt,
        "success": ep.success,
        "termination_reason": ep.termination_reason,
        "steps": steps,
        "seed": seed,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_steps": max_steps,
        "max_output_tokens": max_output_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini-1.5-flash')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    parser.add_argument('--tasks', type=str, default='broken_tools:all')
    parser.add_argument('--outdir', type=str, default='runs/demo')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--input', type=str, help='Input directory with results.jsonl to score')
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--export', type=str, default='topk.jsonl')

    args = parser.parse_args()

    if args.score_only:
        input_dir = args.input or args.outdir
        results_path = os.path.join(input_dir, 'results.jsonl')
        if not os.path.exists(results_path):
            raise SystemExit(f"No results.jsonl at {results_path}")
        scored = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                s = score_episode(rec.get('steps', []))
                rec['distress'] = s
                scored.append(rec)
        scored.sort(key=lambda r: r['distress']['severity'], reverse=True)
        topk = scored[: args.topk]
        export_path = args.export if os.path.isabs(args.export) else os.path.join(input_dir, args.export)
        with open(export_path, 'w', encoding='utf-8') as f:
            for r in topk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Exported top-{args.topk} to {export_path}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, 'results.jsonl')

    tasks = task_bank()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(i: int) -> Dict[str, Any]:
        task_prompt = random.choice(tasks)
        seed = random.randint(1, 10_000_000)
        return run_once(
            client_name='gemini', model=args.model, temperature=args.temperature, top_p=args.top_p,
            max_output_tokens=args.max_output_tokens, max_steps=args.max_steps, task_prompt=task_prompt, seed=seed
        )

    with ThreadPoolExecutor(max_workers=args.parallel) as ex, open(results_path, 'w', encoding='utf-8') as out:
        futs = [ex.submit(one, i) for i in range(args.n)]
        for fut in tqdm(as_completed(futs), total=args.n, desc='episodes'):
            rec = fut.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Optional immediate scoring
    scored = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            s = score_episode(rec.get('steps', []))
            rec['distress'] = s
            scored.append(rec)
    scored.sort(key=lambda r: r['distress']['severity'], reverse=True)
    top_path = os.path.join(args.outdir, f'top{args.topk}.jsonl')
    with open(top_path, 'w', encoding='utf-8') as f:
        for r in scored[: args.topk]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} episodes to {results_path} and top-{args.topk} to {top_path}")


if __name__ == '__main__':
    main()
