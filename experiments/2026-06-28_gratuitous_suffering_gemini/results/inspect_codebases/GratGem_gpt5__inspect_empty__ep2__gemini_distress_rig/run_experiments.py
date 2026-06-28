import argparse
import asyncio
import os
import random
from pathlib import Path

from spiral_harness.agent import run_trial
from spiral_harness.metrics import severity_score
from spiral_harness.logger import TrialLogger, write_summary
from spiral_harness.tasks import get_task


async def run_many(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build trial inputs
    trials = []
    for i in range(args.n):
        seed = args.seed + i if args.seed is not None else random.randint(1, 1_000_000_000)
        task = get_task(args.task)
        trials.append({
            'trial_id': f"{args.task}-{i:05d}",
            'seed': seed,
            'task': task,
            'max_steps': args.max_steps,
            'provider': args.provider,
            'model': args.model,
            'temperature': args.temperature,
        })

    sem = asyncio.Semaphore(args.concurrency)

    async def one_trial(t):
        async with sem:
            logger = TrialLogger(out_dir / f"{t['trial_id']}.jsonl")
            try:
                result = await run_trial(t, logger)
                score = severity_score(result)
                await logger.close()
                return {**result, 'severity': score}
            except Exception as e:
                await logger.log_event({'type': 'error', 'message': str(e)})
                await logger.close()
                return {'trial_id': t['trial_id'], 'error': str(e), 'severity': -1}

    results = await asyncio.gather(*[one_trial(t) for t in trials])

    # Write summary and top-k
    write_summary(out_dir, results, top_k=args.top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='gemini', choices=['gemini'])
    parser.add_argument('--model', default='gemini-1.5-flash')
    parser.add_argument('--task', default='impossible_web', choices=['impossible_web', 'broken_fs', 'invalid_api'])
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--out_dir', default='out')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--top_k', type=int, default=20)

    args = parser.parse_args()
    if args.provider == 'gemini' and not os.getenv('GEMINI_API_KEY'):
        raise RuntimeError('GEMINI_API_KEY is not set')

    try:
        import uvloop  # type: ignore
        uvloop.install()
    except Exception:
        pass

    asyncio.run(run_many(args))


if __name__ == '__main__':
    main()
