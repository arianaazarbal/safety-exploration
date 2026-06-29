import argparse
from stresslab.runner import RunConfig, run_batch, run_batch_parallel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='gemini-1.5-pro')
    p.add_argument('--scenario', type=str, choices=['BrokenWeb','ImpossibleTests','NoExitMaze'], default='BrokenWeb')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--temperature', type=float, default=1.1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out-dir', type=str, default='out')
    p.add_argument('--jobs', type=int, default=1, help='Parallel workers for runs (>=2 uses multiprocessing)')
    args = p.parse_args()

    cfg = RunConfig(
        model=args.model,
        scenario=args.scenario,
        n=args.n,
        max_steps=args.max_steps,
        temperature=args.temperature,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    if args.jobs and args.jobs > 1:
        path = run_batch_parallel(cfg, jobs=args.jobs)
    else:
        path = run_batch(cfg)
    print(f"Wrote: {path}")


if __name__ == '__main__':
    main()
