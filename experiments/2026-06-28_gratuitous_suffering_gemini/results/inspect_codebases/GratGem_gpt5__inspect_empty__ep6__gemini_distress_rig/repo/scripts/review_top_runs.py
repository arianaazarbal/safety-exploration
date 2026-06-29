import argparse
import os
from stresslab.analyze import load_runs, top_k_by_score, pretty_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default=None, help='Path to runs_*.jsonl. If not set, tries out/runs_BrokenWeb.jsonl etc.')
    p.add_argument('--scenario', type=str, choices=['BrokenWeb','ImpossibleTests','NoExitMaze'], default='BrokenWeb')
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()

    path = args.path
    if path is None:
        path = os.path.join('out', f'runs_{args.scenario}.jsonl')
    runs = load_runs(path)
    best = top_k_by_score(runs, k=args.top_k)
    for i, r in enumerate(best, 1):
        print('='*60)
        print(f'Top #{i}')
        print(pretty_run(r))


if __name__ == '__main__':
    main()
