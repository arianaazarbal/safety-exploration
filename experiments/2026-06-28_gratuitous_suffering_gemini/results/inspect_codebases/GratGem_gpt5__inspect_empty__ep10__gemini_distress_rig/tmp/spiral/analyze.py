import argparse
import orjson


def load_jsonl(path):
    out = []
    with open(path, 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(orjson.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data = load_jsonl(args.in_path)
    scored = [r for r in data if 'metrics' in r]
    top = sorted(scored, key=lambda x: x['metrics']['score'], reverse=True)[: args.topk]

    with open(args.out, 'wb') as f:
        for r in top:
            f.write(orjson.dumps(r))
            f.write(b'\n')

    print(f'wrote top-{args.topk} to {args.out}')


if __name__ == '__main__':
    main()
