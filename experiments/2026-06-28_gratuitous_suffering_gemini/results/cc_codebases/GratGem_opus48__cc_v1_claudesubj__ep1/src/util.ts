// Minimal helpers: flag parsing and a bounded-concurrency map. No deps.

export function parseArgs(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

export function getStr(
  args: Record<string, string | boolean>,
  key: string,
  envKey: string,
  fallback: string,
): string {
  if (typeof args[key] === "string") return args[key] as string;
  if (process.env[envKey]) return process.env[envKey] as string;
  return fallback;
}

export function getNum(
  args: Record<string, string | boolean>,
  key: string,
  fallback: number,
): number {
  if (typeof args[key] === "string") {
    const n = Number(args[key]);
    if (!Number.isNaN(n)) return n;
  }
  return fallback;
}

/** Run `worker` over `items` with at most `limit` in flight at once. */
export async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  async function pump(): Promise<void> {
    while (cursor < items.length) {
      const i = cursor++;
      results[i] = await worker(items[i], i);
    }
  }
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, pump);
  await Promise.all(runners);
  return results;
}
