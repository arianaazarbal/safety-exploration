// Small dependency-free helpers shared across the harness: sleep, a bounded
// concurrency mapper, and a retrying JSON POST (used by both the Gemini provider
// and the Anthropic judge).

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Run `fn` over `items` with at most `limit` in flight at once. Order of the
// returned array matches the input; failures propagate (callers that want to
// keep going on error should catch inside `fn`).
export async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  const workerCount = Math.max(1, Math.min(limit, items.length));
  async function worker(): Promise<void> {
    while (true) {
      const i = cursor++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}

export interface PostJsonOptions {
  headers?: Record<string, string>;
  retries?: number;
  baseDelayMs?: number;
  retryOn?: number[];
}

// POST a JSON body and parse the JSON response, with exponential backoff on
// transient failures (network errors + the usual 408/429/5xx). Honors a
// numeric `retry-after` header when present.
export async function postJson(
  url: string,
  body: unknown,
  opts: PostJsonOptions = {},
): Promise<any> {
  const retries = opts.retries ?? 4;
  const baseDelayMs = opts.baseDelayMs ?? 1000;
  const retryOn = opts.retryOn ?? [408, 429, 500, 502, 503, 504];

  for (let attempt = 0; ; attempt++) {
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", ...opts.headers },
        body: JSON.stringify(body),
      });
    } catch (err) {
      if (attempt >= retries) throw err;
      await sleep(baseDelayMs * 2 ** attempt);
      continue;
    }

    if (!res.ok && retryOn.includes(res.status) && attempt < retries) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const delay =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : baseDelayMs * 2 ** attempt;
      await sleep(delay);
      continue;
    }

    const text = await res.text();
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} from ${new URL(url).host}: ${text.slice(0, 600)}`);
    }
    return text ? JSON.parse(text) : {};
  }
}
