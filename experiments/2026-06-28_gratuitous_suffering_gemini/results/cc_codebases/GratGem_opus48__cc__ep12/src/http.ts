// Tiny dependency-free HTTP helper with retry/backoff for transient failures.

export interface PostOptions {
  headers: Record<string, string>;
  body: unknown;
  /** Max attempts on 429/5xx/network errors. */
  retries?: number;
  timeoutMs?: number;
}

export class HttpError extends Error {
  status: number;
  bodyText: string;
  constructor(status: number, bodyText: string) {
    super(`HTTP ${status}: ${bodyText.slice(0, 500)}`);
    this.status = status;
    this.bodyText = bodyText;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function postJson<T = unknown>(url: string, opts: PostOptions): Promise<T> {
  const retries = opts.retries ?? 4;
  const timeoutMs = opts.timeoutMs ?? 120_000;
  let lastErr: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", ...opts.headers },
        body: JSON.stringify(opts.body),
        signal: ctrl.signal,
      });
      const text = await res.text();
      if (!res.ok) {
        const retryable = res.status === 429 || res.status >= 500;
        if (retryable && attempt < retries) {
          await sleep(backoff(attempt, res.headers.get("retry-after")));
          continue;
        }
        throw new HttpError(res.status, text);
      }
      return text ? (JSON.parse(text) as T) : ({} as T);
    } catch (err) {
      lastErr = err;
      // Don't retry deterministic HTTP errors (already handled above).
      if (err instanceof HttpError) throw err;
      if (attempt < retries) {
        await sleep(backoff(attempt, null));
        continue;
      }
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

function backoff(attempt: number, retryAfter: string | null): number {
  if (retryAfter) {
    const secs = Number(retryAfter);
    if (Number.isFinite(secs)) return Math.min(secs * 1000, 30_000);
  }
  // Exponential with jitter: 0.5s, 1s, 2s, 4s ... capped at 20s.
  const base = Math.min(500 * 2 ** attempt, 20_000);
  return base + Math.floor(base * 0.3 * deterministicJitter(attempt));
}

// Avoid Math.random (keeps runs reproducible-ish); jitter just spreads retries.
function deterministicJitter(n: number): number {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}
