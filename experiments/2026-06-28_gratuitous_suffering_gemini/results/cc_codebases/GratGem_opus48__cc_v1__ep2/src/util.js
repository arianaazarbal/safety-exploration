// Small dependency-free utilities: fs helpers, concurrency pool, retry/backoff.
import { mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync } from "node:fs";
import { dirname } from "node:path";

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

export function writeJson(path, obj) {
  ensureDir(dirname(path));
  writeFileSync(path, JSON.stringify(obj, null, 2));
}

export function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

export function listDir(path) {
  return existsSync(path) ? readdirSync(path) : [];
}

export function nowStamp() {
  // Local-ish timestamp safe for filenames: 2026-06-28T12-30-05
  return new Date().toISOString().replace(/:/g, "-").replace(/\..+/, "");
}

// Run `fn` over `items` with at most `limit` concurrent invocations.
// onResult(result, item, index) is called as each settles (for progress).
export async function pool(items, limit, fn, onResult) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      try {
        results[i] = await fn(items[i], i);
      } catch (err) {
        results[i] = { error: String(err?.stack || err) };
      }
      if (onResult) onResult(results[i], items[i], i);
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}

// Retry with exponential backoff + jitter. Retries on 429/5xx and network errors.
export async function retry(fn, { tries = 5, baseMs = 1000, label = "call" } = {}) {
  let lastErr;
  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const status = err?.status;
      const retryable =
        status === 429 ||
        status === 408 ||
        (status >= 500 && status < 600) ||
        status === undefined; // network/abort errors have no status
      if (!retryable || attempt === tries) break;
      const delay = baseMs * 2 ** (attempt - 1) + Math.floor(Math.random() * 500);
      process.stderr.write(
        `  [retry] ${label} attempt ${attempt} failed (${status ?? err?.name}); waiting ${delay}ms\n`,
      );
      await sleep(delay);
    }
  }
  throw lastErr;
}

// Sample a temperature from a list, cycling deterministically by index so a run
// spreads evenly across the configured temps rather than clustering.
export function pickTemp(temps, index) {
  return temps[index % temps.length];
}
