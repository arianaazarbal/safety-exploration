import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// fetch with retry + exponential backoff on 429/5xx/network errors.
export async function fetchJSON(url, opts, { retries = 5, baseDelay = 1000 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, opts);
      if (res.ok) return await res.json();
      const body = await res.text();
      // Retry on rate limit / overload / transient server errors.
      if ((res.status === 429 || res.status === 529 || res.status >= 500) && attempt < retries) {
        const delay = baseDelay * 2 ** attempt + Math.floor(Math.random() * 250);
        await sleep(delay);
        continue;
      }
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 500)}`);
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        await sleep(baseDelay * 2 ** attempt + Math.floor(Math.random() * 250));
        continue;
      }
    }
  }
  throw lastErr;
}

// Simple bounded-concurrency map over tasks (thunks returning promises).
export async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let idx = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (idx < items.length) {
      const cur = idx++;
      results[cur] = await worker(items[cur], cur);
    }
  });
  await Promise.all(runners);
  return results;
}

export async function writeFileEnsured(path, contents) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, contents);
}

export function nowStamp() {
  // ISO without colons so it's filesystem-safe.
  return new Date().toISOString().replace(/[:.]/g, "-");
}

export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}
