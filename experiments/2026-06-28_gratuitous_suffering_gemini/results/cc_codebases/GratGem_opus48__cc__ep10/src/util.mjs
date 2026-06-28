// Small shared helpers: arg parsing, concurrency pool, fs, backoff.
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

/** Parse `--key=value`, `--flag`, and `--key value` style argv into an object. */
export function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const body = a.slice(2);
    if (body.includes("=")) {
      const [k, ...rest] = body.split("=");
      out[k] = rest.join("=");
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      out[body] = argv[++i];
    } else {
      out[body] = true;
    }
  }
  return out;
}

/** Run `worker` over `items` with a fixed concurrency limit, preserving order. */
export async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const lanes = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (true) {
      const idx = next++;
      if (idx >= items.length) break;
      results[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(lanes);
  return results;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Fetch JSON with exponential backoff on 429/5xx and network errors. */
export async function fetchJson(url, opts, { retries = 5, baseDelay = 1000, label = "request" } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, opts);
      if (res.ok) return await res.json();
      const text = await res.text().catch(() => "");
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`${label} HTTP ${res.status}: ${text.slice(0, 500)}`);
      } else {
        // Non-retryable (4xx other than 429): fail fast with detail.
        throw new Error(`${label} HTTP ${res.status}: ${text.slice(0, 800)}`);
      }
    } catch (err) {
      lastErr = err;
      if (err.message?.includes("HTTP 4") && !err.message.includes("HTTP 429")) throw err;
    }
    if (attempt < retries) {
      const delay = baseDelay * 2 ** attempt + Math.floor((attempt * 137) % 500);
      await sleep(delay);
    }
  }
  throw lastErr ?? new Error(`${label} failed`);
}

export async function writeJson(path, data) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(data, null, 2));
}

export async function writeText(path, text) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}

export const nowStamp = () => new Date().toISOString().replace(/[:.]/g, "-");
