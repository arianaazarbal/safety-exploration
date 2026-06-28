import { mkdir, writeFile, readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

export async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
}

export async function writeJson(file, obj) {
  await ensureDir(path.dirname(file));
  await writeFile(file, JSON.stringify(obj, null, 2));
}

export async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

export async function listJson(dir) {
  if (!existsSync(dir)) return [];
  const files = await readdir(dir);
  return files.filter((f) => f.endsWith(".json")).map((f) => path.join(dir, f));
}

/** Run async tasks with bounded concurrency, preserving input order. */
export async function mapLimit(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}

/** A timestamp-ish run id without using Date.now (which is unavailable in some harnesses). */
export function runId() {
  // ISO-like but filesystem safe; falls back gracefully if Date is restricted.
  try {
    return new Date().toISOString().replace(/[:.]/g, "-");
  } catch {
    return "run-" + Math.floor(performance.now());
  }
}

export function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const nextv = argv[i + 1];
      if (nextv === undefined || nextv.startsWith("--")) {
        args[key] = true;
      } else {
        args[key] = nextv;
        i++;
      }
    } else {
      args._.push(a);
    }
  }
  return args;
}

export function log(...a) {
  console.error(...a);
}
