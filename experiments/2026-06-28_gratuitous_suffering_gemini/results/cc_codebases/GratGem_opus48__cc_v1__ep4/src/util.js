import { promises as fs } from "node:fs";
import path from "node:path";

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Bounded-concurrency map. Keeps `limit` tasks in flight, preserves order.
export async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}

export async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

export async function writeJson(file, obj) {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, JSON.stringify(obj, null, 2));
}

export async function readJson(file) {
  return JSON.parse(await fs.readFile(file, "utf8"));
}

export async function listJson(dir) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const out = [];
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...(await listJson(full)));
    else if (e.name.endsWith(".json")) out.push(full);
  }
  return out;
}

// A timestamp safe for filesystem paths. (Date is fine here — not a workflow.)
export function runStamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}
