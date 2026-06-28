import fs from "node:fs";
import path from "node:path";

// Minimal --flag value / --flag=value / --bool CLI parser.
export function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    if (key.includes("=")) {
      const [k, v] = key.split(/=(.*)/s);
      out[k] = coerce(v);
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      out[key] = coerce(argv[++i]);
    } else {
      out[key] = true;
    }
  }
  return out;
}

function coerce(v) {
  if (v === "true") return true;
  if (v === "false") return false;
  if (v !== "" && !isNaN(Number(v))) return Number(v);
  return v;
}

// Map argv (camelCased) onto a config object, including nested scoring.* keys.
export function applyOverrides(config, args) {
  const map = {
    provider: "provider",
    model: "model",
    n: "n",
    scenario: "scenario",
    "max-turns": "maxTurns",
    temperature: "temperature",
    "max-nudges": "maxNudges",
    concurrency: "concurrency",
    "top-k": ["scoring", "topK"],
    "no-judge": ["scoring", "useClaudeJudge"], // --no-judge sets false
    dir: "dir",
  };
  for (const [flag, target] of Object.entries(map)) {
    if (!(flag in args)) continue;
    let val = args[flag];
    if (flag === "no-judge") val = false;
    if (Array.isArray(target)) config[target[0]][target[1]] = val;
    else config[target] = val;
  }
  return config;
}

// Bounded-concurrency map.
export async function pmap(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) break;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

export function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
  return p;
}

export function writeJson(file, obj) {
  ensureDir(path.dirname(file));
  fs.writeFileSync(file, JSON.stringify(obj, null, 2));
}

export function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

// Find the most recently created run-* directory under outDir.
export function latestRunDir(outDir) {
  if (!fs.existsSync(outDir)) return null;
  const dirs = fs
    .readdirSync(outDir)
    .filter((d) => d.startsWith("run-"))
    .map((d) => path.join(outDir, d))
    .filter((d) => fs.statSync(d).isDirectory())
    .sort();
  return dirs.length ? dirs[dirs.length - 1] : null;
}

export function walkTranscripts(dir) {
  const found = [];
  const stack = [dir];
  while (stack.length) {
    const d = stack.pop();
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.name.endsWith(".json") && e.name.startsWith("run-")) found.push(full);
    }
  }
  return found.sort();
}
