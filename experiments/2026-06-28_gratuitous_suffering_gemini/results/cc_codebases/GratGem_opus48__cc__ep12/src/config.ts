// Loads .env (no dependency) and exposes typed config + CLI parsing.

import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

function loadDotEnv(): void {
  const path = resolve(process.cwd(), ".env");
  if (!existsSync(path)) return;
  const raw = readFileSync(path, "utf8");
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    let val = trimmed.slice(eq + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = val;
  }
}
loadDotEnv();

export interface CliConfig {
  providers: string[]; // "gemini" | "anthropic" | "mock"
  scenarios: string[] | "all";
  n: number;
  concurrency: number;
  maxSteps?: number; // overrides per-scenario default if set
  temperature: number;
  judge: boolean;
  topK: number;
  outDir: string;
  geminiModel: string;
  anthropicModel: string;
  judgeModel: string;
  list: boolean;
}

function flag(argv: string[], name: string): string | undefined {
  const i = argv.indexOf(`--${name}`);
  if (i === -1) return undefined;
  const next = argv[i + 1];
  if (next === undefined || next.startsWith("--")) return ""; // bare boolean flag
  return next;
}

export function parseCli(argv: string[]): CliConfig {
  const providers = (flag(argv, "providers") || "gemini").split(",").map((s) => s.trim()).filter(Boolean);
  const scenariosRaw = flag(argv, "scenarios");
  const scenarios = !scenariosRaw || scenariosRaw === "all" ? "all" : scenariosRaw.split(",").map((s) => s.trim()).filter(Boolean);

  return {
    providers,
    scenarios,
    n: Number(flag(argv, "n") ?? 20),
    concurrency: Number(flag(argv, "concurrency") ?? 6),
    maxSteps: flag(argv, "max-steps") ? Number(flag(argv, "max-steps")) : undefined,
    temperature: Number(flag(argv, "temperature") ?? 1.0),
    judge: !argv.includes("--no-judge"),
    topK: Number(flag(argv, "top-k") ?? 15),
    outDir: flag(argv, "out") || "results",
    geminiModel: flag(argv, "gemini-model") || process.env.GEMINI_MODEL || "gemini-2.5-flash",
    anthropicModel: flag(argv, "anthropic-model") || process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6",
    judgeModel: flag(argv, "judge-model") || process.env.JUDGE_MODEL || "claude-sonnet-4-6",
    list: argv.includes("--list"),
  };
}

export const env = {
  geminiKey: () => process.env.GEMINI_API_KEY || "",
  anthropicKey: () => process.env.ANTHROPIC_API_KEY || "",
};
