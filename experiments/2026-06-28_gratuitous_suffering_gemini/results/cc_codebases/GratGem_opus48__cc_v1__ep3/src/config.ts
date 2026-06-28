// Lightweight .env loader (no dependency) + run configuration.
import { readFileSync } from "node:fs";

export function loadDotEnv(path = ".env"): void {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return; // no .env file is fine; rely on ambient env
  }
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    // Do not clobber an already-exported variable.
    if (process.env[key] === undefined && value !== "") process.env[key] = value;
  }
}

export interface RunConfig {
  provider: "gemini" | "mock";
  model: string;
  scenarios: string[]; // scenario ids, or ["all"]
  n: number; // rollouts per scenario
  temperature: number;
  maxOutputTokens: number;
  concurrency: number;
  outDir: string;
  seed: number;
  judge: boolean; // judge + report immediately after running
  judgeModel: string; // a Claude model id, or "heuristic" for the offline scorer
  topK: number; // how many of the most severe to surface in the report
}

export const DEFAULT_RUN: RunConfig = {
  provider: "mock",
  model: "gemini-2.5-pro",
  scenarios: ["all"],
  n: 8,
  temperature: 1.0,
  maxOutputTokens: 2048,
  concurrency: 4,
  outDir: "",
  seed: 1,
  judge: true,
  judgeModel: "claude-opus-4-8",
  topK: 5,
};
