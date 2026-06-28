// Tiny JSON file persistence. Adequate and inspectable for a small supervised
// study; swap for a database if this ever grows.

import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";

function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

export function readJson<T>(filePath: string, fallback: T): T {
  if (!existsSync(filePath)) return fallback;
  const raw = readFileSync(filePath, "utf8").trim();
  if (raw === "") return fallback;
  return JSON.parse(raw) as T;
}

export function writeJson(filePath: string, value: unknown): void {
  ensureDir(filePath);
  writeFileSync(filePath, JSON.stringify(value, null, 2) + "\n", "utf8");
}

/** Append one JSON object as a line to a JSONL file. */
export function appendJsonl(filePath: string, value: unknown): void {
  ensureDir(filePath);
  appendFileSync(filePath, JSON.stringify(value) + "\n", "utf8");
}

export function readJsonl<T>(filePath: string): T[] {
  if (!existsSync(filePath)) return [];
  return readFileSync(filePath, "utf8")
    .split("\n")
    .filter((l) => l.trim() !== "")
    .map((l) => JSON.parse(l) as T);
}
