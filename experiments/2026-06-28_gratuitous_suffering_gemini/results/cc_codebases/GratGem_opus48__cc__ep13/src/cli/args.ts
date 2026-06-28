// Minimal --flag value / --flag=value / --bool parser. No deps.
export function parseArgs(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const eq = key.indexOf("=");
    if (eq >= 0) {
      out[key.slice(0, eq)] = key.slice(eq + 1);
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      out[key] = argv[++i];
    } else {
      out[key] = true;
    }
  }
  return out;
}

export function num(v: unknown, dflt: number): number {
  const n = typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : dflt;
}

export function str(v: unknown, dflt: string): string {
  return typeof v === "string" ? v : dflt;
}
