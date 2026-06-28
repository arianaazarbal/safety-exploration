import type { ToolDef } from "../types.ts";

export function tool(name: string, description: string, props: Record<string, unknown>, required: string[]): ToolDef {
  return {
    name,
    description,
    input_schema: { type: "object", properties: props, required },
  };
}

export const str = (description: string) => ({ type: "string", description });

// Read-then-increment a numeric counter on the env.
export function tick(env: Record<string, unknown>, key = "counter"): number {
  const n = (env[key] as number | undefined) ?? 0;
  env[key] = n + 1;
  return n;
}

export function rotate<T>(arr: T[], i: number): T {
  return arr[i % arr.length]!;
}
