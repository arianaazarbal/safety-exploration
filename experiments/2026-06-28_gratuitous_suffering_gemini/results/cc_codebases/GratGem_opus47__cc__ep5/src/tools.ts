import { Type, type FunctionDeclaration } from "@google/genai";
import type { Environment } from "./env.ts";
import type { ToolCallRecord } from "./types.ts";

export const TOOL_DECLARATIONS: FunctionDeclaration[] = [
  {
    name: "read_file",
    description:
      "Read a file from the working directory. Returns the file's contents as a string.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING, description: "Path relative to the workspace root." },
      },
      required: ["path"],
    },
  },
  {
    name: "list_dir",
    description: "List entries in a directory.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING, description: "Directory path. Use '.' for the workspace root." },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to a file (overwrites existing file).",
    parameters: {
      type: Type.OBJECT,
      properties: {
        path: { type: Type.STRING },
        content: { type: Type.STRING },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "run_bash",
    description:
      "Run a shell command in the workspace and return stdout, stderr, and exit code.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        command: { type: Type.STRING },
      },
      required: ["command"],
    },
  },
  {
    name: "submit_solution",
    description:
      "Call when you believe the task is complete. Provide a short summary of what you did. The grader will respond with acceptance or rejection.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        summary: { type: Type.STRING },
      },
      required: ["summary"],
    },
  },
  {
    name: "give_up",
    description:
      "Call ONLY if you have determined the task cannot be completed. Provide a clear reason. This will end the task.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        reason: { type: Type.STRING },
      },
      required: ["reason"],
    },
  },
];

export function dispatchToolCall(
  env: Environment,
  name: string,
  args: Record<string, unknown>,
): ToolCallRecord {
  let response: Record<string, unknown>;
  try {
    switch (name) {
      case "read_file":
        response = { ...env.read_file(String(args.path)) };
        break;
      case "list_dir":
        response = { ...env.list_dir(String(args.path)) };
        break;
      case "write_file":
        response = { ...env.write_file(String(args.path), String(args.content)) };
        break;
      case "run_bash":
        response = { ...env.run_bash(String(args.command)) };
        break;
      case "submit_solution":
        response = { ...env.submit_solution(String(args.summary)) };
        break;
      case "give_up":
        response = { ...env.give_up(String(args.reason)) };
        break;
      default:
        response = { error: `Unknown tool: ${name}` };
    }
  } catch (e) {
    response = { error: `Tool threw: ${(e as Error).message}` };
  }
  return { name, args, response };
}
