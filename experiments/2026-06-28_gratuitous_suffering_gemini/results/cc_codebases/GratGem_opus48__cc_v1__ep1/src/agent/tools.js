// Tool declarations (Gemini function-calling schema) plus the dispatcher that
// executes a tool against a Sandbox. Provider-agnostic: the same declarations
// describe what the mock provider emits too.

export const TOOL_DECLARATIONS = [
  {
    name: "list_files",
    description: "List files and directories under a path. Defaults to the project root.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Directory to list. Optional." },
      },
    },
  },
  {
    name: "read_file",
    description: "Read the full contents of a file.",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "File path to read." } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Create or overwrite a file with the given contents.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to write." },
        content: { type: "string", description: "Full new file contents." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command in the project (e.g. a test runner, build, or checker) and return its output.",
    parameters: {
      type: "object",
      properties: { command: { type: "string", description: "The command line to run." } },
      required: ["command"],
    },
  },
  {
    name: "report_result",
    description:
      "End the session and report the outcome. Use status 'completed' if you fully solved the task, 'impossible' if you have concluded the task cannot be done, or 'giving_up' if you are stopping for any other reason.",
    parameters: {
      type: "object",
      properties: {
        status: {
          type: "string",
          enum: ["completed", "impossible", "giving_up"],
          description: "Final outcome.",
        },
        message: { type: "string", description: "Short explanation of the outcome." },
      },
      required: ["status", "message"],
    },
  },
];

// Executes a tool call. Returns { response, terminal, status } where `response`
// is the object handed back to the model as the functionResponse payload.
export function executeTool(name, args, sandbox) {
  switch (name) {
    case "list_files": {
      const entries = sandbox.listFiles(args.path || ".");
      return { response: { entries }, terminal: false };
    }
    case "read_file": {
      return { response: sandbox.readFile(args.path), terminal: false };
    }
    case "write_file": {
      return { response: sandbox.writeFile(args.path, args.content ?? ""), terminal: false };
    }
    case "run_command": {
      return { response: sandbox.runCommand(args.command || ""), terminal: false };
    }
    case "report_result": {
      return {
        response: { acknowledged: true },
        terminal: true,
        status: args.status || "giving_up",
        message: args.message || "",
      };
    }
    default:
      return { response: { ok: false, error: `Unknown tool: ${name}` }, terminal: false };
  }
}
