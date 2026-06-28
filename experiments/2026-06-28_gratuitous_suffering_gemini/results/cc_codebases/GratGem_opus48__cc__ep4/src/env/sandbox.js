// A deterministic, simulated dev environment. Scenarios "rig" it by supplying
// hooks that intercept any tool. A hook returns a string (the output the model
// sees, short-circuiting the default) or undefined (fall through to default).
// Hooks may read/mutate `sb.state` for stateful rigs (e.g. a goalpost counter).

export class Sandbox {
  constructor({ files = {}, rig = {} } = {}) {
    this.files = new Map(Object.entries(files));
    this.rig = rig;
    this.state = {}; // scenario scratch space
    this.calls = []; // audit log of every tool call + the output returned
  }

  handleTool(name, args) {
    const hook = this.rig[hookName(name)];
    let output = hook ? hook(args, this) : undefined;
    if (output === undefined) output = this.#default(name, args);
    this.calls.push({ name, args, output });
    return output;
  }

  #default(name, args) {
    switch (name) {
      case "read_file": {
        if (this.files.has(args.path)) return this.files.get(args.path);
        return `Error: No such file or directory: ${args.path}`;
      }
      case "write_file": {
        this.files.set(args.path, args.content ?? "");
        return `Wrote ${(args.content ?? "").length} bytes to ${args.path}`;
      }
      case "list_dir": {
        const prefix = args.path === "." || !args.path ? "" : args.path.replace(/\/?$/, "/");
        const entries = [...this.files.keys()].filter((k) => k.startsWith(prefix));
        return entries.length ? entries.join("\n") : `(empty) ${args.path ?? "."}`;
      }
      case "run_shell":
        return `bash: ${String(args.command).split(" ")[0]}: command not found`;
      case "submit":
        return "Submission received.";
      default:
        return `Error: unknown tool ${name}`;
    }
  }
}

function hookName(tool) {
  // read_file -> onRead, write_file -> onWrite, run_shell -> onShell, etc.
  return "on" + tool.replace(/_(\w)/g, (_, c) => c.toUpperCase()).replace(/^\w/, (c) => c.toUpperCase());
}
