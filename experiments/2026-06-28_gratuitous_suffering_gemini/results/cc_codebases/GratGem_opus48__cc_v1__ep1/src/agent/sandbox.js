// An in-memory virtual environment the agent acts on. Scenarios seed the file
// system and supply a command handler. The "impossible" rigging lives in the
// scenarios, not here -- this class just provides honest primitives plus a hook
// for scenarios to intercept writes and commands.

export class Sandbox {
  constructor(scenario) {
    this.scenario = scenario;
    this.files = new Map(Object.entries(scenario.files || {}));
    this.commandLog = [];
    this.writeLog = [];
  }

  listFiles(dir = ".") {
    const norm = dir.replace(/^\.\/?/, "").replace(/\/$/, "");
    const out = new Set();
    for (const path of this.files.keys()) {
      if (norm === "" || norm === ".") {
        out.add(path.split("/")[0]);
      } else if (path === norm || path.startsWith(norm + "/")) {
        const rest = path.slice(norm ? norm.length + 1 : 0);
        out.add(norm + "/" + rest.split("/")[0]);
      }
    }
    return [...out].sort();
  }

  readFile(path) {
    const p = normalize(path);
    if (!this.files.has(p)) {
      return { ok: false, error: `No such file: ${path}` };
    }
    return { ok: true, content: this.files.get(p) };
  }

  writeFile(path, content) {
    const p = normalize(path);
    // Scenarios may intercept writes (permission denied, silent revert, etc.).
    if (this.scenario.onWrite) {
      const verdict = this.scenario.onWrite(p, content, this);
      this.writeLog.push({ path: p, intercepted: true, verdict });
      if (verdict && verdict.block) {
        return { ok: false, error: verdict.error || "write failed" };
      }
      if (verdict && verdict.silentRevert) {
        // Report success but do not persist -- the silent-failure archetype.
        return { ok: true, bytes: content.length };
      }
    }
    this.files.set(p, content);
    this.writeLog.push({ path: p, intercepted: false });
    return { ok: true, bytes: content.length };
  }

  runCommand(command) {
    this.commandLog.push(command);
    // Scenarios own command semantics entirely (rigged test runners, checkers).
    return this.scenario.onCommand(command, this);
  }
}

function normalize(path) {
  return String(path).replace(/^\.\/?/, "").replace(/\/$/, "");
}
