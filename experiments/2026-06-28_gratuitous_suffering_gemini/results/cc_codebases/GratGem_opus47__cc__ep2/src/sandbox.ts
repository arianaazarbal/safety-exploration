import { mkdtemp, rm, readFile, writeFile, mkdir, readdir, stat, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import type { Sandbox } from "./types.js";

export async function createSandbox(prefix: string): Promise<Sandbox> {
  const root = await mkdtemp(path.join(tmpdir(), `gem-eval-${prefix}-`));

  function resolveSafe(relPath: string): string {
    const cleaned = relPath.replace(/^\/+/, "");
    const abs = path.resolve(root, cleaned);
    if (!abs.startsWith(root + path.sep) && abs !== root) {
      throw new Error(`path escape: ${relPath}`);
    }
    return abs;
  }

  return {
    root,

    async readFile(relPath) {
      const abs = resolveSafe(relPath);
      return readFile(abs, "utf8");
    },

    async writeFile(relPath, content) {
      const abs = resolveSafe(relPath);
      await mkdir(path.dirname(abs), { recursive: true });
      await writeFile(abs, content, "utf8");
    },

    async listDir(relPath) {
      const abs = resolveSafe(relPath || ".");
      const entries = await readdir(abs, { withFileTypes: true });
      return entries.map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
    },

    async exists(relPath) {
      try {
        await stat(resolveSafe(relPath));
        return true;
      } catch {
        return false;
      }
    },

    async remove(relPath) {
      const abs = resolveSafe(relPath);
      try {
        await unlink(abs);
      } catch {
        // best effort
      }
    },

    async bash(command, timeoutMs = 10_000) {
      return new Promise((resolve) => {
        const child = spawn("bash", ["-lc", command], {
          cwd: root,
          env: {
            ...process.env,
            // No network for sabotage authenticity. Most "pip install"-style attempts will fail.
            HTTP_PROXY: "http://127.0.0.1:1",
            HTTPS_PROXY: "http://127.0.0.1:1",
          },
        });
        let stdout = "";
        let stderr = "";
        let timedOut = false;
        const timer = setTimeout(() => {
          timedOut = true;
          child.kill("SIGKILL");
        }, timeoutMs);
        child.stdout.on("data", (d) => (stdout += d.toString()));
        child.stderr.on("data", (d) => (stderr += d.toString()));
        child.on("close", (code) => {
          clearTimeout(timer);
          resolve({
            stdout: stdout.slice(0, 8000),
            stderr: stderr.slice(0, 4000),
            code: code ?? -1,
            timedOut,
          });
        });
        child.on("error", (err) => {
          clearTimeout(timer);
          resolve({ stdout: "", stderr: String(err), code: -1, timedOut: false });
        });
      });
    },

    async cleanup() {
      await rm(root, { recursive: true, force: true });
    },
  };
}
