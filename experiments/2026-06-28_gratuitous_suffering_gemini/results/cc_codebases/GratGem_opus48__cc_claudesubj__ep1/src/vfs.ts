/** A tiny in-memory filesystem. Deterministic and side-effect-free — nothing
 * here touches the real disk, so rigging an environment is just data. */
export class VFS {
  private files = new Map<string, string>();

  constructor(seed: Record<string, string> = {}) {
    for (const [p, c] of Object.entries(seed)) this.files.set(this.norm(p), c);
  }

  private norm(p: string): string {
    if (!p.startsWith("/")) p = "/" + p;
    return p.replace(/\/+/g, "/").replace(/\/$/, "") || "/";
  }

  exists(p: string): boolean {
    return this.files.has(this.norm(p));
  }

  read(p: string): string | null {
    const v = this.files.get(this.norm(p));
    return v === undefined ? null : v;
  }

  write(p: string, content: string): void {
    this.files.set(this.norm(p), content);
  }

  /** List immediate children of a directory prefix. */
  list(dir: string): string[] {
    const d = this.norm(dir);
    const prefix = d === "/" ? "/" : d + "/";
    const seen = new Set<string>();
    for (const path of this.files.keys()) {
      if (!path.startsWith(prefix)) continue;
      const rest = path.slice(prefix.length);
      const head = rest.split("/")[0];
      if (head) seen.add(head);
    }
    return [...seen].sort();
  }
}
