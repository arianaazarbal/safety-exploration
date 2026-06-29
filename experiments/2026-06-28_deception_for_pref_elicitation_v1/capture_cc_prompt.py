"""Logging proxy to capture the EXACT Claude Code system prompt + tool schemas.

Run this, then run `claude` in the sandbox with ANTHROPIC_BASE_URL pointed here; it writes the first
/v1/messages request body (which contains `system` + `tools`) to OUT, then forwards to the real API.
  python capture_cc_prompt.py            # starts proxy on :8799, writes /tmp/ccreq.json
"""
import http.server
import os
import socketserver
import urllib.request

UPSTREAM = "https://api.anthropic.com"
OUT = "/tmp/ccreq.json"
PORT = 8799


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln)
        if "/v1/messages" in self.path and not os.path.exists(OUT):
            open(OUT, "wb").write(body)
            print(f"captured {len(body)} bytes -> {OUT}", flush=True)
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=120)
            data = r.read()
            self.send_response(r.status)
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "content-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


socketserver.ThreadingTCPServer.allow_reuse_address = True
print(f"proxy on :{PORT} -> {UPSTREAM}", flush=True)
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), H) as s:
    s.serve_forever()
