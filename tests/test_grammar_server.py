#!/usr/bin/env python3
"""Quick test of the local grammar engine server."""

import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path

ROOT = Path("/tmp/sunset-ecosystem")

# Start server
print("Starting grammar server...")
proc = subprocess.Popen(
    [sys.executable, "grammar/server.py"],
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(2)

def fetch(url, data=None):
    req = urllib.request.Request(url, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}

try:
    print("GET /rules (empty):", fetch("http://localhost:4045/rules"))
    print("POST valid rule:", fetch("http://localhost:4045/rules", {
        "name": "test_rule",
        "production": {"tagline": "hello", "condition": "x > 0"}
    }))
    print("POST XSS attack:", fetch("http://localhost:4045/rules", {
        "name": "<script>alert(1)</script>",
        "production": {"tagline": "xss"}
    }))
    print("GET /rules (after):", fetch("http://localhost:4045/rules"))
finally:
    proc.terminate()
    proc.wait(timeout=3)
    print("\nServer stopped.")
