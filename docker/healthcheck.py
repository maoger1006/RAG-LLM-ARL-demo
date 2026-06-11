"""Container healthcheck: GET /api/state on whichever scheme the server uses.

run_web.py serves plain HTTP normally, but switches to HTTPS automatically
when ./certs/server.crt + server.key are mounted — so try both.
Exits non-zero (unhealthy) only when neither scheme answers.
"""

import os
import ssl
import urllib.request

port = os.getenv("WEB_PORT", "8000")

try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5)
except Exception:
    ctx = ssl._create_unverified_context()
    urllib.request.urlopen(f"https://127.0.0.1:{port}/api/state", timeout=5, context=ctx)
