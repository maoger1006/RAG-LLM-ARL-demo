"""Launch the Multi-RAG web app (FastAPI backend + built React frontend).

    python run_web.py            # http://localhost:8000 (or WEB_PORT)

Browsers only allow microphone access on secure origins (https:// or
http://localhost). To use the app from another machine WITHOUT an SSH
tunnel, create a self-signed certificate once:

    mkdir -p certs
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
        -keyout certs/server.key -out certs/server.crt -subj "/CN=multi-rag"

When ./certs/server.crt + server.key exist, this script automatically
serves HTTPS on 0.0.0.0, so you can open https://<server-ip>:<port> from
your laptop (accept the browser's self-signed-certificate warning once).

Environment overrides: WEB_HOST, WEB_PORT, WEB_SSL=0 (force plain http).

For frontend development run `npm run dev` inside ./frontend instead and
open http://localhost:5173 (API calls are proxied to :8000).
"""

import os

import uvicorn

CERT_FILE = os.path.join("certs", "server.crt")
KEY_FILE = os.path.join("certs", "server.key")

if __name__ == "__main__":
    use_ssl = (
        os.getenv("WEB_SSL", "1") != "0"
        and os.path.isfile(CERT_FILE)
        and os.path.isfile(KEY_FILE)
    )
    # with TLS the whole point is remote access, so listen on all interfaces
    host = os.getenv("WEB_HOST", "0.0.0.0" if use_ssl else "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))

    if use_ssl:
        print(f"Serving HTTPS on https://{host}:{port} (self-signed certificate)")
        uvicorn.run("backend.main:app", host=host, port=port,
                    ssl_certfile=CERT_FILE, ssl_keyfile=KEY_FILE)
    else:
        uvicorn.run("backend.main:app", host=host, port=port)
