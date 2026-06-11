# RAG-LLM-ARL-demo

## Run with Docker — easiest

The whole web app (FastAPI backend + built React frontend + ffmpeg +
LibreOffice + CPU PyTorch) is packaged into a single image. You need
[Docker](https://docs.docker.com/engine/install/) with the Compose v2 plugin —
no Python, Node, conda or system packages on the host.

### 1  One-time setup

```bash
# OpenAI key (required) — Hume key optional, enables emotion detection
echo "OPENAI_API_KEY=sk-YOURKEYHERE" > .env

# Google Cloud credentials (required only for microphone / video STT)
mkdir -p api
cp ~/Downloads/your-google-creds.json api/
```

Secrets stay on the host: `.dockerignore` keeps `.env` and `api/` out of the
image; the key is injected as an environment variable and the credential JSON
is bind-mounted read-only at runtime.

### 2  Build & run

```bash
docker compose up --build        # first build takes a while (apt + pip deps)
```

Open **http://localhost:8000**. Browsers only expose the microphone on secure
origins, so always use `http://localhost:8000` (or tunnel the port — see the
[Web UI section](#2--run) below). If port 8000 is taken, change the host port
in the `ports:` mapping in `docker-compose.yml` (e.g. `"127.0.0.1:8800:8000"`).

> **Security** — the app has **no authentication**: anyone who can reach the
> port can read live transcripts and Q&A, download every uploaded document,
> spend your OpenAI credits and wipe/stop the server. That is why the compose
> file publishes the port on `127.0.0.1` only. For remote use prefer an SSH
> tunnel (`ssh -L 8000:localhost:8000 user@server`); only publish on all
> interfaces (`"8000:8000"`) on a network you trust.

Notes:

* **First PDF upload** downloads the surya OCR models (~2 GB) into the
  `model_cache` volume; the **first mp4** downloads whisper `base` (~140 MB).
  Both survive rebuilds. To pre-warm the caches:

  ```bash
  docker compose exec app python -c "import sys; sys.path.insert(0, 'vendor/mypackage'); from marker.models import create_model_dict; create_model_dict()"
  docker compose exec app python -c "import whisper; whisper.load_model('base')"
  ```

* **GPU (optional)** — only speeds up local OCR/whisper; the LLM, embeddings,
  TTS and streaming STT are remote APIs. Requires the NVIDIA driver +
  `nvidia-container-toolkit` on the host:

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
  ```

* **Remote access without a tunnel** — generate the self-signed certificate
  into `./certs` (openssl one-liner in the [Web UI section](#2--run)), then in
  `docker-compose.yml` uncomment the `./certs:/app/certs:ro` line **and**
  change the port mapping to `"8000:8000"` (this exposes the unauthenticated
  API to the network — see the security note above). Re-run
  `docker compose up -d` to recreate the container (a plain
  `docker compose restart` does **not** pick up compose-file changes): the
  server switches to HTTPS automatically and the microphone works from
  `https://<server-ip>:8000`.

* **Exit button**: the in-app Exit wipes the upload workspace and vector DB
  and stops the server — by design. With `restart: unless-stopped` the
  container comes back with a fresh, empty workspace.

* `constraints.docker.txt` freezes the pip dependency resolution that is known
  to work (e.g. the `protobuf==3.20.0` pin needs `google-cloud-speech 2.15.1`).
  If you change `requirements.txt`, regenerate it — the command is in the file
  header.

---

## Web UI (React frontend) — manual setup

The original PyQt6 GUI (`gui_beta.py`) has been refactored into a browser app:
a **FastAPI** backend (`backend/`) wraps the existing RAG / STT / video
pipelines and pushes realtime updates over WebSocket, and a **React (Vite)**
frontend (`frontend/`) reproduces the same layout (Transcripts | Q&A | Video
panels, control bar, push-to-talk, ROI selection on video, keyboard
shortcuts). The PyQt version still works and is untouched.

### 1  One-time setup

```bash
# Python deps (same venv/conda env as before)
pip install -r requirements.txt          # now also installs fastapi/uvicorn

# Build the frontend (requires Node.js >= 18)
cd frontend
npm install
npm run build
cd ..
```

Keys are configured exactly like the PyQt version: `OPENAI_API_KEY` in `.env`,
Google credential JSON in `./api/`.

Optional: add `HUME_API_KEY=...` to `.env` to enable speech-prosody emotion
detection in conversation mode — each recognized question is tagged with the
top Hume.ai emotion (`[Emotion: Curiosity]`) so the LLM can adapt its answer;
the Q&A panel shows the label as a chip. Without the key the feature is
silently disabled.

### 2  Run

```bash
python run_web.py        # serves API + built frontend on http://127.0.0.1:8000
```

If port 8000 is taken, pick another one: `WEB_PORT=8800 python run_web.py`.

**Audio is browser-side**: the microphone is captured in the browser and
streamed to the backend over WebSocket for Google streaming STT, and
Read-Aloud answers are synthesized server-side (OpenAI TTS) but played by
the browser. The machine running the backend needs no sound hardware.

Browsers only expose the microphone on secure origins (`https://` or
`http://localhost`), so the rule of thumb is simple — **always open the app
as `http://localhost:<port>`** and the microphone just works, no
certificates needed:

* **Backend on this machine (local run):** open
  `http://localhost:8000` (or your `WEB_PORT`) directly. Done.

* **Backend on a remote machine:** forward the port so it *becomes*
  localhost on your laptop — pick one:

  * **VS Code Remote-SSH (easiest).** Open the **PORTS** panel (next to
    the terminal), *Forward a Port* → e.g. `8800`, then open
    `http://localhost:8800` on your laptop. Ports started from a VS Code
    integrated terminal are usually forwarded automatically. Forwarding
    rides the existing SSH connection, so it also works when a network
    firewall blocks the port itself.
  * **Plain SSH tunnel** (same mechanism without VS Code):
    `ssh -L 8800:localhost:8800 user@server`, then `http://localhost:8800`.

* **Remote without any tunnel (direct HTTPS)** — only works if the network
  firewall lets you reach the port. Generate a self-signed certificate once
  on the server:

  ```bash
  mkdir -p certs
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -keyout certs/server.key -out certs/server.crt -subj "/CN=multi-rag"
  ```

  `run_web.py` detects `./certs` automatically, switches to HTTPS and
  listens on all interfaces — open `https://<server-ip>:<port>` and accept
  the certificate warning once (Advanced → Proceed). Anyone on your network
  can then reach the app, so use it on trusted LANs only. Set `WEB_SSL=0`
  to ignore the certificates and serve plain localhost HTTP again (do this
  when you switch back to tunneled/local access).

While STT is running, a **Share Speaker Audio** button appears: share a
tab/screen *with audio* (Chrome/Edge) and that audio is transcribed as the
"Speaker" channel, which also feeds the Correct Content checker.

### 3  Frontend development mode (hot reload)

```bash
uvicorn backend.main:app --reload        # terminal 1: API on :8000
cd frontend && npm run dev               # terminal 2: UI on :5173 (proxies to :8000)
```

### Keyboard shortcuts (same as PyQt)

`Space` toggle STT · `C` conversation · `R` read aloud · `A` correct content ·
`D` retrieval · `T` detail mode · `Y` concise mode · `S` summary · `E` exit ·
hold `Enter` (or the Voice Input button) for push-to-talk. Shortcuts are
inactive while typing in the question box.

---

## Run with Virtual Env (Windows / conda example)

1. Create a conda environment (Python 3.11.7 is recommended):  
   ```bash
   conda create -n achilles python=3.11.7
   ```

2. Activate the environment:  
   ```bash
   conda activate achilles
   ```

3. Install the project requirements:  
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the GUI:  
   ```bash
   python gui_beta.py
   ```

5. Add your keys:  
   * Paste the **OpenAI** key into a `.env` file (`OPENAI_API_KEY=...`).  
   * Create an `./api` folder and drop your Google credentials JSON inside.

> **Notice** – Before uploading `.docx`, `.pptx`, or `.xlsx` files, save and close any other work to avoid data‑loss.

> **Notice** - If show error: Error loading transcription into LLM: Descriptors cannot be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. If you cannot immediately regenerate your protos, some other possible workarounds are: 1. Downgrade the protobuf package to 3.20.x or lower. 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
>```bash
>pip uninstall protobuf
>pip install protobuf==3.20.0
>```
---

## Running on Ubuntu 20.04 – 24.04 (and other Debian‑based distros)

The project is now Linux‑friendly thanks to a LibreOffice‑based Office‑to‑PDF
converter and PortAudio fixes. Follow these steps:

### 1  System packages (one‑time)

```bash
sudo apt update
sudo apt install -y   python3.11 python3.11-venv build-essential   # Python 3.11 + compiler toolchain
sudo apt install -y   portaudio19-dev libportaudio2 libportaudiocpp0  # headers for PyAudio
sudo apt install -y   espeak libespeak1                              # runtime for pyttsx3 TTS
sudo apt install -y   libreoffice libreoffice-core fonts-liberation   # headless Office → PDF
```

*If `python3.11` isn’t in your repo, add the Deadsnakes PPA first:*  
`sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`

### 2  Virtual‑environment

```bash
cd ~/RAG-LLM-ARL-demo          # or wherever you cloned the repo
python3.11 -m venv venv        # creates ./venv
source venv/bin/activate       # prompt shows (venv)
python -V                      # → Python 3.11.x
```

### 3  Python dependencies (note, on linux, remove win32 from requirements.txt)

```bash
pip install --upgrade pip setuptools wheel
# remove Windows‑only deps if present
sed -i '/pywin32/d' requirements.txt
pip install -r requirements.txt
```

If `pyaudio` still fails, double‑check that `portaudio19-dev` is installed.

> **Notice** – If `playsound` fails to build with
> `OSError: could not get source code` while *Getting requirements to build
> wheel*, that is a known bug in the unmaintained `playsound 1.3.0` sdist
> (its `setup.py` breaks under modern setuptools, regardless of Python
> version — reproduced on 3.11.7). Pin the last version that ships a wheel:
>
> ```bash
> pip install playsound==1.2.2
> ```
>
> or change the `playsound` line in `requirements.txt` to
> `playsound==1.2.2` and re-run `pip install -r requirements.txt`.

### 4  Project setup

```bash
# OpenAI key
echo "OPENAI_API_KEY=sk-YOURKEYHERE" > .env

# Google credentials
mkdir -p api
cp ~/Downloads/your-google-creds.json api/
```
### 5  Office-to-PDF helper — already Linux-ready

`file_conversion/office_2_pdf.py` already ships the LibreOffice/FPDF-based
converter (no more `win32com`/`pywin32`), so **no manual edits are needed**
on a fresh clone — just make sure the LibreOffice packages from step 1 are
installed.

### 6  Run the GUI

```bash
python gui_beta.py
```

### 7  File‑conversion notes

* `file_conversion/office_2_pdf.py` now calls **LibreOffice** in headless mode
  – ensure the packages in step 1 are installed.
* Converted PDFs are moved to `./source/`.
* No more `win32com` / `pywin32` is required.

### 8  Playing nicely with `playsound`

`playsound==1.3.0` (the version pip resolves by default) fails to build with
modern setuptools on any recent Python — 3.11 included, not just 3.12+. Pin
the wheel-only release instead:

```bash
pip install playsound==1.2.2
```

If you ever need a maintained drop-in replacement, the fork also works:

```bash
pip install playsound@git+https://github.com/taconi/playsound
```

---

### Tested environments

* **Windows 11** (conda 23.11, Python 3.11.7)  
* **Ubuntu 22.04.4 LTS** (Python 3.11.9, LibreOffice 7.5)

Enjoy!
