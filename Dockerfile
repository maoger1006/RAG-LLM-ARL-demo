# =============================================================================
# Multi-RAG web app (run_web.py -> FastAPI backend + built React frontend).
#
#   docker compose up --build                 # CPU image (default)
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
#
# The PyQt6 desktop GUI (gui_beta.py) is NOT part of this image — audio is
# captured browser-side, so the container needs no sound hardware.
# =============================================================================

# ---------- Stage 1: build the React frontend (frontend/dist is gitignored) ----------
FROM node:20-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim

# torch is only used by marker/surya OCR and whisper, both auto-fall back to
# CPU. The default CPU wheel keeps the image several GB smaller than the CUDA
# one; docker-compose.gpu.yml overrides this to the regular PyPI index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# git ................ requirements.txt installs whisper from git+https
# build-essential +
# portaudio19-dev .... only to compile PyAudio (desktop-only dep kept in
#                      requirements.txt; never imported by the web backend)
# ffmpeg ............. backend/workers.py video transcription + whisper
# libreoffice-* ...... headless Office->PDF (file_conversion/office_2_pdf.py)
# fonts-* ............ avoid blank glyphs (incl. CJK) in converted PDFs
# libgl1/libglib2.0-0  non-headless opencv-python from requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        portaudio19-dev \
        ffmpeg \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        fonts-liberation \
        fonts-noto-cjk \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first so the requirements step below finds it already satisfied
# instead of pulling the default (CUDA) wheel transitively via marker-pdf
RUN pip install torch --index-url ${TORCH_INDEX_URL}

# constraints.docker.txt freezes the dependency resolution proven to work in
# the dev venv (protobuf==3.20.0 forces old google-cloud-speech/grpcio-status
# versions that a fresh resolve may not find on its own).
# openai-whisper unconditionally depends on triton (~0.5 GB GPU compiler) on
# linux/x86_64, but only imports it inside CUDA-only code paths — strip it
# from the CPU image in the same layer so the space is actually reclaimed
# (the GPU build keeps it: CUDA torch itself depends on triton).
COPY requirements.txt constraints.docker.txt ./
RUN pip install -r requirements.txt -c constraints.docker.txt \
    && case "${TORCH_INDEX_URL}" in */whl/cpu) pip uninstall -y triton ;; esac

# pre-download the gpt2 tokenizer used for text splitting (RAG/rag.py) so the
# first document upload does not depend on huggingface.co being reachable
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

# vendored patched marker-pdf copy — file_conversion/pdf_split.py prepends it
# to sys.path to shadow the pip-installed package; pre-download its PDF font
# so the vendor tree does not have to be writable at runtime
COPY vendor/ vendor/
RUN python -c "import sys; sys.path.insert(0, '/app/vendor/mypackage'); \
from marker.util import download_font; download_font()"

# application code — only the modules the web path imports
# (gui_beta.py / utils/ / Test_files are PyQt-desktop-only)
COPY run_web.py ./
COPY backend/ backend/
COPY RAG/ RAG/
COPY answer_request/ answer_request/
COPY file_conversion/ file_conversion/
COPY stt_functions/ stt_functions/
COPY docker/healthcheck.py docker/healthcheck.py

# built SPA — backend/main.py mounts <repo>/frontend/dist at / when it exists
COPY --from=frontend-build /build/frontend/dist frontend/dist/

# without TLS certs run_web.py defaults to 127.0.0.1, which would be
# unreachable through the container port mapping
ENV WEB_HOST=0.0.0.0 \
    WEB_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]

# Secrets are runtime-only (see docker-compose.yml):
#   OPENAI_API_KEY / HUME_API_KEY  -> environment variables
#   Google service-account JSON   -> bind-mounted into /app/api/
CMD ["python", "run_web.py"]
