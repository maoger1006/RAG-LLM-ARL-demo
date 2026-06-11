"""FastAPI backend for the Multi-RAG web frontend.

Run with:  python run_web.py            (serves the built React app too)
Dev mode:  uvicorn backend.main:app --reload   +   cd frontend && npm run dev
"""

import asyncio
import json
import os
import signal
import threading

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import core, tts
from backend.audio import AudioSession
from backend.bus import bus
from backend.state import state

app = FastAPI(title="Multi-RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("./source", exist_ok=True)
os.makedirs("./docs", exist_ok=True)


@app.on_event("startup")
async def on_startup():
    bus.set_loop(asyncio.get_running_loop())


# ---------------------------------------------------------------- websocket

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await bus.register(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; the client never needs to send
    except WebSocketDisconnect:
        bus.unregister(ws)


@app.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket):
    """Browser microphone / system-audio stream.

    First message: JSON {"target": "transcript"|"question"|"conversation",
    "channel": 1|2}. Then binary PCM16 mono 16 kHz frames. A text "end"
    (or simply disconnecting) flushes the recognizer and finalizes.
    """
    await ws.accept()
    session = None
    try:
        config = json.loads(await ws.receive_text())
        session = await asyncio.to_thread(
            AudioSession, state, bus,
            config.get("target", "transcript"), config.get("channel", 1),
        )
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                session.feed(message["bytes"])
            elif message.get("text") == "end":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        core.system_msg(bus, "transcripts", f"Audio session error: {e}")
    finally:
        if session is not None:
            await asyncio.to_thread(session.close)


@app.get("/api/tts/{tts_id}")
async def get_tts(tts_id: str):
    audio = tts.get_audio(tts_id)
    if audio is None:
        return Response(status_code=404)
    return Response(content=audio, media_type="audio/mpeg")


# ---------------------------------------------------------------- models

class QuestionRequest(BaseModel):
    question: str


class SettingsRequest(BaseModel):
    response_type: str | None = None
    read_aloud: bool | None = None
    correct_content: bool | None = None
    use_retrieval: bool | None = None


class VideoParseRequest(BaseModel):
    filename: str
    roi: list[int] | None = None  # [x, y, w, h] in original video pixels


# ---------------------------------------------------------------- state / settings

@app.get("/api/state")
async def get_state():
    return state.settings_dict()


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    if req.response_type is not None:
        state.response_type = req.response_type
    if req.read_aloud is not None:
        state.read_aloud = req.read_aloud
    if req.correct_content is not None:
        state.correct_content = req.correct_content
    if req.use_retrieval is not None:
        state.use_retrieval = req.use_retrieval
    core.publish_settings(state, bus)
    return state.settings_dict()


# ---------------------------------------------------------------- Q&A

@app.post("/api/question")
async def submit_question(req: QuestionRequest):
    try:
        answer = await asyncio.to_thread(core.answer_question, state, bus, req.question)
        return {"question": req.question, "answer": answer}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        core.system_msg(bus, "qa", f"Error generating answer: {e}")
        return {"error": f"Error generating answer: {e}"}


# ---------------------------------------------------------------- upload / video

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    path = core.save_upload(file.filename, data)

    if path.endswith(".mp4"):
        # the frontend must show the ROI selector first, then call
        # /api/video/process (File Management) or /api/video/parse (Video Parser)
        return {"ok": True, "filename": os.path.basename(path), "needs_roi": True}

    threading.Thread(target=core.process_upload, args=(state, bus, path), daemon=True).start()
    return {"ok": True, "filename": os.path.basename(path), "needs_roi": False}


@app.post("/api/video/process")
async def video_process_endpoint(req: VideoParseRequest):
    """Offline processing for mp4 uploaded through File Management."""
    path = os.path.join("./source", os.path.basename(req.filename))
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File not found: {req.filename}"}
    threading.Thread(target=core.process_video_offline,
                     args=(state, bus, path, req.roi), daemon=True).start()
    return {"ok": True}


@app.post("/api/video/parse")
async def video_parse(req: VideoParseRequest):
    """Realtime Video Parser: frame descriptions + streaming transcription."""
    path = os.path.join("./source", os.path.basename(req.filename))
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File not found: {req.filename}"}
    try:
        core.system_msg(bus, "qa", f"File '{os.path.basename(path)}' uploaded successfully to './source/'.")
        core.start_video_parse(state, bus, path, req.roi)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/video/summary")
async def video_summary():
    core.start_video_summary(state, bus)
    return {"ok": True}


# ---------------------------------------------------------------- exit

@app.post("/api/exit")
async def exit_app():
    """Port of closeEvent(): clear ./source and ./docs, then stop the server."""
    core.clear_workspace()

    def shutdown():
        os.kill(os.getpid(), signal.SIGINT)

    threading.Timer(0.5, shutdown).start()
    return {"ok": True}


# ---------------------------------------------------------------- static files

# uploaded videos are served for the ROI preview player
app.mount("/source", StaticFiles(directory="./source"), name="source")

# serve the built React app when frontend/dist exists (production mode)
_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
