import os
import queue
import subprocess
import threading
import textwrap
import time

from PyQt6.QtCore import QThread, pyqtSignal
from fpdf import FPDF

from stt_providers.assemblyai_provider import RealtimeAssemblyAISession


STOP_STREAM_AFTER_SECONDS = 240
CHUNK_SIZE = 4096
SAMPLE_RATE = 16000


class RealTimeStreamingTranscriptionThread(QThread):
    """
    Read PCM audio from ffmpeg and stream to AssemblyAI realtime STT.
    Preserves existing PyQt signal contract for GUI integration.
    """

    transcript_update = pyqtSignal(str)
    status_update = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, mp4_file_path: str, parent=None):
        super().__init__(parent)
        self.mp4_file_path = mp4_file_path
        self._stop_flag = False
        self._reader_finished = False
        self.chunk_stop = threading.Event()

        self.chunk_index = 1
        self.md_filename = os.path.basename(mp4_file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)

    def stop(self):
        self._stop_flag = True
        self.chunk_stop.set()

    def run(self):
        if not os.path.isfile(self.mp4_file_path):
            self.status_update.emit(f"Error: file does not exist {self.mp4_file_path}")
            self.finished_processing.emit()
            return

        os.makedirs("./source", exist_ok=True)
        self.status_update.emit("Start processing...")

        audio_queue = queue.Queue()

        reader_thread = threading.Thread(
            target=self._read_from_ffmpeg,
            args=(self.mp4_file_path, audio_queue),
            daemon=True,
        )
        reader_thread.start()

        try:
            while not self._stop_flag:
                self.chunk_stop.clear()
                pdf_filename = f"transcription_chunk_{self.chunk_index}.pdf"

                self._recognize_chunk(audio_queue, pdf_filename)
                self.chunk_index += 1

                if self._reader_finished and audio_queue.empty():
                    self.status_update.emit("Audio reading finished.")
                    break

            reader_thread.join(timeout=3)
        except Exception as exc:
            self.status_update.emit(f"Error during transcription: {exc}")
        finally:
            self.status_update.emit("Processing finished.")
            self.finished_processing.emit()

    def _read_from_ffmpeg(self, mp4_file_path, audio_queue):
        cmd = [
            "ffmpeg",
            "-i",
            mp4_file_path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-vn",
            "-",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        try:
            while not self._stop_flag:
                data = process.stdout.read(CHUNK_SIZE)
                if not data:
                    break

                audio_queue.put(data)
                duration_seconds = CHUNK_SIZE / (SAMPLE_RATE * 2)
                time.sleep(duration_seconds)
        finally:
            if process.stdout:
                process.stdout.close()
            process.wait()
            self._reader_finished = True

    def _recognize_chunk(self, audio_queue, pdf_filename: str):
        timer = threading.Timer(STOP_STREAM_AFTER_SECONDS, lambda: self.chunk_stop.set())
        timer.start()

        local_lines = []

        def handle_final_turn(text: str) -> None:
            recognized_text = text.strip()
            if not recognized_text:
                return

            local_lines.append(recognized_text)
            with open(self.md_file_path, "a", encoding="utf-8") as md_file:
                md_file.write("Transcript:")
                md_file.write(recognized_text + "\n")
            self.transcript_update.emit(recognized_text)

        session = RealtimeAssemblyAISession(
            sample_rate=SAMPLE_RATE,
            on_final=handle_final_turn,
            on_error=lambda err: self.status_update.emit(f"Realtime STT error: {err}"),
        )

        try:
            session.start()

            while not self.chunk_stop.is_set() and not self._stop_flag:
                try:
                    data = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if self._reader_finished and audio_queue.empty():
                        break
                    continue

                try:
                    session.stream(data)
                except Exception as exc:
                    self.status_update.emit(f"Streaming error: {exc}")
                    self.chunk_stop.set()
                finally:
                    try:
                        audio_queue.task_done()
                    except ValueError:
                        pass
        finally:
            timer.cancel()
            session.close()

            if local_lines:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                merged = "\n".join(local_lines)
                safe_text = merged.encode("latin-1", "replace").decode("latin-1")
                wrapped = textwrap.fill(safe_text, width=90)
                pdf.multi_cell(0, 10, wrapped)
                pdf.output(os.path.join("./source", pdf_filename))

            with audio_queue.mutex:
                audio_queue.queue.clear()
