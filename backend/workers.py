"""Plain-threading ports of the QThread workers used by the PyQt GUI.

- VideoFrameWorker          <- file_conversion/video_process_realtime.VideoProcessingThread
- VideoTranscriptionWorker  <- file_conversion/audio_in_video_real_unlimit.RealTimeStreamingTranscriptionThread
- run_video_summary         <- answer_request/video_summary.VideoSummaryThread

pyqtSignal(...).emit(x) becomes an on_xxx(x) callback; QTimer becomes
threading.Timer. The processing logic is unchanged.
"""

import base64
import glob
import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env", override=True)


def _noop(*args, **kwargs):
    pass


def mse(imageA, imageB):
    imageA = np.array(imageA).astype("float")
    imageB = np.array(imageB).astype("float")
    return np.mean((imageA - imageB) ** 2)


# ---------------------------------------------------------------- frames

class VideoFrameWorker(threading.Thread):
    """Capture distinct frames from an mp4, describe each with GPT-4o and
    summarize recent frames + transcripts."""

    def __init__(self, file_path, video_status, coordinate_list=None,
                 on_status=_noop, on_frame=_noop, on_description=_noop,
                 on_recent_summary=_noop, on_transcript_frame_finished=_noop,
                 on_finished=_noop):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.coordinate_list = coordinate_list
        self.md_filename = os.path.basename(file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)
        self.video_status = video_status
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.on_status = on_status
        self.on_frame = on_frame
        self.on_description = on_description
        self.on_recent_summary = on_recent_summary
        self.on_transcript_frame_finished = on_transcript_frame_finished
        self.on_finished = on_finished

    def run(self):
        self.on_status("Start Processing frames...")
        video = cv2.VideoCapture(self.file_path)
        recorded_frames = []
        mse_thresh = 750

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 16.0

        descriptions = []

        while video.isOpened():
            # recent video transcript + frame summary
            if self.video_status["video_transcript_flag"] and self.video_status["video_frame_flag"]:
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": f"Summarize the following video content, including recent frames descriptions and transcripts concisely within two sentences:\n{self.video_status['video_last_frame']}\n{self.video_status['video_last_transcript']}"
                    }
                ]
                params = {
                    "model": "gpt-4o",
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1000,
                }
                try:
                    response = self.client.chat.completions.create(**params)
                    recent_summary = response.choices[0].message.content.strip()
                except Exception as e:
                    recent_summary = f"Summary generation failed: {e}"

                self.on_recent_summary(recent_summary)
                self.on_transcript_frame_finished(True)

            ret, frame = video.read()
            if not ret:
                break

            if self.coordinate_list:
                x, y, w, h = self.coordinate_list
                cropped_frame = frame[y:y + h, x:x + w]
            else:
                cropped_frame = frame

            new_frame = False

            if not recorded_frames:
                recorded_frames.append(cropped_frame)
                new_frame = True
            else:
                last_frame = recorded_frames[-1]
                score = mse(last_frame, cropped_frame)
                if score > mse_thresh:
                    recorded_frames.append(cropped_frame)
                    new_frame = True

            if new_frame:
                _, buffer = cv2.imencode('.jpg', cropped_frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                self.on_frame(base64_frame)
                self.on_status("New frame captured and processed.")

                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            "These are frames from a video. Generate a detailed description of this frame:",
                            {"image": base64_frame, "resize": 768},
                        ],
                    },
                ]
                params = {
                    "model": "gpt-4o",
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 8192,
                }
                try:
                    result = self.client.chat.completions.create(**params)
                    frame_description = result.choices[0].message.content
                except Exception as e:
                    frame_description = f"Failed to generate the descrpion{e}"

                with open(self.md_file_path, "a", encoding="utf-8") as md_file:
                    md_file.write(f"Frame {len(recorded_frames)}:")
                    md_file.write(frame_description + "\n")

                self.on_description(frame_description)
                descriptions.append(frame_description)

            time.sleep(1 / fps)

        video.release()
        self.on_status("Video frames processing finished.")

        output_dir = "./source/captured_frames"
        os.makedirs(output_dir, exist_ok=True)
        for idx, frame in enumerate(recorded_frames):
            frame_filename = os.path.join(output_dir, f"captured_frame_{idx + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            self.on_status(f"Save frames:{frame_filename}")

        output_path = "./source/video_summary.md"
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Video Frame Descriptions\n\n")
            for i, desc in enumerate(descriptions, start=1):
                md_file.write(f"## Frame {i}\n{desc}\n\n")
        self.on_finished(output_path)


# ---------------------------------------------------------------- audio transcription

STOP_STREAM_AFTER_SECONDS = 240
CHUNK_SIZE = 4096
SAMPLE_RATE = 16000


class VideoTranscriptionWorker(threading.Thread):
    """Stream the audio track of an mp4 through Google Cloud Speech-to-Text."""

    def __init__(self, mp4_file_path, on_transcript=_noop, on_status=_noop, on_finished=_noop):
        super().__init__(daemon=True)
        from google.cloud import speech
        from google.oauth2 import service_account

        api_dir = os.path.join(os.getcwd(), "api")
        json_files = glob.glob(os.path.join(api_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

        self.speech = speech
        self.mp4_file_path = mp4_file_path
        self._stop_flag = False
        self.md_filename = os.path.basename(mp4_file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)

        self.credentials = service_account.Credentials.from_service_account_file(json_files[0])
        self.client = speech.SpeechClient(credentials=self.credentials)

        self.chunk_index = 1
        self.global_transcript = ""
        self.chunk_stop = threading.Event()

        self.on_transcript = on_transcript
        self.on_status = on_status
        self.on_finished = on_finished

    def stop(self):
        self._stop_flag = True
        self.chunk_stop.set()

    def run(self):
        if not os.path.isfile(self.mp4_file_path):
            self.on_status(f"Error: File not exit {self.mp4_file_path}")
            self.on_finished()
            return

        self.on_status("Start processing...")
        audio_queue = queue.Queue()

        reader_thread = threading.Thread(
            target=self._read_from_ffmpeg,
            args=(self.mp4_file_path, audio_queue),
            daemon=True
        )
        reader_thread.start()

        try:
            while not self._stop_flag:
                self.chunk_stop.clear()
                self._recognize_chunk(audio_queue)
                self.chunk_index += 1

                if self._stop_flag:
                    break

                if not reader_thread.is_alive() and audio_queue.empty():
                    self.on_status("Audio reading finished.")
                    break

            reader_thread.join()
        except Exception as e:
            self.on_status(f"Error during transcript: {e}")
        finally:
            self.on_status("Processing finished.")
            self.on_finished()

    def _read_from_ffmpeg(self, mp4_file_path, audio_queue):
        cmd = [
            "ffmpeg",
            "-i", mp4_file_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-vn",
            "-"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        while not self._stop_flag:
            data = process.stdout.read(CHUNK_SIZE)
            if not data:
                break
            audio_queue.put(data)

            duration = CHUNK_SIZE / (SAMPLE_RATE * 2)
            time.sleep(duration)

        process.stdout.close()
        process.wait()
        self._stop_flag = True

    def _recognize_chunk(self, audio_queue):
        speech = self.speech

        # threading.Timer replaces the QTimer of the Qt version
        timer = threading.Timer(STOP_STREAM_AFTER_SECONDS, self.chunk_stop.set)
        timer.start()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        def request_generator():
            while not self.chunk_stop.is_set():
                try:
                    data = audio_queue.get(timeout=0.5)
                    yield speech.StreamingRecognizeRequest(audio_content=data)
                except queue.Empty:
                    if self._stop_flag:
                        break
                    continue

        responses = self.client.streaming_recognize(
            streaming_config,
            requests=request_generator()
        )

        try:
            for response in responses:
                if self.chunk_stop.is_set():
                    break
                for result in response.results:
                    if result.is_final:
                        recognized_text = result.alternatives[0].transcript

                        with open(self.md_file_path, "a", encoding="utf-8") as md_file:
                            md_file.write("Transcript:")
                            md_file.write(recognized_text + "\n")
                        self.on_transcript(recognized_text)
        except Exception as e:
            self.on_status(f"[分段 {self.chunk_index}] 识别异常: {e}")
        finally:
            timer.cancel()
            with audio_queue.mutex:
                audio_queue.queue.clear()


# ---------------------------------------------------------------- summary

def run_video_summary(video_path, on_status=_noop, on_summary=_noop, on_finished=_noop):
    """Summarize the parsed video markdown plus every .md in ./source."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    on_status("summarizing ..")
    video_content = ""
    if os.path.isfile(video_path):
        try:
            with open(video_path, 'r', encoding='utf-8') as f:
                video_content = f.read()
        except Exception as e:
            video_content = f"[Read {video_path} Fail: {str(e)}]"
    else:
        video_content = f"[Warning: Can't find file {video_path}]"

    md_files = glob.glob(os.path.join("./source", "*.md"))
    all_md_contents = []
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                all_md_contents.append(f.read())
        except Exception as e:
            all_md_contents.append(f"[Read {md_file} Fail:{str(e)}]")

    combined_text = (video_content
                     + "\n\nOther content\n"
                     + "\n\n".join(all_md_contents))

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": (
                "Summarize the following video-related content (including frames descriptions and transcripts) "
                "in at most 2 sentences:\n\n"
                + combined_text
            ),
        }
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1000,
    }

    try:
        response = client.chat.completions.create(**params)
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Summary generation failed: {e}"

    on_summary(summary)
    on_finished("Finished summarizing")
