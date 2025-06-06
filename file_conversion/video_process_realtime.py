
import os
import cv2  # pip install opencv-python
import base64
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def mse(imageA, imageB):
    # Convert images to float arrays and compute the Mean Squared Error
    imageA = np.array(imageA).astype("float")
    imageB = np.array(imageB).astype("float")
    return np.mean((imageA - imageB) ** 2)

class VideoProcessingThread(QThread):
    # Define signals: status update, frame update (base64 encoded), description update, and finished processing result (Markdown file path)
    status_update = pyqtSignal(str)
    frame_update = pyqtSignal(str)  
    description_update = pyqtSignal(str)  # Added for real-time frame description updates
    finished_processing = pyqtSignal(str)
    recent_summary_update = pyqtSignal(str)  # Added for real-time summary updates
    transcript_frame_finished = pyqtSignal(bool)  # Added for real-time transcript and frame summary updates
    

    def __init__(self, file_path, video_status, coordinate_list=None, parent=None):
        """
        :param file_path: video file path
        :param coordinate_list: ROI coordinaes [x, y, w, h], optional
        """
        super().__init__(parent)
        self.file_path = file_path
        self.coordinate_list = coordinate_list
        self.md_filename = os.path.basename(file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)
        self.video_status = video_status

    def run(self):
        self.status_update.emit("Start Processing frames...")
        video = cv2.VideoCapture(self.file_path)
        recorded_frames = []
        mse_thresh = 750

        # Obtain the video frame rate, default to 16 if retrieval fails
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 16.0

        # Used for accumulating descriptions of each frame
        descriptions = []
        
        while video.isOpened():
            
            # this is for recent video transcript and frame summary
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
                    response = client.chat.completions.create(**params)
                    recent_summary = response.choices[0].message.content.strip()
                except Exception as e:
                    recent_summary = f"Summary generation failed: {e}"
                
                
                self.recent_summary_update.emit(recent_summary)
                self.transcript_frame_finished.emit(True)
                
                
            ret, frame = video.read()
            if not ret:
                break

            # Crop the frame based on the provided coordinates
            if self.coordinate_list:
                x, y, w, h = self.coordinate_list
                cropped_frame = frame[y:y+h, x:x+w]
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
                    # Uncomment the following lines if you want to avoid duplicates
                    # duplicate = any(mse(prev_frame, cropped_frame) < mse_thresh for prev_frame in recorded_frames)
                    # if not duplicate:
                    recorded_frames.append(cropped_frame)
                    new_frame = True

            if new_frame:
                # Encoding the frame to JPEG format and converting to base64 string
                _, buffer = cv2.imencode('.jpg', cropped_frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                # Send the base64 encoded frame to the GUI for display
                self.frame_update.emit(base64_frame)
                self.status_update.emit("New frame captured and processed.")

                # Call the OpenAI API to generate a description for the current frame
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
                    result = client.chat.completions.create(**params)
                    frame_description = result.choices[0].message.content
                except Exception as e:
                    frame_description = f"Failed to generate the descrpion{e}"
                
                # Send he description to the GUI for display
                
                with open(self.md_file_path, "a", encoding = "utf-8") as md_file:
                    md_file.write(f"Frame {len(recorded_frames)}:")
                    md_file.write(frame_description + "\n")
                
                self.description_update.emit(frame_description)
                descriptions.append(frame_description)
                
            new_frame = False
            # Wait for the duration of one frame based on the video FPS
            time.sleep(1 / fps)

        video.release()
        self.status_update.emit("Video frames processing finished.")

        # Save each captured frame in the ./source/captured_frames directory as a JPEG file
        output_dir = "./source/captured_frames"
        os.makedirs(output_dir, exist_ok=True)
        for idx, frame in enumerate(recorded_frames):
            frame_filename = os.path.join(output_dir, f"captured_frame_{idx+1}.jpg")
            cv2.imwrite(frame_filename, frame)
            self.status_update.emit(f"Save frames:{frame_filename}")

        # Write the descriptions to a Markdown file
        output_path = "./source/video_summary.md"
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Video Frame Descriptions\n\n")
            for i, desc in enumerate(descriptions, start=1):
                md_file.write(f"## Frame {i}\n{desc}\n\n")
        self.finished_processing.emit(output_path)