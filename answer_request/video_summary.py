
import os
import cv2  # pip install opencv-python
import base64
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from openai import OpenAI
from dotenv import load_dotenv
import glob

load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VideoSummaryThread(QThread):
    # Summarize the video frames and transcripts using LLM and feedback to the UI
    status_update = pyqtSignal(str)
    frame_update = pyqtSignal(str)  
    summary_update = pyqtSignal(str)  # update the summary of the video content
    recent_summary_update = pyqtSignal(str)  # update the recent summary of the video content
    transcript_frame_finished = pyqtSignal(bool)  # update the transcript frame
    finished_processing = pyqtSignal(str)
    
    def __init__(self, video_path, parent=None):
        """
        :param video_content: The content of the video to be summarized        
        """
        super().__init__(parent)
        self.video_path = video_path

        # with open(video_path, 'r', encoding='utf-8') as file:
        #     self.video_content = file.read()
        
    def run(self):
        self.status_update.emit("summarizing ..")
        video_content = ""
        if os.path.isfile(self.video_path):
            try:
                with open(self.video_path, 'r', encoding='utf-8') as f:
                    video_content = f.read()
            except Exception as e:
                video_content = f"[Read {self.video_path} Fail: {str(e)}]"
        else:
            video_content = f"[Warning: Can't find file {self.video_path}]"

        # 2) Find all .md files in the ./source/ directory
        md_files = glob.glob(os.path.join("./source", "*.md"))
        all_md_contents = []
        for md_file in md_files:
            #
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_md_contents.append(content)
            except Exception as e:
                all_md_contents.append(f"[Read {md_file} Fail:{str(e)}]")

        # 3) Combine the video content and all .md contents into a single string
        combined_text = (video_content 
            + "\n\nOther content\n" 
            + "\n\n".join(all_md_contents)
        )


        # 4) prompt
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

        # Emit the summary and processing completion to the UI
        self.summary_update.emit(summary)
        self.finished_processing.emit("Finished summarizing")
        

