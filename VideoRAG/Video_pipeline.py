import os 
import sys
from PyQt6.QtCore import QThread, pyqtSignal
import requests

class VideoUploadThread(QThread):
    # 返回 video_id, video_save_name, video_filename
    finished = pyqtSignal(str, str, str)  # video_id, video_save_name, video_filename
    error = pyqtSignal(str)
    
    def __init__(self, video_path, server_ip="10.160.200.119", port=8000):
        super().__init__()
        self.video_path = video_path
        self.server_ip = server_ip
        self.port = port
    
    def run(self):
        url = f"http://{self.server_ip}:{self.port}/upload_video"
        try:
            with open(self.video_path, "rb") as f:
                files = {"file": f}
                r = requests.post(url, files=files, timeout=180)
            r.raise_for_status()
            resp = r.json()
            # 获取三个关键字段
            video_id = resp.get("video_id", "")
            video_save_name = resp.get("video_save_name", "")
            video_filename = resp.get("video_filename", "")
            self.finished.emit(video_id, video_save_name, video_filename)
        except Exception as e:
            self.error.emit(str(e))
        
        