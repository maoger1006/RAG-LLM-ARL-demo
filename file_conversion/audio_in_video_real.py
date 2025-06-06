import subprocess
import time
from PyQt6.QtCore import QThread, pyqtSignal
from google.cloud import speech
from google.oauth2 import service_account
import os
import glob

api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

credential = service_account.Credentials.from_service_account_file(json_files[0])

class RealTimeStreamingTranscriptionThread(QThread):
    # 定义信号，用于更新转录文本、状态信息和结束标志
    transcript_update = pyqtSignal(str)
    status_update = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, mp4_file_path, sample_rate=16000, chunk_size=4096, parent=None):
        """
        :param mp4_file_path: MP4 文件路径
        :param sample_rate: 音频采样率，默认 16000
        :param chunk_size: 每次读取的字节数
        """
        super().__init__(parent)
        self.mp4_file_path = mp4_file_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.md_filename = os.path.basename(mp4_file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)

    def generate_requests(self):
        """
        利用 ffmpeg 将 MP4 文件中的音频提取为 LINEAR16 格式的原始 PCM 数据，
        并分块发送，每块数据后等待对应时长以模拟实时播放速率。
        """
        ffmpeg_command = [
            "ffmpeg",
            "-i", self.mp4_file_path,
            "-f", "s16le",            # 输出原始 PCM 数据，16位小端
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",               # 单声道
            "-vn",                    # 不输出视频流
            "-"                       # 输出到标准输出
        ]
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while True:
            data = process.stdout.read(self.chunk_size)
            if not data:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)
            # 计算数据块对应的时长（每个样本 2 字节）
            duration = self.chunk_size / (self.sample_rate * 2)
            time.sleep(duration)
        process.stdout.close()
        process.wait()

    def run(self):
        client = speech.SpeechClient(credentials=credential)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-US",             # 根据需要设置语言
            enable_automatic_punctuation=True   # 启用自动标点
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True                # 若需要中间结果，设置为 True
        )

        self.status_update.emit("Starting real-time transcription...")
        requests = self.generate_requests()
        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        full_transcript = ""
        try:
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        # 累计最终确定的转录文本
                        
                        with open(self.md_file_path, "a", encoding = "utf-8") as md_file:
                            md_file.write("Transcript:")
                            md_file.write(result.alternatives[0].transcript + "\n")
                            
                        full_transcript += result.alternatives[0].transcript + "\n"
                        self.transcript_update.emit(result.alternatives[0].transcript)    
                            
                            
                        # self.status_update.emit("实时转录中...")
        except Exception as e:
            self.status_update.emit(f"转录过程中出错：{e}")
        self.status_update.emit("Realtime transcripts finished.")
        self.finished_processing.emit()