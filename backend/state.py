import threading


class AppState:
    """Server-side equivalent of the instance attributes of the PyQt ConvoAid widget."""

    def __init__(self):
        self.current_chunk_number = 0  # + 1 when Start STT and Stop STT, no change when submit question
        self.history_transcript = ""
        self.rag_instance = None
        self.response_type = "concise mode"  # "concise mode" | "detail mode"
        self.history_length = 3  # Number of previous interactions to consider

        self.loaded_files = {}  # files already ingested into the RAG (path -> mtime)
        self.history_text = []  # history conversation awareness

        # toggles that used to be radio buttons in the Qt control bar
        self.read_aloud = False
        self.correct_content = False
        self.use_retrieval = True  # "Vector Retrieval (Adaptive)"

        self.stt_running = False
        self.conversation_running = False
        self.voice_inputting = False
        self.reading_text = False  # True while TTS playback is in progress

        self.voice_thread = None
        self.voice_transcript = ""

        self.video_path = ""  # markdown file the video parser writes to
        self.video_status = {
            "video_transcript_flag": False,
            "video_frame_flag": False,
            "video_last_transcript": "",
            "video_last_frame": "",
        }

        self.RAG_thresh = 0.5  # adaptive RAG similarity threshold

        # refresh_rag / answer generation may be triggered concurrently from
        # video + STT worker threads; the Qt version serialized them on the
        # main thread, we serialize them with a lock instead.
        self.rag_lock = threading.RLock()

    def settings_dict(self):
        return {
            "response_type": self.response_type,
            "read_aloud": self.read_aloud,
            "correct_content": self.correct_content,
            "use_retrieval": self.use_retrieval,
            "stt_running": self.stt_running,
            "conversation_running": self.conversation_running,
        }


state = AppState()
