import os
import shutil
from fpdf import FPDF

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit, QFrame, QRadioButton, QButtonGroup, QTextEdit, QSplitter,QFileDialog, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QSize, QRect, QUrl
from PyQt6.QtGui import QKeySequence, QShortcut, QPainter, QPen
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer
import cv2
import time
import stat
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QAudioFormat
import requests


def upload_mp4_with_roi(file_path):
    # Using the provided file_path. No re-selection is necessary.
    if file_path and file_path.endswith(".mp4"):
        dialog = VideoPreviewDialog(file_path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            roi = dialog.get_roi()
            if roi:
                
                # print("Return ROI Coordinates:", roi.x(), roi.y(), roi.width(), roi.height())
                
                return list(roi)  
                
            else:
                print("no choose ROI。")
        else:
            print("video preview dialog was canceled.")
    else:
        print("Invalid file path or not an MP4 file.")
        

class OverlayWidget(QWidget):
    # signal when the ROI is selected
    roiSelected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        #
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.startPos = None
        self.currentRect = QRect()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.startPos = event.position().toPoint()
            self.currentRect = QRect(self.startPos, QSize())
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.startPos:
            self.currentRect = QRect(self.startPos, event.position().toPoint()).normalized()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.startPos:
            self.currentRect = QRect(self.startPos, event.position().toPoint()).normalized()
            self.startPos = None
            self.update()
            self.roiSelected.emit(self.currentRect)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.currentRect)
        

# use QDialog to preview video and select ROI
class VideoPreviewDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Preview and ROI Selection")
        self.resize(1280, 720)
        self.roi = None

        layout = QVBoxLayout(self)
        self.videoWidget = QVideoWidget(self)
        layout.addWidget(self.videoWidget)

        # Initialize the media player
        self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
        self.mediaPlayer.play()

        # Add a transparent overlay for ROI selection
        self.overlay = OverlayWidget(self.videoWidget)
        self.overlay.setGeometry(self.videoWidget.rect())
        self.overlay.raise_()  # Ensure overlay is on top
        self.overlay.roiSelected.connect(self.on_roi_selected)


        # Ensure the overlay resizes with the video widget
        self.videoWidget.resizeEvent = self.on_video_resize
        
    # @staticmethod
    def get_video_size(self, file_path):
        cap = cv2.VideoCapture(file_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        return width, height

    def on_video_resize(self, event):
        self.overlay.setGeometry(self.videoWidget.rect())
        QVideoWidget.resizeEvent(self.videoWidget, event)

    def on_roi_selected(self, rect):
        video_width, video_height = self.get_video_size(self.mediaPlayer.source().toLocalFile())
        print("Video Size:", video_width, video_height)

        # Get the current display widget size
        display_width = self.videoWidget.width()
        display_height = self.videoWidget.height()
        
        print("Display Size:", display_width, display_height)

        # Calculate the scaling factors
        scale_x = video_width / display_width
        scale_y = video_height / display_height

        # Convert the rectangle coordinates to the original video size
        true_x = int(rect.x() * scale_x)
        true_y = int(rect.y() * scale_y)
        true_w = int(rect.width() * scale_x)
        true_h = int(rect.height() * scale_y)

        self.roi = [true_x, true_y, true_w, true_h]
        
        print("Chosen ROI:", self.roi)
        self.accept()

    def get_roi(self):
        return self.roi
    
    
def init_ui(instance):
    """Initialize the user interface."""
    # Main layout
    main_layout = QVBoxLayout(instance)
            
    
    # Title label
    title_label = QLabel("Multi-RAG")  #Conversational Aid System
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("font-size:20px; font-weight:bold; background-color: lightgrey; color:black;")
    main_layout.addWidget(title_label)
    
    # Control frame
    control_frame = QFrame()
    control_frame_layout = QHBoxLayout(control_frame)
    control_frame.setFrameShape(QFrame.Shape.StyledPanel)
    control_frame.setStyleSheet("background-color: lightgrey; border:1px solid black; color:black;")
    main_layout.addWidget(control_frame)
                
    # In init_ui() - inside control_frame_layout:
    instance.toggle_stt_button = QPushButton("Start STT")
    instance.toggle_stt_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.toggle_stt_button.setCheckable(True)  # Make it toggleable
    instance.toggle_stt_button.clicked.connect(instance.toggle_stt)
    control_frame_layout.addWidget(instance.toggle_stt_button)
    
    # Conversation Mode button
    instance.toggle_conv_button = QPushButton("Talk with AI (Emotion Detection)")
    instance.toggle_conv_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.toggle_conv_button.setCheckable(True)  # Make it toggleable
    instance.toggle_conv_button.clicked.connect(instance.toggle_conversation)
    control_frame_layout.addWidget(instance.toggle_conv_button)
    
    # Read Aloud Radio Button
    instance.read_aloud_button = QRadioButton("Read Aloud")
    instance.read_aloud_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.read_aloud_button.setEnabled(True)
    instance.read_aloud_button.setAutoExclusive(False)  # Disable mutual exclusivity
    # instance.read_aloud_button.toggled.connect(instance.set_response_type)
    control_frame_layout.addWidget(instance.read_aloud_button)

    # Correct Content Radio Button
    instance.correct_button = QRadioButton("Correct Content")
    instance.correct_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.correct_button.setEnabled(True)
    instance.correct_button.setAutoExclusive(False)  # Disable mutual exclusivity
    # Optionally connect to your correction function:
    # instance.correct_button.toggled.connect(instance.correct_button)
    control_frame_layout.addWidget(instance.correct_button)


    instance.upload_file_button = QPushButton("File Management")
    instance.upload_file_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.upload_file_button.clicked.connect(instance.upload_file)
    control_frame_layout.addWidget(instance.upload_file_button)
    
    # video parser button
    instance.video_parser_button = QPushButton("Video Parser")
    instance.video_parser_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.video_parser_button.clicked.connect(instance.video_parser)
    control_frame_layout.addWidget(instance.video_parser_button)
    
    instance.summary_button = QPushButton("Summary")
    instance.summary_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.summary_button.clicked.connect(instance.video_summary)
    control_frame_layout.addWidget(instance.summary_button)

    instance.VideoRAG_button = QPushButton("Video RAG")
    instance.VideoRAG_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.VideoRAG_button.clicked.connect(instance.VideoRAG_manage)
    control_frame_layout.addWidget(instance.VideoRAG_button)    
    

    instance.exit_button = QPushButton("Exit")
    instance.exit_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.exit_button.clicked.connect(instance.close)
    control_frame_layout.addWidget(instance.exit_button)


    # Question input and response area
    system_display_frame = QFrame()
    system_display_layout = QHBoxLayout(system_display_frame)
    main_layout.addWidget(system_display_frame)

    # Question area
    Transcripts_area = QVBoxLayout()
    system_display_layout.addLayout(Transcripts_area, 5)  # Set stretch factor to 4


    Transcripts_label = QLabel("Transcripts")
    Transcripts_label.setStyleSheet("font-size:14px; font-weight:bold;")
    Transcripts_area.addWidget(Transcripts_label)

    instance.Transcripts_area = QTextEdit()
    instance.Transcripts_area.setReadOnly(True)
    instance.Transcripts_area.setStyleSheet("font-size:12px; background-color:white; color:black;")
    Transcripts_area.addWidget(instance.Transcripts_area)

    # Answer area
    QA_area = QVBoxLayout()
    system_display_layout.addLayout(QA_area, 5)

    QA_label = QLabel("Q&A")
    QA_label.setStyleSheet("font-size:14px; font-weight:bold;")
    QA_area.addWidget(QA_label)

    instance.QA_area = QTextEdit()
    instance.QA_area.setReadOnly(True)
    instance.QA_area.setStyleSheet("font-size:12px; background-color:white; color:black;")
    QA_area.addWidget(instance.QA_area)
    
   
    
    # Video(the most right)的垂直布局
    video_parser_area = QVBoxLayout()
    system_display_layout.addLayout(video_parser_area, 5)  # Set stretch factor = 5

    # ========== Summary==========
    summary_label = QLabel("Video Summary")
    summary_label.setStyleSheet("font-size:14px; font-weight:bold;")
    video_parser_area.addWidget(summary_label)

    instance.summary_area = QTextEdit()
    instance.summary_area.setReadOnly(True)
    instance.summary_area.setStyleSheet("font-size:12px; background-color:white; color:black;")
    video_parser_area.addWidget(instance.summary_area)

    # ========== Video==========
    video_label = QLabel("Video Transcripts / Frames")
    video_label.setStyleSheet("font-size:14px; font-weight:bold;")
    video_parser_area.addWidget(video_label)

    instance.video_area = QTextEdit()
    instance.video_area.setReadOnly(True)
    instance.video_area.setStyleSheet("font-size:12px; background-color:white; color:black;")
    video_parser_area.addWidget(instance.video_area)
    
    
    
    # Question input frame
    question_frame = QFrame()
    question_frame.setFrameShape(QFrame.Shape.StyledPanel)
    question_frame.setStyleSheet("background-color: lightgrey; border:1px solid black; color:black;")
    question_layout = QGridLayout(question_frame)
    main_layout.addWidget(question_frame)

    question_label = QLabel("Response Type:")
    question_label.setStyleSheet("font-size:14px; color:black;")
    question_layout.addWidget(question_label, 0, 0)

    # Radio buttons for response type: Concise Mode
    instance.concise_button = QRadioButton("Concise Mode")
    instance.concise_button.setChecked(True)  # Default option
    instance.concise_button.toggled.connect(instance.set_response_type)
    question_layout.addWidget(instance.concise_button, 0, 1)

    # Radio buttons for response type: Detail Mode
    instance.detailed_response_button = QRadioButton("Detail Mode")
    instance.detailed_response_button.toggled.connect(instance.set_response_type)
    question_layout.addWidget(instance.detailed_response_button, 0, 2)
    

    instance.response_type_group = QButtonGroup()
    instance.response_type_group.addButton(instance.concise_button)
    instance.response_type_group.addButton(instance.detailed_response_button)

    # Radio buttons for response type: if need retrieval
    instance.need_retrival_button = QRadioButton("Vector Retrieval (Adaptive)")
    
    instance.video_rag_answer_button = QRadioButton("Video-RAG Retrieval")
    
    instance.mode_button_group = QButtonGroup()
    instance.mode_button_group.addButton(instance.need_retrival_button)
    instance.mode_button_group.addButton(instance.video_rag_answer_button)

    instance.need_retrival_button.setChecked(True)  # Default option
    # instance.need_retrival_button.toggled.connect(instance.set_response_type)
    question_layout.addWidget(instance.need_retrival_button, 0, 3)
    
    # Radio buttons for the Graph Retrieval Retrieval
    # instance.graph_retrieval_button = QRadioButton("Graph Retrieval")
    # question_layout.addWidget(instance.graph_retrieval_button, 0, 4)

    # Add Video-RAG answer radio button
    question_layout.addWidget(instance.video_rag_answer_button, 0, 4)
    
    
    
    # Question input field
    instance.question_entry = QLineEdit()
    instance.question_entry.setStyleSheet("font-size:14px; background-color:white; color:black;")
    question_layout.addWidget(instance.question_entry, 1, 0, 1, 2)
    
    
    # Voice Input Button (Push-to-Talk)
    instance.voice_input_button = QPushButton("Voice Input")
    instance.voice_input_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.voice_input_button.setCheckable(True)  # Enables toggling effect
    instance.voice_input_button.pressed.connect(instance.start_voice_input)  # Pressing starts STT
    instance.voice_input_button.released.connect(instance.stop_voice_input)  # Releasing stops STT
    question_layout.addWidget(instance.voice_input_button, 1, 2)  # Place it before Submit button


    instance.submit_question_button = QPushButton("Submit")
    instance.submit_question_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")
    instance.submit_question_button.clicked.connect(instance.submit_question)
    question_layout.addWidget(instance.submit_question_button, 1, 3)
    
    return main_layout  # Return the layout for the main window


# Append messages to the conversation area
def append_system_message(message, Area):
    """Append a system message to the conversation area (left-aligned)."""
    message_html = f"<p style='text-align:left; color:black;'><b>System:</b> {message}</p>"
    Area.append(message_html)

    
def append_recognition(message, Transcripts_area):
    """
    Append a recognition message (with included label) to the conversation area.
    Makes the label part bold and keeps text left-aligned.
    Expected message format: "[Label]: Actual text"
    """
    label = "Unknown:" # Default label if parsing fails
    content = message  # Default content if parsing fails

    # Try to parse the label and content
    # Option 1: Simple split (assumes first colon separates label/content)
    parts = message.split(':', 1) # Split only on the first colon
    if len(parts) == 2:
        label = parts[0].strip() + ":" # Include colon, strip potential whitespace
        content = parts[1].strip()     # Strip leading/trailing whitespace from content
    else:
        # Fallback if no colon found, treat the whole message as content without a specific label
        # Or you could keep the original message as content and use the "Unknown:" label
        print(f"Warning: Could not parse label from message: '{message}'")
        content = message # Keep original message if format is unexpected
        label = "" # No label in this case

    # Construct the HTML - Make the parsed label bold
    # Add color:black; as previously, though black is usually default
    if label: # Only add bold tag if label was parsed
        message_html = f"<p style='text-align:left; color:black;'><b>{label}</b> {content}</p>"
    else: # If no label was parsed, just append content
         message_html = f"<p style='text-align:left; color:black;'>{content}</p>"

    Transcripts_area.append(message_html)

def append_recognition_speaker(message, Transcripts_area):
    """Append a recognition message to the conversation area (left-aligned)."""
    message_html = f"<p style='text-align:left; color:black;'><b>Speaker:</b> {message}</p>"
    Transcripts_area.append(message_html)
    
def append_recognition_user(message, Transcripts_area):
    """Append a recognition message to the conversation area (left-aligned)."""
    message_html = f"<p style='text-align:left; color:black;'><b>User:</b> {message}</p>"
    Transcripts_area.append(message_html)
    
        
def append_conversation(QA_area, question, answer):
    """Append question (right-aligned) and answer (left-aligned) to the conversation area."""
    conversation_html = f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">  <!-- 总体块之间的间距 -->
        <div style="flex: 1; text-align:left; color:blue; padding-right: 10px; margin-bottom: 5px;">  <!-- Q 部分间距 -->
            <b>Q:</b> {question}
        </div>
        <div style="flex: 1; text-align:left; color:green; padding-left: 10px; margin-top: 5px;">  <!-- A 部分间距 -->
            <b>A:</b> {answer}
        </div>
    </div>
    """
    QA_area.append(conversation_html)
    
    
    
def append_Video(head_name, message, video_parser_area):
    """Append a system message to the conversation area (left-aligned)."""
    message_html = f"<p style='text-align:left; color:black;'><b>{head_name}:</b> {message}</p>"
    video_parser_area.append(message_html)   
    

def force_remove_readonly(func, path, exc_info):
    """
    Delete a read-only file by changing its permissions.
    """
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception as e:
        print(f"❌ Force remove failed for {path}: {e}")

def safe_delete_dir(path, retries=3):
    """
    Try to delete an entire directory, retrying a few times in case of errors.
    """
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                print(f"🧹 Attempting to delete: {path}")
                shutil.rmtree(path, onerror=force_remove_readonly)
                print(f"✅ Deleted: {path}")
            break
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {path}: {e}")
            time.sleep(0.5)
    else:
        print(f"❌ Final failure: Could not delete {path} after {retries} attempts.")

    # 重建空目录
    os.makedirs(path, exist_ok=True)

def clear_source_directory():
    """
    Remove all contents in the source and docs directories robustly, then recreate empty directories.
    """
    directories = ["./source/", "./docs/"]
    for d in directories:
        safe_delete_dir(d)

def clean_server_workdir(server_ip="", port=8000):
    url = f"http://{server_ip}:{port}/clean_workdir"
    try:
        r = requests.post(url, timeout=2)  
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[Warning] Could not connect to server: {e}")
        return None

def get_videorag_answer(question, server_ip="", port=8000):
    url = f"http://{server_ip}:{port}/query"
    payload = {"question": question}
    try:
        r = requests.post(url, data=payload, timeout=120) #timeout
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[Warning] Could not connect to server: {e}")
        return None

def save_transcription(history_transcript, current_chunk_number):
    """Save the transcription to a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Conversation {current_chunk_number}", ln=True, align="C")
    pdf.ln(10)  # Add a line break
    pdf.multi_cell(0, 10, history_transcript)
    pdf.output(f"./source/transcription_chunk_{current_chunk_number}.pdf")
    print(f"Transcription saved as transcription_chunk_{current_chunk_number}.pdf") 
    