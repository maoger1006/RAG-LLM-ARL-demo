"""File helpers copied from utils/utils_beta.py without the PyQt6 imports,
so the web backend can run headless."""

import os
import shutil
import stat
import time

from fpdf import FPDF


def force_remove_readonly(func, path, exc_info):
    """Delete a read-only file by changing its permissions."""
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception as e:
        print(f"❌ Force remove failed for {path}: {e}")


def safe_delete_dir(path, retries=3):
    """Try to delete an entire directory, retrying a few times in case of errors."""
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

    os.makedirs(path, exist_ok=True)


def clear_source_directory():
    """Remove all contents in the source and docs directories, then recreate them."""
    directories = ["./source/", "./docs/"]
    for d in directories:
        safe_delete_dir(d)


def save_transcription(history_transcript, current_chunk_number):
    """Save the transcription to a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Conversation {current_chunk_number}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, history_transcript)
    pdf.output(f"./source/transcription_chunk_{current_chunk_number}.pdf")
    print(f"Transcription saved as transcription_chunk_{current_chunk_number}.pdf")
