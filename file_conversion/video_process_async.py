import cv2
import base64
import os
import openai
import time
from dotenv import load_dotenv
import subprocess
from openai import AsyncOpenAI
import asyncio
import pathlib

# --- Configuration ---
load_dotenv(dotenv_path=".env", override=True)
client_async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use a synchronous client for the audio part as the library recommends
client_sync = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VISION_MODEL = "gpt-4.1-mini" # gpt-4.1-mini is not a valid model name, gpt-4o is a great choice
SUMMARY_MODEL = "gpt-4.1-mini"
TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_OUTPUT_DIR = "./source"
SECONDS_PER_FRAME_SAMPLE = 2
CONCURRENCY = 15

###
# New function to extract audio with FFmpeg and transcribe it with Whisper
###
def extract_and_transcribe_audio(video_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Extracts audio from video, transcribes it, and cleans up the temp file.
    This is a synchronous function intended to be run in a separate thread.
    """
    print("Starting audio extraction and transcription...")
    temp_audio_path = os.path.join(output_dir, "temp_audio.mp3")
    transcript_text = None
    error_message = None

    try:
        # 1. Extract audio using FFmpeg
        # The '-y' flag overwrites the file if it exists. '-vn' means no video.
        command = f"ffmpeg -i \"{video_path}\" -y -vn -acodec libmp3lame -q:a 2 \"{temp_audio_path}\""
        print("Extracting audio with FFmpeg...")
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print("Audio extracted successfully.")

        # 2. Transcribe the extracted audio file
        print(f"Transcribing audio file '{os.path.basename(temp_audio_path)}'...")
        with open(temp_audio_path, "rb") as audio_file:
            start_time = time.time()
            transcription_response = client_sync.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file
            )
            end_time = time.time()
            transcript_text = transcription_response.text
            print(f"Transcription successful ({end_time - start_time:.2f}s).")

    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg error during audio extraction: {e.stderr.decode()}"
        print(error_message)
    except openai.OpenAIError as e:
        error_message = f"OpenAI API error during transcription: {e}"
        print(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred during audio processing: {e}"
        print(error_message)
    finally:
        # 3. Clean up temporary audio file
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except OSError as e:
            print(f"Warning: Could not remove temporary audio file '{temp_audio_path}': {e}")

    return transcript_text, error_message

async def describe_frame_async(b64: str, frame_num: int, ts_sec: float, sem: asyncio.Semaphore) -> dict:
    """Send one Vision prompt. Semaphore controls max concurrent calls."""
    prompt = [{"role": "user", "content": [
        "Describe this video frame concisely and capture the key information:",
        {"image": b64, "resize": 768}
    ]}]
    async with sem:
        try:
            res = await client_async.chat.completions.create(
                model=VISION_MODEL,
                messages=prompt,
                max_tokens=200, # Reduced for single frame
                temperature=0.1
            )
            desc = res.choices[0].message.content.strip()
            return {"frame_num": frame_num, "timestamp": round(ts_sec, 2), "description": desc}
        except Exception as e:
            return {"frame_num": frame_num, "timestamp": round(ts_sec, 2), "description": f"Error: {type(e).__name__}: {e}"}

async def sample_and_describe_frames_async(video_path: str) -> list[dict]:
    """Samples frames and describes them in parallel."""
    print("Starting frame sampling and description...")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback for videos with invalid FPS metadata
        print("Warning: Cannot determine FPS, defaulting to 30.")
        fps = 30

    every_n = int(round(fps * SECONDS_PER_FRAME_SAMPLE)) or 1
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    frame_idx = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % every_n != 0:
            continue

        ts_sec = (frame_idx / fps) # More accurate timestamp calculation
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            continue
        b64 = base64.b64encode(buf).decode("utf-8")
        tasks.append(asyncio.create_task(describe_frame_async(b64, frame_idx, ts_sec, sem)))

    video.release()
    print(f"Submitting {len(tasks)} frames for description...")
    results = await asyncio.gather(*tasks)
    print("Frame description complete.")
    return results

###
# New function to generate the final summary from combined text
###
async def summarise_text(text: str) -> str:
    """Generates a final summary from the combined transcript and frame descriptions."""
    print("Generating final summary...")
    prompt = [
        {"role": "system", "content": "You are an expert video analyst. You will be given the full audio transcript and a series of timestamped visual descriptions from a video. Your task is to synthesize all this information into a single, coherent summary paragraph."},
        {"role": "user", "content": f"Please summarize the following video content, combining the audio transcript and visual descriptions:\n\n{text}"}
    ]
    resp = await client_async.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=prompt,
        max_tokens=500,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

###
# Main orchestrator function, now updated to handle all steps
###
async def process_video_combined_async(video_path: str, output_dir: str = DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # --- Run audio transcription and frame description in parallel ---
    # Use asyncio.to_thread to run the synchronous audio function
    audio_task = asyncio.to_thread(extract_and_transcribe_audio, video_path, output_dir)
    frames_task = sample_and_describe_frames_async(video_path)

    # The return_exceptions=True flag is crucial
    results = await asyncio.gather(audio_task, frames_task, return_exceptions=True)

    # --- Process results and print any exceptions for debugging ---
    audio_result, frames_result = results
    
    transcript_text, transcript_err = (None, None)
    frame_descriptions, frame_err = (None, None)

    # Check the audio task result
    if isinstance(audio_result, Exception):
        # Print the actual exception to the console
        print(f"\n--- 🔴 ERROR IN AUDIO TASK 🔴 ---\n{audio_result}\n----------------------------------\n")
        transcript_err = str(audio_result)
    else:
        transcript_text, transcript_err = audio_result

    # Check the frame description task result
    if isinstance(frames_result, Exception):
        # Print the actual exception to the console
        print(f"\n--- 🔴 ERROR IN FRAME TASK 🔴 ---\n{frames_result}\n---------------------------------\n")
        frame_err = str(frames_result)
    else:
        frame_descriptions = frames_result

    # --- Combine text for summary ---
    # (This part remains the same)
    combined_text = ""
    if transcript_text:
        combined_text += "---TRANSCRIPT---\n" + transcript_text + "\n\n"
    if frame_descriptions:
        combined_text += "---VISUALS---\n"
        for item in frame_descriptions:
            combined_text += f"Time {item['timestamp']:.2f}s: {item['description']}\n"

    # --- Generate Summary ---
    # (This part remains the same)
    summary = ""
    if combined_text.strip():
        summary = await summarise_text(combined_text)
        print("Summary generated successfully.")
    
    # --- Write final Markdown file ---
    # (This part remains the same)
    base_name = os.path.basename(video_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    output_md_path = os.path.join(output_dir, f"{file_name_no_ext}_analysis.md")

    print(f"\nSaving combined results to: {output_md_path}")
    # ... (the rest of the file writing logic is the same) ...
    try:
        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Video Analysis: {base_name}\n\n")

            if summary:
                md_file.write("## Summary\n\n")
                md_file.write(summary + "\n\n")
                md_file.write("---\n\n")

            md_file.write("## Frame-by-Frame Description\n\n")
            if frame_err:
                md_file.write(f"**Error during frame sampling:**\n```\n{frame_err}\n```\n\n")
            elif not frame_descriptions:
                md_file.write("*No frames were sampled or described.*\n\n")
            else:
                for item in frame_descriptions:
                    md_file.write(f"**Time {item['timestamp']:.2f}s:** {item['description']}\n")
                md_file.write("\n")

            md_file.write("---\n\n")
            md_file.write("## Full Transcript\n\n")
            if transcript_err:
                md_file.write(f"**Error during transcription:**\n```\n{transcript_err}\n```\n\n")
            elif not transcript_text or not transcript_text.strip():
                md_file.write("*Transcription failed or video has no audio.*\n")
            else:
                md_file.write(transcript_text + "\n")

        print("Combined analysis saved successfully.")
    except IOError as e:
        print(f"Error writing Markdown file '{output_md_path}': {e}")


if __name__ == "__main__":
    start_time = time.time()
    # Replace with the path to your video file
    video_to_process = "C:\\Users\\22770\\OneDrive - Johns Hopkins\\GitHub\\RAG-LLM-ARL-demo-2\\Test_files\\ucIDF_ZHdhY.mp4"

    if os.path.exists(video_to_process):
        print(f"\n--- Starting Combined Processing for: {video_to_process} ---")
        asyncio.run(process_video_combined_async(video_to_process))
        print("\n--- Processing Finished ---")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds.")
    else:
        print(f"\nSkipping example: Video file '{video_to_process}' not found.")  