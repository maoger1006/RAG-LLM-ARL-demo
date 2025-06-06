import whisper
import os
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from dotenv import load_dotenv

def mse(imageA, imageB):
    # convert the images to numpy arrays and float type
    imageA = np.array(imageA).astype("float")
    imageB = np.array(imageB).astype("float")
    # calculate the mean squared error between the two images
    return np.mean((imageA - imageB) ** 2)



def video_transcripts_frames(file_path, coordinate_list = None):
    """
    Process a video file to extract its transcript and key frames, then generate a Markdown file with the results.
    """
    # get the video file name and create a markdown file name
    video_name = os.path.basename(file_path)
    md_filename = os.path.splitext(video_name)[0] + ".md"
    
    # get transcript from the video using Whisper model
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    transcript = result["text"]
    
    # write the content in Markdown format
    os.makedirs("./source", exist_ok=True)
    output_path = os.path.join("./source", md_filename)
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# Video Analysis for {video_name}\n\n")
        md_file.write("## Transcription\n\n")
        md_file.write(transcript)
        md_file.write("\n\n")
    
    print(f"Transcripts saved {output_path}")
    
    # Processing video frames
    # extract key frames from the video and convert them to base64 for GPT-4 analysis
    video = cv2.VideoCapture(file_path)
    base64Frames = []
    recorded_frames = []
    mse_thresh = 750
    # coordinate_list format [x, y, w, h]
    if coordinate_list:
        x, y, w, h = coordinate_list
        print("receive coordinate list")
    while video.isOpened():
        ret, frame = video.read()
        same_flag = False
        if not ret:
            break
        if coordinate_list:
            # get the cropped frame from the frame using the provided coordinates
            cropped_frame = frame[y:y+h, x:x+w]
        else:
            cropped_frame = frame
        
        if len(recorded_frames) == 0:
            # the first frame is always recorded
            recorded_frames.append(cropped_frame)
            _, buffer = cv2.imencode('.jpg', cropped_frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        else:
            last_recorded_frame = recorded_frames[-1]
            score = mse(last_recorded_frame, cropped_frame)
            
            if score > mse_thresh:
                
                for frame_recorded in recorded_frames:
                    # Check if the current frame is similar to any of the previously recorded frames
                    # This is to avoid duplicates in case of minor changes in the frame
                    if mse(frame_recorded, cropped_frame) < mse_thresh:
                        same_flag = True
                
                if not same_flag:

                    recorded_frames.append(cropped_frame)
                    _, buffer = cv2.imencode('.jpg', cropped_frame)
                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                
    video.release()
    print(len(base64Frames), "frames read.")
    # Save each captured frame in the ./source directory as a JPEG file
    output_dir = "./source/captured_frames"
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(recorded_frames):
        frame_filename = os.path.join(output_dir, f"captured_frame_{idx+1}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want you to describe. Generate a detailed description of each frame.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 8192,
    }
    
    result = client.chat.completions.create(**params)
    summary = result.choices[0].message.content
    print("Summary:", summary)
    
    # Use the topic as a header in the markdown file for key frames understanding
    with open(output_path, "a", encoding="utf-8") as md_file:
        md_file.write("## Topic\n\n")
        md_file.write(summary)
        md_file.write("\n")



  
if __name__ == '__main__':
    video_file = "./Test_files/Writing a Literature Review Meeting Recording.mp4"  
    # video_transcripts(video_file)
    video_transcripts_frames(video_file)


