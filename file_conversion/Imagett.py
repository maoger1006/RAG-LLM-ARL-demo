import base64
from openai import OpenAI
from fpdf import FPDF
import os
import textwrap 
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def image_to_pdf(image_path):
    """Convert an image to a PDF and add GPT-4 analysis."""
    
    pdf = FPDF()
    pdf.add_page()
    
    # Set a simple font (Arial)
    pdf.set_font("Arial", "B", 16)

    # Extract image name and define output file
    image_name = os.path.basename(image_path)
    pdf_filename = os.path.splitext(image_name)[0] + ".pdf"

    # Add image title
    pdf.multi_cell(0, 10, image_name, align="C")
    pdf.ln(10)

    # Function to encode image in Base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Encode image
    base64_image = encode_image(image_path)

    # GPT-4 image analysis
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What can you tell me about this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )

    # Get response and ensure safe encoding
    response_text = response.choices[0].message.content
    response_text = response_text.encode("latin-1", "ignore").decode("latin-1")  # Skip unsupported characters

    # Add GPT-4 output to PDF
    pdf.multi_cell(0, 10, response_text)

    # Ensure the output directory exists
    os.makedirs("./source", exist_ok=True)

    # Save the PDF
    output_path = f"./source/{pdf_filename}"
    pdf.output(output_path)

    print(f"Transcription saved to {output_path}")
    

def image_to_txt(image_path):
    """Convert an image to a text file with GPT-4 analysis."""

   
    image_name = os.path.basename(image_path)
    txt_filename = os.path.splitext(image_name)[0] + ".txt"

    
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What can you tell me about this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )

    
    response_text = response.choices[0].message.content
    response_text = response_text.encode("latin-1", "ignore").decode("latin-1")  

    
    os.makedirs("./source", exist_ok=True)

    
    output_path = f"./source/{txt_filename}"

    
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Image Name: {image_name}\n\n")
        txt_file.write(response_text)

    print(f"Analysis saved to {output_path}")