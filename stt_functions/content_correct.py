from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def content_correct(content):
    """Get a direct response from the LLM with corrections if needed."""
    # Optionally, modify the prompt to instruct the model to correct errors.
    # prompt = f"Please analyze the following statement and concisely identify any common-sense mistakes (in no more than 10 words). If there are none, simply return 'correct'.\n\n{content}"
    prompt = f"""Follow these rules STRICTLY:
            1. If the input is EMPTY (no text), say 'correct'
            2. If the statement has OBVIOUS factual errors (e.g. 'sun rises from west'), 
            respond ONLY with the corrected phrase (MAX 10 words)
            3. For incomplete sentences/uncertain claims/no clear errors, say 'correct'
            4. DO NOT write explanations or prefixes like 'Correction:'

            Examples:
            Input: "Fire is cold" → Response: "Fire is hot"
            Input: "Cats can fly" → Response: "Cats cannot fly"
            Input: "" → Response: "correct"
            Input: "It might rain" → Response: "correct"

            Now analyze: "{content}"
            """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-4o",
    )
    
    return chat_completion.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    # Example statement to correct
    original_statement = "have to leave now, because I"
    corrected_response = content_correct(original_statement)
    print("Corrected response:", corrected_response)