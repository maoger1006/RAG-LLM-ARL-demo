from openai import OpenAI
from dotenv import load_dotenv
import time
import os

load_dotenv(dotenv_path=".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_direct_response(question):
    """Get a direct response from the LLM."""

    # query = "What's RAG?"
    # question = "If you are confident about the following content. You give A, otherwise you give B. The content is: " + query

    # print(question)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        model="gpt-4o",
    )
    
    return chat_completion.choices[0].message.content
