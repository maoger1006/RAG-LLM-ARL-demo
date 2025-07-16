"""
QUESTIONS: CLASSIFICATION
"""

import os 
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import re


CP = ['Video Style', 'Video Scene', 'Video Emotion', 'Video Topic' ] # Coarse_Perception
HL = ['Hallucination'] # Hallucination
FP_S = ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting']  # Fine-grained Perception single instance
FP_C = ['Human Interaction', 'Human-object Interaction','Spatial Relationship']

LR = ['Mathematical Calculation', 'Structuralized Image-Text Understanding'] # Logical Reasoning
AR = ['Identity Reasoning', 'Functional Reasoning', 'Physical Property'] # Attribute Reasoning
RR = ['Social Relation', 'Physical Relation', 'Natural Relation'] # Relation Reasoning
CSR = ['Common Sense Reasoning'] # Common Sense Reasoning
TR = ['Future Prediction', 'Causal Reasoning', 'Counterfactual Reasoning'] # Temporal Reasoning



load_dotenv(dotenv_path="./.env", override=True)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_client = None
try:
    # The client automatically looks for the OPENAI_API_KEY environment variable.
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Perform a simple test call to verify the key (optional but recommended)
    openai_client.models.list() 
    print("OpenAI client initialized successfully using gpt-4o.")
except OpenAIError as e:
    print(f"Error initializing OpenAI client or invalid API key: {e}")
    print("Please ensure the OPENAI_API_KEY environment variable is set correctly.")
except Exception as e:
    print(f"An unexpected error occurred during OpenAI client setup: {e}")
    

PROMPT_EVALUATION = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a
                    given correct answer. The question itself, the correct 'groundtruth' answer, and
                    the candidate answer will be provided to you. Your assessment should range from
                    0 to 3, based solely on the semantic similarity between the groundtruth and the
                    candidate answer, disregarding any grammatical differences. A rating of 0 suggests
                    no similarity, implying the candidate answer is entirely incorrect. A rating of
                    1 suggests low similarity, meaning the candidate answer is largely incorrect. A
                    rating of 2 suggests high similarity, meaning the candidate answer is largely
                    correct. Lastly, a rating of 3 indicates complete similarity, which means the
                    candidate answer is entirely correct. Your response should be a single integer from
                    0, 1, 2, or 3.
                    Question: [QUESTION]
                    Groundtruth answer: [ANNOTATED ANSWER]
                    Candidate answer: [CANDIDATE ANSWER]
                    Your response:
                    """
                    
                    
def get_llm_grade(question: str, standard_answer: str, candidate_answer: str) -> int:
    """
    Gets a 0-3 similarity grade from OpenAI's gpt-4o. Returns -1 on error.
    """
    if not openai_client:
        print("Error: OpenAI Client is not available or failed to initialize.")
        return -1

    try:
        # 1. Format the prompt
        prompt = PROMPT_EVALUATION.replace("[QUESTION]", question)
        prompt = prompt.replace("[ANNOTATED ANSWER]", standard_answer)
        prompt = prompt.replace("[CANDIDATE ANSWER]", candidate_answer)

        # 2. Call the OpenAI API using gpt-4o
        response = openai_client.chat.completions.create(
            model="gpt-4", # Specify the GPT-4o model （GPT 4 Turbo）
            messages=[
                # Optional: You could add a system message here if needed
                # {"role": "system", "content": "You are an evaluation assistant responding with only a single digit 0-3."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Set temperature to 0 for deterministic output
            max_tokens=5      # Expecting only a single digit, so keep tokens low
        )

        response_text = response.choices[0].message.content

        if not response_text:
             print("Error: GPT-4o returned an empty response.")
             return -1

        # 3. Parse the response (using regex is safer)
        match = re.search(r'\b([0-3])\b', response_text.strip())
        if match:
            score = int(match.group(1))
            print(f"GPT-4o raw response: '{response_text.strip()}', Parsed score: {score}")
            return score
        else:
            print(f"Error: Could not find score (0-3) in GPT-4o response: '{response_text.strip()}'")
            return -1

    except OpenAIError as e:
        print(f"An OpenAI API error occurred: {e}")
        return -1
    except Exception as e:
        # Catch other potential errors (network, parsing issues)
        print(f"An unexpected error occurred during grading: {e}")
        return -1

# --- Example Usage ---
if __name__ == "__main__":
    # Make sure your OPENAI_API_KEY is set in your environment variables!

    if openai_client: # Only run examples if client initialized
        q = "Is Silicon Valley Bank facing an opportunity or a challenge now?"
        std_ans = "Silicon Valley Bank encountered a financial turmoil and fell into a liquidity crisis."
        cand_ans_good = "Silicon Valley Bank is facing a challenge, as it is experiencing a liquidity crisis and has been closed by regulatory authorities."
        cand_ans_bad = "Berlin is a major city in Europe."
        cand_ans_partial = "The primary city is Paris."

        print("\n--- Evaluating Good Answer (using gpt-4o) ---")
        score1 = get_llm_grade(q, std_ans, cand_ans_good)
        print(f"Final Score: {score1}") # Expecting 2 or 3

        # print("\n--- Evaluating Bad Answer (using gpt-4o) ---")
        # score2 = get_llm_grade(q, std_ans, cand_ans_bad)
        # print(f"Final Score: {score2}")  # Expecting 0 or 1

        # print("\n--- Evaluating Partial Answer (using gpt-4o) ---")
        # score3 = get_llm_grade(q, std_ans, cand_ans_partial)
        # print(f"Final Score: {score3}") # Expecting 1 or 2
    else:
        print("\nSkipping examples because OpenAI client failed to initialize.")
        print("Please check your API key and network connection.")                    
                    

        