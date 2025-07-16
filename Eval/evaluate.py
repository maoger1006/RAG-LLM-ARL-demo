# Abalation study: with or without audio transcript to the video

import os 
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import re
# from video_process import process_video_combined
from llm_grades import get_llm_grade
from generate_answer import generate_answer, build_db
from video_process_without_audio import process_video_combined
# from video_process_async import process_video_combined_async
from video_process_async_insert import process_video_combined_async
import json
import shutil
from RAG_LLM.RAG.rag import RAG_pipeline
import ast
import asyncio

CP = ['Video Style', 'Video Scene', 'Video Emotion', 'Video Topic' ] # Coarse_Perception
HL = ['Hallucination'] # Hallucination
FP_S = ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting']  # Fine-grained Perception single instance
FP_C = ['Human Interaction', 'Human-object Interaction','Spatial Relationship']


LR = ['Mathematical Calculation', 'Structuralized Image-Text Understanding'] # Logical Reasoning
AR = ['Identity Reasoning', 'Functional Reasoning', 'Physical Property'] # Attribute Reasoning
RR = ['Social Relation', 'Physical Relation', 'Natural Relation'] # Relation Reasoning
CSR = ['Common Sense Reasoning'] # Common Sense Reasoning
TR = ['Future Prediction', 'Causal Reasoning', 'Counterfactual Reasoning'] # Temporal Reasoning

# Concatenate all subcategory lists into a single list
All_category = CP + HL + FP_S + FP_C + LR + AR + RR + CSR + TR

# Initialize an empty dictionary to store scores
scores_dict = {}

# Iterate through the combined list of all subcategories
for subcategory in All_category:
    # Add the subcategory as a key to the dictionary
    # Initialize the score to an empty list to store multiple values
    scores_dict[subcategory] = []
    
    
question_file = "/home/mingyang/video_benchmark/MMBench-Video/MMBench-Video_q.json"
answer_file = "/home/mingyang/video_benchmark/MMBench-Video/MMBench-Video_a.json"

VIDEO_ANALYSIS_SOURCE = "/home/mingyang/video_benchmark/video_analysis_output_0.5fps_only"
START_NUM = 0
QUESTION_NUM = 1998
CHUNK_SIZE = 2048
K_for_RAG = 5


def get_unique_video_names(json_file_path: str) -> list[str] | None:

    unique_names_set = set() # Use a set internally for efficient uniqueness check


    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Load the entire JSON structure (expected to be a list)
        data = json.load(f)

        # Ensure the loaded data is a list
        if not isinstance(data, list):
            print(f"Error: JSON file '{json_file_path}' does not contain a list.")
            return None # Indicate critical error

        # Iterate through each dictionary in the list
        for item in data:
            # Check if the item is a dictionary and has the 'video_name' key
            if isinstance(item, dict):
                video_name = item.get('video_name') # Use .get() for safety
                # Add to set only if video_name is a non-empty string
                if isinstance(video_name, str) and video_name:
                    unique_names_set.add(video_name)
            else:
                # Optional: Print a warning if an item is not a dictionary
                print(f"Warning: Skipping non-dictionary item in JSON list: {item}")

        # Convert the set of unique names to a list before returning
    return list(unique_names_set)

def clean_db_and_source(docs_dir='docs/', analysis_dir='./video_analysis_output'):
    """Cleans directories using shutil."""
    print("INFO: Cleaning directories...")
    for dir_path in [docs_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed: {dir_path}")
            except OSError as e:
                print(f"Error removing {dir_path}: {e}")
    try:
        # Recreate necessary directories
        os.makedirs(os.path.join(docs_dir, 'chroma'), exist_ok=True)
        # os.makedirs(analysis_dir, exist_ok=True)
        print("Clean directories recreated.")
    except OSError as e:
        print(f"Error creating directories: {e}")


def benchmark(START_NUM: int, QUESTION_NUM: int, question_file, answer_file) -> None:
    """
    Benchmark the performance of the Conversational Aid System (CAS) in video understanding tasks.
    """
    #Initialize 
    # clean_db_and_source()
    load_or_initialize_scores()   # <<< 新加这一行
    
    global scores_dict
    
    # load_or_initialize_scores()   # <<< 新加这一行
    with open(question_file, 'r', encoding='utf-8') as f:
        # Load the entire JSON structure (expected to be a list)
        question_data = json.load(f)
    
    with open(answer_file, 'r', encoding='utf-8') as f:
        # Load the entire JSON structure (expected to be a list)
        answer_data = json.load(f)
    # video_list = get_unique_video_names(question_file)
    current_video = None
    processed_video = 0
    clean_db_and_source()
    
    for i in range(START_NUM, QUESTION_NUM):
        
        video_name = question_data[i]['video_name']
        
        if video_name != current_video:
            # If the video name has changed, clean the database and source files
            
            video_path = "/home/mingyang/video_benchmark/MMBench-Video/video/" + video_name + ".mp4"
            # process_video_combined_async(video_path)
            # asyncio.run(process_video_combined_async(video_path))
            analyzer = build_db(video_name, base_dir= VIDEO_ANALYSIS_SOURCE, chunk_size=CHUNK_SIZE)
            current_video = video_name
            processed_video += 1
        
        question = question_data[i]['question']
        print(f"Question is: {question}")
        target_question_id = question_data[i]['question_id']
        # dimensions = question_data[i]['dimensions']
        dimensions_raw = question_data[i]['dimensions']
        try:
            dimensions = ast.literal_eval(dimensions_raw)
            if not isinstance(dimensions, list):
                print(f"Warning: Parsed dimensions are not a list: {dimensions}")
                dimensions = []
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing dimensions: {e}")
            dimensions = []        
                
        # Load the questions from the question file

        for item in answer_data:
            # Check if the item is a dictionary and has the 'question_id' key
            if isinstance(item, dict):
                question_id = item.get('question_id')
                # Check if the question_id matches the target
                if question_id == target_question_id:
                    # Retrieve the 'answer' safely using .get()
                    std_answer = item.get('answer')
                    # Return the answer (could be None if 'answer' key is missing)
                    # Or return None immediately if answer is None/empty string?
                    # Let's return it as is for now.
                    # return std_answer        
        
        # Generate the answer for the user-provided question
        llm_answer,_,_,_ = analyzer.generate_answer(question, k=K_for_RAG)
        # llm_answer = generate_answer(question, video_name)
        
        # Save llm_answer to a JSON file based on question_id
        answers_file = 'generated_answers.json'
        if os.path.exists(answers_file):
            try:
                with open(answers_file, 'r', encoding='utf-8') as f:
                    saved_answers = json.load(f)
            except json.JSONDecodeError:
                saved_answers = {}
        else:
            saved_answers = {}

        saved_answers[target_question_id] = llm_answer

        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(saved_answers, f, ensure_ascii=False, indent=4)
            
            
        print(f"LLM Answer: {llm_answer}")
        
        score = get_llm_grade(question, std_answer, llm_answer)
        print(f"Quetion: {i}, Score: {score}")
        
        for dim in dimensions:
            if dim in All_category:
                # Append the score to the corresponding subcategory in the dictionary
                scores_dict[dim].append(score)
                
        with open('scores_results.json', 'w', encoding='utf-8') as f:
            json.dump(scores_dict, f, ensure_ascii=False, indent=4) 


    print(f"Benchmark completed. Processed {processed_video} videos.")
                


def load_or_initialize_scores(scores_file='scores_results.json'):
    """Load existing scores if the file exists, otherwise initialize a new scores dictionary."""
    global scores_dict
    
    if os.path.exists(scores_file):
        print(f"INFO: Loading existing scores from {scores_file}...")
        try:
            with open(scores_file, 'r', encoding='utf-8') as f:
                scores_dict = json.load(f) 
            print("INFO: Existing scores loaded.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load scores file due to error: {e}. Initializing empty scores.")
            scores_dict = {subcategory: [] for subcategory in All_category} 
    else:
        print("INFO: No existing scores file found. Initializing new scores.")
        scores_dict = {subcategory: [] for subcategory in All_category}

# # for i in range(QUESTION_NUM):

if __name__ == "__main__":
    # Load questions from the question file

    # Run the benchmark using the loaded questions
    benchmark(START_NUM, QUESTION_NUM, question_file, answer_file)
 