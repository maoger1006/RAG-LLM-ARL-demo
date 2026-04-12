# Abalation study: with or without audio transcript to the video

import argparse
import os
import json
import shutil
import ast

try:
    from .llm_grades import get_llm_grade
    from .generate_answer import build_db
    from .path_config import (
        default_answer_file,
        default_analysis_dir,
        default_question_file,
        ensure_dir,
        ensure_file,
    )
except ImportError:
    from llm_grades import get_llm_grade
    from generate_answer import build_db
    from path_config import (
        default_answer_file,
        default_analysis_dir,
        default_question_file,
        ensure_dir,
        ensure_file,
    )

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
    
    
DEFAULT_QUESTION_FILE = str(default_question_file())
DEFAULT_ANSWER_FILE = str(default_answer_file())
DEFAULT_VIDEO_ANALYSIS_SOURCE = str(default_analysis_dir("video_analysis_output_0.5fps_only"))
DEFAULT_START_NUM = int(os.getenv("EVAL_START_NUM", "0"))
DEFAULT_QUESTION_NUM = int(os.getenv("EVAL_QUESTION_NUM", "1998"))
DEFAULT_CHUNK_SIZE = int(os.getenv("EVAL_CHUNK_SIZE", "2048"))
DEFAULT_K_FOR_RAG = int(os.getenv("EVAL_K_FOR_RAG", "5"))


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


def benchmark(
    start_num: int,
    question_num: int,
    question_file: str,
    answer_file: str,
    video_analysis_source: str,
    chunk_size: int,
    k_for_rag: int,
) -> None:
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
    
    end_index = min(question_num, len(question_data))
    for i in range(start_num, end_index):
        
        video_name = question_data[i]['video_name']
        
        if video_name != current_video:
            # If the video name has changed, clean the database and source files

            analyzer = build_db(
                video_name,
                base_dir=video_analysis_source,
                chunk_size=chunk_size,
            )
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

        std_answer = ""
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
        llm_answer,_,_,_ = analyzer.generate_answer(question, k=k_for_rag)
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
    parser = argparse.ArgumentParser(description="Run MMBench video evaluation.")
    parser.add_argument("--question-file", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--answer-file", default=DEFAULT_ANSWER_FILE)
    parser.add_argument("--analysis-source", default=DEFAULT_VIDEO_ANALYSIS_SOURCE)
    parser.add_argument("--start-num", type=int, default=DEFAULT_START_NUM)
    parser.add_argument("--question-num", type=int, default=DEFAULT_QUESTION_NUM)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--k-for-rag", type=int, default=DEFAULT_K_FOR_RAG)
    args = parser.parse_args()

    question_file = ensure_file(
        args.question_file,
        "Question JSON",
        "--question-file",
        "MMBENCH_Q_JSON",
    )
    answer_file = ensure_file(
        args.answer_file,
        "Answer JSON",
        "--answer-file",
        "MMBENCH_A_JSON",
    )
    analysis_source = ensure_dir(
        args.analysis_source,
        "Video analysis source",
        "--analysis-source",
        "EVAL_ANALYSIS_DIR",
    )

    benchmark(
        start_num=args.start_num,
        question_num=args.question_num,
        question_file=question_file,
        answer_file=answer_file,
        video_analysis_source=analysis_source,
        chunk_size=args.chunk_size,
        k_for_rag=args.k_for_rag,
    )
 