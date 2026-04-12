import argparse
import asyncio
import ast
import functools
import json
import os
import shutil
from collections import defaultdict

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

DEFAULT_QUESTION_FILE = str(default_question_file())
DEFAULT_ANSWER_FILE = str(default_answer_file())
DEFAULT_ANALYSIS_SOURCE = str(default_analysis_dir("video_analysis_output_with_audio_1fps_all"))
DEFAULT_START_NUM = int(os.getenv("EVAL_START_NUM", "0"))
DEFAULT_K_FOR_RAG = int(os.getenv("EVAL_K_FOR_RAG", "5"))


async def run_sync(func, *args, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kw))


CONCURRENCY = 8          # tune to taste, stay < org-wide limit
SEM         = asyncio.Semaphore(CONCURRENCY)

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

async def grade_one(analyzer, q_item, std_answer, k_for_rag: int):
    """
    • call LLM to answer
    • call LLM to grade
    • return (dimensions, score)
    """
    question   = q_item["question"]
    dims_raw   = q_item["dimensions"]

    # --- generate answer (async or threaded) -----------------------------
    async with SEM:   # limit concurrent OpenAI calls
        try:
            llm_answer, *_ = await run_sync(analyzer.generate_answer, question, k=k_for_rag)
        except Exception as e:
            print("ERR answer:", e)
            llm_answer = ""

    # --- grade answer ----------------------------------------------------
    async with SEM:
        try:
            score = await run_sync(get_llm_grade, question, std_answer, llm_answer)
        except Exception as e:
            print("ERR grade:", e)
            score = 0

    # --- parse dimensions list safely ------------------------------------
    try:
        dims = ast.literal_eval(dims_raw)
        if not isinstance(dims, list):
            dims = []
    except Exception:
        dims = []

    return dims, score


async def benchmark_async(
    q_json: str,
    a_json: str,
    analysis_source: str,
    start_num: int,
    question_num: int | None,
    k_for_rag: int,
):
    with open(q_json, encoding="utf-8") as f:
        q_data = json.load(f)
    with open(a_json, encoding="utf-8") as f:
        a_map  = {d["question_id"]: d["answer"] for d in json.load(f)}

    scores = defaultdict(list)
    current_video, analyzer = None, None

    # Keep parity with other benchmark scripts: reset DB once before processing.
    clean_db_and_source()

    end_index = question_num if question_num is not None else len(q_data)
    for q in q_data[start_num:end_index]:
        vid = q["video_name"]

        # build DB only when video changes
        if vid != current_video:
            analyzer = build_db(vid, base_dir=analysis_source)
            current_video = vid

        std_answer = a_map.get(q["question_id"], "")

        # run grading task
        dims, score = await grade_one(analyzer, q, std_answer, k_for_rag)
        for d in dims:
            if d in All_category:
                scores[d].append(score)

    # save once at the end
    with open("scores_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("DONE - processed", len(q_data[start_num:end_index]), "questions")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run async MMBench video evaluation.")
    parser.add_argument("--question-file", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--answer-file", default=DEFAULT_ANSWER_FILE)
    parser.add_argument("--analysis-source", default=DEFAULT_ANALYSIS_SOURCE)
    parser.add_argument("--start-num", type=int, default=DEFAULT_START_NUM)
    parser.add_argument("--question-num", type=int, default=None)
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

    asyncio.run(
        benchmark_async(
            question_file,
            answer_file,
            analysis_source,
            args.start_num,
            args.question_num,
            args.k_for_rag,
        )
    )