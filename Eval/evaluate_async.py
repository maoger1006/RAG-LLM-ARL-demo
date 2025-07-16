import asyncio, json, os, ast, shutil, pathlib
from collections import defaultdict
import functools
from llm_grades import get_llm_grade
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
retrieved_k = 5


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

async def grade_one(analyzer, q_item, std_answer):
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
            llm_answer, *_ = await run_sync(analyzer.generate_answer, question, k=retrieved_k)
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


async def benchmark_async(q_json, a_json):
    with open(q_json, encoding="utf-8") as f:
        q_data = json.load(f)
    with open(a_json, encoding="utf-8") as f:
        a_map  = {d["question_id"]: d["answer"] for d in json.load(f)}

    scores = defaultdict(list)
    current_video, analyzer = None, None

    for q in q_data:
        vid = q["video_name"]

        # build DB only when video changes
        if vid != current_video:
            clean_db_and_source()
            analyzer = build_db(vid, base_dir="./video_analysis_output_with_audio_1fps_all")
            current_video = vid

        std_answer = a_map.get(q["question_id"], "")

        # run grading task
        dims, score = await grade_one(analyzer, q, std_answer)
        for d in dims:
            if d in All_category:
                scores[d].append(score)

    # save once at the end
    with open("scores_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("DONE - processed", len(q_data), "questions")
    
    
if __name__ == "__main__":
    asyncio.run(
        benchmark_async(
            "/home/mingyang/video_benchmark/MMBench-Video/MMBench-Video_q.json",
            "/home/mingyang/video_benchmark/MMBench-Video/MMBench-Video_a.json",
        )
    )