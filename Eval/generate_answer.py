import os
import sys
from pathlib import Path


# Ensure repository root is importable when running from Eval/ as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RAG.rag import RAG_pipeline


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _collect_files(video_name: str, base_dir: str) -> list[str]:
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")

    candidates = [
        p for p in base_path.glob(f"{video_name}*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # Common layout: <base_dir>/<video_name>/...files
    nested_dir = base_path / video_name
    if nested_dir.exists() and nested_dir.is_dir():
        candidates.extend([
            p for p in nested_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

    unique_sorted = sorted({str(p.resolve()) for p in candidates})
    if not unique_sorted:
        raise FileNotFoundError(
            f"No supported analysis files found for video '{video_name}' under {base_path}"
        )

    return unique_sorted


def build_db(video_name: str, base_dir: str = "./video_analysis_output", chunk_size: int = 2048) -> RAG_pipeline:
    files = _collect_files(video_name=video_name, base_dir=base_dir)

    analyzer = RAG_pipeline()
    docs = analyzer.load_documents(files)
    splits = analyzer.split_documents(docs, chunk_size=chunk_size)
    analyzer.create_vector_db(splits, persist_directory=str(REPO_ROOT / "docs" / "chroma"))
    analyzer.build_qa_chain("concise mode")
    return analyzer


def generate_answer(
    question: str,
    video_name: str,
    base_dir: str = "./video_analysis_output",
    chunk_size: int = 2048,
    k: int = 5,
):
    analyzer = build_db(video_name=video_name, base_dir=base_dir, chunk_size=chunk_size)
    return analyzer.generate_answer(question, k=k)
