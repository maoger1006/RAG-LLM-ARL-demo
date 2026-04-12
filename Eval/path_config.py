import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _from_env_or_default(env_name: str, default_path: Path) -> Path:
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default_path.expanduser().resolve()


def default_dataset_root() -> Path:
    env_root = os.getenv("MMBENCH_VIDEO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates = [
        REPO_ROOT / "MMBench-Video",
        REPO_ROOT / "datasets" / "MMBench-Video",
        Path.cwd() / "MMBench-Video",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def default_question_file() -> Path:
    return _from_env_or_default(
        "MMBENCH_Q_JSON",
        default_dataset_root() / "MMBench-Video_q.json",
    )


def default_answer_file() -> Path:
    return _from_env_or_default(
        "MMBENCH_A_JSON",
        default_dataset_root() / "MMBench-Video_a.json",
    )


def default_analysis_dir(default_subdir: str) -> Path:
    return _from_env_or_default("EVAL_ANALYSIS_DIR", REPO_ROOT / default_subdir)


def ensure_file(path: str, label: str, arg_flag: str, env_name: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} not found: {resolved}. Provide {arg_flag} or set {env_name}."
        )
    return str(resolved)


def ensure_dir(path: str, label: str, arg_flag: str, env_name: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{label} directory not found: {resolved}. Provide {arg_flag} or set {env_name}."
        )
    return str(resolved)
