from __future__ import annotations

import os
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()  # use the system certificate store
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR can be moved to a faster disk, e.g. MODELS_DIR=/content/models in Colab.
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
RESULTS_DIR = BASE_DIR / "results"

# Model files are downloaded into ./models, not into the user's home folder.
# This must be set before transformers is imported.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))

# Hardware. Default is CPU only. Set DEVICE=cuda or DEVICE=auto to allow a GPU.
DEVICE = os.getenv("DEVICE", "cpu").lower()
CPU_THREADS = int(os.getenv("CPU_THREADS", min(os.cpu_count() or 4, 8)))

# Speech-to-Text (faster-whisper model names).
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 = fast on CPU
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))  # 5 = geriausia kokybė    
USE_VAD = True             # skip silence before decoding
WHISPER_MODEL_CHOICES = ("tiny", "base", "small", "medium", "large-v3")

# Sentiment models (Hugging Face ids). Both return negative / neutral / positive.
SENTIMENT_MODELS = {
    "multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "english": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}
DEFAULT_SENTIMENT_MODEL = "multilingual"
MAX_TEXT_TOKENS = 512
SENTIMENT_CHUNK_TOKENS = 500
# Dynamic int8 quantization of the sentiment model: faster on CPU, may change scores slightly.
# On by default; applies only on CPU. Set QUANTIZE_SENTIMENT=0 for the exact original scores.
QUANTIZE_SENTIMENT = os.getenv("QUANTIZE_SENTIMENT", "1") == "1"

# Language choices: UI label -> Whisper language code (None = auto-detect).
LANGUAGE_CHOICES = {"Automatinis": None, "Lietuvių": "lt", "English": "en"}
SUPPORTED_LANGUAGE_CODES = ("lt", "en")

# Audio limits.
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")
TARGET_SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.3
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "0"))  # 0 = be ribos
SILENCE_PEAK_THRESHOLD = 1e-4


def _has_snapshot(repo_dir: Path, filenames: tuple[str, ...]) -> bool:
    """True if a Hugging Face cache folder already holds one of the given weight files."""
    return any(
        (snapshot / name).is_file()
        for snapshot in repo_dir.glob("snapshots/*")
        for name in filenames
    )


def whisper_model_cached(model_name: str) -> bool:
    repo = MODELS_DIR / "whisper" / f"models--Systran--faster-whisper-{model_name}"
    return _has_snapshot(repo, ("model.bin",))


def sentiment_model_cached(model_id: str) -> bool:
    repo = MODELS_DIR / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    return _has_snapshot(repo, ("pytorch_model.bin", "model.safetensors"))


def sentiment_safetensors_path(model_id: str) -> Path | None:
    """A downloaded .safetensors copy of the model, if any (loads faster than pytorch_model.bin)."""
    repo = MODELS_DIR / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    found = sorted(repo.glob("snapshots/*/model.safetensors"))
    return found[0] if found else None


def get_device() -> str:
    """Return "cpu" by default; "cuda" only when DEVICE=cuda, or DEVICE=auto and a GPU exists."""
    if DEVICE == "cuda":
        return "cuda"
    if DEVICE == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cpu"
