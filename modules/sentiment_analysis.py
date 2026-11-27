from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

LABELS = ("positive", "neutral", "negative")

_LABEL_ALIASES = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


class SentimentError(RuntimeError):
    pass


def normalize_label(raw_label: str) -> str:
    key = str(raw_label).strip().lower()
    if key not in _LABEL_ALIASES:
        raise SentimentError(f"Nežinoma modelio klasė: {raw_label}")
    return _LABEL_ALIASES[key]


@dataclass
class SentimentResult:
    label: str
    scores: dict
    model_name: str

    @property
    def confidence(self) -> float:
        return self.scores[self.label]


class SentimentAnalyzer:
    def __init__(self, model_key: str | None = None, device: str | None = None):
        key = model_key or config.DEFAULT_SENTIMENT_MODEL
        if key not in config.SENTIMENT_MODELS:
            raise SentimentError(f"Nežinomas sentimentų modelis: {key}")
        self.model_key = key
        self.model_name = config.SENTIMENT_MODELS[key]
        self.device = device or config.get_device()
        self._pipe = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def _ensure_model(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise SentimentError(
                "Nerastas paketas transformers. Įdiekite: pip install transformers torch"
            ) from exc
        logger.info("Loading '%s' on %s", self.model_name, self.device)
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
        )
        if self.device == "cpu":
            self._tune_for_cpu()

    def _tune_for_cpu(self) -> None:
        import torch

        torch.set_num_threads(config.CPU_THREADS)
        if config.QUANTIZE_SENTIMENT:
            try:
                self._pipe.model = torch.ao.quantization.quantize_dynamic(
                    self._pipe.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Sentiment model quantized to int8")
            except Exception as exc:
                logger.warning("Quantization skipped: %s", exc)

    def analyze(self, text: str) -> SentimentResult:
        text = (text or "").strip()
        if not text:
            raise SentimentError("Tekstas tuščias.")
        with self._lock:
            self._ensure_model()
            try:
                outputs = self._pipe(
                    text,
                    top_k=None,
                    truncation=True,
                    max_length=config.MAX_TEXT_TOKENS,
                )
            except Exception as exc:
                raise SentimentError(f"Sentimentų modelio klaida: {exc}") from exc

        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]

        scores = {label: 0.0 for label in LABELS}
        for item in outputs:
            scores[normalize_label(item["label"])] = round(item["score"] * 100, 2)
        dominant = max(scores, key=scores.get)
        return SentimentResult(dominant, scores, self.model_name)