from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import config
from modules.audio_input import AudioData

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    model_name: str
    segments: list[dict] = field(default_factory=list)


class SpeechToText:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.WHISPER_MODEL_NAME
        self.device = device or config.get_device()
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "Nerastas paketas faster-whisper. Įdiekite: pip install faster-whisper"
            ) from exc
        compute_type = config.WHISPER_COMPUTE_TYPE if self.device == "cpu" else "float16"
        logger.info("Loading Whisper '%s' on %s (%s)", self.model_name, self.device, compute_type)
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=config.CPU_THREADS,
            download_root=str(config.MODELS_DIR / "whisper"),
        )

    def transcribe(
        self, audio: AudioData, language: str | None = None
    ) -> TranscriptionResult:
        if language is not None and language not in config.SUPPORTED_LANGUAGE_CODES:
            raise TranscriptionError(f"Nepalaikoma kalba: {language}")
        with self._lock:
            self._ensure_model()
            try:
                segment_iter, info = self._model.transcribe(
                    audio.samples,
                    language=language,
                    task="transcribe",
                    beam_size=config.WHISPER_BEAM_SIZE,
                    vad_filter=config.USE_VAD,
                )
                segments = [
                    {"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segment_iter
                ]
            except Exception as exc:
                raise TranscriptionError(f"Whisper klaida: {exc}") from exc

        text = " ".join(s["text"] for s in segments if s["text"]).strip()
        if not text:
            raise TranscriptionError("Kalbos atpažinti nepavyko: tekstas tuščias.")
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None) or language or "unknown",
            duration=audio.duration,
            model_name=self.model_name,
            segments=segments,
        )