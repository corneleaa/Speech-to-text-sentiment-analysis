"""Integration of the modules and a command line interface.

Usage:
    python main.py path/to/audio.wav --language lt
"""
from __future__ import annotations

import argparse
import logging
import sys

import config
from modules.audio_input import AudioData, AudioValidationError, load_audio_path
from modules.sentiment_analysis import SentimentAnalyzer, SentimentError, SentimentResult
from modules.speech_to_text import SpeechToText, TranscriptionError, TranscriptionResult
from modules.visualization import build_record, save_results


class AnalysisPipeline:
    """Holds both models, so they are loaded once and reused."""

    def __init__(
        self,
        whisper_model: str | None = None,
        sentiment_model: str | None = None,
        device: str | None = None,
    ):
        self.device = device or config.get_device()
        self.stt = SpeechToText(whisper_model, self.device)
        self.sentiment = SentimentAnalyzer(sentiment_model, self.device)

    def transcribe(self, audio: AudioData, language: str | None) -> TranscriptionResult:
        return self.stt.transcribe(audio, language)

    def analyze_text(self, text: str) -> SentimentResult:
        return self.sentiment.analyze(text)

    def make_record(
        self, *, source: str, duration: float, language: str,
        transcript: str, result: SentimentResult,
    ) -> dict:
        return build_record(
            source=source, duration=duration, language=language,
            transcript=transcript, sentiment=result.label, scores=result.scores,
            whisper_model=self.stt.model_name, sentiment_model=result.model_name,
            device=self.device,
        )

    def run(self, audio: AudioData, language: str | None) -> tuple[dict, SentimentResult]:
        transcription = self.transcribe(audio, language)
        result = self.analyze_text(transcription.text)
        record = self.make_record(
            source=audio.source_name, duration=audio.duration,
            language=transcription.language, transcript=transcription.text,
            result=result,
        )
        return record, result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech-to-Text + sentimentų analizė")
    parser.add_argument("audio", help="garso failo kelias")
    parser.add_argument("--language", choices=["auto", "lt", "en"], default="auto")
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL_NAME,
                        choices=config.WHISPER_MODEL_CHOICES)
    parser.add_argument("--sentiment-model", default=config.DEFAULT_SENTIMENT_MODEL,
                        choices=list(config.SENTIMENT_MODELS))
    parser.add_argument("--no-save", action="store_true", help="nesaugoti rezultatų į results/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    language = None if args.language == "auto" else args.language
    try:
        audio = load_audio_path(args.audio)
        pipeline = AnalysisPipeline(args.whisper_model, args.sentiment_model)
        record, result = pipeline.run(audio, language)
    except (AudioValidationError, TranscriptionError, SentimentError) as exc:
        print(f"Klaida: {exc}", file=sys.stderr)
        return 1

    print(f"Kalba: {record['language']}")
    print(f"Tekstas: {record['transcript']}")
    print(f"Sentimentas: {result.label} ({result.confidence:.1f}%)")
    for label, value in result.scores.items():
        print(f"  {label}: {value:.2f}%")
    if not args.no_save:
        paths = save_results(record)
        print("Išsaugota:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())