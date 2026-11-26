"""Results module: result record and saving to CSV and JSON."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import config

LABELS_LT = {"positive": "Teigiamas", "neutral": "Neutralus", "negative": "Neigiamas"}

CSV_FIELDS = [
    "timestamp", "source", "duration_sec", "language", "transcript",
    "sentiment", "positive_pct", "neutral_pct", "negative_pct",
    "whisper_model", "sentiment_model", "device",
]


def build_record(
    *, source: str, duration: float, language: str, transcript: str,
    sentiment: str, scores: dict, whisper_model: str, sentiment_model: str, device: str,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "duration_sec": round(duration, 2),
        "language": language,
        "transcript": transcript,
        "sentiment": sentiment,
        "positive_pct": scores["positive"],
        "neutral_pct": scores["neutral"],
        "negative_pct": scores["negative"],
        "whisper_model": whisper_model,
        "sentiment_model": sentiment_model,
        "device": device,
    }


def record_to_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)


def record_to_csv(record: dict) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerow(record)
    return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel reads Lithuanian letters


def save_results(record: dict, directory: Path | None = None) -> dict:
    """Write JSON and CSV files to the results folder."""
    directory = Path(directory or config.RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
    paths = {"json": directory / f"{stem}.json", "csv": directory / f"{stem}.csv"}
    paths["json"].write_text(record_to_json(record), encoding="utf-8")
    paths["csv"].write_bytes(record_to_csv(record))
    return paths
