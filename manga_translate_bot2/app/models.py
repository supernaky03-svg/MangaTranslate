from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ImageJob:
    user_id: int
    chat_id: int
    source_message_id: int
    input_path: Path
    original_filename: str
    mime_type: str
    submission_seq: int
    index_in_submission: int
    media_group_id: str | None = None
    reply_to_message_id: int | None = None


@dataclass(slots=True)
class Submission:
    seq: int
    user_id: int
    chat_id: int
    jobs: list[ImageJob] = field(default_factory=list)
    ready: bool = False
    media_group_id: str | None = None


@dataclass(slots=True)
class OCRRegion:
    polygon: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    write_bbox: tuple[int, int, int, int]
    text: str
    confidence: float


@dataclass(slots=True)
class RenderedRegion:
    source_text: str
    translated_text: str
    bbox: tuple[int, int, int, int]
    write_bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class ProcessedImageResult:
    output_path: Path
    regions: list[RenderedRegion]
    had_text: bool
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
