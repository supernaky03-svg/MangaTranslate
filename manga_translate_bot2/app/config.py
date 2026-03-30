from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values


DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansMyanmarUI-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
    "/usr/share/fonts/truetype/padauk/PadaukBook-Regular.ttf",
    "/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf",
]


@dataclass(slots=True)
class Settings:
    telegram_bot_token: str
    admin_user_ids: set[int]
    temp_root: Path
    log_level: str
    media_group_collect_delay: float
    ocr_languages: tuple[str, ...]
    ocr_gpu: bool
    ocr_confidence_threshold: float
    max_text_regions_per_image: int
    mm_font_path: str | None
    min_font_size: int
    max_font_size: int
    send_original_when_no_text: bool
    notify_on_image_failure: bool
    translation_backend: str
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str
    llm_timeout_seconds: int

    @property
    def incoming_dir(self) -> Path:
        return self.temp_root / "incoming"

    @property
    def outgoing_dir(self) -> Path:
        return self.temp_root / "outgoing"

    @property
    def failed_dir(self) -> Path:
        return self.temp_root / "failed"

    def ensure_dirs(self) -> None:
        for path in [self.temp_root, self.incoming_dir, self.outgoing_dir, self.failed_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def cleanup_stale_temp_files(self, max_age_hours: int = 24) -> None:
        cutoff = __import__("time").time() - (max_age_hours * 3600)
        for directory in [self.incoming_dir, self.outgoing_dir]:
            if not directory.exists():
                continue
            for item in directory.iterdir():
                try:
                    if item.is_file() and item.stat().st_mtime < cutoff:
                        item.unlink()
                except Exception:
                    pass

    def resolve_font_path(self) -> str:
        candidates: list[str] = []
        if self.mm_font_path:
            candidates.append(self.mm_font_path)
        candidates.extend(DEFAULT_FONT_CANDIDATES)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate

        raise FileNotFoundError(
            "No Myanmar font found. Set MM_FONT_PATH to a readable Myanmar font such as "
            "Noto Sans Myanmar or Padauk."
        )


def _parse_admin_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    admin_ids: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        admin_ids.add(int(chunk))
    return admin_ids


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_languages(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("en",)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(items) if items else ("en",)


def load_settings(env_file: str | None = None) -> Settings:
    env_map = {k: v for k, v in (dotenv_values(env_file).items() if env_file else [])}

    def get(name: str, default: str | None = None) -> str | None:
        if name in env_map:
            return env_map.get(name)
        return os.getenv(name, default)

    token = (get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required.")

    settings = Settings(
        telegram_bot_token=token,
        admin_user_ids=_parse_admin_ids(get("ADMIN_USER_IDS")),
        temp_root=Path(get("TEMP_ROOT", "./tmp") or "./tmp").resolve(),
        log_level=(get("LOG_LEVEL", "INFO") or "INFO").upper(),
        media_group_collect_delay=float(get("MEDIA_GROUP_COLLECT_DELAY", "1.2") or "1.2"),
        ocr_languages=_parse_languages(get("OCR_LANGUAGES", "en")),
        ocr_gpu=_parse_bool(get("OCR_GPU"), False),
        ocr_confidence_threshold=float(get("OCR_CONFIDENCE_THRESHOLD", "0.35") or "0.35"),
        max_text_regions_per_image=int(get("MAX_TEXT_REGIONS_PER_IMAGE", "80") or "80"),
        mm_font_path=get("MM_FONT_PATH"),
        min_font_size=int(get("MIN_FONT_SIZE", "14") or "14"),
        max_font_size=int(get("MAX_FONT_SIZE", "42") or "42"),
        send_original_when_no_text=_parse_bool(get("SEND_ORIGINAL_WHEN_NO_TEXT"), True),
        notify_on_image_failure=_parse_bool(get("NOTIFY_ON_IMAGE_FAILURE"), True),
        translation_backend=(get("TRANSLATION_BACKEND", "auto") or "auto").strip().lower(),
        openai_api_key=get("OPENAI_API_KEY"),
        openai_model=get("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
        gemini_api_key=get("GEMINI_API_KEY"),
        gemini_model=get("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        llm_timeout_seconds=int(get("LLM_TIMEOUT_SECONDS", "45") or "45"),
    )
    settings.ensure_dirs()
    if not settings.admin_user_ids:
        raise ValueError("ADMIN_USER_IDS is required and must contain at least one Telegram user ID.")
    settings.cleanup_stale_temp_files()
    return settings
