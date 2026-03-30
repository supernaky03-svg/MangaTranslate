from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final

import requests

from .config import Settings

logger = logging.getLogger(__name__)


TRANSLATION_INSTRUCTIONS: Final[str] = (
    "Translate the English text into natural conversational Burmese suitable for manga/comic dialogue. "
    "Preserve tone, emotion, and speaker intent. Avoid literal or textbook Burmese. Keep the wording concise "
    "and easy to fit inside a speech bubble. If the line is angry, make it forceful. If it is sad, hesitant, "
    "or inner monologue, make it softer and more natural. Prefer spoken Burmese over formal written Burmese. "
    "Return only the Burmese line with no explanation, no quotes, and no romanization."
)

SHORTEN_INSTRUCTIONS: Final[str] = (
    "Rewrite the Burmese line so it is shorter and more natural for a manga speech bubble while preserving meaning, "
    "tone, and emotion. Keep only the Burmese line. No explanations."
)


class TranslationError(RuntimeError):
    pass


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        raise NotImplementedError

    def shorten(self, text: str, target_chars: int) -> str:
        return text


class NullTranslator(Translator):
    def translate(self, text: str) -> str:
        return text


@dataclass(slots=True)
class OpenAITranslator(Translator):
    api_key: str
    model: str
    timeout_seconds: int
    _client: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TranslationError(
                "openai package is not installed. Add it to requirements or use another backend."
            ) from exc

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def _run(self, instructions: str, input_text: str, max_output_tokens: int = 180) -> str:
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=max_output_tokens,
            )
            output = (response.output_text or "").strip()
            if not output:
                raise TranslationError("OpenAI returned empty output.")
            return output
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise TranslationError(f"OpenAI translation failed: {exc}") from exc

    def translate(self, text: str) -> str:
        prompt = f"English dialogue:\n{text.strip()}"
        return self._run(TRANSLATION_INSTRUCTIONS, prompt)

    def shorten(self, text: str, target_chars: int) -> str:
        prompt = f"Target length: around {target_chars} characters or less when possible.\nBurmese line:\n{text.strip()}"
        return self._run(SHORTEN_INSTRUCTIONS, prompt, max_output_tokens=120)


@dataclass(slots=True)
class GeminiTranslator(Translator):
    api_key: str
    model: str
    timeout_seconds: int

    def _request(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 180,
            },
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise TranslationError(f"Gemini returned no candidates: {data}")
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            output = "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
            if not output:
                raise TranslationError(f"Gemini returned empty text: {data}")
            return output
        except requests.RequestException as exc:  # pragma: no cover - network/runtime dependent
            raise TranslationError(f"Gemini request failed: {exc}") from exc

    def translate(self, text: str) -> str:
        prompt = f"{TRANSLATION_INSTRUCTIONS}\n\nEnglish dialogue:\n{text.strip()}"
        return self._request(prompt)

    def shorten(self, text: str, target_chars: int) -> str:
        prompt = (
            f"{SHORTEN_INSTRUCTIONS}\n\nTarget length: around {target_chars} characters or less when possible.\n"
            f"Burmese line:\n{text.strip()}"
        )
        return self._request(prompt)


class FallbackRuleTranslator(Translator):
    """Safety net only. This keeps the bot alive but is not a substitute for an LLM backend."""

    _direct_map = {
        "get out": "ထွက်သွား!",
        "get out!": "ထွက်သွား!",
        "i said get out": "ထွက်သွားစမ်း!",
        "please stop": "တော်ပါတော့…",
        "please... stop": "တော်ပါတော့…",
        "this is bad": "မကောင်းတော့ဘူး…",
        "no": "မဟုတ်ဘူး!",
        "what": "ဘာလဲ?",
        "why": "ဘာလို့လဲ?",
        "run": "ပြေး!",
        "help": "ကူညီပါဦး!",
        "wait": "ခဏနေ!",
        "damn": "တောက်!",
        "dammit": "တောက်!",
        "im sorry": "တောင်းပန်ပါတယ်…",
        "i'm sorry": "တောင်းပန်ပါတယ်…",
        "thank you": "ကျေးဇူးပဲ။",
        "thanks": "ကျေးဇူးပဲ။",
        "okay": "အင်း။",
        "ok": "အင်း။",
        "huh": "ဟင်?",
        "who are you": "မင်းဘယ်သူလဲ?",
        "leave me alone": "ငါ့ကိုတစ်ယောက်တည်းထားပါ။",
        "shut up": "ပါးစပ်ပိတ်ထား!",
    }

    _cleanup_rules = [
        (re.compile(r"\bI am\b", re.IGNORECASE), "ငါက"),
        (re.compile(r"\bI\b", re.IGNORECASE), "ငါ"),
        (re.compile(r"\byou\b", re.IGNORECASE), "မင်း"),
        (re.compile(r"\bwe\b", re.IGNORECASE), "တို့"),
        (re.compile(r"\bwhat\b", re.IGNORECASE), "ဘာ"),
        (re.compile(r"\bwhy\b", re.IGNORECASE), "ဘာလို့"),
        (re.compile(r"\bno\b", re.IGNORECASE), "မဟုတ်ဘူး"),
        (re.compile(r"\byes\b", re.IGNORECASE), "အင်း"),
    ]

    def translate(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = normalized.replace("—", "-")
        if normalized in self._direct_map:
            return self._direct_map[normalized]

        working = text.strip()
        for pattern, replacement in self._cleanup_rules:
            working = pattern.sub(replacement, working)

        working = working.strip()
        if not working:
            return text

        if re.search(r"[A-Za-z]", working):
            return working
        return working


class TranslationService(Translator):
    def __init__(self, primary: Translator, fallback: Translator | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or NullTranslator()

    def translate(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        try:
            output = self.primary.translate(cleaned).strip()
            return self._sanitize(output) or cleaned
        except Exception as exc:
            logger.warning("Primary translator failed for %r: %s", cleaned, exc)
            fallback = self.fallback.translate(cleaned).strip()
            return self._sanitize(fallback) or cleaned

    def shorten(self, text: str, target_chars: int) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        try:
            output = self.primary.shorten(cleaned, target_chars).strip()
            return self._sanitize(output) or cleaned
        except Exception as exc:
            logger.warning("Shorten pass failed for %r: %s", cleaned, exc)
            return self._heuristic_shorten(cleaned, target_chars)

    @staticmethod
    def _sanitize(text: str) -> str:
        text = text.strip()
        text = text.strip("\"'“”‘’` ")
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _heuristic_shorten(text: str, target_chars: int) -> str:
        if len(text) <= target_chars:
            return text
        shortened = text
        shortened = shortened.replace("ကျွန်တော်", "ငါ")
        shortened = shortened.replace("ကျွန်မ", "ငါ")
        shortened = shortened.replace("ပါသည်", "တယ်")
        shortened = shortened.replace("ပါတော့", "တော့")
        shortened = shortened.replace("မဟုတ်ပါဘူး", "မဟုတ်ဘူး")
        shortened = shortened.replace("နေပါတယ်", "နေတယ်")
        shortened = re.sub(r"\s+", " ", shortened).strip()
        if len(shortened) <= target_chars:
            return shortened
        return shortened[: max(4, target_chars - 1)].rstrip() + "…"


def build_translation_service(settings: Settings) -> TranslationService:
    backend = settings.translation_backend
    fallback = FallbackRuleTranslator()

    if backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("TRANSLATION_BACKEND=openai but OPENAI_API_KEY is missing.")
        return TranslationService(
            OpenAITranslator(settings.openai_api_key, settings.openai_model, settings.llm_timeout_seconds),
            fallback=fallback,
        )

    if backend == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("TRANSLATION_BACKEND=gemini but GEMINI_API_KEY is missing.")
        return TranslationService(
            GeminiTranslator(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds),
            fallback=fallback,
        )

    if backend == "fallback":
        return TranslationService(fallback, fallback=NullTranslator())

    if settings.openai_api_key:
        try:
            return TranslationService(
                OpenAITranslator(settings.openai_api_key, settings.openai_model, settings.llm_timeout_seconds),
                fallback=fallback,
            )
        except Exception as exc:
            logger.warning("OpenAI translator unavailable, falling back: %s", exc)

    if settings.gemini_api_key:
        try:
            return TranslationService(
                GeminiTranslator(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds),
                fallback=fallback,
            )
        except Exception as exc:
            logger.warning("Gemini translator unavailable, falling back: %s", exc)

    logger.warning("No LLM translator configured. Using fallback translator only.")
    return TranslationService(fallback, fallback=NullTranslator())
