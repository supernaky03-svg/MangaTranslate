from __future__ import annotations

import logging
import math
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Settings
from .models import OCRRegion, ProcessedImageResult, RenderedRegion
from .ocr import OCRService, load_image_bgr, save_image_bgr
from .translators import TranslationService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TextLayout:
    lines: list[str]
    font: ImageFont.FreeTypeFont
    line_height: int
    total_height: int
    max_line_width: int
    stroke_width: int


class MangaImageProcessor:
    def __init__(
        self,
        settings: Settings,
        ocr_service: OCRService,
        translator: TranslationService,
    ) -> None:
        self.settings = settings
        self.ocr_service = ocr_service
        self.translator = translator
        self.font_path = settings.resolve_font_path()

    def process_image(self, input_path: Path, output_path: Path) -> ProcessedImageResult:
        warnings: list[str] = []
        image = load_image_bgr(input_path)

        try:
            regions = self.ocr_service.detect_regions(image)
        except Exception as exc:
            logger.exception("OCR failed for %s", input_path)
            shutil.copy2(input_path, output_path)
            return ProcessedImageResult(
                output_path=output_path,
                regions=[],
                had_text=False,
                warnings=[f"OCR failed: {exc}"],
            )

        if not regions:
            shutil.copy2(input_path, output_path)
            return ProcessedImageResult(output_path=output_path, regions=[], had_text=False)

        working = image.copy()
        rendered_regions: list[RenderedRegion] = []
        for region in regions:
            try:
                translated = self.translator.translate(region.text)
                translated = translated.strip() or region.text
            except Exception as exc:
                logger.warning("Translation failed for %r: %s", region.text, exc)
                warnings.append(f"Translation failed for region: {region.text}")
                translated = region.text

            try:
                trial = self._remove_original_text(working.copy(), [region])
                self._prepare_background_for_regions(trial, [region])
                pil_image = Image.fromarray(cv2.cvtColor(trial, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                written_text = self._draw_region(draw, pil_image, region, translated)
                working = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                logger.exception("Render failed for region %r in %s", region.text, input_path)
                warnings.append(f"Render failed for region: {region.text}")
                continue

            rendered_regions.append(
                RenderedRegion(
                    source_text=region.text,
                    translated_text=written_text,
                    bbox=region.bbox,
                    write_bbox=region.write_bbox,
                    confidence=region.confidence,
                )
            )

        if not rendered_regions:
            shutil.copy2(input_path, output_path)
            return ProcessedImageResult(
                output_path=output_path,
                regions=[],
                had_text=False,
                warnings=warnings or ["Detected text but could not render any translated regions."],
                debug={"regions_detected": len(regions)},
            )

        save_image_bgr(output_path, working)
        return ProcessedImageResult(
            output_path=output_path,
            regions=rendered_regions,
            had_text=True,
            warnings=warnings,
            debug={"regions_detected": len(regions)},
        )

    @staticmethod
    def _remove_original_text(image: np.ndarray, regions: list[OCRRegion]) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in regions:
            polygon = np.array(region.polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    def _prepare_background_for_regions(self, image: np.ndarray, regions: list[OCRRegion]) -> None:
        for region in regions:
            x1, y1, x2, y2 = region.write_bbox
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            std = float(np.std(gray))
            if mean < 150 or std > 55:
                continue
            fill_color = self._estimate_fill_color(image, region.write_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), fill_color, thickness=-1)

    @staticmethod
    def _estimate_fill_color(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
        x1, y1, x2, y2 = bbox
        border_samples: list[np.ndarray] = []
        if y1 > 0:
            border_samples.append(image[max(0, y1 - 2):y1, x1:x2])
        if y2 < image.shape[0]:
            border_samples.append(image[y2:min(image.shape[0], y2 + 2), x1:x2])
        if x1 > 0:
            border_samples.append(image[y1:y2, max(0, x1 - 2):x1])
        if x2 < image.shape[1]:
            border_samples.append(image[y1:y2, x2:min(image.shape[1], x2 + 2)])
        if not border_samples:
            return (255, 255, 255)
        merged = np.concatenate([sample.reshape(-1, 3) for sample in border_samples if sample.size > 0], axis=0)
        if merged.size == 0:
            return (255, 255, 255)
        median = np.median(merged, axis=0)
        return tuple(int(v) for v in median.tolist())

    def _draw_region(
        self,
        draw: ImageDraw.ImageDraw,
        pil_image: Image.Image,
        region: OCRRegion,
        translated: str,
    ) -> str:
        x1, y1, x2, y2 = region.write_bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        padding_x = max(6, int(box_w * 0.06))
        padding_y = max(4, int(box_h * 0.08))
        inner_w = max(10, box_w - padding_x * 2)
        inner_h = max(10, box_h - padding_y * 2)

        layout = self._fit_text(draw, translated, inner_w, inner_h)
        final_text = translated
        if layout is None:
            target_chars = max(4, int((inner_w * inner_h) / 1500))
            shortened = self.translator.shorten(translated, target_chars)
            layout = self._fit_text(draw, shortened, inner_w, inner_h)
            if layout is not None:
                final_text = shortened

        if layout is None:
            layout = self._fit_text(draw, translated, inner_w, inner_h, allow_tiny=True)
            if layout is None:
                raise ValueError(f"Unable to fit text in region: {translated!r}")

        text_color, stroke_fill = self._choose_text_colors(pil_image, region.write_bbox)
        start_y = y1 + padding_y + max(0, (inner_h - layout.total_height) // 2)
        for index, line in enumerate(layout.lines):
            line_bbox = draw.textbbox((0, 0), line, font=layout.font, stroke_width=layout.stroke_width)
            line_w = line_bbox[2] - line_bbox[0]
            line_h = line_bbox[3] - line_bbox[1]
            line_x = x1 + padding_x + max(0, (inner_w - line_w) // 2)
            line_y = start_y + index * layout.line_height
            draw.text(
                (line_x, line_y),
                line,
                font=layout.font,
                fill=text_color,
                stroke_width=layout.stroke_width,
                stroke_fill=stroke_fill,
            )
        return final_text

    def _fit_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        max_height: int,
        allow_tiny: bool = False,
    ) -> TextLayout | None:
        min_size = max(10 if allow_tiny else self.settings.min_font_size, 8)
        max_size = max(min(self.settings.max_font_size, max_height), min_size)
        for font_size in range(max_size, min_size - 1, -1):
            font = ImageFont.truetype(self.font_path, font_size)
            lines = self._wrap_text(draw, text, font, max_width)
            if not lines:
                continue
            stroke_width = max(1, font_size // 14)
            line_spacing = max(2, int(font_size * 0.18))
            line_height = self._measure_line_height(draw, font, stroke_width) + line_spacing
            max_line_width = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                max_line_width = max(max_line_width, bbox[2] - bbox[0])
            total_height = line_height * len(lines) - line_spacing
            if max_line_width <= max_width and total_height <= max_height:
                return TextLayout(
                    lines=lines,
                    font=font,
                    line_height=line_height,
                    total_height=total_height,
                    max_line_width=max_line_width,
                    stroke_width=stroke_width,
                )
        return None

    @staticmethod
    def _measure_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> int:
        bbox = draw.textbbox((0, 0), "မြန်မာAa", font=font, stroke_width=stroke_width)
        return bbox[3] - bbox[1]

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        text = text.strip()
        if not text:
            return []

        tokens = text.split()
        if len(tokens) <= 1:
            return self._wrap_by_characters(draw, text, font, max_width)

        lines: list[str] = []
        current = ""
        for token in tokens:
            candidate = token if not current else f"{current} {token}"
            width = self._text_width(draw, candidate, font)
            if width <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = token
            else:
                lines.extend(self._wrap_by_characters(draw, token, font, max_width))
                current = ""
        if current:
            lines.append(current)

        normalized: list[str] = []
        for line in lines:
            if self._text_width(draw, line, font) <= max_width:
                normalized.append(line)
            else:
                normalized.extend(self._wrap_by_characters(draw, line, font, max_width))
        return normalized

    def _wrap_by_characters(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        lines: list[str] = []
        current = ""
        for unit in self._iter_wrap_units(text):
            candidate = unit if not current else current + unit
            if self._text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current.rstrip())
                    current = unit.lstrip()
                else:
                    lines.append(unit.rstrip())
                    current = ""
        if current:
            lines.append(current.rstrip())
        return [line for line in lines if line]

    @staticmethod
    def _iter_wrap_units(text: str) -> list[str]:
        units: list[str] = []
        current = ""
        for ch in text:
            if ch.isspace():
                if current:
                    units.append(current)
                    current = ""
                units.append(ch)
                continue
            is_combining = bool(unicodedata.combining(ch)) or ch in {"္", "်", "‍"}
            if not current:
                current = ch
            elif is_combining:
                current += ch
            else:
                units.append(current)
                current = ch
        if current:
            units.append(current)
        return units

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    @staticmethod
    def _choose_text_colors(
        pil_image: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        x1, y1, x2, y2 = bbox
        crop = pil_image.crop((x1, y1, x2, y2))
        arr = np.array(crop)
        if arr.size == 0:
            return (0, 0, 0), (255, 255, 255)
        luminance = float(np.mean(0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]))
        if luminance >= 150:
            return (10, 10, 10), (255, 255, 255)
        return (255, 255, 255), (0, 0, 0)
