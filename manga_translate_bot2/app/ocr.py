from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import Settings
from .models import OCRRegion

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RawOCRResult:
    polygon: list[tuple[int, int]]
    text: str
    confidence: float


class OCRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self.reader = None  # lazy load

    def _ensure_reader(self) -> None:
        if self.reader is not None:
            return

        with self._lock:
            if self.reader is not None:
                return

            logger.info("About to import easyocr")
            import easyocr

            logger.info(
                "Initializing EasyOCR with languages=%s gpu=%s",
                self.settings.ocr_languages,
                self.settings.ocr_gpu,
            )
            self.reader = easyocr.Reader(
                list(self.settings.ocr_languages),
                gpu=self.settings.ocr_gpu,
            )
            logger.info("EasyOCR ready")

    def detect_regions(self, image: np.ndarray) -> list[OCRRegion]:
        self._ensure_reader()

        prepared, scale = self._prepare_for_ocr(image)
        raw = self._readtext(prepared)
        filtered = self._filter_and_normalize(raw, image.shape[1], image.shape[0], scale)
        merged = self._merge_adjacent(filtered)
        ordered = self._sort_reading_order(merged)
        return ordered[: self.settings.max_text_regions_per_image]

    def _readtext(self, image: np.ndarray) -> list[RawOCRResult]:
        with self._lock:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=5,
                text_threshold=0.65,
                low_text=0.35,
                link_threshold=0.25,
                width_ths=0.5,
                y_ths=0.35,
                height_ths=0.6,
                mag_ratio=1.2,
            )

        normalized: list[RawOCRResult] = []
        for item in results:
            if len(item) < 3:
                continue
            box, text, confidence = item[0], item[1], float(item[2])
            polygon = [(int(p[0]), int(p[1])) for p in box]
            normalized.append(RawOCRResult(polygon=polygon, text=str(text), confidence=confidence))
        return normalized

    @staticmethod
    def _prepare_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        scale = 1.0
        if min(gray.shape[:2]) < 900:
            scale = max(1.0, 1100 / float(min(gray.shape[:2])))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return gray, scale

    def _filter_and_normalize(
        self,
        results: list[RawOCRResult],
        width: int,
        height: int,
        scale: float = 1.0,
    ) -> list[OCRRegion]:
        regions: list[OCRRegion] = []

        for result in results:
            text = self._cleanup_text(result.text)
            if not text:
                continue
            if result.confidence < self.settings.ocr_confidence_threshold:
                continue
            if not self._looks_like_english(text):
                continue

            inv_scale = 1.0 / max(scale, 1e-6)
            polygon = [
                (
                    int(round(min(max(x * inv_scale, 0.0), width - 1))),
                    int(round(min(max(y * inv_scale, 0.0), height - 1))),
                )
                for x, y in result.polygon
            ]

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, x2 = max(0, min(xs)), min(width - 1, max(xs))
            y1, y2 = max(0, min(ys)), min(height - 1, max(ys))

            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            bbox = (x1, y1, x2, y2)
            write_bbox = self._estimate_write_bbox(bbox, width, height)

            regions.append(
                OCRRegion(
                    polygon=polygon,
                    bbox=bbox,
                    write_bbox=write_bbox,
                    text=text,
                    confidence=result.confidence,
                )
            )

        return regions

    @staticmethod
    def _cleanup_text(text: str) -> str:
        cleaned = text.replace("\n", " ").replace("\t", " ").strip()
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def _looks_like_english(text: str) -> bool:
        letters = sum(ch.isalpha() for ch in text)
        latin = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
        if letters == 0 or latin == 0:
            return False
        if latin / max(1, letters) < 0.6:
            return False
        if len(re.sub(r"[^A-Za-z]", "", text)) < 2:
            return False
        return True

    @staticmethod
    def _estimate_write_bbox(
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(10, int(bw * 0.45))
        pad_y = max(8, int(bh * 0.65))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width - 1, x2 + pad_x),
            min(height - 1, y2 + pad_y),
        )

    def _merge_adjacent(self, regions: list[OCRRegion]) -> list[OCRRegion]:
        if not regions:
            return []

        merged: list[OCRRegion] = []
        consumed: set[int] = set()

        for idx, region in enumerate(regions):
            if idx in consumed:
                continue

            current = region
            for jdx in range(idx + 1, len(regions)):
                if jdx in consumed:
                    continue
                other = regions[jdx]
                if self._should_merge(current, other):
                    current = self._merge_two(current, other)
                    consumed.add(jdx)

            merged.append(current)

        return merged

    @staticmethod
    def _should_merge(a: OCRRegion, b: OCRRegion) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        ah = ay2 - ay1
        bh = by2 - by1

        same_row = abs(((ay1 + ay2) / 2) - ((by1 + by2) / 2)) <= max(ah, bh) * 0.7
        horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0)

        overlaps_x = min(ax2, bx2) - max(ax1, bx1) > -max(ah, bh) * 0.4
        vertical_gap = max(by1 - ay2, ay1 - by2, 0)
        close_vertically = vertical_gap <= max(ah, bh) * 0.5 and overlaps_x
        close_horizontally = same_row and horizontal_gap <= max(ah, bh) * 1.8

        return close_horizontally or close_vertically

    @staticmethod
    def _merge_two(a: OCRRegion, b: OCRRegion) -> OCRRegion:
        x1 = min(a.bbox[0], b.bbox[0])
        y1 = min(a.bbox[1], b.bbox[1])
        x2 = max(a.bbox[2], b.bbox[2])
        y2 = max(a.bbox[3], b.bbox[3])

        wx1 = min(a.write_bbox[0], b.write_bbox[0])
        wy1 = min(a.write_bbox[1], b.write_bbox[1])
        wx2 = max(a.write_bbox[2], b.write_bbox[2])
        wy2 = max(a.write_bbox[3], b.write_bbox[3])

        ordered = sorted(
            [a, b],
            key=lambda region: (((region.bbox[1] + region.bbox[3]) // 2) // 24, region.bbox[0]),
        )
        text = " ".join(region.text for region in ordered)
        polygon = a.polygon + b.polygon
        confidence = min(a.confidence, b.confidence)

        return OCRRegion(
            polygon=polygon,
            bbox=(x1, y1, x2, y2),
            write_bbox=(wx1, wy1, wx2, wy2),
            text=text.strip(),
            confidence=confidence,
        )

    @staticmethod
    def _sort_reading_order(regions: list[OCRRegion]) -> list[OCRRegion]:
        def row_key(region: OCRRegion) -> tuple[int, int]:
            x1, y1, x2, y2 = region.bbox
            center_y = (y1 + y2) // 2
            return (center_y // 24, x1)

        return sorted(regions, key=row_key)


def load_image_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def save_image_bgr(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    success, encoded = cv2.imencode(
        ext if ext in {".png", ".jpg", ".jpeg", ".webp"} else ".png",
        image,
    )
    if not success:
        raise ValueError(f"Failed to encode output image: {path}")
    path.write_bytes(encoded.tobytes())            if self.reader is not None:
                return

            logger.info("About to import easyocr")
            import easyocr
            logger.info(
                "Initializing EasyOCR with languages=%s gpu=%s",
                self.settings.ocr_languages,
                self.settings.ocr_gpu,
            )
            self.reader = easyocr.Reader(
                list(self.settings.ocr_languages),
                gpu=self.settings.ocr_gpu,
            )

    def detect_regions(self, image):
        self._ensure_reader()
        prepared, scale = self._prepare_for_ocr(image)
        raw = self._readtext(prepared)
        filtered = self._filter_and_normalize(raw, image.shape[1], image.shape[0], scale)
        merged = self._merge_adjacent(filtered)
        ordered = self._sort_reading_order(merged)
        return ordered[: self.settings.max_text_regions_per_image]

    def _readtext(self, image: np.ndarray) -> list[RawOCRResult]:
        with self._lock:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=5,
                text_threshold=0.65,
                low_text=0.35,
                link_threshold=0.25,
                width_ths=0.5,
                y_ths=0.35,
                height_ths=0.6,
                mag_ratio=1.2,
            )
        normalized: list[RawOCRResult] = []
        for item in results:
            if len(item) < 3:
                continue
            box, text, confidence = item[0], item[1], float(item[2])
            polygon = [(int(p[0]), int(p[1])) for p in box]
            normalized.append(RawOCRResult(polygon=polygon, text=str(text), confidence=confidence))
        return normalized

    @staticmethod
    def _prepare_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        scale = 1.0
        if min(gray.shape[:2]) < 900:
            scale = max(1.0, 1100 / float(min(gray.shape[:2])))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return gray, scale

    def _filter_and_normalize(self, results: list[RawOCRResult], width: int, height: int, scale: float = 1.0) -> list[OCRRegion]:
        regions: list[OCRRegion] = []

        for result in results:
            text = self._cleanup_text(result.text)
            if not text:
                continue
            if result.confidence < self.settings.ocr_confidence_threshold:
                continue
            if not self._looks_like_english(text):
                continue

            inv_scale = 1.0 / max(scale, 1e-6)
            polygon = [
                (
                    int(round(min(max(x * inv_scale, 0.0), width - 1))),
                    int(round(min(max(y * inv_scale, 0.0), height - 1))),
                )
                for x, y in result.polygon
            ]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, x2 = max(0, min(xs)), min(width - 1, max(xs))
            y1, y2 = max(0, min(ys)), min(height - 1, max(ys))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            bbox = (x1, y1, x2, y2)
            write_bbox = self._estimate_write_bbox(bbox, width, height)
            regions.append(
                OCRRegion(
                    polygon=polygon,
                    bbox=bbox,
                    write_bbox=write_bbox,
                    text=text,
                    confidence=result.confidence,
                )
            )
        return regions

    @staticmethod
    def _cleanup_text(text: str) -> str:
        cleaned = text.replace("\n", " ").replace("\t", " ").strip()
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def _looks_like_english(text: str) -> bool:
        letters = sum(ch.isalpha() for ch in text)
        latin = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
        if letters == 0 or latin == 0:
            return False
        if latin / max(1, letters) < 0.6:
            return False
        if len(re.sub(r"[^A-Za-z]", "", text)) < 2:
            return False
        return True

    @staticmethod
    def _estimate_write_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(10, int(bw * 0.45))
        pad_y = max(8, int(bh * 0.65))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width - 1, x2 + pad_x),
            min(height - 1, y2 + pad_y),
        )

    def _merge_adjacent(self, regions: list[OCRRegion]) -> list[OCRRegion]:
        if not regions:
            return []

        merged: list[OCRRegion] = []
        consumed: set[int] = set()
        for idx, region in enumerate(regions):
            if idx in consumed:
                continue
            current = region
            for jdx in range(idx + 1, len(regions)):
                if jdx in consumed:
                    continue
                other = regions[jdx]
                if self._should_merge(current, other):
                    current = self._merge_two(current, other)
                    consumed.add(jdx)
            merged.append(current)
        return merged

    @staticmethod
    def _should_merge(a: OCRRegion, b: OCRRegion) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ah = ay2 - ay1
        bh = by2 - by1
        same_row = abs(((ay1 + ay2) / 2) - ((by1 + by2) / 2)) <= max(ah, bh) * 0.7
        horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0)
        overlaps_x = min(ax2, bx2) - max(ax1, bx1) > -max(ah, bh) * 0.4
        vertical_gap = max(by1 - ay2, ay1 - by2, 0)
        close_vertically = vertical_gap <= max(ah, bh) * 0.5 and overlaps_x
        close_horizontally = same_row and horizontal_gap <= max(ah, bh) * 1.8
        return close_horizontally or close_vertically

    @staticmethod
    def _merge_two(a: OCRRegion, b: OCRRegion) -> OCRRegion:
        x1 = min(a.bbox[0], b.bbox[0])
        y1 = min(a.bbox[1], b.bbox[1])
        x2 = max(a.bbox[2], b.bbox[2])
        y2 = max(a.bbox[3], b.bbox[3])
        wx1 = min(a.write_bbox[0], b.write_bbox[0])
        wy1 = min(a.write_bbox[1], b.write_bbox[1])
        wx2 = max(a.write_bbox[2], b.write_bbox[2])
        wy2 = max(a.write_bbox[3], b.write_bbox[3])
        ordered = sorted([a, b], key=lambda region: (((region.bbox[1] + region.bbox[3]) // 2) // 24, region.bbox[0]))
        text = " ".join(region.text for region in ordered)
        polygon = a.polygon + b.polygon
        confidence = min(a.confidence, b.confidence)
        return OCRRegion(
            polygon=polygon,
            bbox=(x1, y1, x2, y2),
            write_bbox=(wx1, wy1, wx2, wy2),
            text=text.strip(),
            confidence=confidence,
        )

    @staticmethod
    def _sort_reading_order(regions: list[OCRRegion]) -> list[OCRRegion]:
        def row_key(region: OCRRegion) -> tuple[int, int]:
            x1, y1, x2, y2 = region.bbox
            center_y = (y1 + y2) // 2
            return (center_y // 24, x1)

        return sorted(regions, key=row_key)


def load_image_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def save_image_bgr(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    success, encoded = cv2.imencode(ext if ext in {'.png', '.jpg', '.jpeg', '.webp'} else '.png', image)
    if not success:
        raise ValueError(f"Failed to encode output image: {path}")
    path.write_bytes(encoded.tobytes())
