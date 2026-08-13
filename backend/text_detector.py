"""PaddleOCR based text detection + recognition adapter."""

from __future__ import annotations

# ---------------------------------------------------------
# IMPORTANT:
# PaddleOCR/PaddleX CPU oneDNN/PIR compatibility settings
# ---------------------------------------------------------
import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

# ---------------------------------------------------------
# Standard imports
# ---------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
from config import (
    OCR_DEVICE,
    OCR_LANG,
    OCR_SCORE_THRESHOLD,
    MAX_LINES,
    DEBUG_DIR,
)

# ---------------------------------------------------------
# PaddleOCR optional import
# ---------------------------------------------------------
try:
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover
    PaddleOCR = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: List[int]


class HandwritingOCR:
    def __init__(self) -> None:
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR is not installed or could not be imported. "
                "Install backend/requirements.txt first. "
                f"Original import error: {_IMPORT_ERROR}"
            )

        self._ocr = None
        self._create()

    def _create(self):
        """Create PaddleOCR using the working CPU configuration."""

        kwargs = {
            "lang": OCR_LANG,

            # Use CPU unless project config explicitly specifies otherwise.
            "device": OCR_DEVICE,

            # Disable unnecessary document processing.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,

            # IMPORTANT:
            # This fixed the oneDNN/PIR error in our standalone test.
            "enable_mkldnn": False,
        }

        try:
            self._ocr = PaddleOCR(**kwargs)

        except TypeError:
            # Fallback if a PaddleOCR version does not accept
            # enable_mkldnn or another newer option.
            fallback_kwargs = {
                "lang": OCR_LANG,
                "device": OCR_DEVICE,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }

            self._ocr = PaddleOCR(**fallback_kwargs)

    @staticmethod
    def _poly_to_box(poly: Any) -> List[int]:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

        x1, y1 = np.floor(pts.min(axis=0)).astype(int)
        x2, y2 = np.ceil(pts.max(axis=0)).astype(int)

        return [
            int(x1),
            int(y1),
            int(max(1, x2 - x1)),
            int(max(1, y2 - y1)),
        ]

    @staticmethod
    def _sort_lines(lines: List[OCRLine]) -> List[OCRLine]:
        """
        Robust reading order:
        primarily top-to-bottom,
        secondarily left-to-right.
        """

        lines = sorted(
            lines,
            key=lambda r: (r.box[1], r.box[0])
        )

        grouped: List[List[OCRLine]] = []

        for line in lines:
            cy = line.box[1] + line.box[3] / 2
            placed = False

            for group in grouped:
                gy = np.mean(
                    [
                        x.box[1] + x.box[3] / 2
                        for x in group
                    ]
                )

                gh = np.median(
                    [
                        x.box[3]
                        for x in group
                    ]
                )

                if abs(cy - gy) <= max(12, 0.55 * gh):
                    group.append(line)
                    placed = True
                    break

            if not placed:
                grouped.append([line])

        ordered: List[OCRLine] = []

        for group in sorted(
            grouped,
            key=lambda g: np.mean([x.box[1] for x in g])
        ):
            ordered.extend(
                sorted(
                    group,
                    key=lambda x: x.box[0]
                )
            )

        return ordered

    def predict(
        self,
        image: np.ndarray,
        debug: bool = False
    ) -> List[OCRLine]:

        if self._ocr is None:
            raise RuntimeError("PaddleOCR model is not initialized.")

        # -------------------------------------------------
        # Run PaddleOCR
        # -------------------------------------------------
        result = self._ocr.predict(image)

        records: List[OCRLine] = []

        # -------------------------------------------------
        # Extract OCR results
        # -------------------------------------------------
        for res in result:

            data = getattr(res, "json", None)

            if callable(data):
                data = data()

            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            if not isinstance(data, dict):
                data = getattr(res, "_data", {}) or {}

            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []

            polys = (
                data.get("rec_polys")
                or data.get("dt_polys")
                or []
            )

            boxes = (
                data.get("rec_boxes")
                or data.get("dt_boxes")
                or []
            )

            n = min(
                len(texts),
                len(scores),
                max(len(polys), len(boxes))
            )

            for i in range(n):

                text = str(texts[i]).strip()
                score = float(scores[i])

                if not text:
                    continue

                if score < OCR_SCORE_THRESHOLD:
                    continue

                if len(polys) > i:
                    source_box = polys[i]
                else:
                    source_box = boxes[i]

                box = self._poly_to_box(source_box)

                records.append(
                    OCRLine(
                        text=text,
                        confidence=score,
                        box=box,
                    )
                )

        # -------------------------------------------------
        # Sort and limit results
        # -------------------------------------------------
        records = self._sort_lines(records)[:MAX_LINES]

        # -------------------------------------------------
        # Debug image
        # -------------------------------------------------
        if debug:

            os.makedirs(
                DEBUG_DIR,
                exist_ok=True
            )

            canvas = image.copy()

            for line in records:

                x, y, w, h = line.box

                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + w, y + h),
                    (0, 200, 0),
                    2,
                )

            cv2.imwrite(
                os.path.join(
                    DEBUG_DIR,
                    "detected_text_regions.png"
                ),
                canvas,
            )

        return records