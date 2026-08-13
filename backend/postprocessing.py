"""Conservative OCR post-processing. It never invents words."""
from __future__ import annotations

import re
from typing import Iterable, List

from text_detector import OCRLine


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep normal printable Unicode, spaces and line breaks.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def build_response(lines: List[OCRLine]):
    if not lines:
        return "", 0.0, []
    text = clean_text("\n".join(x.text for x in lines if x.text.strip()))
    confidence = sum(x.confidence for x in lines) / len(lines)
    return text, float(confidence), [
        {"text": x.text, "confidence": round(x.confidence, 4), "box": x.box}
        for x in lines
    ]
