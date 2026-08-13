import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEBUG_DIR = os.path.join(BASE_DIR, "debug")

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_SIDE = 5000
MIN_IMAGE_SIDE = 32

OCR_LANG = os.getenv("HCR_OCR_LANG", "en")
OCR_DEVICE = os.getenv("HCR_OCR_DEVICE", "cpu")
OCR_USE_DOC_ORIENTATION = os.getenv("HCR_OCR_DOC_ORIENTATION", "false").lower() == "true"
OCR_USE_UNWARPING = os.getenv("HCR_OCR_UNWARP", "false").lower() == "true"
OCR_USE_TEXTLINE_ORIENTATION = os.getenv("HCR_OCR_TEXTLINE_ORIENTATION", "false").lower() == "true"
OCR_SCORE_THRESHOLD = float(os.getenv("HCR_OCR_SCORE_THRESHOLD", "0.35"))
MAX_LINES = int(os.getenv("HCR_MAX_LINES", "100"))

DEBUG_ENABLED = os.getenv("HCR_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
