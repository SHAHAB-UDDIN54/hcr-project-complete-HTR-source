from __future__ import annotations

import os
import time
import traceback

from flask import Flask, jsonify, request, send_from_directory

import preprocessing
from config import DEBUG_ENABLED, MAX_IMAGE_BYTES
from text_detector import HandwritingOCR
from postprocessing import build_response


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

_ocr = None


def get_ocr():
    global _ocr

    if _ocr is None:
        print("[TIME] Loading PaddleOCR model...")
        start = time.perf_counter()

        _ocr = HandwritingOCR()

        elapsed = time.perf_counter() - start
        print(f"[TIME] PaddleOCR model loading: {elapsed:.2f} seconds")

    return _ocr


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_file(path):
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found."}), 404

    full = os.path.join(FRONTEND_DIR, path)

    if os.path.isfile(full):
        return send_from_directory(FRONTEND_DIR, path)

    return jsonify({"error": "Frontend file not found."}), 404


@app.get("/api/health")
def health():
    try:
        get_ocr()

        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "engine": "PaddleOCR PP-OCRv6"
        })

    except Exception as exc:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(exc)
        }), 500


@app.route("/api/recognize", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def recognize():

    if request.method == "OPTIONS":
        return ("", 204)

    total_start = time.perf_counter()

    try:

        # --------------------------------------------------
        # STEP 1: Receive image
        # --------------------------------------------------

        step_start = time.perf_counter()

        file = request.files.get("image") or request.files.get("file")

        if file is None:
            return jsonify({
                "success": False,
                "text": "",
                "error": "No image uploaded. Use form field 'image'."
            }), 400

        image_bytes = file.read()

        print(
            f"[TIME] Image received: "
            f"{len(image_bytes) / 1024:.1f} KB"
        )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            return jsonify({
                "success": False,
                "text": "",
                "error": "Image is too large."
            }), 413

        print(
            f"[TIME] Image reading: "
            f"{time.perf_counter() - step_start:.2f} seconds"
        )

        # --------------------------------------------------
        # STEP 2: Preprocessing
        # --------------------------------------------------

        step_start = time.perf_counter()

        debug = (
            request.args.get("debug", "0").lower()
            in {"1", "true", "yes", "on"}
            or DEBUG_ENABLED
        )

        image = preprocessing.prepare_image(
            image_bytes,
            debug=debug
        )

        print(
            f"[TIME] Preprocessing: "
            f"{time.perf_counter() - step_start:.2f} seconds"
        )

        # --------------------------------------------------
        # STEP 3: Get OCR model
        # --------------------------------------------------

        step_start = time.perf_counter()

        ocr = get_ocr()

        print(
            f"[TIME] Get OCR: "
            f"{time.perf_counter() - step_start:.2f} seconds"
        )

        # --------------------------------------------------
        # STEP 4: PaddleOCR prediction
        # --------------------------------------------------

        step_start = time.perf_counter()

        lines = ocr.predict(
            image,
            debug=debug
        )

        print(
            f"[TIME] PaddleOCR prediction: "
            f"{time.perf_counter() - step_start:.2f} seconds"
        )

        print(
            f"[OCR] Detected lines: {len(lines)}"
        )

        for i, line in enumerate(lines):
            print(
                f"[OCR] {i + 1}: "
                f"{line.text} "
                f"(confidence={line.confidence:.4f})"
            )

        # --------------------------------------------------
        # STEP 5: Post-processing
        # --------------------------------------------------

        step_start = time.perf_counter()

        text, confidence, line_details = build_response(
            lines
        )

        print(
            f"[TIME] Postprocessing: "
            f"{time.perf_counter() - step_start:.2f} seconds"
        )

        # --------------------------------------------------
        # TOTAL TIME
        # --------------------------------------------------

        total_time = time.perf_counter() - total_start

        print(
            f"[TIME] ============================="
        )

        print(
            f"[TIME] TOTAL RECOGNITION: "
            f"{total_time:.2f} seconds"
        )

        print(
            f"[TIME] ============================="
        )

        if not text:
            return jsonify({
                "success": True,
                "text": "",
                "confidence": 0.0,
                "lines": [],
                "message": "No readable handwritten text detected.",
                "processing_time": round(total_time, 2)
            })

        return jsonify({
            "success": True,
            "text": text,
            "confidence": round(confidence, 4),
            "lines": line_details,
            "line_count": len(line_details),
            "engine": "PaddleOCR PP-OCRv6",
            "processing_time": round(total_time, 2)
        })

    except ValueError as exc:

        return jsonify({
            "success": False,
            "text": "",
            "error": str(exc)
        }), 400

    except Exception as exc:

        print("[HTR] Unexpected error:")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "text": "",
            "error": f"OCR failed: {exc}"
        }), 500


if __name__ == "__main__":

    print("Handwritten Text Recognition server")
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )