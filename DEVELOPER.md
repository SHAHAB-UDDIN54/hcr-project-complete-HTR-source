# DEVELOPER.MD — Handwritten Character & Text Recognition (HCR / HTR) System

## 1. System Architecture & Overview

This project is an end-to-end **Handwritten Character & Text Recognition (HTR/HCR)** system composed of a Flask REST backend, a modern browser frontend with canvas overlays, and a two-tier OCR model infrastructure (production PaddleOCR PP-OCR text-level sequence engine + custom CRNN-BiLSTM-CTC training pipeline).

```
                      +-----------------------------+
                      |     Browser Frontend        |
                      |  (HTML5 / Canvas / Vanilla) |
                      +--------------+--------------+
                                     |
                       HTTP POST     | Multipart Form-Data
                       (/api/        | Image file
                        recognize)   v
                      +-----------------------------+
          
                      |     Flask Backend API       |
                      |  (app.py / CORS / Config)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Image Preprocessing      |
                      | (Border Trim + CLAHE in LAB)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  PaddleOCR Detection/Recog  |
                      | (PP-OCRv6 Text Detector +   |
                      |  Sequence Text Recognizer)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Spatial Line Sorting Engine |
                      | (Top-to-Bottom / Left-to-R) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Post-Processing Sanitizer|
                      | (Unicode Cleaning / Conf.)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | JSON Response to Client     |
                      | (Text, Bounding Boxes, Conf)|
                      +-----------------------------+
```

---

## 2. Directory Structure & File Map

```text
d:\hcr-project-complete-HTR-source\
├── backend/
│   ├── app.py                  # Main Flask entrypoint & REST API controller
│   ├── config.py               # Global environment & runtime configuration
│   ├── preprocessing.py        # Image decoding, resizing, border trim & CLAHE
│   ├── text_detector.py        # PaddleOCR wrapper & 2D spatial line clustering
│   ├── postprocessing.py       # Unicode text cleanup & response formatting
│   ├── requirements.txt        # Backend dependencies
│   ├── debug/                  # Output directory for debug intermediate images
│   ├── model/                  # Legacy character-level model artifacts
│   │   ├── README.md           # Model directory notice
│   │   ├── hcr_model.keras     # Legacy 62-class CNN model (EMNIST-ByClass)
│   │   ├── label_map.json      # 62-class label mapping (0-9, A-Z, a-z)
│   │   └── train_model.py      # Legacy CNN character model training script
│   └── training/               # Custom line-level HTR training framework
│       ├── README.md           # Line-level HTR training guidelines
│       ├── dataset.py          # PyTorch/NumPy style LineDataset loader for IAM
│       └── train_htr.py        # Reference CRNN + BiLSTM + CTC Keras model builder
├── frontend/
│   ├── index.html              # Modern single-page web interface
│   ├── script.js               # Event handlers, API caller & Canvas bounding box renderer
│   └── style.css               # Clean styling, responsive layout, animations
├── dataset/
│   └── README.md               # Directory specification for IAM line dataset
├── hcr_model.keras             # Root copy of legacy EMNIST character model
├── label_map.json              # Root copy of 62-class character label mapping
├── shkahn.jpeg                 # Sample test image
├── test_ocr.py                 # Standalone script for testing PaddleOCR engine
├── README.md                   # User-facing summary & installation guide
└── DEVELOPER.md                # Comprehensive technical documentation (This file)
```

---

## 3. The Core Transition: Character Classifier vs. Sequence OCR

### Legacy Architecture (Character Segmentation)
* **Pipeline:** OpenCV Binarization $\rightarrow$ Contour Extraction $\rightarrow$ Resize each contour to $28 \times 28$ $\rightarrow$ Feed to EMNIST-ByClass CNN (62 classes: 0-9, A-Z, a-z).
* **Failure Modes:**
  * Cannot process cursive or continuous handwriting (contours merge letters together).
  * Fails on disconnected dots and accents (the dot on 'i' or 'j' gets detected as an isolated character).
  * Vulnerable to lined notebook paper, smudges, and colored paper backgrounds.
  * Preserved in `backend/model/` for academic reference only.

### Active Production Architecture (Text-Level Sequence Recognition)
* **Pipeline:** Original/Normalized Color Image $\rightarrow$ Deep Learning Text Detection (DBNet / PP-OCR) $\rightarrow$ Text Sequence Recognition (CRNN / SVTR) $\rightarrow$ 2D Spatial Reading Order $\rightarrow$ Output text.
* **Advantages:**
  * Handles arbitrary handwriting styles, connected cursive, notebook lines, and varied lighting.
  * No manual character slicing; words and sentences are recognized as continuous temporal sequences.
  * Returns line bounding boxes and confidence scores directly.

---

## 4. End-to-End Processing Pipeline & Working Mechanics

### Step 1: Request Ingestion (`backend/app.py`)
1. User uploads an image via `POST /api/recognize` (or `/api/predict`).
2. Flask checks file presence under key `image` or `file`.
3. Image payload size validated against `MAX_IMAGE_BYTES` (default 15 MB).

### Step 2: Image Preprocessing (`backend/preprocessing.py`)
1. **Byte Decoding:** `cv2.imdecode` safely converts bytes into a 3-channel BGR NumPy array.
2. **Dimension Guards:**
   * Rejects images if `min(height, width) < 32px` (`MIN_IMAGE_SIDE`).
   * Proportionally downscales images if `max(height, width) > 5000px` (`MAX_IMAGE_SIDE`) using `cv2.INTER_AREA`.
3. **Uniform Border Trimming (`_trim_uniform_border`):**
   * Detects artificial scanner borders or solid camera margins.
   * Examines up to 250 boundary pixels on all four sides.
   * Drops edge rows/cols only if standard deviation $< 8.0$ and mean difference from median image intensity $< 18$.
   * Never crops ordinary ink or text.
4. **Illumination Normalization (`normalize_for_detection`):**
   * Converts BGR to LAB color space.
   * Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) **only to the Luminance (L) channel** (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
   * Leaves 'A' and 'B' color channels intact to preserve ink-to-paper color contrasts.
   * Converts back to BGR.
   * If `debug=True`, outputs `backend/debug/normalized.png`.

### Step 3: Text Detection & Recognition (`backend/text_detector.py`)
1. **Engine Initialization (`HandwritingOCR`):**
   * Instantiated as a singleton in `app.py` on first request (`get_ocr()`).
   * Configures PaddleOCR with:
     * `lang="en"`
     * `device="cpu"` (or configured GPU)
     * `use_doc_orientation_classify=False`, `use_doc_unwarping=False`, `use_textline_orientation=False`
     * `enable_mkldnn=False` (Critical: Prevents Intel oneDNN PIR executor crashes on Windows).
2. **Inference Execution:**
   * Calls `self._ocr.predict(image)`.
   * Unpacks JSON response payloads (`rec_texts`, `rec_scores`, `dt_polys` / `rec_polys`).
3. **Box Normalization & Filtering:**
   * Converts polygonal coordinates into standard rectangular bounding boxes: `[x, y, width, height]`.
   * Filters out any line with confidence $< 0.35$ (`OCR_SCORE_THRESHOLD`).

### Step 4: 2D Spatial Reading-Order Sorting (`_sort_lines`)
OCR detectors return text regions in arbitrary order. The engine restores human reading order (top-to-bottom, left-to-right):
1. Initial sort of all boxes by vertical coordinate ($Y$) then horizontal ($X$).
2. **Adaptive Row Grouping:**
   * Calculates the vertical center $c_y = y + h / 2$ for each box.
   * Compares $c_y$ against existing line groups.
   * Merges line into an existing group if $|c_y - g_y| \le \max(12, 0.55 \times g_h)$, where $g_y$ is the group's mean center and $g_h$ is the median box height.
   * Creates a new group if no match is found.
3. Groups are sorted by mean $Y$ coordinate (top to bottom).
4. Within each group, items are sorted by $X$ coordinate (left to right).
5. Limits output to `MAX_LINES` (default 100).
6. If `debug=True`, draws green bounding boxes and writes `backend/debug/detected_text_regions.png`.

### Step 5: Postprocessing & Formatting (`backend/postprocessing.py`)
1. **Text Normalization (`clean_text`):**
   * Converts `\r\n` and `\r` to standard `\n`.
   * Strips non-printable characters while preserving valid Unicode and whitespace.
   * Deduplicates inline spaces (`[ \t]+` $\rightarrow$ `" "`).
   * Normalizes surrounding line break spacing.
2. **Response Aggregation (`build_response`):**
   * Joins line texts with newlines.
   * Calculates mean confidence: $\frac{1}{N}\sum \text{confidence}_i$.
   * Packages detailed line dictionary: `{"text": ..., "confidence": ..., "box": [x, y, w, h]}`.

### Step 6: Frontend Rendering (`frontend/script.js`)
1. Drag-and-drop or file picker loads the local image via `FileReader`.
2. Image drawn to `#previewImg`.
3. `POST /api/recognize` dispatched via `fetch`.
4. On success:
   * Formatted transcription rendered in `#recognizedText`.
   * Confidence progress bar filled to match percentage.
   * Per-line badge chips displayed in `#lineGrid`.
   * **Canvas Overlay:** HTML5 `#overlayCanvas` dynamically resized to natural image dimensions. Draws rectangular bounding boxes (`#10b981`) and labels (`Line 1`, `Line 2`) directly over detected handwriting.
   * Copy button enabled with 1-click clipboard writing.

---

## 5. API Reference

### 1. Health Check
* **Endpoint:** `GET /api/health`
* **Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "engine": "PaddleOCR PP-OCRv6"
}
```

### 2. Text Recognition
* **Endpoint:** `POST /api/recognize` (Alias: `POST /api/predict`)
* **Headers:** `Content-Type: multipart/form-data`
* **Query Parameters:**
  * `debug` (optional, boolean `1`/`0`): Enables saving intermediate debug images to `backend/debug/`.
* **Form Body:**
  * `image` (or `file`): Binary image file (PNG, JPEG, WebP, BMP).
* **Success Response (200 OK):**
```json
{
  "success": true,
  "text": "Hello world\nThis is handwritten text",
  "confidence": 0.9245,
  "line_count": 2,
  "lines": [
    {
      "text": "Hello world",
      "confidence": 0.9412,
      "box": [45, 80, 320, 48]
    },
    {
      "text": "This is handwritten text",
      "confidence": 0.9078,
      "box": [48, 142, 510, 52]
    }
  ],
  "engine": "PaddleOCR PP-OCRv6",
  "processing_time": 0.84
}
```
* **No Text Detected Response (200 OK):**
```json
{
  "success": true,
  "text": "",
  "confidence": 0.0,
  "lines": [],
  "message": "No readable handwritten text detected.",
  "processing_time": 0.32
}
```
* **Error Response (400 Bad Request / 413 Too Large / 500 Server Error):**
```json
{
  "success": false,
  "text": "",
  "error": "Image is too large."
}
```

---

## 6. Configuration & Environment Variables

All parameters are centrally managed in [backend/config.py](file:///d:/hcr-project-complete-HTR-source/backend/config.py):

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MAX_IMAGE_BYTES` | Integer | `15728640` (15 MB) | Maximum upload payload size allowed by Flask. |
| `MAX_IMAGE_SIDE` | Integer | `5000` | Downscales images exceeding this side length (prevents OOM). |
| `MIN_IMAGE_SIDE` | Integer | `32` | Rejects images smaller than this pixel resolution. |
| `HCR_OCR_LANG` | String | `"en"` | Language code passed to PaddleOCR (e.g., `en`, `ch`, `french`). |
| `HCR_OCR_DEVICE` | String | `"cpu"` | Hardware device for OCR (`cpu` or `gpu`). |
| `HCR_OCR_SCORE_THRESHOLD` | Float | `0.35` | Minimum recognition score required to include a text box. |
| `HCR_MAX_LINES` | Integer | `100` | Maximum number of text lines returned in one request. |
| `HCR_DEBUG` | Boolean | `0` (`false`) | Set to `1` or `true` to save debug images in `backend/debug/`. |

---

## 7. Setup, Installation & Execution Guide

### Prerequisites
* Windows 10/11 or Linux
* Python 3.10 – 3.12
* PowerShell or Bash

### Installation (PowerShell)

```powershell
# 1. Navigate to project root
cd d:\hcr-project-complete-HTR-source

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If execution policy blocks activation:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### Running the Server

```powershell
cd backend
python app.py
```

* **Web UI:** Navigate to `http://127.0.0.1:5000`
* **API Health Check:** `http://127.0.0.1:5000/api/health`

### Testing via Command Line

```powershell
# Standalone model test
python test_ocr.py

# cURL recognition test
curl.exe -X POST -F "image=@shkahn.jpeg" http://127.0.0.1:5000/api/recognize
```

---

## 8. Custom HTR Model Training (CRNN + BiLSTM + CTC)

For domain-specific handwriting recognition where a custom neural network is required, the project provides a training framework in `backend/training/`.

### Architecture (`backend/training/train_htr.py`)
```
Input Image (Height: 48, Width: Variable, Channels: 1)
  │
  ├── Conv2D(64, 3x3) + BatchNorm + MaxPool(2x2)
  ├── Conv2D(128, 3x3) + BatchNorm + MaxPool(2x2)
  ├── Conv2D(256, 3x3) + BatchNorm + MaxPool(2x2)
  │
  ├── Permute & Reshape (Feature Sequence extraction)
  │
  ├── Bidirectional LSTM (256 units, dropout 0.2)
  ├── Bidirectional LSTM (256 units, dropout 0.2)
  │
  └── Dense (Num_Classes + 1 [CTC blank token], Softmax/Logits)
```

### Dataset Structure (`dataset/`)
Prepare line-level handwriting data (e.g. IAM dataset):
```text
dataset/
├── train/
│   ├── images/
│   │   ├── line001.png
│   │   └── line002.png
│   └── labels.csv       # Columns: filename,text
└── validation/
    ├── images/
    └── labels.csv
```

### Generating Model Architecture:
```powershell
python backend/training/train_htr.py --charset label_map.json --output backend/model/htr_model.keras
```

---

## 9. Critical Technical Notes & Troubleshooting

### 1. PaddlePaddle oneDNN / PIR CPU Crash on Windows
* **Symptom:** Unhandled C++ access violation or memory error when initializing `PaddleOCR` on Windows CPU.
* **Resolution:** Four mandatory environment flags are injected before importing `paddleocr` in [backend/text_detector.py](file:///d:/hcr-project-complete-HTR-source/backend/text_detector.py#L9-L15) and [test_ocr.py](file:///d:/hcr-project-complete-HTR-source/test_ocr.py#L4-L7):
  ```python
  os.environ["FLAGS_use_mkldnn"] = "0"
  os.environ["FLAGS_enable_pir_api"] = "0"
  os.environ["FLAGS_enable_pir_in_executor"] = "0"
  os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
  ```
  Additionally, `enable_mkldnn=False` is passed to the `PaddleOCR` constructor.

### 2. Blank or Low-Contrast Image Handling
* Naive HCR classifiers predict random noise characters on blank pages.
* This pipeline handles blank inputs safely: if the detector finds no text regions or all scores fall below `OCR_SCORE_THRESHOLD` (0.35), the system returns:
  ```json
  {"success": true, "text": "", "confidence": 0.0, "lines": [], "message": "No readable handwritten text detected."}
  ```

### 3. Debug Mode Inspection
* Set `$env:HCR_DEBUG="1"` before running `python app.py`, or append `?debug=1` to the request URL.
* Check `backend/debug/normalized.png` to inspect the CLAHE color normalization.
* Check `backend/debug/detected_text_regions.png` to view the exact bounding boxes identified by the detector.
