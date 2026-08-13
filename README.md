# Handwritten Text Recognition (HTR)

This project replaces the old contour → 28x28 → EMNIST character classifier with a text-level OCR pipeline.

## Runtime architecture

```text
Uploaded image
  -> validation
  -> conservative color/illumination normalization
  -> PP-OCRv5 deep-learning text detection/recognition
  -> line ordering
  -> conservative text cleanup
  -> final text
```

The active API does **not** assume one contour equals one character and does not use the legacy EMNIST model.

## Why this is better

The old system could only classify isolated character crops. It therefore failed on cursive/connected writing, dots, touching characters, notebook lines and colored backgrounds. The new system detects text regions and recognizes text sequences instead of forcing character segmentation.

## Install on Windows PowerShell

```powershell
cd hcr-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Run

```powershell
cd backend
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/api/health
```

## API

```powershell
curl.exe -X POST -F "image=@C:\path\to\handwriting.png" http://127.0.0.1:5000/api/recognize
```

Response:

```json
{
  "success": true,
  "text": "Hello World",
  "confidence": 0.91,
  "lines": [
    {"text":"Hello World", "confidence":0.91, "box":[10,20,400,60]}
  ]
}
```

A blank/no-text image returns an empty text string and a clear message instead of random EMNIST characters.

## Debug

Run:

```powershell
$env:HCR_DEBUG="1"
python app.py
```

Or append `?debug=1` to the API request. Debug images are written to `backend/debug/`.

## Custom model

`backend/training/` contains the CRNN + BiLSTM + CTC architecture. Train it on line-level handwriting data such as IAM. EMNIST-ByClass is not a sufficient final HTR dataset.

## Limitations

No OCR system can guarantee perfect recognition of arbitrary handwriting. Very faint, severely occluded, extremely cursive or unseen scripts require appropriate training data. The first runtime is English-focused.
