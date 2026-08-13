# Models

The legacy EMNIST model and label map remain here only for reference. They are **not** used by `/api/recognize`.

The active runtime uses PP-OCRv5 and downloads/cache its pretrained detector/recognizer through PaddleOCR on first use.

A custom CRNN+BiLSTM+CTC model can be trained from `backend/training/` using line-level handwriting data.
