import os

# Paddle/PIR/oneDNN compatibility settings
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

print("Starting PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)

print("PaddleOCR initialized successfully!")

print("Running OCR on shkahn.jpeg...")

result = ocr.predict("shkahn.jpeg")

print("OCR completed successfully!")

for item in result:
    print(item)