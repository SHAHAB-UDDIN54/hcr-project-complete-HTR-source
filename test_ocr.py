import os
import time

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

print("Starting PaddleOCR...")

start = time.perf_counter()

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)

print(
    f"OCR model loaded in "
    f"{time.perf_counter() - start:.2f} seconds"
)

print("Running OCR...")

start = time.perf_counter()

result = ocr.predict("shkahn.jpeg")

elapsed = time.perf_counter() - start

print(f"OCR prediction time: {elapsed:.2f} seconds")

for item in result:
    print(item)