"""Tải model ASR tiếng Việt (Zipformer-30M-RNNT-6000h, int8, ~26MB nén) cho
selfhost/asr.py. Chạy một lần: python scripts/download_asr_model.py
"""

import tarfile
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
)
DEST_DIR = Path(__file__).resolve().parent.parent / "models" / "asr-vi"
EXTRACTED_NAME = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"


def main() -> None:
    model_dir = DEST_DIR / EXTRACTED_NAME
    if (model_dir / "tokens.txt").exists():
        print(f"Model đã có sẵn: {model_dir}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DEST_DIR / "model.tar.bz2"

    print(f"Đang tải {URL} ...")
    urllib.request.urlretrieve(URL, archive_path)

    print("Đang giải nén...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(DEST_DIR)

    archive_path.unlink()
    print(f"Xong. Model tại: {model_dir}")
    print(f"Đặt ASR_MODEL_DIR={model_dir} trong .env (đường dẫn tương đối cũng được).")


if __name__ == "__main__":
    main()
