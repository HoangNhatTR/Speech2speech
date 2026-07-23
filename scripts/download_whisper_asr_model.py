"""Tải model Whisper large-v3-turbo dạng ONNX (bản int8, đã convert sẵn bởi k2-fsa, từ
`csukuangfj/sherpa-onnx-whisper-turbo` trên HuggingFace, ~1GB) cho selfhost/asr.py —
lựa chọn ASR local THỨ HAI bên cạnh Zipformer-30M-RNNT (xem docs/roadmap.md mục 9,
"Đánh giá và các hướng ra paper" — đo trước khi đổi model, không đoán). Whisper
large-v3-turbo là model đa ngôn ngữ tổng quát (KHÔNG train riêng cho tiếng Việt như
Zipformer) — CHƯA có benchmark WER tiếng Việt công khai nào, nên dùng
`eval/asr_wer.py` để tự đo so với Zipformer trước khi coi đây là lựa chọn mặc định.

Chạy một lần: python scripts/download_whisper_asr_model.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main"
FILES = ["turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"]
DEST_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-turbo"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = DEST_DIR / filename
        if dest.exists():
            print(f"Đã có sẵn: {dest}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Đang tải {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)

    print(f"\nXong. Đặt trong .env:")
    print(f"  ASR_LOCAL_ENGINE=whisper")
    print(f"  WHISPER_ASR_MODEL_DIR={DEST_DIR}")
    print(
        "\nLưu ý: Whisper large-v3-turbo là model đa ngôn ngữ tổng quát, KHÔNG train "
        "riêng cho tiếng Việt như Zipformer — chạy `python -m eval.asr_wer` với từng "
        "engine để tự đo WER trước khi coi đây là lựa chọn mặc định."
    )


if __name__ == "__main__":
    main()
