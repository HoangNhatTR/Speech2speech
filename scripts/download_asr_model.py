"""Tải model ASR tiếng Việt cho selfhost/asr.py. Hai lựa chọn — mặc định là bản lớn hơn:

  large (mặc định) — Zipformer train trên ~70.000 giờ dữ liệu (zzasdf/viet_iter3_pseudo_label
    qua k2-fsa/sherpa-onnx, ~68MB nén). ĐÃ ĐO bằng eval/asr_wer.py: WER tổng 2.7% (so với
    3.5% của bản small), code-switch VN-EN 14.3% (so với 18.1%), latency ~65ms/câu (so với
    ~40ms) — chấp nhận chậm hơn một chút để đổi lấy độ chính xác tốt hơn rõ rệt, đặc biệt
    quan trọng cho giọng nói thật/tự nhiên (bộ dữ liệu train lớn hơn ~10x nên khái quát hoá
    tốt hơn nhiều so với benchmark tổng hợp gTTS thể hiện).
  small — Zipformer-30M-RNNT train trên ~6.000 giờ (~26MB nén), nhanh hơn một chút
    (~40ms/câu) nhưng WER kém hơn trên mọi domain đã đo — chỉ dùng nếu máy rất yếu.

Chạy: python scripts/download_asr_model.py [--variant large|small]
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

VARIANTS = {
    "large": {
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2"
        ),
        "extracted_name": "sherpa-onnx-zipformer-vi-int8-2025-04-20",
        # Tên file thật trong archive khác quy ước selfhost/asr.py mong đợi — đổi tên
        # sau khi giải nén để code không cần biết chi tiết từng bản model.
        "rename": {
            "encoder-epoch-12-avg-8.int8.onnx": "encoder.int8.onnx",
            "decoder-epoch-12-avg-8.onnx": "decoder.onnx",
            "joiner-epoch-12-avg-8.int8.onnx": "joiner.int8.onnx",
        },
    },
    "small": {
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
        ),
        "extracted_name": "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        "rename": {},
    },
}

DEST_DIR = Path(__file__).resolve().parent.parent / "models" / "asr-vi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS.keys(), default="large")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    model_dir = DEST_DIR / variant["extracted_name"]
    if (model_dir / "tokens.txt").exists():
        print(f"Model đã có sẵn: {model_dir}")
        print(f"Đặt ASR_MODEL_DIR={model_dir} trong .env (đường dẫn tương đối cũng được).")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DEST_DIR / "model.tar.bz2"

    print(f"Đang tải {variant['url']} ...")
    urllib.request.urlretrieve(variant["url"], archive_path)

    print("Đang giải nén...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(DEST_DIR)
    archive_path.unlink()

    for old_name, new_name in variant["rename"].items():
        old_path = model_dir / old_name
        if old_path.exists():
            old_path.rename(model_dir / new_name)

    print(f"Xong. Model tại: {model_dir}")
    print(f"Đặt ASR_MODEL_DIR={model_dir} trong .env (đường dẫn tương đối cũng được).")


if __name__ == "__main__":
    main()
