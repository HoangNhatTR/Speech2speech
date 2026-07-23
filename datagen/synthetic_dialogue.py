"""Giai đoạn 3 — Sinh dữ liệu hội thoại full-duplex 2 kênh tiếng Việt TỔNG HỢP
(docs/roadmap.md mục 8, điểm 1): "Hội thoại full-duplex 2 kênh tiếng Việt cho giai đoạn
3 — cách làm giống Moshi: ... sinh hội thoại tổng hợp (LLM viết kịch bản có ngắt lời,
backchannel → TTS đa giọng render thành 2 stream)."

Đây là việc chuẩn bị DỮ LIỆU, không phải huấn luyện model — không cần GPU, không cần
chọn trước lựa chọn (a)/(b)/(c) ở roadmap.md mục 2 (dữ liệu này cần cho cả 3), nên khác
với LLM/TTS tự host của Giai đoạn 1, phần này có thể chạy thật NGAY nếu có
ANTHROPIC_API_KEY + ELEVENLABS_API_KEY (tốn phí 2 API đó theo số hội thoại sinh ra).

Cách hoạt động:
  1. Xin Claude (qua services/llm_client.py::complete(), đã verify chạy trong Track B)
     viết kịch bản hội thoại ngắn giữa "user" và "assistant", đánh dấu mỗi lượt là
     normal / backchannel / interrupts_previous.
  2. Tổng hợp audio từng lượt bằng ElevenLabs (gọi thẳng REST API, KHÔNG qua
     pipecat.services.elevenlabs vì dịch vụ đó là streaming frame-based cho hội thoại
     real-time, không hợp để sinh file hàng loạt ở đây). URL/header/payload/cách đọc
     response (JSON-lines có "audio_base64") sao chép đúng từ
     `pipecat/services/elevenlabs/tts.py` (đã đọc source trong .venv của repo này khi
     viết file này — không đoán field name).
  3. Xếp audio 2 giọng vào 1 file WAV stereo (kênh trái = user, kênh phải = assistant)
     theo mốc thời gian THẬT, có chồng lấn (overlap) ở lượt backchannel/interrupt thay
     vì chỉ nối chuỗi — để file có tín hiệu "nói chồng" giống hội thoại full-duplex thật,
     dùng làm nhãn huấn luyện được (mỗi lượt kèm start_s/duration_s/type trong .json).

CHƯA chạy thử trong lần build này (tốn phí API thật mỗi lần chạy — không tự ý gọi khi
chưa được xác nhận). Đã verify: (a) logic ghép JSON kịch bản + dựng buffer stereo bằng
audio giả (xem test khi build file này); (b) request shape ElevenLabs khớp
`pipecat/services/elevenlabs/tts.py` cài trong .venv. CHƯA verify: chất lượng kịch bản
Claude sinh ra có thực tế/đa dạng đủ để làm dữ liệu huấn luyện hay không — cần chạy thử
một mẻ nhỏ rồi đọc lại bằng tai/mắt trước khi sinh số lượng lớn.

Chạy: python -m datagen.synthetic_dialogue --n 5 --out datagen/output
"""

import argparse
import asyncio
import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import aiohttp
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from loguru import logger

from services.llm_client import complete

load_dotenv(override=True)

SAMPLE_RATE = 24000
GAP_NORMAL_S = 0.3
OVERLAP_BACKCHANNEL_S = 0.4
OVERLAP_INTERRUPT_S = 0.6

TurnType = Literal["normal", "backchannel", "interrupts_previous"]

SCRIPT_SYSTEM_PROMPT = """Bạn viết kịch bản hội thoại thoại tiếng Việt ngắn (6-10 lượt)
giữa NGƯỜI DÙNG và TRỢ LÝ giọng nói của một app ngân hàng, mô phỏng hội thoại full-duplex
tự nhiên có xen backchannel ("dạ", "vâng ạ", "ừm") và đôi khi người dùng ngắt lời trợ lý
giữa câu.

Trả lời DUY NHẤT một JSON array (không markdown, không giải thích thêm), mỗi phần tử có
đúng 3 khoá:
{"speaker": "user"|"assistant", "text": "...", "type": "normal"|"backchannel"|"interrupts_previous"}

Quy tắc:
- "backchannel": lượt rất ngắn (1-3 từ, vd "dạ", "vâng ạ", "ừm") chen vào khi BÊN KIA
  đang nói, không mang nội dung mới, không cắt ngang ý của họ.
- "interrupts_previous": lượt cắt ngang lượt ngay trước — bên bị cắt xem như đang nói dở
  câu (nội dung lượt bị cắt vẫn viết trọn vẹn, phần cắt xử lý ở khâu dựng audio).
- "normal": lượt bình thường, nói sau khi bên kia nói xong.
- Chủ đề: ngân hàng/tài chính cá nhân (chuyển tiền, tra số dư, lịch sử giao dịch, hoá
  đơn, thẻ...). Xưng hô tự nhiên như hội thoại call center Việt Nam thật.
"""


@dataclass
class Turn:
    speaker: Literal["user", "assistant"]
    text: str
    type: TurnType
    start_s: float = 0.0
    duration_s: float = 0.0


def parse_script(raw: str) -> list[Turn]:
    """Tách JSON kịch bản ra khỏi text LLM trả về (Claude đôi khi bọc thêm ```json)."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    data = json.loads(cleaned.strip())
    return [
        Turn(speaker=item["speaker"], text=item["text"], type=item.get("type", "normal"))
        for item in data
    ]


async def generate_script(topic_hint: str = "") -> list[Turn]:
    """Sinh kịch bản hội thoại bằng LLM (Claude qua services/llm_client.py, đã verify
    chạy được ở Track B của docs/platform-architecture.md)."""
    prompt = f"Chủ đề gợi ý: {topic_hint}" if topic_hint else "Tự chọn một chủ đề ngân hàng phù hợp."
    raw = await complete(SCRIPT_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=1500)
    return parse_script(raw)


async def synthesize_wav(session: aiohttp.ClientSession, text: str, voice_id: str) -> np.ndarray:
    """Gọi ElevenLabs REST API trực tiếp — request/response shape sao chép từ
    `HttpTTSService` thật trong pipecat/services/elevenlabs/tts.py (endpoint
    /stream/with-timestamps, JSON-lines chứa "audio_base64" mỗi dòng), KHÔNG đoán."""
    api_key = os.environ["ELEVENLABS_API_KEY"]
    model = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream/with-timestamps"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model, "language_code": "vi"}
    params = {"output_format": f"pcm_{SAMPLE_RATE}"}

    audio_chunks = []
    async with session.post(url, json=payload, headers=headers, params=params) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"ElevenLabs lỗi ({response.status}): {body}")
        async for line in response.content:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            if data.get("audio_base64"):
                audio_chunks.append(base64.b64decode(data["audio_base64"]))

    raw = b"".join(audio_chunks)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def place_dialogue_timeline(turns_and_audio: list[tuple[Turn, np.ndarray]]) -> np.ndarray:
    """Xếp audio từng lượt vào buffer stereo (trái=user, phải=assistant) theo mốc thời
    gian có chồng lấn thật cho backchannel/interrupt. Tách khỏi phần gọi mạng để test
    được bằng audio giả (xem test khi build file này)."""
    t = 0.0
    prev_duration = 0.0
    placed: list[tuple[Turn, np.ndarray, float]] = []

    for turn, audio in turns_and_audio:
        duration = len(audio) / SAMPLE_RATE
        if turn.type == "backchannel":
            start = max(0.0, t - min(OVERLAP_BACKCHANNEL_S, prev_duration * 0.5))
        elif turn.type == "interrupts_previous":
            start = max(0.0, t - min(OVERLAP_INTERRUPT_S, prev_duration * 0.5))
        else:
            start = t

        turn.start_s = round(start, 3)
        turn.duration_s = round(duration, 3)
        placed.append((turn, audio, start))
        t = start + duration + GAP_NORMAL_S
        prev_duration = duration

    total_samples = int((t + 1.0) * SAMPLE_RATE) or 1
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    for turn, audio, start in placed:
        channel = 0 if turn.speaker == "user" else 1
        start_sample = int(start * SAMPLE_RATE)
        end_sample = start_sample + len(audio)
        stereo[start_sample:end_sample, channel] += audio

    return stereo


async def render_dialogue(
    turns: list[Turn], user_voice_id: str, assistant_voice_id: str
) -> np.ndarray:
    async with aiohttp.ClientSession() as session:
        turns_and_audio = []
        for turn in turns:
            voice_id = user_voice_id if turn.speaker == "user" else assistant_voice_id
            audio = await synthesize_wav(session, turn.text, voice_id)
            turns_and_audio.append((turn, audio))
    return place_dialogue_timeline(turns_and_audio)


async def generate_one(index: int, out_dir: Path, user_voice_id: str, assistant_voice_id: str) -> None:
    turns = await generate_script()
    stereo = await render_dialogue(turns, user_voice_id, assistant_voice_id)

    wav_path = out_dir / f"dialogue_{index:04d}.wav"
    json_path = out_dir / f"dialogue_{index:04d}.json"

    sf.write(wav_path, stereo, SAMPLE_RATE, subtype="PCM_16")
    json_path.write_text(
        json.dumps([asdict(t) for t in turns], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Đã sinh: {wav_path} ({len(turns)} lượt)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="Số hội thoại cần sinh")
    parser.add_argument("--out", type=Path, default=Path("datagen/output"))
    args = parser.parse_args()

    user_voice_id = os.environ["DATAGEN_USER_VOICE_ID"]
    assistant_voice_id = os.getenv("ELEVENLABS_VOICE_ID") or os.environ["DATAGEN_ASSISTANT_VOICE_ID"]

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        await generate_one(i, args.out, user_voice_id, assistant_voice_id)


if __name__ == "__main__":
    asyncio.run(main())
