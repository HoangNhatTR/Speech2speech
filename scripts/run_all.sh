#!/usr/bin/env bash
# Khởi động toàn bộ hệ thống trong 1 lệnh, thay cho 5 terminal thủ công (xem README.md):
# NATS (binary, không cần Docker) -> runtime.dispatcher -> services.monitoring_service ->
# gateway.main -> frontend (SvelteKit dev server). Ctrl+C dừng tất cả cùng lúc, kể cả
# tiến trình con (npm spawn sh spawn node) — xem hàm start_bg/cleanup bên dưới về lý do
# dùng setsid.
#
# Chạy: ./scripts/run_all.sh [--skip-nats] [--skip-frontend]
# Log từng tiến trình: logs/<tên>.log (ghi đè mỗi lần chạy)
#
# Yêu cầu đã cài trước (xem README.md phần cài đặt):
#   - .venv với `pip install -r requirements.txt`
#   - frontend/node_modules với `cd frontend && npm install`
#   - bin/nats-server — chạy `./scripts/install_nats_server.sh` một lần (tải binary
#     NATS, không cần Docker/sudo). Nếu bạn tự quản lý NATS bằng cách khác (systemd,
#     Docker, máy chủ riêng), dùng `--skip-nats`.
#
# Chỉ dành cho Linux (dùng .venv/bin/activate + setsid, đặc thù util-linux) — trên
# Windows/macOS dùng terminal thủ công như README.md mô tả.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_NATS=false
SKIP_FRONTEND=false
for arg in "$@"; do
	case "$arg" in
		--skip-nats) SKIP_NATS=true ;;
		--skip-frontend) SKIP_FRONTEND=true ;;
		*)
			echo "Cờ không rõ: $arg (chỉ hỗ trợ --skip-nats, --skip-frontend)" >&2
			exit 1
			;;
	esac
done

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

HAVE_SETSID=true
if ! command -v setsid &>/dev/null; then
	HAVE_SETSID=false
	echo "Cảnh báo: không có 'setsid' — Ctrl+C có thể để sót tiến trình con (npm/vite)." >&2
fi

PIDS=()
CLEANED_UP=false

# Chạy 1 lệnh nền, gộp cả cây tiến trình con (npm -> sh -> node...) vào CÙNG 1 process
# group qua setsid, để cleanup() dừng được toàn bộ chỉ bằng 1 lần kill vào group đó —
# nếu chỉ kill PID gốc, tiến trình cháu (vd node/vite bị npm/sh fork ra) sẽ mồ côi và
# sống sót sau khi script thoát.
#
# QUAN TRỌNG: đặt PID vào biến toàn cục LAST_BG_PID thay vì `echo "$!"` để gọi viết
# `pid=$(start_bg ...)` — command substitution `$(...)` chạy trong SUBSHELL, nên lệnh
# `&` bên trong sẽ là job của subshell đó, không phải của shell chính đang chạy script.
# Hậu quả (đã xảy ra thật): `wait` ở cuối script không thấy job nào của chính nó để
# đợi (subshell đã thoát ngay sau khi echo xong), trả về ngay lập tức → script tưởng
# đã xong, chạy hết trap EXIT → tự dừng toàn bộ dịch vụ dù không ai bấm Ctrl+C.
start_bg() {
	local log_file="$1"
	shift
	if [ "$HAVE_SETSID" = true ]; then
		setsid "$@" >"$log_file" 2>&1 &
	else
		"$@" >"$log_file" 2>&1 &
	fi
	LAST_BG_PID=$!
}

# Cảnh báo ngay nếu tiến trình chết trong vài giây đầu (vd cổng đã bị chiếm) thay vì để
# lỗi chôn trong file log — nếu không sẽ tưởng "khởi động thành công" trong khi service
# đã chết ngay sau đó.
check_alive() {
	local name="$1" pid="$2" log_file="$3"
	if ! kill -0 "$pid" 2>/dev/null; then
		echo "Lỗi: [$name] đã thoát ngay sau khi khởi động — xem $log_file. Vài dòng cuối:" >&2
		tail -5 "$log_file" >&2 || true
		return 1
	fi
	return 0
}

# Kiểm tra cổng TCP đang có ai lắng nghe không, dùng /dev/tcp có sẵn của bash — không
# cần cài thêm netcat/nc (không phải máy server nào cũng có sẵn).
port_open() {
	timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

# Chỉ in banner "sẵn sàng" sau khi model server thực sự mở cổng. TTS phải load +
# warm-up; vLLM phải load checkpoint + compile/capture CUDA graph nên có thể mất vài
# phút ở lần đầu. In tiến độ mỗi 30s để không tạo cảm giác script bị treo.
wait_for_port() {
	local name="$1" port="$2" timeout_s="$3" log_file="$4"
	local started_at=$SECONDS last_update=$SECONDS
	while ! port_open "$port"; do
		if ((SECONDS - started_at >= timeout_s)); then
			echo "[$name] Cảnh báo: chưa sẵn sàng sau ${timeout_s}s — xem $log_file" >&2
			return 1
		fi
		if ((SECONDS - last_update >= 30)); then
			echo "[$name] Vẫn đang khởi động ($((SECONDS - started_at))s), xem $log_file..."
			last_update=$SECONDS
		fi
		sleep 2
	done
	echo "[$name] Sẵn sàng trên cổng $port ($((SECONDS - started_at))s chờ)."
}

cleanup() {
	if [ "$CLEANED_UP" = true ]; then
		return
	fi
	CLEANED_UP=true
	echo ""
	echo "Đang dừng tất cả tiến trình..."
	for pid in "${PIDS[@]}"; do
		if [ "$HAVE_SETSID" = true ]; then
			kill -TERM -- "-$pid" 2>/dev/null || true
		else
			pkill -P "$pid" 2>/dev/null || true
			kill -TERM "$pid" 2>/dev/null || true
		fi
	done
	sleep 1
	for pid in "${PIDS[@]}"; do
		if [ "$HAVE_SETSID" = true ]; then
			kill -KILL -- "-$pid" 2>/dev/null || true
		else
			kill -KILL "$pid" 2>/dev/null || true
		fi
	done
	wait 2>/dev/null || true
	echo "Đã dừng."
}
trap cleanup EXIT INT TERM

# --- 1. NATS (binary, không cần Docker) ---
NATS_BIN=""
if [ -x "$ROOT_DIR/bin/nats-server" ]; then
	NATS_BIN="$ROOT_DIR/bin/nats-server"
elif command -v nats-server &>/dev/null; then
	NATS_BIN="$(command -v nats-server)"
fi

if [ "$SKIP_NATS" = true ]; then
	echo "[nats] Bỏ qua (--skip-nats) — giả định bạn đã tự chạy NATS ở nats://localhost:4222."
elif port_open 4222; then
	echo "[nats] Đã có tiến trình lắng nghe cổng 4222 — coi như NATS đã chạy."
elif [ -z "$NATS_BIN" ]; then
	echo "[nats] Cảnh báo: chưa có nats-server. Chạy './scripts/install_nats_server.sh' rồi" >&2
	echo "        chạy lại (hoặc dùng --skip-nats nếu tự quản lý NATS bằng cách khác)." >&2
	echo "        Tool calling sẽ không hoạt động cho tới khi NATS chạy." >&2
else
	echo "[nats] Khởi động $NATS_BIN..."
	mkdir -p "$ROOT_DIR/data/nats"
	start_bg "$LOG_DIR/nats.log" "$NATS_BIN" -js -sd "$ROOT_DIR/data/nats" -p 4222 -m 8222
	PIDS+=("$LAST_BG_PID")
	sleep 1
	check_alive nats "$LAST_BG_PID" "$LOG_DIR/nats.log" || true
fi

# --- 2. Python venv ---
if [ ! -f "$ROOT_DIR/.venv/bin/activate" ]; then
	echo "Lỗi: chưa có .venv — chạy 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt' trước." >&2
	exit 1
fi
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

# print() mặc định full-buffered khi stdout không phải TTY (ghi ra file log) — nếu
# process bị kill trước khi buffer đầy/thoát sạch, log sẽ trống dù có lỗi thật xảy ra.
# Tắt buffering cho mọi tiến trình Python chạy nền bên dưới để log luôn phản ánh đúng.
export PYTHONUNBUFFERED=1

if [ ! -f "$ROOT_DIR/.env" ]; then
	echo "Cảnh báo: chưa có .env (cp .env.example .env rồi điền API key)." >&2
	echo "          Dashboard/Settings/benchmark miễn phí vẫn chạy được, nhưng voice chat thật sẽ lỗi." >&2
fi

# --- TTS tự host (VieNeu-TTS), chỉ khi .env bật TTS_BACKEND=local ---
# venv riêng (.venv-tts) — xem selfhost/tts_server.py về lý do tách venv. Không dùng
# start_bg (setsid) trực tiếp trên "source ... && python ..." vì cần source activate
# script trước; bọc trong "bash -c" để cả hai lệnh chạy trong CÙNG process group.
if [ -f "$ROOT_DIR/.env" ] && grep -qx "TTS_BACKEND=local" "$ROOT_DIR/.env"; then
	if port_open 8100; then
		echo "[tts-local] Đã có tiến trình lắng nghe cổng 8100 — coi như TTS server đã chạy."
	elif [ ! -f "$ROOT_DIR/.venv-tts/bin/activate" ]; then
		echo "[tts-local] Cảnh báo: TTS_BACKEND=local nhưng chưa có .venv-tts. Chạy:" >&2
		echo "        python -m venv .venv-tts && .venv-tts/bin/pip install -r requirements-tts.txt" >&2
		echo "        (model VieNeu-TTS tự tải về ở lần chạy server đầu tiên, không cần script riêng)" >&2
		echo "        Voice chat sẽ lỗi TTS cho tới khi chạy xong các bước trên." >&2
	else
		echo "[tts-local] Khởi động selfhost/tts_server.py (.venv-tts, tải model lần đầu có thể mất chút)..."
		start_bg "$LOG_DIR/tts_local.log" bash -c \
			"source '$ROOT_DIR/.venv-tts/bin/activate' && exec python '$ROOT_DIR/selfhost/tts_server.py'"
		PIDS+=("$LAST_BG_PID")
		sleep 3
		check_alive tts-local "$LAST_BG_PID" "$LOG_DIR/tts_local.log" || true
	fi
fi

# --- LLM tự host (Qwen3-8B-AWQ qua vLLM), chỉ khi .env bật LLM_BACKEND=local ---
# venv riêng (.venv-vllm) — xem docs/platform-architecture.md mục "Giai đoạn 1". Khởi
# động song song với các service khác; trước banner cuối, script sẽ đợi cổng 8000 thật
# sự sẵn sàng để người dùng không mở voice chat trong lúc model còn compile.
if [ -f "$ROOT_DIR/.env" ] && grep -qx "LLM_BACKEND=local" "$ROOT_DIR/.env"; then
	if port_open 8000; then
		echo "[vllm] Đã có tiến trình lắng nghe cổng 8000 — coi như vLLM server đã chạy."
	elif [ ! -f "$ROOT_DIR/.venv-vllm/bin/activate" ]; then
		echo "[vllm] Cảnh báo: LLM_BACKEND=local nhưng chưa có .venv-vllm. Chạy:" >&2
		echo "        python -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements-vllm.txt" >&2
		echo "        Voice chat sẽ lỗi LLM cho tới khi chạy xong bước trên." >&2
	else
		vllm_model=$(grep -m1 "^VLLM_MODEL=" "$ROOT_DIR/.env" | cut -d= -f2-)
		vllm_model="${vllm_model:-Qwen/Qwen3-8B-AWQ}"
		vllm_mem=$(grep -m1 "^VLLM_GPU_MEMORY_UTILIZATION=" "$ROOT_DIR/.env" | cut -d= -f2-)
		vllm_mem="${vllm_mem:-0.15}"
		echo "[vllm] Khởi động $vllm_model (.venv-vllm) — có thể mất vài phút biên dịch kernel;" \
			"readiness gate sẽ đợi cổng 8000, xem $LOG_DIR/vllm.log để theo dõi tiến độ..."
		start_bg "$LOG_DIR/vllm.log" bash -c \
			"source '$ROOT_DIR/.venv-vllm/bin/activate' && exec vllm serve '$vllm_model' \
			--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
			--default-chat-template-kwargs '{\"enable_thinking\": false}' \
			--gpu-memory-utilization '$vllm_mem' --max-model-len 8192"
		PIDS+=("$LAST_BG_PID")
		sleep 3
		check_alive vllm "$LAST_BG_PID" "$LOG_DIR/vllm.log" || true
	fi
fi

echo "[dispatcher] Khởi động..."
start_bg "$LOG_DIR/dispatcher.log" python -m runtime.dispatcher
dispatcher_pid="$LAST_BG_PID"
PIDS+=("$dispatcher_pid")

echo "[monitoring] Khởi động..."
start_bg "$LOG_DIR/monitoring.log" python -m services.monitoring_service
monitoring_pid="$LAST_BG_PID"
PIDS+=("$monitoring_pid")

sleep 2 # để dispatcher/monitoring kết nối NATS xong trước khi gateway nhận request đầu tiên
check_alive dispatcher "$dispatcher_pid" "$LOG_DIR/dispatcher.log" || true
check_alive monitoring "$monitoring_pid" "$LOG_DIR/monitoring.log" || true

echo "[gateway] Khởi động..."
start_bg "$LOG_DIR/gateway.log" python -m gateway.main
PIDS+=("$LAST_BG_PID")
sleep 2
check_alive gateway "$LAST_BG_PID" "$LOG_DIR/gateway.log" || true

# --- 3. Frontend (SvelteKit) ---
if [ "$SKIP_FRONTEND" = true ]; then
	echo "[frontend] Bỏ qua (--skip-frontend)."
elif [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
	echo "Lỗi: frontend/node_modules chưa có — chạy 'cd frontend && npm install' trước." >&2
	exit 1
else
	echo "[frontend] Khởi động..."
	start_bg "$LOG_DIR/frontend.log" npm --prefix "$ROOT_DIR/frontend" run dev -- --host 127.0.0.1
	PIDS+=("$LAST_BG_PID")
	sleep 2
	check_alive frontend "$LAST_BG_PID" "$LOG_DIR/frontend.log" || true
fi

# Model readiness gate — các service đã được khởi động song song ở trên nên thời gian
# chờ ở đây chỉ là phần warm-up còn lại. Timeout không kill cả dashboard/gateway; banner
# vẫn hiện cùng cảnh báo rõ ràng và đường dẫn log để debug.
if [ -f "$ROOT_DIR/.env" ] && grep -qx "TTS_BACKEND=local" "$ROOT_DIR/.env"; then
	wait_for_port tts-local 8100 300 "$LOG_DIR/tts_local.log" || true
fi
if [ -f "$ROOT_DIR/.env" ] && grep -qx "LLM_BACKEND=local" "$ROOT_DIR/.env"; then
	wait_for_port vllm 8000 1200 "$LOG_DIR/vllm.log" || true
fi

# vLLM load/compile sau TTS có thể làm lượt CUDA đầu tiếp theo của VieNeu chậm lại dù
# TTS đã warm-up trước đó. Khi cả hai cùng local, chạy một câu bỏ đi SAU readiness vLLM
# để lời chào thật của người dùng không chịu cold outlier này.
if [ -f "$ROOT_DIR/.env" ] \
	&& grep -qx "TTS_BACKEND=local" "$ROOT_DIR/.env" \
	&& grep -qx "LLM_BACKEND=local" "$ROOT_DIR/.env" \
	&& command -v curl &>/dev/null \
	&& port_open 8100 \
	&& port_open 8000; then
	echo "[tts-local] Warm-up lại sau khi vLLM sẵn sàng..."
	if ! curl -fsS --max-time 120 -X POST \
		-H "Content-Type: application/json" \
		-d '{"text":"Xin chào."}' \
		-o /dev/null http://127.0.0.1:8100/synthesize/stream; then
		echo "[tts-local] Cảnh báo: post-vLLM warm-up lỗi; xem $LOG_DIR/tts_local.log" >&2
	fi
fi

echo ""
echo "=================================================================="
echo " Tất cả tiến trình đã khởi động (xem cảnh báo/lỗi ở trên nếu có)."
echo " Log chi tiết: $LOG_DIR/*.log"
echo ""
echo "  Voice chat (WebRTC):      http://localhost:7860/client"
echo "  Dashboard/Settings API:   http://localhost:7860/api/status"
if [ "$SKIP_FRONTEND" = false ]; then
	echo "  Dashboard/Settings (FE):  xem $LOG_DIR/frontend.log để lấy URL (thường http://127.0.0.1:5173)"
fi
if [ -f "$ROOT_DIR/.env" ] && grep -qx "LLM_BACKEND=local" "$ROOT_DIR/.env"; then
	echo "  vLLM (LLM local):         http://localhost:8000 (đã qua readiness gate)"
fi
echo ""
echo " Nhấn Ctrl+C để dừng toàn bộ."
echo "=================================================================="

wait
