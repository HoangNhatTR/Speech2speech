#!/usr/bin/env bash
# Tải NATS server dạng binary (KHÔNG cần Docker) — dùng cho máy server không cài được
# Docker. Thay cho `docker run ... nats:latest` trong README.md cũ. Chạy một lần, lưu
# vào bin/nats-server (thư mục riêng của repo, không cần sudo, không đụng hệ thống).
# scripts/run_all.sh tự tìm và chạy file này nếu có.
#
# Chạy: ./scripts/install_nats_server.sh
#
# Phiên bản cố định (không dùng "latest") để không bị đứt khi GitHub đổi cấu trúc asset
# giữa các release — nâng cấp: sửa NATS_VERSION bên dưới rồi chạy lại.

set -euo pipefail

NATS_VERSION="2.14.3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
DEST="$BIN_DIR/nats-server"

if [ -x "$DEST" ]; then
	echo "Đã có sẵn: $DEST"
	"$DEST" -v
	exit 0
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
	linux | darwin) ;;
	*)
		echo "Lỗi: hệ điều hành '$OS' chưa được script này hỗ trợ tự tải." >&2
		echo "Tải thủ công tại: https://github.com/nats-io/nats-server/releases/tag/v$NATS_VERSION" >&2
		exit 1
		;;
esac

MACHINE="$(uname -m)"
case "$MACHINE" in
	x86_64) ARCH="amd64" ;;
	aarch64 | arm64) ARCH="arm64" ;;
	armv7l) ARCH="arm7" ;;
	armv6l) ARCH="arm6" ;;
	i386 | i686) ARCH="386" ;;
	*)
		echo "Lỗi: kiến trúc '$MACHINE' chưa được script này hỗ trợ tự tải." >&2
		echo "Tải thủ công tại: https://github.com/nats-io/nats-server/releases/tag/v$NATS_VERSION" >&2
		exit 1
		;;
esac

ASSET="nats-server-v${NATS_VERSION}-${OS}-${ARCH}"
URL="https://github.com/nats-io/nats-server/releases/download/v${NATS_VERSION}/${ASSET}.tar.gz"

echo "Tải $URL ..."
mkdir -p "$BIN_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl -fSL "$URL" -o "$TMP_DIR/nats-server.tar.gz"
tar -xzf "$TMP_DIR/nats-server.tar.gz" -C "$TMP_DIR"
cp "$TMP_DIR/$ASSET/nats-server" "$DEST"
chmod +x "$DEST"

echo "Đã cài: $DEST"
"$DEST" -v
