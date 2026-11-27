#!/usr/bin/env sh
# One-command launcher for cmod-inference on Google Colab / Linux NVIDIA L4 (24 GB VRAM)
# with automatic CUDA install, model download + SHA-256 verification, and HTTP/2 Cloudflare Tunnel.
#
# Usage:
#   sh colab-l4.sh
#   CMOD_API_KEY="rahasia" CMOD_CTX=65536 CMOD_KV_QUANT=q4_0 sh colab-l4.sh
#   CF_TUNNEL_TOKEN="eyJ..." sh colab-l4.sh   # Optional: permanent Cloudflare Named Tunnel

set -eu

MODEL_ID="${CMOD_MODEL_ID:-qwen3-14b-doc}"
CTX="${CMOD_CTX:-32768}"
SLOTS="${CMOD_SLOTS:-2}"
KV_QUANT="${CMOD_KV_QUANT:-q8_0}"
KV_RAM_MB="${CMOD_KV_RAM_MB:-4096}"
PORT="${CMOD_PORT:-8790}"
API_KEY="${CMOD_API_KEY:-cmod123}"
CF_TOKEN="${CF_TUNNEL_TOKEN:-}"
MODELS_DIR="${CMOD_INFERENCE_MODELS_DIR:-$HOME/.cmod/models}"

echo "==> [1/5] Memeriksa cmod-inference..."
if ! command -v cmod-inference >/dev/null 2>&1; then
  curl -fsSL https://raw.githubusercontent.com/agungtp22/cmod-inference/main/install.sh | CMOD_ACCEL=cuda sh
fi

echo "==> [2/5] Memeriksa model ($MODEL_ID)..."
mkdir -p "$MODELS_DIR"
MODEL_FILE="$(ls "$MODELS_DIR"/*14B*.gguf 2>/dev/null | head -n 1 || true)"
if [ -z "$MODEL_FILE" ] || [ "$MODEL_ID" != "qwen3-14b-doc" ]; then
  cmod-inference pull "$MODEL_ID" --models-dir "$MODELS_DIR"
  MODEL_FILE="$(ls -t "$MODELS_DIR"/*.gguf 2>/dev/null | grep -v 'mmproj' | head -n 1 || true)"
fi

if [ -z "$MODEL_FILE" ]; then
  echo "Error: File model .gguf tidak ditemukan di $MODELS_DIR" >&2
  exit 1
fi

echo "==> [3/5] Memeriksa cloudflared..."
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

echo "==> [4/5] Menjalankan cmod-inference di background (port $PORT)..."
pkill -f "cmod-inference serve" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

nohup cmod-inference serve \
  --backend native \
  --model-path "$MODEL_FILE" \
  --ctx "$CTX" \
  --slots "$SLOTS" \
  --kv-cache-quant "$KV_QUANT" \
  --kv-cache-ram-mb "$KV_RAM_MB" \
  --verify-model \
  --bind 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  >/tmp/cmod.log 2>&1 &

READY=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "Error: cmod-inference gagal siap dalam 60 detik. Cek /tmp/cmod.log:" >&2
  tail -n 30 /tmp/cmod.log >&2 || true
  exit 1
fi
echo "    cmod-inference SIAP di :$PORT (model: $(basename "$MODEL_FILE"))"

echo "==> [5/5] Menjalankan Cloudflare Tunnel (HTTP/2 anti-drop)..."
if [ -n "$CF_TOKEN" ]; then
  nohup sh -c "while true; do cloudflared tunnel --no-autoupdate --protocol http2 run --token '$CF_TOKEN'; sleep 3; done" >/tmp/cloudflared.log 2>&1 &
  echo "========================================="
  echo "Mode     : Cloudflare Named Tunnel (permanen, auto-reconnect)"
  echo "API Key  : $API_KEY"
  echo "Log LLM  : tail -f /tmp/cmod.log"
  echo "Log CF   : tail -f /tmp/cloudflared.log"
  echo "========================================="
else
  nohup cloudflared tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:$PORT" >/tmp/cloudflared.log 2>&1 &
  URL=""
  for _ in $(seq 1 20); do
    URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -n 1 || true)"
    [ -n "$URL" ] && break
    sleep 1
  done
  echo "========================================="
  echo "Web UI   : ${URL:-Cek /tmp/cloudflared.log}"
  echo "Base URL : ${URL:-<URL>}/v1/"
  echo "API Key  : $API_KEY"
  echo "Log LLM  : tail -f /tmp/cmod.log"
  echo "Log CF   : tail -f /tmp/cloudflared.log"
  echo "========================================="
fi
