#!/usr/bin/env bash
# Stage 2: launch one vLLM server, sweep client concurrency against it, shut it down.
#
# One server process per model, so the KV-cache / state footprint scraped from
# /metrics belongs unambiguously to one architecture.
#
#   ./scripts/run_stage2.sh state-spaces/mamba2-2.7b
#   MODEL_ARGS="--enforce-eager" ./scripts/run_stage2.sh moonshotai/Kimi-Linear-48B-A3B-Instruct
#
# On SM12x, CUDA-graph capture paths written for datacenter Blackwell may fail; if the
# server aborts during capture, re-run with MODEL_ARGS="--enforce-eager".
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:?usage: run_stage2.sh <model-id> [extra client args...]}"
shift || true

PORT="${PORT:-8000}"
ENDPOINT="http://localhost:${PORT}/v1"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64}"
INPUT_TOKENS="${INPUT_TOKENS:-4096}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-results/raw/stage2}"
MODEL_ARGS="${MODEL_ARGS:-}"
TAG="${TAG:-$(echo "${MODEL}" | tr '/' '_')}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/${TAG}.server.log}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

mkdir -p "${OUTPUT_DIR}"

echo "== Launching vLLM: ${MODEL} on port ${PORT}"
# shellcheck disable=SC2086
uv run vllm serve "${MODEL}" \
  --trust-remote-code \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${MODEL_ARGS} > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "== Stopping vLLM (pid ${SERVER_PID})"; kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

echo "== Waiting for the server to become ready (log: ${SERVER_LOG})"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "error: vLLM exited during startup; last lines of ${SERVER_LOG}:" >&2
    tail -40 "${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "error: server not ready after ${STARTUP_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done
echo "== Server ready"

# vLLM prints the KV/state cache size at startup; pass it through so the client can
# turn the reported usage fraction into an absolute GB figure.
# vLLM 0.27 logs "Available KV cache memory: N.NN GiB"; older builds logged the size
# on the "GPU KV cache size" line. Try the current format first, then the legacy one.
KV_CACHE_GB="$(grep -oE 'Available KV cache memory: [0-9.]+' "${SERVER_LOG}" | tail -1 | grep -oE '[0-9.]+' || true)"
if [[ -z "${KV_CACHE_GB}" ]]; then
  KV_CACHE_GB="$(grep -oE 'GPU KV cache size: [0-9.]+ GiB' "${SERVER_LOG}" | tail -1 | grep -oE '[0-9.]+' || true)"
fi
KV_ARG=()
if [[ -n "${KV_CACHE_GB}" ]]; then
  KV_ARG=(--kv-cache-size-gb "${KV_CACHE_GB}")
  echo "== Reported KV/state cache size: ${KV_CACHE_GB} GB"
fi

uv run python -m src.stage2_serving.benchmark_vllm \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --input-tokens "${INPUT_TOKENS}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  "${KV_ARG[@]}" \
  "$@"

echo
echo "== Rendering figures"
uv run python -m src.utils.plotting \
  --stage2 "${OUTPUT_DIR}/${TAG}.json" \
  --output-dir "${PLOT_DIR:-results/plots}"
