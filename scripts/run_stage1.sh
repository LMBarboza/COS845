#!/usr/bin/env bash
# Stage 1: component-level microbenchmark sweep, then figures.
#
# Every mechanism is correctness-checked against a reference implementation before it
# is timed; a mechanism whose kernel is unavailable on this hardware is recorded as
# "unsupported" rather than silently dropped.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-results/raw/stage1}"
PLOT_DIR="${PLOT_DIR:-results/plots}"
TAG="${TAG:-stage1_gb10}"
MIN_SEQ="${MIN_SEQ:-512}"
MAX_SEQ="${MAX_SEQ:-65536}"
DTYPE="${DTYPE:-bfloat16}"

echo "== Stage 1 sweep: S=${MIN_SEQ}..${MAX_SEQ}, dtype=${DTYPE}"
uv run python -m src.stage1_micro.runner \
  --device cuda \
  --dtype "${DTYPE}" \
  --min-seq "${MIN_SEQ}" \
  --max-seq "${MAX_SEQ}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  "$@"

echo
echo "== Rendering figures"
uv run python -m src.utils.plotting \
  --stage1 "${OUTPUT_DIR}/${TAG}.json" \
  --output-dir "${PLOT_DIR}"
