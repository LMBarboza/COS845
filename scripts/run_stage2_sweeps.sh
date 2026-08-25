#!/usr/bin/env bash
# Stage 2 driver: run the full model x input-length matrix, one server at a time.
#
#   ./scripts/run_stage2_sweeps.sh b        # 2.5-3B tier
#   ./scripts/run_stage2_sweeps.sh a        # 7-9B tier
#   ./scripts/run_stage2_sweeps.sh b a      # both, B first
#
# Every run disables prefix caching. The client sends one identical prompt to every
# request in a level, so with prefix caching on, a transformer would serve all but the
# first request out of a cached prefix and report a near-zero TTFT -- while the SSMs,
# which have no prefix cache at all, would pay full prefill every time. Leaving it on
# would compare a cache-hit path against a compute path and call the difference
# architecture.
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
OUT="${OUT:-results/raw/stage2}"
LOGDIR="${LOGDIR:-${OUT}/logs}"
mkdir -p "${OUT}" "${LOGDIR}"

MAMBA2_LOCAL="${MAMBA2_LOCAL:-${HOME}/models/mamba2-2.7b-vllm}"

# tag | model | max_model_len | input-token passes (space separated) | extra server args
tier_b() {
  cat <<SPEC
qwen2.5-3b-instruct|Qwen/Qwen2.5-3B-Instruct|8192|2048 4096|
deepseek-v2-lite|deepseek-ai/DeepSeek-V2-Lite|8192|2048 4096|
mamba2-2.7b|${MAMBA2_LOCAL}|8192|2048 4096|
zamba2-2.7b-instruct|Zyphra/Zamba2-2.7B-instruct|4096|2048|
SPEC
}
tier_a() {
  cat <<SPEC
qwen2.5-7b-instruct|Qwen/Qwen2.5-7B-Instruct|8192|2048 4096|
mamba-codestral-7b|mistralai/Mamba-Codestral-7B-v0.1|8192|2048 4096|
falcon-h1-7b-instruct|tiiuae/Falcon-H1-7B-Instruct|8192|2048 4096|
SPEC
}

run_one() {
  local tag="$1" model="$2" maxlen="$3" intok="$4" extra="$5" eager="$6"
  local full_tag="${tag}__in${intok}"
  local args="--no-enable-prefix-caching ${extra}"
  [[ "${eager}" == "eager" ]] && args="${args} --enforce-eager" && full_tag="${full_tag}_eager"
  # Derive the log path *after* the eager suffix is applied: sharing one path between the
  # graph attempt and its eager retry makes the retry truncate the log that recorded why
  # the first attempt died, which is the only evidence of the failure.
  local log="${LOGDIR}/${full_tag}.server.log"

  echo "### ${full_tag}  (model=${model} maxlen=${maxlen} in=${intok})"
  MODEL_ARGS="${args}" \
  MAX_MODEL_LEN="${maxlen}" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  CONCURRENCY="${CONCURRENCY}" \
  INPUT_TOKENS="${intok}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  OUTPUT_DIR="${OUT}" \
  TAG="${full_tag}" \
  SERVER_LOG="${log}" \
  STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" \
  ./scripts/run_stage2.sh "${model}"
}

for tier in "$@"; do
  case "${tier}" in
    a|A) spec="$(tier_a)" ;;
    b|B) spec="$(tier_b)" ;;
    *) echo "unknown tier: ${tier} (expected a or b)" >&2; exit 2 ;;
  esac

  while IFS='|' read -r tag model maxlen passes extra; do
    [[ -z "${tag}" ]] && continue
    for intok in ${passes}; do
      if ! run_one "${tag}" "${model}" "${maxlen}" "${intok}" "${extra}" "graph"; then
        # CUDA-graph capture paths written for datacenter Blackwell can abort on SM12x;
        # a second attempt in eager mode distinguishes "the model does not run here"
        # from "graph capture does not run here".
        echo "!!! ${tag} in=${intok} failed with CUDA graphs; retrying --enforce-eager" >&2
        run_one "${tag}" "${model}" "${maxlen}" "${intok}" "${extra}" "eager" \
          || echo "!!! ${tag} in=${intok} FAILED in both modes -- see ${LOGDIR}" >&2
      fi
      sleep 10
    done
  done <<< "${spec}"
done

echo "== all sweeps attempted; results in ${OUT}"
