#!/usr/bin/env bash
set -euo pipefail

dir=$(cd -- "$(dirname -- "$0")" && pwd)
source_dir=$dir
if [[ ! -d $source_dir/vllm ]]; then
  source_dir=$(cd -- "$dir/../.." && pwd)
fi
base_image='lazymio/vllm-backport:latest-sm80'
image=${VLLM_IMAGE:-$base_image}
site=${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}
port=${PORT:-18182}
action=${1:---check}
[[ $# -le 1 && $port =~ ^[0-9]{1,5}$ ]] && (( 10#$port >= 1 && 10#$port <= 65535 )) || exit 2
case "$action" in
  --check|--start) ;;
  *) echo "Usage: $0 [--check | --start]" >&2; exit 2 ;;
esac

case "${PREFILL_PROFILE:-default}" in
  default) batch=3072; utilization=0.975 ;;
  capacity) batch=1280; utilization=0.965 ;;
  *) echo 'PREFILL_PROFILE must be default or capacity.' >&2; exit 2 ;;
esac

context=524288
sequences=6
drafts=${DSPARK_TOKENS:-5}
case "$drafts" in
  5) sizes='[5,6,10,12,15,18,20,24,25,30,36]' ;;
  3) sizes='[3,4,6,8,9,12,15,16,18,20,24]' ;;
  0) sizes='[1,2,3,4,5,6]' ;;
  *) echo 'DSPARK_TOKENS must be 0, 3 or 5.' >&2; exit 2 ;;
esac
largest=$((sequences * (drafts + 1)))
spec=()
if (( drafts )); then
  spec=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$drafts,\"use_local_argmax_reduction\":true,\"enable_adaptive_verification\":false}")
fi

args=(serve /model
  --host 0.0.0.0 --port 8000 --served-model-name dsv41-w4a8-512k
  --tensor-parallel-size 4 --distributed-executor-backend mp --disable-custom-all-reduce
  --quantization compressed-tensors --moe-backend marlin --cpu-offload-gb 0
  --max-model-len "$context" --max-num-seqs "$sequences" --max-num-batched-tokens "$batch"
  --gpu-memory-utilization "$utilization"
  --engram-config '{"cpu_offload":true}' --kv-cache-dtype fp8_ds_mla
  --enable-prefix-caching --prefix-match-unit 8 --kv-offloading-backend native --kv-offloading-size 128
  --kv-transfer-config '{"kv_role":"kv_both","kv_connector_extra_config":{"private_rank_arena":true}}'
  --mm-encoder-tp-mode weights --limit-mm-per-prompt "{\"image\":$context}"
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$sizes,\"max_cudagraph_capture_size\":$largest}"
  --safetensors-load-strategy lazy --tokenizer-mode deepseek_v41
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  "${spec[@]}"
)

# Each mount replaces a Python source file and leaves the image's compiled extensions in place.
mounts=()
while IFS= read -r file; do
  [[ $file == vllm/*.py && $file != *'..'* && -f $source_dir/$file ]]
  mounts+=(--mount "type=bind,src=$source_dir/$file,dst=$site/$file,readonly")
done < "$dir/python-files.txt"

runtime=(--init "${mounts[@]}"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
  -e VLLM_USE_SIMPLE_KV_OFFLOAD=0 -e VLLM_EXPERIMENT_DSV41_CAP_ROPE=1
  -e VLLM_EXPERIMENT_DSV41_INDEXER_BF16_K_WS=1
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 -e VLLM_DSV4_LOGITS_ROW_CHUNK=32
  -e OMP_NUM_THREADS=2 -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo)

if [[ $action == --check ]]; then
  docker run --rm --network none "${runtime[@]}" --entrypoint /usr/bin/python3 "$image" \
    -c 'from pathlib import Path; import vllm; from vllm.config.speculative import SpeculativeConfig; from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec; p = Path(vllm.__file__).parent; assert list(p.glob("_moe_C*.so")); assert "Native private KV arena" in (p / "v1/kv_offload/cpu/spec.py").read_text(); print("vLLM Python overlay and existing MoE extension found", vllm.__file__)'
  printf '%q ' vllm "${args[@]}"
  printf '\n'
  exit 0
fi

model=$(cd -- "${MODEL_DIR:?Set MODEL_DIR to the downloaded W4A8 checkpoint}" && pwd)
test -r "$model/config.json"
test -r "$model/model.safetensors.index.json"
gpus=${GPU_DEVICES:-0,1,2,3}
IFS=, read -r -a devices <<< "$gpus"
if (( ${#devices[@]} != 4 )); then
  echo 'GPU_DEVICES must name four GPUs separated by commas.' >&2
  exit 2
fi

log_root=${LOG_DIR:-$PWD/dsv41-logs}
mkdir -p -- "$log_root"
log_root=$(cd -- "$log_root" && pwd)
run=$(mktemp -d "$log_root/run.XXXXXXXX")
name="dsv41-w4a8-${run##*.}"

printf '%q ' vllm "${args[@]}" > "$run/argv.txt"
printf '\n' >> "$run/argv.txt"
docker run -d --name "$name" --cidfile "$run/container.cid" \
  --gpus "\"device=$gpus\"" -p "127.0.0.1:$port:8000" --shm-size 8g --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$model,dst=/model,readonly" \
  --mount "type=bind,src=$run,dst=/logs" "${runtime[@]}" --entrypoint /bin/bash "$image" \
  -c 'exec > >(tee -a /logs/server.log) 2>&1; exec vllm "$@"' dsv41 "${args[@]}" \
  2>&1 | tee "$run/startup.log"
printf 'Container %s\nLogs %s\nHealth http://127.0.0.1:%s/health\n' "$name" "$run" "$port"
