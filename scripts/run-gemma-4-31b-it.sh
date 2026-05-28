#!/bin/bash
# Single-node (8x H200) RL smoke test for Google Gemma 4 31B-it DENSE.
# Mirrors scripts/run-gemma-4-26b-a4b-it.sh but for the dense (non-MoE) 31B:
# parallelism is TP=4 x PP=1 x DP=2 (no expert parallelism), and the MoE-only
# flags (--num-experts, --moe-*, --sglang-moe-runner-backend, --use-miles-router,
# --use-rollout-routing-replay) are removed.
#
# Requires the radixark/Megatron-Bridge `zhichen/gemma4-dense` branch (dense
# support: Gemma4VLBridge unblock for hidden_size_per_layer_input==0, the
# attention_k_eq_v K=V tying fix, and the dense GatedMLP weight mappings).
#
# Model: google/gemma-4-31B-it (VLM repo; the AutoBridge resolves it as
# Gemma4VLModelProvider in dense mode — the LM portion is what RL trains).

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
ray stop --force
sleep 3

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gemma-4-31b-it.sh"

MODELS_DIR=${MODELS_DIR:-/storage/models}
DATASETS_DIR=${DATASETS_DIR:-/cluster_public/miles_data/datasets}
LLM_CKPT=${LLM_CKPT:-/storage/models/google/gemma-4-31B-it}

CKPT_ARGS=(
   --hf-checkpoint $LLM_CKPT
   --ref-load $LLM_CKPT
   --save $MODELS_DIR/google/gemma-4-31B-it_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 10
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 768
   --rollout-temperature 1
   --global-batch-size 8
   --balance-data
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   --max-tokens-per-gpu 512
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # No KL: coef was 0 anyway, and the reference model is a full extra copy of
   # the dense 31B (~15.5GB/GPU, no expert parallelism to shard it) that pushes
   # the colocate train footprint past what fits alongside sglang.
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   # Dense 31B colocate: sglang's KV pool (mem-fraction) is held for the whole
   # run, so keep it small (the smoke only needs tiny KV for 256-token responses)
   # to leave room for the large dense train footprint on the same GPUs.
   --sglang-mem-fraction-static 0.5
   # Gemma 4 has head_dim=512 on global layers, above the FlashAttention
   # cap of 256. Triton attention backend handles any head_dim.
   --sglang-attention-backend triton
   --sglang-disable-custom-all-reduce
   --sglang-disable-cuda-graph
   --sglang-disable-overlap-schedule
   --sglang-disable-radix-cache
   # Keep both resident (like the 26B) — the torch_memory_saver offload path
   # crashes during update_weights for this model. Fits because mem-fraction is
   # low (sglang ~56GB) leaving room for the dense train footprint (~63GB).
   --no-offload-train
   --no-offload-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --no-gradient-accumulation-fusion
   --no-check-for-nan-in-loss-and-grad
   --attention-softmax-in-fp32
   --attention-backend unfused
   --qkv-format bshd
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --colocate \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
