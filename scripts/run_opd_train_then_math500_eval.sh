#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-opd_dedup_math500_${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-logs/${RUN_NAME}.train.log}"
MERGE_LOG="${MERGE_LOG:-logs/${RUN_NAME}.merge.log}"
EVAL_LOG="${EVAL_LOG:-logs/${RUN_NAME}.eval.log}"
CKPT_PATH="${CKPT_PATH:-checkpoint/${RUN_NAME}}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${CKPT_PATH}/hf_step20}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-${RUN_NAME}}"

mkdir -p \
  "$(dirname "$TRAIN_LOG")" \
  "$(dirname "$MERGE_LOG")" \
  "$(dirname "$EVAL_LOG")" \
  "$(dirname "$MERGED_MODEL_PATH")"
MERGED_MODEL_PATH="$(cd -- "$(dirname -- "$MERGED_MODEL_PATH")" && pwd)/$(basename -- "$MERGED_MODEL_PATH")"
export PYTHONPATH="$ROOT_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
# The OPD shell resolves model snapshots from HF_HOME.  Keep the launcher
# independent of the caller's current shell environment.
export HF_HOME="${HF_HOME:-/apdcephfs_szcf/share_304335953/halanchen/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

echo "[$(date)] starting OPD training"
echo "  train_log=$TRAIN_LOG"
echo "  ckpt_path=$CKPT_PATH"
echo "  train_dataset=$ROOT_DIR/datasets/dapo-math-17k-dedup.parquet"
echo "  eval_dataset=$ROOT_DIR/datasets/test_data/MATH-500/test.parquet"

set +e
env \
  TRAIN_DATASET="$ROOT_DIR/datasets/dapo-math-17k-dedup.parquet" \
  TRAIN_DATASET_NAME=DAPO-Math-17k-dedup \
  ACTOR_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
  REWARD_MODEL_NAME=JustRL-DeepSeek-1.5B \
  N_GPUS_PER_NODE=8 \
  NNODES=1 \
  MINI_BATCH_SIZE=64 \
  N_RESPONSES=1 \
  MAX_PROMPT_LENGTH=1024 \
  MAX_RESP_LENGTH=7168 \
  MAX_VAL_RESP_LENGTH=7168 \
  PPO_MAX_TOKEN_LEN_PER_GPU=16384 \
  ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=16384 \
  REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=16384 \
  TEACHER_MAX_TOKEN_LEN_PER_GPU=16384 \
  VLLM_MAX_NUM_BATCHED_TOKENS=65536 \
  VLLM_MAX_NUM_SEQS=16 \
  VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  MODEL_DTYPE=bfloat16 \
  VAL_N=1 \
  TEST_FREQ=-1 \
  SAVE_FREQ=20 \
  TOTAL_TRAINING_STEPS=20 \
  TRAINER_LOGGER="['console']" \
  IS_PLOT=False \
  CKPT_PATH="$CKPT_PATH" \
  EXPERIMENT_NAME="$RUN_NAME" \
  bash "$ROOT_DIR/on_policy_distillation.sh" > >(tee -a "$TRAIN_LOG") 2>&1
TRAIN_STATUS=$?
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "[$(date)] training failed with exit code $TRAIN_STATUS" >&2
  exit "$TRAIN_STATUS"
fi

ACTOR_CKPT="$CKPT_PATH/global_step_20/actor"
if [ ! -d "$ACTOR_CKPT" ]; then
  echo "[$(date)] missing final actor checkpoint: $ACTOR_CKPT" >&2
  exit 3
fi

echo "[$(date)] merging final FSDP actor checkpoint" | tee -a "$MERGE_LOG"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CKPT" \
  --target_dir "$MERGED_MODEL_PATH" \
  --use_cpu_initialization 2>&1 | tee -a "$MERGE_LOG"

if [ ! -f "$MERGED_MODEL_PATH/config.json" ]; then
  echo "[$(date)] merged Hugging Face model is missing config.json" >&2
  exit 4
fi

echo "[$(date)] generating MATH-500 with the final checkpoint" | tee -a "$EVAL_LOG"
(
  cd "$ROOT_DIR/scripts/val/eval"
  EVAL_MODEL_PATH="$MERGED_MODEL_PATH" \
  EVAL_MODEL_NAME="$EVAL_MODEL_NAME" \
  EVAL_BATCH_SIZE=16 \
  EVAL_GPUS=0,1,2,3,4,5,6,7 \
  EVAL_MAX_TOKENS=7168 \
  EVAL_MAX_MODEL_LEN=8192 \
  python gen_vllm.py --disable-thinking
  EVAL_NAME="$EVAL_MODEL_NAME" \
  EVAL_MODEL_PATH="$MERGED_MODEL_PATH" \
  python grade.py
) 2>&1 | tee -a "$EVAL_LOG"

EVAL_OUTPUT_DIR="$ROOT_DIR/scripts/val/eval/justrl_eval_outputs/$EVAL_MODEL_NAME"
EVAL_OUTPUT_FILE="$(find "$EVAL_OUTPUT_DIR" -maxdepth 1 -type f -name '*.jsonl' -print -quit 2>/dev/null)"
if [ -z "$EVAL_OUTPUT_FILE" ]; then
  echo "[$(date)] MATH-500 evaluation produced no jsonl output" >&2
  exit 5
fi
EVAL_LINE_COUNT="$(wc -l < "$EVAL_OUTPUT_FILE")"
if [ "$EVAL_LINE_COUNT" -ne 500 ]; then
  echo "[$(date)] MATH-500 evaluation produced $EVAL_LINE_COUNT rows; expected 500" >&2
  exit 6
fi
if [ ! -f "$EVAL_OUTPUT_DIR/grading_results.json" ]; then
  echo "[$(date)] grading_results.json was not produced" >&2
  exit 7
fi

echo "[$(date)] training and MATH-500 evaluation completed"
echo "  checkpoint=$CKPT_PATH"
echo "  merged_model=$MERGED_MODEL_PATH"
echo "  eval_results=$ROOT_DIR/scripts/val/eval/justrl_eval_outputs/$EVAL_MODEL_NAME/grading_results.json"
