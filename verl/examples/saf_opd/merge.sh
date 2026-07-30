#!/usr/bin/env bash
# ============================================================
# Merge FSDP sharded actor checkpoints into a single HuggingFace-format model.
#
# Usage:
#   CHECKPOINT_PATH=checkpoints/saf-opd-math/mixed_opd1.0_topk20_tanh01_warmup100_kldelta02_anneal200/global_step_200/actor \
#     bash merge.sh
# ============================================================

set -x

cd "$(dirname "$0")/../.." || exit 1

CHECKPOINT_PATH=${CHECKPOINT_PATH:?"Set CHECKPOINT_PATH to the actor checkpoint directory (.../global_step_N/actor)"}
TARGET_PATH=${TARGET_PATH:-"${CHECKPOINT_PATH}/huggingface"}

python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$CHECKPOINT_PATH" \
    --target_dir "$TARGET_PATH"
