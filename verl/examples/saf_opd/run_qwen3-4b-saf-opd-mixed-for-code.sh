#!/usr/bin/env bash
# ============================================================
# SAF (Stable Advantage Fusion) — Code Generation
#
# Reproduces the GRPO / OPD-only / GRPO+OPD(fixed) / SAF settings from the
# paper "SAF: Stable Advantage Fusion for On-Policy Distillation" on the
# Eurus-2-RL-Code code-generation data.
#
# TRAIN_MODE controls which of the paper's four training modes to run:
#   grpo   : GRPO-only (RL only). Advantage fusion disabled entirely.
#   opd    : OPD-only.  grpo_coef=0, opd_coef=1, SAF stages disabled.
#            OPD advantage: log pi_teacher(y_t) - log pi_student(y_t) per token.
#   mixed  : GRPO + OPD fusion (default). Coefficients set by GRPO_COEF / OPD_COEF.
#            With all SAF stages disabled this is the naive fixed-coefficient
#            fusion baseline; enabling the SAF_* flags below turns it into SAF.
#
# SAF's four independently switchable stages (only meaningful when TRAIN_MODE=mixed):
#   Stage 1 — TOPK_PERCENT      : per-sequence top-k% sparsification of the OPD advantage.
#                                  100 = disabled (default).
#   Stage 2 — TANH_COMPRESS/TANH_COEF : bounded tanh compression A_opd_tanh = c*tanh(A_opd).
#   Stage 3 — OPD_WARMUP_STEPS/KL_DROP_DELTA : KL-triggered linear warm-up of the OPD coefficient.
#   Stage 4 — OPD_ANNEAL/ANNEAL_STEPS/ANNEAL_MIN/ANNEAL_TOTAL_STEPS : linear annealing.
#
# Usage examples:
#   TRAIN_MODE=grpo bash run_qwen3-4b-saf-opd-mixed-for-code.sh
#   TRAIN_MODE=opd  bash run_qwen3-4b-saf-opd-mixed-for-code.sh
#   TRAIN_MODE=mixed GRPO_COEF=1.0 OPD_COEF=1.0 bash run_qwen3-4b-saf-opd-mixed-for-code.sh   # fixed fusion
#   TRAIN_MODE=mixed TOPK_PERCENT=20 TANH_COMPRESS=True TANH_COEF=0.1 \
#     OPD_WARMUP_STEPS=100 KL_DROP_DELTA=0.2 OPD_ANNEAL=True \
#     bash run_qwen3-4b-saf-opd-mixed-for-code.sh                                             # full SAF
# ============================================================

set -x

cd "$(dirname "$0")/../.." || exit 1

export RAY_memory_monitor_refresh_ms=0
export PYTHONFAULTHANDLER=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}

# ---- Paths (override via environment variables) ----
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"Qwen/Qwen3-4B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}

# Training/validation data: Eurus-2-RL-Code (comma-separated shard list).
CODE_DATA_DIR=${CODE_DATA_DIR:-"data/eurus2-rl-code"}
train_files="['$CODE_DATA_DIR/train-00000-of-00007.parquet','$CODE_DATA_DIR/train-00001-of-00007.parquet','$CODE_DATA_DIR/train-00002-of-00007.parquet','$CODE_DATA_DIR/train-00003-of-00007.parquet','$CODE_DATA_DIR/train-00004-of-00007.parquet','$CODE_DATA_DIR/train-00005-of-00007.parquet','$CODE_DATA_DIR/train-00006-of-00007.parquet']"
test_files="['$CODE_DATA_DIR/validation-00000-of-00001.parquet']"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints/saf-opd-code"}
WANDB_PROJECT=${WANDB_PROJECT:-"saf-opd"}

# ---- Training mode ----
TRAIN_MODE=${TRAIN_MODE:-mixed}   # grpo | opd | mixed

# ---- Mixing coefficients (TRAIN_MODE=mixed) ----
GRPO_COEF=${GRPO_COEF:-1.0}
OPD_COEF=${OPD_COEF:-1.0}

# ---- SAF Stage 1: top-k% sparsification ----
TOPK_PERCENT=${TOPK_PERCENT:-100}

# ---- SAF Stage 2: bounded tanh compression ----
TANH_COMPRESS=${TANH_COMPRESS:-False}
TANH_COEF=${TANH_COEF:-0.1}

# ---- SAF Stage 3: KL-triggered linear warm-up ----
OPD_WARMUP_STEPS=${OPD_WARMUP_STEPS:-0}
KL_DROP_DELTA=${KL_DROP_DELTA:-0}

# ---- SAF Stage 4: linear annealing over the remaining training budget ----
OPD_ANNEAL=${OPD_ANNEAL:-False}
ANNEAL_STEPS=${ANNEAL_STEPS:-200}
ANNEAL_MIN=${ANNEAL_MIN:-0.0}
ANNEAL_TOTAL_STEPS=${ANNEAL_TOTAL_STEPS:-0}

# ----------------------------------------------------------------
# Resolve per-mode config from TRAIN_MODE
# ----------------------------------------------------------------
if [ "$TRAIN_MODE" = "grpo" ]; then
    token_kl_reg_enable=False
    grpo_coef=1.0
    opd_coef=0.0
    experiment_suffix="grpo_only"

elif [ "$TRAIN_MODE" = "opd" ]; then
    token_kl_reg_enable=True
    grpo_coef=0.0
    opd_coef=$OPD_COEF
    experiment_suffix="opd_only"

else
    # mixed (default): fixed fusion, or full SAF when the Stage flags below are set.
    token_kl_reg_enable=True
    grpo_coef=$GRPO_COEF
    opd_coef=$OPD_COEF
    experiment_suffix="mixed_opd${OPD_COEF}"
fi

# Append suffixes describing which SAF stages are active, for experiment tracking.
if [ "$TOPK_PERCENT" != "100" ]; then
    experiment_suffix="${experiment_suffix}_topk${TOPK_PERCENT}"
fi
if [ "$TANH_COMPRESS" = "True" ]; then
    experiment_suffix="${experiment_suffix}_tanh$(echo $TANH_COEF | tr -d '.')"
fi
if [ "$OPD_WARMUP_STEPS" != "0" ]; then
    experiment_suffix="${experiment_suffix}_warmup${OPD_WARMUP_STEPS}"
    if [ "$KL_DROP_DELTA" != "0" ]; then
        experiment_suffix="${experiment_suffix}_kldelta$(echo $KL_DROP_DELTA | tr -d '.')"
    fi
fi
if [ "$OPD_ANNEAL" = "True" ]; then
    experiment_suffix="${experiment_suffix}_anneal${ANNEAL_STEPS}"
    if [ "$ANNEAL_MIN" != "0.0" ]; then
        experiment_suffix="${experiment_suffix}_min$(echo $ANNEAL_MIN | tr -d '.')"
    fi
fi

export WANDB_DIR=${WANDB_DIR:-"${CHECKPOINT_DIR}/wandb/${experiment_suffix}"}

# ----------------------------------------------------------------
# Launch training
# ----------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=128 \
        data.max_prompt_length=2048 \
        data.max_response_length=8190 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        +algorithm.token_kl_reg.enable=$token_kl_reg_enable \
        +algorithm.token_kl_reg.grpo_coef=$grpo_coef \
        +algorithm.token_kl_reg.opd_coef=$opd_coef \
        +algorithm.token_kl_reg.topk_percent=$TOPK_PERCENT \
        +algorithm.token_kl_reg.tanh_compress=$TANH_COMPRESS \
        +algorithm.token_kl_reg.tanh_coef=$TANH_COEF \
        +algorithm.token_kl_reg.opd_warmup_steps=$OPD_WARMUP_STEPS \
        +algorithm.token_kl_reg.kl_drop_delta=$KL_DROP_DELTA \
        +algorithm.token_kl_reg.opd_anneal=$OPD_ANNEAL \
        +algorithm.token_kl_reg.anneal_steps=$ANNEAL_STEPS \
        +algorithm.token_kl_reg.anneal_min=$ANNEAL_MIN \
        +algorithm.token_kl_reg.anneal_total_steps=$ANNEAL_TOTAL_STEPS \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name="$WANDB_PROJECT" \
        "trainer.experiment_name=saf_opd_code_${experiment_suffix}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        "trainer.default_local_dir=${CHECKPOINT_DIR}/${experiment_suffix}" \
        trainer.test_freq=-1 \
        trainer.total_epochs=3 $@
