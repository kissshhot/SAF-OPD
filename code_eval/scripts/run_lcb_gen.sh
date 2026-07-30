#!/bin/bash
pip install -e code_eval/coding/LiveCodeBench

# Default values
MODEL_NAME="Qwen3-4B-NonThinking"
LOCAL_MODEL_PATH="<PATH_TO_MODEL_CHECKPOINT>"
CUDA_GPU_ID="0"
RESULTS_DIR="./lcb_results/main-debug/1.7b-main"

export LCB_OUTPUT_ROOT="$RESULTS_DIR"
NUM_GPUS=1
BATCH_SIZE=128
N=4
TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKENS=16384

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_NAME="$2"
      shift 2
      ;;
    -l|--local_model_path)
      LOCAL_MODEL_PATH="$2"
      shift 2
      ;;
    -g|--gpu)
      CUDA_GPU_ID="$2"
      shift 2
      ;;
    -n|--n)
      N="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -p|--top_p)
      TOP_P="$2"
      shift 2
      ;;
    -b|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -k|--max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

export LCB_LOCAL_MODEL_PATH="$LOCAL_MODEL_PATH"

cd code_eval/coding/LiveCodeBench

# Run LiveCodeBench with the AZR template and a local model
CUDA_VISIBLE_DEVICES=$CUDA_GPU_ID python -m lcb_runner.runner.main \
  --model $MODEL_NAME \
  ${LOCAL_MODEL_PATH:+--local_model_path "$LOCAL_MODEL_PATH"} \
  --trust_remote_code \
  --scenario codegeneration \
  --release_version v6 \
  --tensor_parallel_size $NUM_GPUS \
  --use_cache \
  --n $N \
  --temperature $TEMPERATURE \
  --max_tokens $MAX_TOKENS \
  --custom_output_save_name $MODEL_NAME \
  --top_p $TOP_P \
  --timeout 60 \
  --num_process_evaluate 4 \
  --evaluate --continue_existing --continue_existing_with_eval
