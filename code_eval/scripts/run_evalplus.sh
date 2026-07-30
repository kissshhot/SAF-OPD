#!/bin/bash
pip install -e code_eval/coding/evalplus
export HUMANEVAL_OVERRIDE_PATH=code_eval/data/HumanEvalPlus.jsonl
export MBPP_OVERRIDE_PATH=code_eval/data/MbppPlus.jsonl

# Set defaults if not specified - fix argument assignments
# DATASET: humaneval | mbpp | all (default: all)
DATASET_ARG=${1:-all}
MODEL=${2:-"<PATH_TO_MODEL_CHECKPOINT>"}
GREEDY=${3:-1}
TEMP=${4:-0.8}
TOP_P=${5:-0.9}
N_SAMPLES=${6:-1}
GPU_IDS=${7:-0}
RESULTS_DIR="./evalplus_results/main-debug/1.7b-main"

export CUDA_VISIBLE_DEVICES=$GPU_IDS

# If greedy mode, force n_samples to 1
if [ "$GREEDY" -eq 1 ]; then
    N_SAMPLES=1
    TEMP_VAL="0.0"
else
    TEMP_VAL="$TEMP"
fi

# Expand "all" to both datasets
if [ "$DATASET_ARG" = "all" ]; then
    DATASETS="humaneval mbpp"
else
    DATASETS="$DATASET_ARG"
fi

# Extract model identifier for output file
MODEL_BASE=$(basename "$MODEL")

# evalplus derives the output filename from $MODEL by replacing '/' with '--'
# (codegen.py: identifier = model.strip('./').replace('/', '--') + ...). Long HF
# checkpoint paths produce a basename > 255 bytes → OSError [Errno 36]. Point
# MODEL at a short symlink so the derived filename stays under NAME_MAX.
SHORT_MODEL="/tmp/evalplus_${MODEL_BASE}_$$"
ln -sfn "$MODEL" "$SHORT_MODEL"
MODEL="$SHORT_MODEL"

echo "Datasets: $DATASETS"
echo "Model: $MODEL"
echo "Greedy: $GREEDY (1=yes, 0=no)"
echo "Temperature: $TEMP_VAL"
echo "Top-P: $TOP_P"
echo "Number of samples: $N_SAMPLES"
echo "GPU IDs: $GPU_IDS"
echo "Model base: $MODEL_BASE"

for DATASET in $DATASETS; do
    echo ""
    echo "===== Dataset: $DATASET ====="

    # Execute command directly without quoting the arguments
    if [ "$GREEDY" -eq 1 ]; then
        python3 code_eval/coding/evalplus/evalplus/codegen.py --model "$MODEL" \
                        --dataset $DATASET \
                        --backend vllm \
                        --trust_remote_code \
                        --root "$RESULTS_DIR" \
                        --greedy
    else
        echo "Running non-greedy mode"
        python3 code_eval/coding/evalplus/evalplus/codegen.py --model "$MODEL" \
                        --dataset $DATASET \
                        --backend vllm \
                        --temperature $TEMP \
                        --top-p $TOP_P \
                        --trust_remote_code \
                        --n-samples $N_SAMPLES \
                        --root "$RESULTS_DIR"
    fi

    # The actual output file - use a glob pattern to find the file
    echo "Waiting for output file to be generated..."
    sleep 2  # Give some time for the file to be created

    # Use find to locate the file with a more flexible pattern that matches actual filename format
    OUTPUT_FILE=$(find "${RESULTS_DIR}/${DATASET}" -name "*${MODEL_BASE}*_vllm_temp_${TEMP_VAL}.jsonl" ! -name "*.raw.jsonl" -type f | head -n 1)

    # Run evaluation with found file
    python3 -m evalplus.evaluate --dataset "$DATASET" \
        --samples "$OUTPUT_FILE" \
        --output_file "${RESULTS_DIR}/${DATASET}/${MODEL_BASE}_eval_results.json" \
        --min-time-limit 10.0 \
        --gt-time-limit-factor 8.0

    echo "Evaluation complete. Results saved to ${RESULTS_DIR}/${DATASET}/${MODEL_BASE}_eval_results.json"
done