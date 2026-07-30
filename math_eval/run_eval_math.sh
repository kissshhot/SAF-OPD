MODEL="Qwen3-8B-main"
MODEL_PATH="<PATH_TO_MODEL_CHECKPOINT>"
MODEL_NAME=$MODEL
RESULTS_DIR="./eval_outputs/8B"
DATA_DIR="../data"

echo $MODEL_PATH
echo $MODEL_NAME

# Create output directories if they don't exist
mkdir -p "${RESULTS_DIR}/aime24"
mkdir -p "${RESULTS_DIR}/aime25"
mkdir -p "${RESULTS_DIR}/hmmt25_feb"
mkdir -p "${RESULTS_DIR}/hmmt25_nov"

# aime24
CUDA_VISIBLE_DEVICES=0,1 python3 eval_math.py \
    --input_file "${DATA_DIR}/aime24/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "${RESULTS_DIR}/aime24/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# aime25
CUDA_VISIBLE_DEVICES=2,3 python3 eval_math.py \
    --input_file "${DATA_DIR}/aime25/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "${RESULTS_DIR}/aime25/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Feb
CUDA_VISIBLE_DEVICES=4,5 python3 eval_math.py \
    --input_file "${DATA_DIR}/hmmt25_feb/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "${RESULTS_DIR}/hmmt25_feb/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Nov
CUDA_VISIBLE_DEVICES=6,7 python3 eval_math.py \
    --input_file "${DATA_DIR}/hmmt25_nov/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "${RESULTS_DIR}/hmmt25_nov/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &

wait
echo "Model $MODEL_NAME done!"
