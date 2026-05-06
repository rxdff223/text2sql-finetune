#!/bin/bash
set -e

MODEL_PATH="${1:-outputs/text2sql-qlora/final}"
BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
EVAL_FILE="data/eval.jsonl"

echo "=== Text-to-SQL Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Eval file: $EVAL_FILE"

python src/evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --eval_file "$EVAL_FILE" \
    --output_file outputs/eval_results.json
