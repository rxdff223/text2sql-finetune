#!/bin/bash
set -e

MODEL_PATH="${1:-outputs/text2sql-qlora/final}"
BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

echo "=== Text-to-SQL Inference Demo ==="

python src/inference.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --question "Find the names of employees who earn more than the average salary" \
    --schema "CREATE TABLE employees (id INT PRIMARY KEY, name VARCHAR, department VARCHAR, salary DECIMAL);"

echo ""
echo "--- Chinese example ---"

python src/inference.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --question "查找每个部门工资最高的员工" \
    --schema "CREATE TABLE employees (id INT PRIMARY KEY, name VARCHAR, department VARCHAR, salary DECIMAL);"
