#!/bin/bash
set -e

CONFIG="configs/train_config.yaml"

echo "=== Text-to-SQL QLoRA Training ==="
echo "Config: $CONFIG"

python src/train.py --config "$CONFIG"
