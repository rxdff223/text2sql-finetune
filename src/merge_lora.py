"""
LoRA 权重合并脚本
将 LoRA adapter 合并到基座模型中，生成完整模型用于部署

Usage:
    python src/merge_lora.py --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
                             --lora_path outputs/text2sql-qlora/final \
                             --output_path outputs/merged_model
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="outputs/text2sql-qlora/final")
    parser.add_argument("--output_path", type=str, default="outputs/merged_model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print("Done!")


if __name__ == "__main__":
    main()
