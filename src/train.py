"""
QLoRA 微调训练脚本 v2 — 抗过拟合版本
基于 Qwen2.5-Coder-7B-Instruct 进行 Text-to-SQL 微调

优化点：
  - Early Stopping: eval_loss 连续 3 次不降就停
  - 更密集的 eval/save: 每 100 步评估，保留更多 checkpoint
  - 模型加载时指定 dtype 替代已废弃的 torch_dtype
  - 训练结束后自动选取 best checkpoint 保存

Usage:
    python src/train.py --config configs/train_config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: str, max_samples: int = None) -> Dataset:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]
    return Dataset.from_list(samples)


def formatting_func(example, tokenizer):
    """将 messages 格式转换为模型输入文本"""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


class BestCheckpointCallback(TrainerCallback):
    """训练结束后，将 best checkpoint 复制到 final 目录"""

    def on_train_end(self, args, state, control, **kwargs):
        if state.best_model_checkpoint:
            best_dir = Path(state.best_model_checkpoint)
            final_dir = Path(args.output_dir) / "final"
            print(f"\nBest checkpoint: {best_dir} (eval_loss={state.best_metric:.4f})")

            # 复制 best 到 final
            if best_dir.exists():
                import shutil
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(str(best_dir), str(final_dir))
                print(f"Best model copied to: {final_dir}")
        else:
            # 没有 best checkpoint（未触发 early stopping），用最后一个
            print("\nNo best checkpoint recorded, saving last model as final.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 — 用 dtype 替代已废弃的 torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    train_dataset = load_dataset_from_jsonl(
        data_cfg["train_file"], data_cfg.get("max_samples")
    )
    eval_dataset = load_dataset_from_jsonl(data_cfg["eval_file"])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Early Stopping — eval_loss 连续 patience 次不降就停
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
        early_stopping_threshold=0.001,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=lambda example: formatting_func(example, tokenizer),
        callbacks=[early_stopping, BestCheckpointCallback()],
    )

    # 开始训练
    print("Starting training (v2 — anti-overfitting)...")
    print(f"  Epochs: {train_cfg['num_epochs']}")
    print(f"  LoRA rank: {lora_cfg['rank']}, alpha: {lora_cfg['alpha']}, dropout: {lora_cfg['dropout']}")
    print(f"  LR: {train_cfg['learning_rate']}, warmup: {train_cfg['warmup_ratio']}")
    print(f"  Early stopping patience: {train_cfg.get('early_stopping_patience', 3)}")
    print(f"  Save every {train_cfg['save_steps']} steps, eval every {train_cfg['eval_steps']} steps")

    trainer.train()

    # 如果 early stopping 没触发，也保存 final
    final_dir = Path(train_cfg["output_dir"]) / "final"
    if not final_dir.exists():
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Model saved to: {final_dir}")

    # 打印训练摘要
    log_history = trainer.state.log_history
    eval_losses = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]
    if eval_losses:
        best_step, best_loss = min(eval_losses, key=lambda x: x[1])
        print(f"\n=== Training Summary ===")
        print(f"Total steps: {trainer.state.global_step}")
        print(f"Best eval_loss: {best_loss:.4f} at step {best_step}")
        print(f"Final eval_loss: {eval_losses[-1][1]:.4f} at step {eval_losses[-1][0]}")


if __name__ == "__main__":
    main()