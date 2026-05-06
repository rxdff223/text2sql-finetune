"""
评估脚本
计算模型在 Text-to-SQL 任务上的 Execution Accuracy 和 Exact Match

Usage:
    python src/evaluate.py --model_path outputs/text2sql-qlora/final \
                           --eval_file data/eval.jsonl \
                           --db_dir data/spider/database
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str, base_model: str = None):
    """加载微调后的模型"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not base_model else base_model,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """生成 SQL"""
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result


def normalize_sql(sql: str) -> str:
    """SQL 归一化，用于 Exact Match 比较"""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.lower()
    return sql


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> list | None:
    """在 SQLite 数据库上执行 SQL，返回结果"""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception:
        return None


def compute_execution_accuracy(
    pred_sql: str, gold_sql: str, db_path: str
) -> bool:
    """比较两条 SQL 的执行结果是否一致"""
    pred_result = execute_sql(db_path, pred_sql)
    gold_result = execute_sql(db_path, gold_sql)

    if pred_result is None or gold_result is None:
        return False

    return set(map(tuple, pred_result)) == set(map(tuple, gold_result))


def classify_difficulty(sql: str) -> str:
    """按 SQL 复杂度分级"""
    sql_upper = sql.upper()
    nested = sql_upper.count("SELECT") - 1
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_set_op = any(op in sql_upper for op in ["UNION", "INTERSECT", "EXCEPT"])

    score = nested * 2 + has_join + has_group + has_having * 2 + has_set_op * 2
    if score == 0:
        return "easy"
    elif score <= 2:
        return "medium"
    elif score <= 4:
        return "hard"
    return "extra_hard"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl")
    parser.add_argument("--db_dir", type=str, default="data/spider/database")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)

    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating on {len(samples)} samples...")

    results = {"exact_match": 0, "execution_accuracy": 0, "total": 0, "by_difficulty": {}}
    predictions = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        gold_sql = messages[-1]["content"]
        difficulty = classify_difficulty(gold_sql)

        pred_sql = generate_sql(model, tokenizer, messages)

        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        results["exact_match"] += int(em)
        results["total"] += 1

        if difficulty not in results["by_difficulty"]:
            results["by_difficulty"][difficulty] = {"em": 0, "total": 0}
        results["by_difficulty"][difficulty]["total"] += 1
        results["by_difficulty"][difficulty]["em"] += int(em)

        predictions.append({
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "difficulty": difficulty,
        })

        if (i + 1) % 50 == 0:
            em_rate = results["exact_match"] / results["total"] * 100
            print(f"  [{i+1}/{len(samples)}] EM: {em_rate:.1f}%")

    em_rate = results["exact_match"] / results["total"] * 100
    print(f"\n=== Results ===")
    print(f"Exact Match: {em_rate:.2f}% ({results['exact_match']}/{results['total']})")

    print(f"\nBy Difficulty:")
    for diff, stats in sorted(results["by_difficulty"].items()):
        rate = stats["em"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff}: {rate:.1f}% ({stats['em']}/{stats['total']})")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "predictions": predictions}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
