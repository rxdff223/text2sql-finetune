"""
快速验证脚本 - 方案B
用现有 checkpoint 做 EM 评估和预测质量分析，不依赖数据库目录

Usage:
    python src/quick_eval.py --model_path outputs/text2sql-qlora/checkpoint-4400 \
                             --base_model D:/text2sql-finetune/pretrained_models/Qwen2.5-Coder-7B-Instruct \
                             --eval_file data/eval.jsonl \
                             --max_samples 200 \
                             --output_file outputs/quick_eval_report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str, base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.lower()
    return sql


def classify_difficulty(sql: str) -> str:
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


def detect_source(messages: list[dict]) -> str:
    """Detect dataset source from user message content"""
    user_msg = messages[1]["content"]
    if "你是一个SQL专家" in messages[0]["content"]:
        return "cspider"
    return "spider_or_bird"


def check_sql_validity(sql: str) -> dict:
    """Basic SQL validity checks"""
    sql_upper = sql.upper()
    checks = {
        "has_select": "SELECT" in sql_upper,
        "has_from": "FROM" in sql_upper,
        "no_chinese_in_sql": not any("\u4e00" <= c <= "\u9fff" for c in sql),
        "reasonable_length": 10 <= len(sql) <= 500,
        "no_explanation": not any(w in sql for w in ["这是", "这个查询", "The query", "This SQL"]),
    }
    checks["is_valid"] = all(checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="outputs/quick_eval_report.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    print(f"Base model: {args.base_model}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.\n")

    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating on {len(samples)} samples...")

    results = {
        "exact_match": 0,
        "total": 0,
        "by_difficulty": {},
        "by_source": {},
        "sql_validity": {"valid_count": 0, "invalid_count": 0, "details": {}},
    }
    predictions = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        gold_sql = messages[-1]["content"]
        difficulty = classify_difficulty(gold_sql)
        source = detect_source(messages)

        pred_sql = generate_sql(model, tokenizer, messages)

        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        results["exact_match"] += int(em)
        results["total"] += 1

        # By difficulty
        if difficulty not in results["by_difficulty"]:
            results["by_difficulty"][difficulty] = {"em": 0, "total": 0}
        results["by_difficulty"][difficulty]["total"] += 1
        results["by_difficulty"][difficulty]["em"] += int(em)

        # By source
        if source not in results["by_source"]:
            results["by_source"][source] = {"em": 0, "total": 0}
        results["by_source"][source]["total"] += 1
        results["by_source"][source]["em"] += int(em)

        # SQL validity
        validity = check_sql_validity(pred_sql)
        if validity["is_valid"]:
            results["sql_validity"]["valid_count"] += 1
        else:
            results["sql_validity"]["invalid_count"] += 1
            for key, passed in validity.items():
                if key != "is_valid" and not passed:
                    if key not in results["sql_validity"]["details"]:
                        results["sql_validity"]["details"][key] = 0
                    results["sql_validity"]["details"][key] += 1

        predictions.append({
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "difficulty": difficulty,
            "source": source,
            "sql_valid": validity["is_valid"],
        })

        if (i + 1) % 50 == 0:
            em_rate = results["exact_match"] / results["total"] * 100
            valid_rate = results["sql_validity"]["valid_count"] / results["total"] * 100
            print(f"  [{i+1}/{len(samples)}] EM: {em_rate:.1f}% | Valid SQL: {valid_rate:.1f}%")

    em_rate = results["exact_match"] / results["total"] * 100
    valid_rate = results["sql_validity"]["valid_count"] / results["total"] * 100

    print(f"\n=== Results ===")
    print(f"Exact Match: {em_rate:.2f}% ({results['exact_match']}/{results['total']})")
    print(f"SQL Validity: {valid_rate:.2f}% ({results['sql_validity']['valid_count']}/{results['total']})")

    print(f"\nBy Difficulty:")
    for diff, stats in sorted(results["by_difficulty"].items()):
        rate = stats["em"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff}: EM {rate:.1f}% ({stats['em']}/{stats['total']})")

    print(f"\nBy Source:")
    for src, stats in sorted(results["by_source"].items()):
        rate = stats["em"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {src}: EM {rate:.1f}% ({stats['em']}/{stats['total']})")

    if results["sql_validity"]["details"]:
        print(f"\nInvalid SQL breakdown:")
        for key, count in sorted(results["sql_validity"]["details"].items()):
            print(f"  {key}: {count}")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "predictions": predictions}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Show some examples
    print(f"\n=== Sample Predictions (first 5) ===")
    for p in predictions[:5]:
        print(f"\nGold: {p['gold_sql']}")
        print(f"Pred: {p['pred_sql']}")
        print(f"EM: {p['exact_match']} | Difficulty: {p['difficulty']} | Valid: {p['sql_valid']}")

    # Show worst predictions (first 3 EM=false that are valid SQL)
    worst = [p for p in predictions if not p["exact_match"] and p["sql_valid"]][:3]
    print(f"\n=== Interesting Misses (valid SQL, wrong answer) ===")
    for p in worst:
        print(f"\nGold: {p['gold_sql']}")
        print(f"Pred: {p['pred_sql']}")
        print(f"Difficulty: {p['difficulty']} | Source: {p['source']}")


if __name__ == "__main__":
    main()