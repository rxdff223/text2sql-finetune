"""
Quick evaluation script - 用现有checkpoint对eval数据做推理并分析
"""

import argparse
import json
import re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from collections import defaultdict
from sqlparse import format as sql_format


import sqlite3

import os


import tempfile


def normalize_sql(sql: str) -> str:
    """SQL normalization for Exact Match"""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.lower()
    return sql


def extract_sql_from_output(pred: str) -> str:
    """Extract actual SQL from model output, cleaning up artifacts"""
    pred = pred.strip()
    # Remove markdown-like artifacts
    pred = re.sub(r'```sql\s*```', '', pred)
    pred = re.sub(r'```.*?```', '', pred)
    pred = re.sub(r'^--.*$', '', pred)
    # Remove common preamble patterns
    for prefix in ['Here is the SQL query:', 'The SQL query is:', 'SQL:', 'Answer:', 'Query:']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
    return pred.strip()
def classify_difficulty(sql: str) -> str:
    """Classify SQL by complexity"""
    sql_upper = sql.upper()
    nested = sql_upper.count("SELECT") - 1
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_set_op = any(op in sql_upper for op in ["UNION", "INTERSECT", "EXCEPT"])
    score = nested * 2 + has_join * 1 + has_group * 1 + has_having * 2 + has_set_op * 2
    if score == 0:
        return "easy"
    elif score <= 2:
        return "medium"
    elif score <= 4:
        return "hard"
    return "extra_hard"


def execute_sql_on_schema(sql: str, schema: str) -> dict:
    """Execute SQL on in-memory database built from schema DDL"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        # Create tables from schema
        for statement in schema.split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement + ";")
        # Execute the query
        cursor.execute(sql)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {"success": True, "results": results, "columns": columns}
    except Exception as e:
        conn.close()
        return {"success": False, "error": str(e)}
def compute_execution_accuracy(pred_sql: str, gold_sql: str, schema: str) -> dict:
    """Compare execution results of pred vs gold SQL"""
    pred_result = execute_sql_on_schema(pred_sql, schema)
    gold_result = execute_sql_on_schema(gold_sql, schema)
    if not pred_result["success"] or not gold_result["success"]:
        return {"match": False, "reason": "execution_failed"}
    if pred_result["results"] is None and gold_result["results"] is None:
        return {"match": True, "reason": "both_empty"}
    try:
        # Compare as sets of tuples for order-independent comparison
        pred_set = set(map(tuple, pred_result["results"]))
        gold_set = set(map(tuple, gold_result["results"]))
        return {"match": pred_set == gold_set, "reason": "result_sets_match"}
    except Exception:
        return {"match": False, "reason": "comparison_error"}
def extract_schema_from_messages(messages: list[dict]) -> str:
    """Extract schema DDL from user message in messages format"""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            schema_match = re.search(r'### Database Schema:\n(.+)', content, re.DOTALL)
            if schema_match:
                return schema_match.group(1).strip()
    return ""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="D:/text2sql-finetune/outputs/text2sql-qlora/checkpoint-4400")
    parser.add_argument("--base_model", type=str, default="D:/text2sql-finetune/pretrained_models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval_file", type=str, default="D:/text2sql-finetune/data/eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="D:/text2sql-finetune/outputs/quick_eval_results.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.")

    # Load eval data
    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Evaluating on {len(samples)} samples...")

    # Run inference
    predictions = []
    results = {
        "exact_match": 0,
        "normalized_match": 0,
        "execution_accuracy": 0,
        "total": 0,
        "by_difficulty": defaultdict(lambda: int({"em": 0, "nm": 0, "ex": 0, "total": 0})),
        "by_source": defaultdict(lambda: int({"em": 0, "nm": 0, "ex": 0, "total": 0})),
    }
    for i, sample in tqdm(enumerate(samples)):
        messages = sample["messages"]
        gold_sql = messages[-1]["content"]
        difficulty = classify_difficulty(gold_sql)
        schema = extract_schema_from_messages(messages)
        pred_sql_raw = generate_sql(model, tokenizer, messages)
        pred_sql = extract_sql_from_output(pred_sql_raw)
        em_exact = pred_sql == gold_sql
        nm_exact = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        # Execution accuracy
        ex_match = False
        if schema and pred_sql:
            ex_result = compute_execution_accuracy(pred_sql, gold_sql, schema)
            ex_match = ex_result["match"]
        predictions.append({
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "pred_sql_raw": pred_sql_raw,
            "exact_match": em_exact,
            "normalized_match": nm_exact,
            "execution_accuracy": ex_match,
            "difficulty": difficulty,
            "source": sample.get("source", "unknown"),
        })
        if em_exact:
            results["exact_match"] += 1
        if nm_exact:
            results["normalized_match"] += 1
        if ex_match:
            results["execution_accuracy"] += 1
        results["total"] += 1
        results["by_difficulty"][difficulty]["em"] += int(em_exact)
        results["by_difficulty"][difficulty]["nm"] += int(nm_exact)
        results["by_difficulty"][difficulty]["ex"] += int(ex_match)
        results["by_difficulty"][difficulty]["total"] += 1
        source = sample.get("source", "unknown")
        results["by_source"][source]["em"] += int(em_exact)
        results["by_source"][source]["nm"] += int(nm_exact)
        results["by_source"][source]["ex"] += int(ex_match)
        results["by_source"][source]["total"] += 1
        if (i + 1) % 20 == 0:
            em_rate = results["exact_match"] / results["total"] * 100
            nm_rate = results["normalized_match"] / results["total"] * 100
            ex_rate = results["execution_accuracy"] / results["total"] * 100
            print(f"  [{i}/{len(samples)}] EM: {em_rate:.1f}% NM: {nm_rate:.1f}% EX: {ex_rate:.1f}%")

    # Final summary
    em_rate = results["exact_match"] / results["total"] * 100
    nm_rate = results["normalized_match"] / results["total"] * 100
    ex_rate = results["execution_accuracy"] / results["total"] * 100
    print(f"\n{'='*25}")
    print(f"Exact Match (EM): {em_rate:.2f}% ({results['exact_match']}/{results['total']})")
    print(f"Normalized Match (NM): {nm_rate:.2f}% ({results['normalized_match']}/{results['total']})")
    print(f"Execution Accuracy (EX): {ex_rate:.2f}% ({results['execution_accuracy']}/{results['total']})")
    print(f"\nBy Difficulty:")
    for diff in sorted(results["by_difficulty"].items()):
        stats = results["by_difficulty"][diff]
        em = stats["em"] / stats["total"] * 100 if stats["total"] > 0 else 0
        nm = stats["nm"] / stats["total"] * 100 if stats["total"] > 0 else 0
        ex = stats["ex"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff}: EM={em:.1f}% NM={nm:.1f}% EX={ex:.1f}% ({stats['em']}/{stats['total']})")
    print(f"\nBy Source:")
    for source in sorted(results["by_source"].items()):
        stats = results["by_source"][source]
        em = stats["em"] / stats["total"] * 100 if stats["total"] > 0 else 0
        nm = stats["nm"] / stats["total"] * 100 if stats["total"] > 0 else 0
        ex = stats["ex"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {source}: EM={em:.1f}% NM={nm:.1f}% EX={ex:.1f}% ({stats['em']}/{stats['total']})")
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "predictions": predictions}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    return results, predictions
if __name__ == "__main__":
    main()