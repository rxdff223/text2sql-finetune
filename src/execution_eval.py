"""
Phase 1: Execution Accuracy 评估
用 schema DDL 构建 in-memory SQLite，执行预测 SQL 和 Gold SQL，对比执行结果
只评估 BIRD 数据（有 create_table），Spider/CSpider 用 schema DDL 构建临时表

Usage:
    python src/execution_eval.py --model_path outputs/text2sql-qlora/checkpoint-4400 \
                              --base_model pretrained_models/Qwen2.5-Coder-7B-Instruct \
                              --eval_file data/eval.jsonl \
                              --max_samples 200
"""

import argparse
import json
import re
import sqlite3
import warnings
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
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, messages, max_new_tokens=512) -> str:
    prompt = [m for m in messages if m["role"] != "assistant"]
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def extract_schema_from_messages(messages):
    """从 messages 中提取 schema DDL"""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            m = re.search(r"### Database Schema:\n(.+?)(?:\n\n### Question:|$)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""


def extract_question(messages):
    for msg in messages:
        if msg["role"] == "user":
            m = re.search(r"### Question:\n(.+)$", msg["content"], re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""


def build_inmemory_db(schema_ddl: str) -> sqlite3.Connection:
    """从 DDL 构建 in-memory SQLite 数据库"""
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    # 只执行 CREATE TABLE 语句
    for stmt in schema_ddl.split(";"):
        stmt = stmt.strip()
        if stmt.upper().startswith("CREATE TABLE"):
            try:
                conn.execute(stmt)
            except Exception:
                pass
    conn.commit()
    return conn


def execute_query(conn: sqlite3.Connection, sql: str) -> dict:
    """执行 SQL，返回结果或错误"""
    if not sql or not sql.strip():
        return {"success": False, "error": "empty_sql"}
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return {"success": True, "rows": rows, "cols": cols}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}


def compare_results(pred_result: dict, gold_result: dict) -> dict:
    """比较两个执行结果"""
    if not pred_result["success"]:
        return {"match": False, "reason": "pred_failed", "detail": pred_result.get("error", "")}
    if not gold_result["success"]:
        return {"match": False, "reason": "gold_failed", "detail": gold_result.get("error", "")}

    pred_rows = set(map(tuple, pred_result["rows"]))
    gold_rows = set(map(tuple, gold_result["rows"]))
    if pred_rows == gold_rows:
        return {"match": True, "reason": "exact_match"}
    else:
        # 计算交集比例
        if gold_rows:
            overlap = len(pred_rows & gold_rows)
            recall = overlap / len(gold_rows)
            precision = overlap / len(pred_rows) if pred_rows else 0
            return {
                "match": False, "reason": "partial_match",
                "recall": recall, "precision": precision,
                "gold_count": len(gold_rows), "pred_count": len(pred_rows)
            }
        return {"match": False, "reason": "result_mismatch"}


def classify_difficulty(sql: str) -> str:
    sql_upper = sql.upper()
    nested = sql_upper.count("SELECT") - 1
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    score = nested * 2 + has_join + has_group + has_having * 2
    if score == 0: return "easy"
    elif score <= 2: return "medium"
    elif score <= 4: return "hard"
    return "extra_hard"


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="D:/text2sql-finetune/outputs/text2sql-qlora/checkpoint-4400")
    parser.add_argument("--base_model", type=str,
                        default="D:/text2sql-finetune/pretrained_models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval_file", type=str,
                        default="D:/text2sql-finetune/data/eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_file", type=str,
                        default="D:/text2sql-finetune/outputs/execution_accuracy_report.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.\n")

    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    samples = samples[: args.max_samples]
    print(f"Evaluating {len(samples)} samples with Execution Accuracy metric...\n")

    results = {
        "exact_match": 0, "execution_accuracy": 0,
        "em_total": 0, "ex_total": 0,
        "by_difficulty": {},
        "reason_breakdown": {},
        "db_build_fail": 0,
    }
    predictions = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        gold_sql = messages[-1]["content"]
        schema = extract_schema_from_messages(messages)
        difficulty = classify_difficulty(gold_sql)

        # 生成预测 SQL
        pred_sql = generate_sql(model, tokenizer, messages)

        # EM
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        if em:
            results["exact_match"] += 1
        results["em_total"] += 1

        # EX — 构建数据库并执行
        if not schema:
            ex_match = False
            ex_reason = "no_schema"
            results["db_build_fail"] += 1
        else:
            conn = build_inmemory_db(schema)
            pred_result = execute_query(conn, pred_sql)
            gold_result = execute_query(conn, gold_sql)
            comparison = compare_results(pred_result, gold_result)
            ex_match = comparison["match"]
            ex_reason = comparison["reason"]
            conn.close()

        if ex_match:
            results["execution_accuracy"] += 1
        results["ex_total"] += 1

        # 原因分布
        results["reason_breakdown"][ex_reason] = results["reason_breakdown"].get(ex_reason, 0) + 1

        # 按难度分层
        if difficulty not in results["by_difficulty"]:
            results["by_difficulty"][difficulty] = {"em": 0, "ex": 0, "total": 0}
        results["by_difficulty"][difficulty]["em"] += int(em)
        results["by_difficulty"][difficulty]["ex"] += int(ex_match)
        results["by_difficulty"][difficulty]["total"] += 1

        predictions.append({
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "schema": schema[:200],
            "exact_match": em,
            "execution_accuracy": ex_match,
            "ex_reason": ex_reason,
            "difficulty": difficulty,
            "source": sample.get("source", "unknown"),
        })

        if (i + 1) % 50 == 0:
            em_r = results["exact_match"] / results["em_total"] * 100
            ex_r = results["execution_accuracy"] / results["ex_total"] * 100
            print(f"  [{i+1}/{len(samples)}] EM: {em_r:.1f}% | EX: {ex_r:.1f}%")

    # 汇总
    em_rate = results["exact_match"] / results["em_total"] * 100
    ex_rate = results["execution_accuracy"] / results["ex_total"] * 100
    print(f"\n{'='*40}")
    print(f"  Exact Match (EM):      {em_rate:.1f}% ({results['exact_match']}/{results['em_total']})")
    print(f"  Execution Accuracy (EX): {ex_rate:.1f}% ({results['execution_accuracy']}/{results['ex_total']})")
    print(f"  EX - EM gap:           +{ex_rate - em_rate:.1f}%")
    print(f"\nReason breakdown:")
    for reason, count in sorted(results["reason_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/results['ex_total']*100:.1f}%)")
    print(f"\nBy Difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard"]:
        s = results["by_difficulty"].get(diff, {"em":0,"ex":0,"total":0})
        if s["total"]:
            print(f"  {diff:12s}: EM={s['em']/s['total']*100:.1f}%  EX={s['ex']/s['total']*100:.1f}%  ({s['ex']}/{s['total']})")

    # 保存
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "predictions": predictions[:10]}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output_file}")

    # 打印示例
    print(f"\n{'='*40}")
    print("Sample predictions (EX match, first 3):")
    for p in [x for x in predictions if x["execution_accuracy"]][:3]:
        print(f"\n  Gold: {p['gold_sql'][:120]}")
        print(f"  Pred: {p['pred_sql'][:120]}")
        print(f"  EX: {p['execution_accuracy']} | {p['ex_reason']}")

    print(f"\nSample predictions (EX failed, first 3):")
    for p in [x for x in predictions if not x["execution_accuracy"]][:3]:
        print(f"\n  Gold: {p['gold_sql'][:120]}")
        print(f"  Pred: {p['pred_sql'][:120]}")
        print(f"  Reason: {p['ex_reason']}")


if __name__ == "__main__":
    main()
