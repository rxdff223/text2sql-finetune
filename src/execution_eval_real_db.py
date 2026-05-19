"""
真实数据库 EX 评估脚本
使用 Spider 提供的 .sqlite 数据库文件，在真实数据上执行预测 SQL 和 Gold SQL，对比执行结果

Usage:
    python src/execution_eval_real_db.py \
        --model_path outputs/text2sql-qlora-v2/final \
        --base_model pretrained_models/Qwen2.5-Coder-7B-Instruct \
        --eval_file data/eval.jsonl \
        --spider_dev_file data/spider/dev.json \
        --db_dir full_CSpider/full_CSpider/CSpider/database \
        --max_samples 1034 \
        --output_file outputs/execution_accuracy_real_db_report.json
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


def extract_question_from_messages(messages):
    for msg in messages:
        if msg["role"] == "user":
            m = re.search(r"### Question:\n(.+)$", msg["content"], re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""


def classify_difficulty(sql: str) -> str:
    sql_upper = sql.upper()
    nested = sql_upper.count("SELECT") - 1
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    score = nested * 2 + has_join + has_group + has_having * 2
    if score == 0:
        return "easy"
    elif score <= 2:
        return "medium"
    elif score <= 4:
        return "hard"
    return "extra_hard"


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def execute_query(conn: sqlite3.Connection, sql: str) -> dict:
    if not sql or not sql.strip():
        return {"success": False, "error": "empty_sql"}
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return {"success": True, "rows": rows, "cols": cols}
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


def compare_results(pred_result: dict, gold_result: dict) -> dict:
    if not pred_result["success"]:
        return {"match": False, "reason": "pred_failed", "detail": pred_result.get("error", "")}
    if not gold_result["success"]:
        return {"match": False, "reason": "gold_failed", "detail": gold_result.get("error", "")}
    pred_rows = set(map(tuple, pred_result["rows"]))
    gold_rows = set(map(tuple, gold_result["rows"]))
    if pred_rows == gold_rows:
        return {"match": True, "reason": "exact_match"}
    else:
        if gold_rows:
            overlap = len(pred_rows & gold_rows)
            recall = overlap / len(gold_rows)
            return {"match": False, "reason": "result_mismatch",
                    "recall": recall, "gold_count": len(gold_rows), "pred_count": len(pred_rows)}
        return {"match": False, "reason": "result_mismatch"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="D:/text2sql-finetune/outputs/text2sql-qlora-v2/final")
    parser.add_argument("--base_model", type=str,
                        default="D:/text2sql-finetune/pretrained_models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval_file", type=str,
                        default="D:/text2sql-finetune/data/eval.jsonl")
    parser.add_argument("--spider_dev_file", type=str,
                        default="D:/text2sql-finetune/data/spider/dev.json")
    parser.add_argument("--db_dir", type=str,
                        default="D:/text2sql-finetune/full_CSpider/full_CSpider/CSpider/database")
    parser.add_argument("--max_samples", type=int, default=1034)
    parser.add_argument("--output_file", type=str,
                        default="D:/text2sql-finetune/outputs/execution_accuracy_real_db_report.json")
    args = parser.parse_args()

    # Load dev.json to get db_id -> .sqlite path mapping
    with open(args.spider_dev_file) as f:
        dev_data = json.load(f)

    # Build db_id -> sqlite_path mapping
    db_id_to_sqlite = {}
    for item in dev_data:
        db_id = item["db_id"]
        sqlite_path = str(Path(args.db_dir) / db_id / f"{db_id}.sqlite")
        db_id_to_sqlite[db_id] = sqlite_path

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.\n")

    # Load eval samples
    samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[: args.max_samples]

    # Build eval-db_id mapping by matching gold SQL in dev.json
    # eval.jsonl has 3602 samples (Spider+CSpider+BIRD mixed)
    # dev.json has 1034 Spider dev entries with known db_id
    gold_sql_to_db_id = {}
    for item in dev_data:
        gold_sql_to_db_id[item["query"]] = item["db_id"]

    # Filter to only samples whose gold SQL matches a Spider dev entry
    eval_samples = []
    db_not_found = 0
    for sample in samples:
        messages = sample["messages"]
        gold_sql = messages[-1]["content"].strip()
        question = extract_question_from_messages(messages)
        db_id = gold_sql_to_db_id.get(gold_sql, "")
        if db_id:
            sqlite_path = str(Path(args.db_dir) / db_id / f"{db_id}.sqlite")
            if Path(sqlite_path).exists():
                eval_samples.append({
                    "messages": messages,
                    "gold_sql": gold_sql,
                    "question": question,
                    "db_id": db_id,
                    "sqlite_path": sqlite_path,
                })
            else:
                db_not_found += 1
        else:
            db_not_found += 1

    print(f"Total samples: {len(samples)}, Eval-eligible (matched + DB found): {len(eval_samples)}, Skipped: {db_not_found}")

    results = {
        "total": 0, "exact_match": 0, "execution_accuracy": 0,
        "em_total": 0, "ex_total": 0,
        "by_difficulty": {},
        "reason_breakdown": {},
        "db_not_found": db_not_found,
        "pred_failed_detail": {},
    }
    predictions = []

    for i, sample in enumerate(eval_samples):
        gold_sql = sample["gold_sql"]
        sqlite_path = sample["sqlite_path"]
        db_id = sample["db_id"]
        difficulty = classify_difficulty(gold_sql)

        # Generate prediction
        pred_sql = generate_sql(model, tokenizer, sample["messages"])

        # EM
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        if em:
            results["exact_match"] += 1
        results["em_total"] += 1

        # EX on real database
        try:
            conn = sqlite3.connect(sqlite_path)
            conn.text_factory = str
            pred_result = execute_query(conn, pred_sql)
            gold_result = execute_query(conn, gold_sql)
            comparison = compare_results(pred_result, gold_result)
            ex_match = comparison["match"]
            ex_reason = comparison["reason"]
            conn.close()
        except Exception as e:
            ex_match = False
            ex_reason = "db_error"
            comparison = {"match": False, "reason": "db_error", "detail": str(e)[:100]}

        if ex_match:
            results["execution_accuracy"] += 1
        results["ex_total"] += 1

        # Reason breakdown
        results["reason_breakdown"][ex_reason] = results["reason_breakdown"].get(ex_reason, 0) + 1

        # Track pred_failed details
        if ex_reason == "pred_failed":
            err = comparison.get("detail", "unknown")
            err_key = err[:50]
            results["pred_failed_detail"][err_key] = results["pred_failed_detail"].get(err_key, 0) + 1

        # By difficulty
        if difficulty not in results["by_difficulty"]:
            results["by_difficulty"][difficulty] = {"em": 0, "ex": 0, "total": 0}
        results["by_difficulty"][difficulty]["em"] += int(em)
        results["by_difficulty"][difficulty]["ex"] += int(ex_match)
        results["by_difficulty"][difficulty]["total"] += 1

        predictions.append({
            "db_id": db_id,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "execution_accuracy": ex_match,
            "ex_reason": ex_reason,
            "difficulty": difficulty,
        })

        if (i + 1) % 20 == 0:
            em_r = results["exact_match"] / results["em_total"] * 100 if results["em_total"] else 0
            ex_r = results["execution_accuracy"] / results["ex_total"] * 100 if results["ex_total"] else 0
            print(f"  [{i+1}/{len(eval_samples)}] EM: {em_r:.1f}% | EX: {ex_r:.1f}%")

    # Summary
    em_rate = results["exact_match"] / results["em_total"] * 100 if results["em_total"] else 0
    ex_rate = results["execution_accuracy"] / results["ex_total"] * 100 if results["ex_total"] else 0

    print(f"\n{'='*50}")
    print(f"  Total eval samples:   {len(eval_samples)}")
    print(f"  DB not found:         {db_not_found}")
    print(f"  Exact Match (EM):     {em_rate:.1f}% ({results['exact_match']}/{results['em_total']})")
    print(f"  Execution Accuracy (EX): {ex_rate:.1f}% ({results['execution_accuracy']}/{results['ex_total']})")
    print(f"  EX - EM gap:          +{ex_rate - em_rate:.1f}%")
    print(f"\nReason breakdown:")
    for reason, count in sorted(results["reason_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/results['ex_total']*100:.1f}%)")
    print(f"\nBy Difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard"]:
        s = results["by_difficulty"].get(diff, {"em": 0, "ex": 0, "total": 0})
        if s["total"]:
            print(f"  {diff:12s}: EM={s['em']/s['total']*100:.1f}%  EX={s['ex']/s['total']*100:.1f}%  (n={s['total']})")

    # Save
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": {k: v for k, v in results.items() if k != "pred_failed_detail"},
        "pred_failed_detail": results["pred_failed_detail"],
        "summary": {
            "em": f"{em_rate:.1f}%",
            "ex": f"{ex_rate:.1f}%",
            "total": len(eval_samples),
            "db_not_found": db_not_found,
        }
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output_file}")

    print(f"\nSample predictions (EX match, first 2):")
    for p in [x for x in predictions if x["execution_accuracy"]][:2]:
        print(f"\n  DB: {p['db_id']} | Diff: {p['difficulty']}")
        print(f"  Gold: {p['gold_sql'][:100]}")
        print(f"  Pred: {p['pred_sql'][:100]}")

    print(f"\nSample predictions (EX failed, first 2):")
    for p in [x for x in predictions if not x["execution_accuracy"]][:2]:
        print(f"\n  DB: {p['db_id']} | Diff: {p['difficulty']} | Reason: {p['ex_reason']}")
        print(f"  Gold: {p['gold_sql'][:100]}")
        print(f"  Pred: {p['pred_sql'][:100]}")


if __name__ == "__main__":
    main()