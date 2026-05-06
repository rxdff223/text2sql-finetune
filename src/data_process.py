"""
数据预处理脚本
将 Spider / BIRD / CSpider 数据集统一转换为 SFT 训练格式 (jsonl)

Usage:
    python src/data_process.py --data_dir data --output_dir data
"""

import argparse
import json
import random
from pathlib import Path


def build_schema_from_tables(tables_data: list) -> dict:
    """从 tables.json 构建 db_id -> schema 文本的映射"""
    db_schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]

        lines = []
        for i, tname in enumerate(table_names):
            cols = []
            for col_idx, (table_id, col_name) in enumerate(column_names):
                if table_id == i:
                    cols.append(f"  {col_name} {column_types[col_idx]}")
            if cols:
                lines.append(f"CREATE TABLE {tname} (\n" + ",\n".join(cols) + "\n);")

        fk_lines = []
        for fk in db.get("foreign_keys", []):
            c1_table_id, c1_name = column_names[fk[0]]
            c2_table_id, c2_name = column_names[fk[1]]
            fk_lines.append(
                f"-- {table_names[c1_table_id]}.{c1_name} = "
                f"{table_names[c2_table_id]}.{c2_name}"
            )

        schema_text = "\n\n".join(lines)
        if fk_lines:
            schema_text += "\n\n" + "\n".join(fk_lines)
        db_schemas[db_id] = schema_text

    return db_schemas


def load_spider(data_dir: Path, split: str) -> list[dict]:
    """加载 Spider 数据集"""
    spider_dir = data_dir / "spider"
    tables_file = spider_dir / "tables.json"
    data_file = spider_dir / ("train_spider.json" if split == "train" else "dev.json")

    if not data_file.exists():
        print(f"[Spider] {data_file} not found, skipping.")
        return []

    with open(tables_file) as f:
        db_schemas = build_schema_from_tables(json.load(f))

    with open(data_file) as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        schema = db_schemas.get(item["db_id"], "")
        if schema:
            samples.append({
                "question": item["question"],
                "sql": item["query"],
                "schema": schema,
                "db_id": item["db_id"],
                "source": "spider",
                "lang": "en",
            })
    return samples


def load_bird(data_dir: Path, split: str) -> list[dict]:
    """加载 BIRD 数据集"""
    bird_dir = data_dir / "bird"
    data_file = bird_dir / split / f"{split}.json"

    if not data_file.exists():
        print(f"[BIRD] {data_file} not found, skipping.")
        return []

    with open(data_file) as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        schema = item.get("create_table", "")
        if not schema:
            schema = f"-- Database: {item.get('db_id', 'unknown')}"
        samples.append({
            "question": item["question"],
            "sql": item["SQL"],
            "schema": schema,
            "db_id": item.get("db_id", ""),
            "source": "bird",
            "lang": "en",
        })
    return samples


def load_cspider(data_dir: Path, split: str) -> list[dict]:
    """加载 CSpider 中文数据集（与 Spider 共享 schema）"""
    cspider_dir = data_dir / "cspider"
    spider_dir = data_dir / "spider"

    data_file = cspider_dir / ("train.json" if split == "train" else "dev.json")
    tables_file = spider_dir / "tables.json"

    if not data_file.exists():
        print(f"[CSpider] {data_file} not found, skipping.")
        return []

    with open(tables_file) as f:
        db_schemas = build_schema_from_tables(json.load(f))

    with open(data_file) as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        schema = db_schemas.get(item["db_id"], "")
        if schema:
            samples.append({
                "question": item["question"],
                "sql": item["query"],
                "schema": schema,
                "db_id": item["db_id"],
                "source": "cspider",
                "lang": "zh",
            })
    return samples


def format_to_messages(sample: dict) -> dict:
    """转换为 Qwen chat messages 格式"""
    if sample["lang"] == "zh":
        system = "你是一个SQL专家。根据给定的数据库表结构，将用户的自然语言问题转换为正确的SQL查询。只输出SQL，不要解释。"
    else:
        system = ("You are a SQL expert. Given the database schema, convert the natural "
                  "language question into a correct SQL query. Output only SQL, no explanation.")

    user_content = f"### Database Schema:\n{sample['schema']}\n\n### Question:\n{sample['question']}"

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["sql"]},
        ]
    }


def save_jsonl(samples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train, all_eval = [], []

    for loader, name in [
        (load_spider, "Spider"),
        (load_bird, "BIRD"),
        (load_cspider, "CSpider"),
    ]:
        train = loader(data_dir, "train")
        dev = loader(data_dir, "dev")
        all_train.extend(train)
        all_eval.extend(dev)
        print(f"{name}: {len(train)} train, {len(dev)} eval")

    train_msgs = [format_to_messages(s) for s in all_train]
    eval_msgs = [format_to_messages(s) for s in all_eval]

    random.seed(42)
    random.shuffle(train_msgs)

    save_jsonl(train_msgs, str(output_dir / "train.jsonl"))
    save_jsonl(eval_msgs, str(output_dir / "eval.jsonl"))
    print(f"\nTotal: {len(train_msgs)} train, {len(eval_msgs)} eval")


if __name__ == "__main__":
    main()
