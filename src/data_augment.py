"""
数据增强脚本
- 对英文问题生成中文翻译版本
- 对问题进行 paraphrase 增强
- 按 SQL 复杂度标注难度
"""

import json
import re
import random
from pathlib import Path


def classify_difficulty(sql: str) -> str:
    """根据 SQL 复杂度分级"""
    sql_upper = sql.upper()

    nested = sql_upper.count("SELECT") - 1
    has_join = "JOIN" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_subquery = nested > 0
    has_union = "UNION" in sql_upper or "INTERSECT" in sql_upper or "EXCEPT" in sql_upper

    score = 0
    score += nested * 2
    score += has_join * 1
    score += has_group * 1
    score += has_having * 2
    score += has_union * 2

    if score == 0:
        return "easy"
    elif score <= 2:
        return "medium"
    elif score <= 4:
        return "hard"
    else:
        return "extra_hard"


def augment_with_difficulty(input_path: str, output_path: str):
    """为数据集添加难度标签"""
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            sql = sample["messages"][-1]["content"]
            sample["difficulty"] = classify_difficulty(sql)
            samples.append(sample)

    difficulty_counts = {}
    for s in samples:
        d = s["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    print(f"Difficulty distribution:")
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d}: {c} ({c/len(samples)*100:.1f}%)")

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


PARAPHRASE_TEMPLATES = [
    "请帮我写一个SQL查询：{question}",
    "我想查询：{question}",
    "帮我用SQL实现：{question}",
    "Write a SQL query to: {question}",
    "I need a query that: {question}",
    "Can you help me write SQL for: {question}",
]


def augment_with_paraphrase(input_path: str, output_path: str, ratio: float = 0.3):
    """通过模板对问题进行 paraphrase 增强"""
    random.seed(42)
    samples = []
    augmented = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    num_augment = int(len(samples) * ratio)
    selected = random.sample(samples, min(num_augment, len(samples)))

    for sample in selected:
        new_sample = json.loads(json.dumps(sample))
        messages = new_sample["messages"]
        user_msg = messages[1]["content"]

        question_match = re.search(r"### Question:\n(.+?)$", user_msg, re.DOTALL)
        if not question_match:
            continue

        original_question = question_match.group(1).strip()
        template = random.choice(PARAPHRASE_TEMPLATES)
        new_question = template.format(question=original_question)

        messages[1]["content"] = user_msg.replace(
            f"### Question:\n{original_question}",
            f"### Question:\n{new_question}"
        )
        augmented.append(new_sample)

    all_samples = samples + augmented
    random.shuffle(all_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Augmented: {len(samples)} -> {len(all_samples)} samples (+{len(augmented)})")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    train_path = str(data_dir / "train.jsonl")
    augmented_path = str(data_dir / "train_augmented.jsonl")

    if not Path(train_path).exists():
        print(f"Train file not found: {train_path}")
        print("Run data_process.py first.")
        return

    print("=== Adding difficulty labels ===")
    augment_with_difficulty(train_path, train_path)

    print("\n=== Paraphrase augmentation ===")
    augment_with_paraphrase(train_path, augmented_path, ratio=0.3)

    print(f"\nDone. Augmented data saved to: {augmented_path}")
    print("You can use train_augmented.jsonl in train_config.yaml for training.")


if __name__ == "__main__":
    main()
