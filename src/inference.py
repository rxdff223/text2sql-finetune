"""
推理脚本
加载微调后的模型，对单条或批量输入进行 SQL 生成

Usage:
    python src/inference.py --model_path outputs/text2sql-qlora/final \
                            --question "查找销售额最高的城市" \
                            --schema "CREATE TABLE sales (id INT, city VARCHAR, amount DECIMAL);"
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str, base_model: str = None):
    """加载模型"""
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
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer


def predict(model, tokenizer, question: str, schema: str, lang: str = "auto") -> str:
    """生成 SQL 查询"""
    if lang == "auto":
        lang = "zh" if any("一" <= c <= "鿿" for c in question) else "en"

    if lang == "zh":
        system = "你是一个SQL专家。根据给定的数据库表结构，将用户的自然语言问题转换为正确的SQL查询。只输出SQL，不要解释。"
    else:
        system = ("You are a SQL expert. Given the database schema, convert the natural "
                  "language question into a correct SQL query. Output only SQL, no explanation.")

    user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/text2sql-qlora/final")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.\n")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print("Format: first enter schema, then enter question.\n")
        while True:
            print("-" * 40)
            schema = input("Schema (or 'quit'): ").strip()
            if schema.lower() == "quit":
                break
            question = input("Question: ").strip()
            if not question:
                continue
            sql = predict(model, tokenizer, question, schema)
            print(f"\nGenerated SQL:\n  {sql}\n")
    elif args.question and args.schema:
        sql = predict(model, tokenizer, args.question, args.schema)
        print(f"Question: {args.question}")
        print(f"SQL: {sql}")
    else:
        print("Provide --question and --schema, or use --interactive mode.")
        print("\nExample:")
        schema = "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);"
        question = "Find the average salary for each department"
        sql = predict(model, tokenizer, question, schema)
        print(f"  Schema: {schema}")
        print(f"  Question: {question}")
        print(f"  SQL: {sql}")


if __name__ == "__main__":
    main()
