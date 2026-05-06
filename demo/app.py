"""
Gradio Web Demo
Text-to-SQL 交互式演示

Usage:
    python demo/app.py --model_path outputs/text2sql-qlora/final
"""

import argparse
import sqlite3
import tempfile

import gradio as gr
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
    model.eval()
    return model, tokenizer


def predict_sql(question: str, schema: str, model, tokenizer) -> str:
    if not question.strip() or not schema.strip():
        return "Please provide both a question and schema."

    if any("一" <= c <= "鿿" for c in question):
        system = "你是一个SQL专家。根据给定的数据库表结构，将用户的自然语言问题转换为正确的SQL查询。只输出SQL，不要解释。"
    else:
        system = ("You are a SQL expert. Given the database schema, convert the natural "
                  "language question into a correct SQL query. Output only SQL, no explanation.")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"### Database Schema:\n{schema}\n\n### Question:\n{question}"},
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


def execute_sql_on_schema(sql: str, schema: str) -> str:
    """在临时数据库上执行 SQL"""
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        for statement in schema.split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement + ";")

        cursor.execute(sql)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not results:
            return "Query executed successfully. No results returned."

        header = " | ".join(columns)
        separator = "-" * len(header)
        rows = "\n".join(" | ".join(str(v) for v in row) for row in results[:20])

        output = f"{header}\n{separator}\n{rows}"
        if len(results) > 20:
            output += f"\n... ({len(results)} rows total, showing first 20)"
        return output
    except Exception as e:
        return f"Execution error: {str(e)}"


def create_demo(model, tokenizer):
    with gr.Blocks(title="Text-to-SQL Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🗃️ Text-to-SQL Demo")
        gr.Markdown("输入数据库表结构和自然语言问题，生成 SQL 查询语句。支持中英双语。")

        with gr.Row():
            with gr.Column():
                schema_input = gr.Textbox(
                    label="Database Schema (DDL)",
                    placeholder="CREATE TABLE employees (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  salary DECIMAL\n);",
                    lines=8,
                )
                question_input = gr.Textbox(
                    label="Question (自然语言问题)",
                    placeholder="查找工资最高的员工姓名",
                    lines=2,
                )
                generate_btn = gr.Button("Generate SQL", variant="primary")

            with gr.Column():
                sql_output = gr.Textbox(label="Generated SQL", lines=5)
                execute_btn = gr.Button("Execute SQL")
                result_output = gr.Textbox(label="Execution Result", lines=8)

        gr.Markdown("### Examples")
        examples = gr.Examples(
            examples=[
                [
                    "CREATE TABLE students (id INT, name VARCHAR, age INT, major VARCHAR);\nCREATE TABLE courses (id INT, course_name VARCHAR, credits INT);\nCREATE TABLE enrollments (student_id INT, course_id INT, grade CHAR(1));",
                    "Find students who are enrolled in more than 3 courses",
                ],
                [
                    "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, order_date DATE);\nCREATE TABLE customers (id INT, name VARCHAR, city VARCHAR);",
                    "查找每个城市的总订单金额，按金额降序排列",
                ],
                [
                    "CREATE TABLE products (id INT, name VARCHAR, category VARCHAR, price DECIMAL);\nCREATE TABLE sales (id INT, product_id INT, quantity INT, sale_date DATE);",
                    "What are the top 5 best-selling products by total quantity?",
                ],
            ],
            inputs=[schema_input, question_input],
        )

        generate_btn.click(
            fn=lambda q, s: predict_sql(q, s, model, tokenizer),
            inputs=[question_input, schema_input],
            outputs=sql_output,
        )

        execute_btn.click(
            fn=execute_sql_on_schema,
            inputs=[sql_output, schema_input],
            outputs=result_output,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/text2sql-qlora/final")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded.")

    demo = create_demo(model, tokenizer)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
