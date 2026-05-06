"""
工具函数
"""

import re
import sqlparse


def normalize_sql(sql: str) -> str:
    """SQL 归一化"""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.lower()
    return sql


def format_sql(sql: str) -> str:
    """格式化 SQL 输出"""
    return sqlparse.format(sql, reindent=True, keyword_case="upper")


def detect_language(text: str) -> str:
    """检测文本语言"""
    if any("一" <= c <= "鿿" for c in text):
        return "zh"
    return "en"


def count_tokens(tokenizer, text: str) -> int:
    """计算 token 数量"""
    return len(tokenizer.encode(text))
