# build_data.py

import json
import re
from pathlib import Path


INPUT_PATH = Path("data/qa.jsonl")
OUTPUT_PATH = Path("data/chat.txt")


def normalize_text(s: str) -> str:
    """
    用于去重的简单归一化：
    1. 去掉前后空格
    2. 多个空格合并成一个
    3. 中文问号和英文问号统一
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("?", "？")
    return s


def load_qa_pairs(path: Path):
    pairs = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行 JSON 格式错误: {e}")

            user = obj.get("user", "").strip()
            assistant = obj.get("assistant", "").strip()

            if not user or not assistant:
                print(f"跳过第 {line_no} 行：user 或 assistant 为空")
                continue

            pairs.append({
                "user": user,
                "assistant": assistant
            })

    return pairs


def deduplicate_pairs(pairs):
    """
    按 user 问题去重。
    如果出现重复问题，只保留第一次出现的问答。
    """
    seen_questions = set()
    unique_pairs = []

    for pair in pairs:
        key = normalize_text(pair["user"])

        if key in seen_questions:
            continue

        seen_questions.add(key)
        unique_pairs.append(pair)

    return unique_pairs


def build_chat_text(pairs):
    """
    生成 GPT 训练文本格式：
    <user> ...
    <assistant> ...
    <end>
    """
    chunks = []

    for pair in pairs:
        chunk = (
            f"<user> {pair['user']}\n"
            f"<assistant> {pair['assistant']}\n"
            f"<end>\n"
        )
        chunks.append(chunk)

    return "\n".join(chunks)


def main():
    pairs = load_qa_pairs(INPUT_PATH)
    unique_pairs = deduplicate_pairs(pairs)
    chat_text = build_chat_text(unique_pairs)

    OUTPUT_PATH.write_text(chat_text, encoding="utf-8")

    print("原始问答数量:", len(pairs))
    print("去重后问答数量:", len(unique_pairs))
    print("已生成:", OUTPUT_PATH)
    print("训练文本字符数:", len(chat_text))


if __name__ == "__main__":
    main()