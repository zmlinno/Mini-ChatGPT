# eval.py

import json
import torch
from difflib import SequenceMatcher

from config import device, block_size
from tokenizer import CharTokenizer
from model import GPTLanguageModel


def load_qa_pairs(path="data/qa.jsonl"):
    """
    读取 qa.jsonl
    """
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            user = obj.get("user", "").strip()
            assistant = obj.get("assistant", "").strip()

            if user and assistant:
                pairs.append({
                    "user": user,
                    "assistant": assistant
                })

    return pairs


def load_model():
    """
    加载 tokenizer 和模型
    """
    tokenizer = CharTokenizer.load("vocab.json")

    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
    model.eval()

    return model, tokenizer


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens=120):
    """
    使用 greedy decoding 生成回答。

    greedy decoding:
    每一步都选择概率最高的 token。
    这样结果最稳定，适合做评估。
    """

    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)

        # 只取最后一个 token 的预测结果
        logits = logits[:, -1, :]

        # 直接选择分数最高的 token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        full_text = tokenizer.decode(idx[0].tolist())

        if "<end>" in full_text[len(prompt):]:
            break

    full_text = tokenizer.decode(idx[0].tolist())

    if "<assistant>" in full_text:
        answer = full_text.split("<assistant>")[-1]
    else:
        answer = full_text

    if "<end>" in answer:
        answer = answer.split("<end>")[0]

    return answer.strip()


def similarity(a, b):
    """
    计算两个字符串相似度，范围 0~1。
    1 表示完全一样。
    """
    return SequenceMatcher(None, a, b).ratio()


def main():
    print("正在加载模型...")
    model, tokenizer = load_model()
    print("模型加载完成。")
    print("-" * 60)

    pairs = load_qa_pairs()

    total_score = 0.0

    for i, pair in enumerate(pairs, start=1):
        user = pair["user"]
        target = pair["assistant"]

        prompt = f"<user> {user}\n<assistant> "

        pred = greedy_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=120
        )

        score = similarity(pred, target)
        total_score += score

        print(f"\n[{i}] 问题：{user}")
        print(f"标准答案：{target}")
        print(f"模型回答：{pred}")
        print(f"相似度：{score:.4f}")
        print("-" * 60)

    avg_score = total_score / len(pairs)

    print("\n评估完成")
    print(f"问答数量：{len(pairs)}")
    print(f"平均相似度：{avg_score:.4f}")


if __name__ == "__main__":
    main()