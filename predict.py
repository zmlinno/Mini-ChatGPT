# predict.py

import torch
from torch.nn import functional as F

from config import device, block_size
from tokenizer import CharTokenizer
from model import GPTLanguageModel


def show_token(token: str) -> str:
    """
    为了让特殊字符显示得更清楚。
    """
    if token == "\n":
        return "\\n"
    if token == " ":
        return "<space>"
    if token == "\t":
        return "\\t"
    return token


def load_model():
    """
    加载 tokenizer 和训练好的模型。
    """
    tokenizer = CharTokenizer.load("vocab.json")

    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
    model.eval()

    return model, tokenizer


@torch.no_grad()
def predict_next_token(model, tokenizer, text, top_n=5):
    """
    给定一段文本，预测下一个 token 的 top_n 概率。
    """

    # 1. 文本 -> token id
    ids = tokenizer.encode(text)

    # 2. 转 tensor，形状 [1, T]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 3. 只保留最后 block_size 个 token
    idx_cond = idx[:, -block_size:]

    # 4. 模型前向传播
    logits, _ = model(idx_cond)

    # 5. 只看最后一个位置的输出
    logits = logits[:, -1, :]  # [1, vocab_size]

    # 6. softmax 转成概率
    probs = F.softmax(logits, dim=-1)

    # 7. 取概率最高的 top_n 个 token
    top_probs, top_ids = torch.topk(probs, k=top_n, dim=-1)

    results = []

    for prob, token_id in zip(top_probs[0], top_ids[0]):
        token_id = token_id.item()
        token = tokenizer.decode([token_id])

        results.append({
            "token": token,
            "token_id": token_id,
            "prob": prob.item(),
            "logit": logits[0, token_id].item()
        })

    return results


@torch.no_grad()
def explain_generation(model, tokenizer, prompt, steps=20, top_n=5):
    """
    逐步展示模型生成过程。
    每一步都打印 top_n 个候选 token。
    """

    current_text = prompt

    print("=" * 60)
    print("初始输入：")
    print(current_text)
    print("=" * 60)

    for step in range(1, steps + 1):
        results = predict_next_token(
            model,
            tokenizer,
            current_text,
            top_n=top_n
        )

        print(f"\n第 {step} 步：")
        print("当前文本末尾：")
        print(current_text[-80:])

        print("\n模型预测下一个 token 的 top 候选：")

        for i, item in enumerate(results, start=1):
            token = show_token(item["token"])
            token_id = item["token_id"]
            prob = item["prob"]
            logit = item["logit"]

            print(
                f"{i}. token='{token}' "
                f"id={token_id} "
                f"prob={prob:.4f} "
                f"logit={logit:.4f}"
            )

        # 为了方便观察，这里选择概率最高的 token，也就是 greedy decoding
        next_token = results[0]["token"]
        current_text += next_token

        print(f"\n本步选择：'{show_token(next_token)}'")

        if "<end>" in current_text[len(prompt):]:
            print("\n检测到 <end>，生成结束。")
            break

    print("\n" + "=" * 60)
    print("最终生成文本：")
    print(current_text)
    print("=" * 60)


def main():
    print("正在加载模型...")
    model, tokenizer = load_model()
    print("模型加载完成。")
    print("-" * 60)

    while True:
        user_input = input("请输入问题，输入 exit 退出：").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("退出 predict.py")
            break

        if user_input == "":
            continue

        prompt = f"<user> {user_input}\n<assistant> "

        explain_generation(
            model,
            tokenizer,
            prompt,
            steps=30,
            top_n=5
        )


if __name__ == "__main__":
    main()