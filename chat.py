# chat.py

# chat.py

import torch
from torch.nn import functional as F

from config import device, block_size
from tokenizer import CharTokenizer
from model import GPTLanguageModel


def top_k_top_p_filtering(logits, top_k=20, top_p=0.9):
    """
    对 logits 做 top-k 和 top-p 过滤。

    logits shape: [B, vocab_size]
    """

    # 1. top-k：只保留概率最高的 k 个 token
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))

        values, _ = torch.topk(logits, top_k)

        # 第 k 大的分数
        min_values = values[:, -1].unsqueeze(-1)

        # 比第 k 大还小的分数全部设为 -inf
        logits = logits.masked_fill(logits < min_values, float("-inf"))

    # 2. top-p：只保留累计概率达到 top_p 的 token
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累计概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p

        # 右移一位，保证第一个超过 top_p 的 token 也被保留
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 把排序后的 mask 映射回原始 vocab 顺序
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def generate_reply(
    model,
    tokenizer,
    prompt,
    max_new_tokens=120,
    temperature=0.7,
    top_k=20,
    top_p=0.9
):
    """
    根据用户输入生成回答。
    """

    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token
            idx_cond = idx[:, -block_size:]

            # 模型预测
            logits, _ = model(idx_cond)

            # 只取最后一个位置的输出
            logits = logits[:, -1, :]  # [B, vocab_size]

            # temperature 控制随机程度
            logits = logits / temperature

            # top-k / top-p 过滤
            logits = top_k_top_p_filtering(
                logits,
                top_k=top_k,
                top_p=top_p
            )

            # 转成概率
            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接新 token
            idx = torch.cat((idx, idx_next), dim=1)

            # 如果生成到了 <end>，提前停止
            generated_text = tokenizer.decode(idx[0].tolist())

            if "<end>" in generated_text[len(prompt):]:
                break

    full_text = tokenizer.decode(idx[0].tolist())

    # 只保留 assistant 的回答部分
    if "<assistant>" in full_text:
        answer = full_text.split("<assistant>")[-1]
    else:
        answer = full_text

    # 遇到 <end> 截断
    if "<end>" in answer:
        answer = answer.split("<end>")[0]

    return answer.strip()


def main():
    print("正在加载 MiniChatGPT...")

    # 1. 加载 tokenizer
    tokenizer = CharTokenizer.load("vocab.json")

    # 2. 创建模型
    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    # 3. 加载训练好的参数
    model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
    model.eval()

    print("MiniChatGPT 加载完成。")
    print("输入 exit 或 quit 可以退出。")
    print("-" * 40)

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("MiniChatGPT：再见！")
            break

        if user_input == "":
            continue

        prompt = f"<user> {user_input}\n<assistant> "

        reply = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=120,
            temperature=0.7,
            top_k=20,
            top_p=0.9
        )

        print("MiniChatGPT：", reply)


if __name__ == "__main__":
    main()