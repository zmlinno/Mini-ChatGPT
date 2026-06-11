# chat.py

import torch
from torch.nn import functional as F

from config import device, block_size
from data_loader import tokenizer
from model import GPTLanguageModel


def generate_reply(model, prompt, max_new_tokens=100, temperature=0.8):
    """
    根据用户输入生成回答。

    prompt: 拼接好的提示词，例如：
        <user> 你好
        <assistant>

    max_new_tokens: 最多生成多少个新 token
    temperature: 温度，越低越稳定，越高越随机
    """

    # 1. 编码输入文本
    input_ids = tokenizer.encode(prompt)

    # 2. 转成 tensor，形状是 [1, T]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token
            idx_cond = idx[:, -block_size:]

            # 模型预测
            logits, _ = model(idx_cond)

            # 只取最后一个位置的输出
            logits = logits[:, -1, :]

            # temperature 控制随机程度
            logits = logits / temperature

            # 转成概率
            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接新 token
            idx = torch.cat((idx, idx_next), dim=1)

            # 如果生成到了 <end>，就可以提前停止
            generated_text = tokenizer.decode(idx[0].tolist())
            if "<end>" in generated_text[len(prompt):]:
                break

    # 3. 解码完整文本
    full_text = tokenizer.decode(idx[0].tolist())

    # 4. 只截取 assistant 的回答部分
    if "<assistant>" in full_text:
        answer = full_text.split("<assistant>")[-1]
    else:
        answer = full_text

    # 5. 如果生成了 <end>，就截断
    if "<end>" in answer:
        answer = answer.split("<end>")[0]

    return answer.strip()


def main():
    print("正在加载 MiniChatGPT...")

    # 1. 创建模型结构
    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    # 2. 加载训练好的参数
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

        # 注意：当前 tokenizer 只能识别训练数据中出现过的字符
        try:
            prompt = f"<user> {user_input}\n<assistant> "
            reply = generate_reply(
                model,
                prompt,
                max_new_tokens=120,
                temperature=0.8
            )
            print("MiniChatGPT：", reply)

        except KeyError as e:
            print("MiniChatGPT：你输入的字符不在训练数据词表里。")
            print("建议把相关问答加入 data/chat.txt，然后重新训练模型。")
            print("无法识别的字符：", e)


if __name__ == "__main__":
    main()