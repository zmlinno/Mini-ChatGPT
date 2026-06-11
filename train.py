# train.py

import torch

from config import max_iters, eval_intreval, learning_rate, eval_iters, device
from data_loader import get_batch, tokenizer
from model import GPTLanguageModel


@torch.no_grad()
def estimate_loss(model):
    """
    评估训练集和验证集上的 loss
    """
    out = {}

    # 切换到评估模式
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    # 切回训练模式
    model.train()

    return out


def main():
    # 1. 创建模型
    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    # 2. 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f} M")

    # 3. 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. 开始训练
    for iter in range(max_iters):

        # 每隔一段时间评估一次 loss
        if iter % eval_intreval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        # 获取一批训练数据
        x, y = get_batch("train")

        # 前向传播
        logits, loss = model(x, y)

        # 梯度清零
        optimizer.zero_grad(set_to_none=True)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

    # 5. 训练结束后保存模型
    torch.save(model.state_dict(), "mini_gpt.pth")
    print("训练完成，模型已保存到 mini_gpt.pth")


if __name__ == "__main__":
    main()