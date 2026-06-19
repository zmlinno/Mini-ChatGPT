# train.py

# train.py

# train.py

import csv
from pathlib import Path

import torch

from config import (
    max_iters,
    eval_intreval,
    learning_rate,
    eval_iters,
    device,
    save_interval,
    resume_from_checkpoint,
    checkpoint_path,
)
from data_loader import get_batch, tokenizer
from model import GPTLanguageModel


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOSS_LOG_PATH = LOG_DIR / "loss_log.csv"


@torch.no_grad()
def estimate_loss(model):
    """
    评估训练集和验证集上的 loss
    """
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


def init_loss_log(resume=False):
    """
    初始化 loss 日志文件。
    如果是从 checkpoint 恢复训练，就不覆盖原来的日志。
    """
    if resume and LOSS_LOG_PATH.exists():
        return

    with open(LOSS_LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])


def write_loss_log(step, train_loss, val_loss):
    """
    追加一行 loss 日志
    """
    with open(LOSS_LOG_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, train_loss, val_loss])


def save_checkpoint(model, optimizer, step, path):
    """
    保存 checkpoint。
    不只保存模型参数，也保存 optimizer 状态和当前 step。
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "vocab_size": tokenizer.vocab_size,
    }

    torch.save(checkpoint, path)
    print(f"checkpoint 已保存: {path}")


def load_checkpoint(model, optimizer, path):
    """
    加载 checkpoint。
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint["step"] + 1

    print(f"已从 checkpoint 恢复训练: {path}")
    print(f"继续训练起点 step: {start_step}")

    return start_step


def main():
    # 1. 创建模型
    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    # 2. 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f} M")

    # 3. 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. 是否从 checkpoint 恢复
    start_step = 0

    ckpt_path = Path(checkpoint_path)

    if resume_from_checkpoint and ckpt_path.exists():
        start_step = load_checkpoint(model, optimizer, ckpt_path)
        init_loss_log(resume=True)
    else:
        init_loss_log(resume=False)

    # 5. 开始训练
    for step in range(start_step, max_iters + 1):

        # 评估 loss
        if step % eval_intreval == 0:
            losses = estimate_loss(model)

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"step {step}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

            write_loss_log(step, train_loss, val_loss)

        # 保存 checkpoint
        if step > 0 and step % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                path=checkpoint_path
            )

            # 额外保存一个带 step 编号的 checkpoint，方便回退
            numbered_path = CHECKPOINT_DIR / f"step_{step}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                path=numbered_path
            )

        # 最后一轮只评估和保存，不再训练
        if step == max_iters:
            break

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

    # 6. 保存最终模型和词表
    tokenizer.save("vocab.json")
    torch.save(model.state_dict(), "mini_gpt.pth")

    # 7. 保存最终 checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=max_iters,
        path=checkpoint_path
    )

    print("训练完成")
    print("最终模型已保存到 mini_gpt.pth")
    print("词表已保存到 vocab.json")
    print(f"最终 checkpoint 已保存到 {checkpoint_path}")
    print(f"loss 日志已保存到 {LOSS_LOG_PATH}")


if __name__ == "__main__":
    main()