# plot_loss.py

import csv
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_LOG_PATH = Path("logs/loss_log.csv")
OUTPUT_PATH = Path("logs/loss_curve.png")


def load_loss_log(path):
    steps = []
    train_losses = []
    val_losses = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    return steps, train_losses, val_losses


def main():
    if not LOSS_LOG_PATH.exists():
        print("没有找到 logs/loss_log.csv，请先运行 python train.py")
        return

    steps, train_losses, val_losses = load_loss_log(LOSS_LOG_PATH)

    plt.figure(figsize=(8, 5))

    plt.plot(steps, train_losses, marker="o", label="train loss")
    plt.plot(steps, val_losses, marker="o", label="val loss")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("MiniChatGPT Training Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)

    print(f"loss 曲线已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()