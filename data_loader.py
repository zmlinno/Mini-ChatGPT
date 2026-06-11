import torch
from tokenizer import CharTokenizer
from config import batch_size, block_size, device


# 1. 读取训练文本
with open("data/chat.txt", "r", encoding="utf-8") as f:
    text = f.read()


# 2. 创建 tokenizer
tokenizer = CharTokenizer(text)


# 3. 把完整文本转换成 token id
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)


# 4. 划分训练集和验证集
# 前 90% 用来训练，后 10% 用来验证
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """
    生成一批训练数据。

    split:
        "train" 表示训练集
        "val" 表示验证集

    返回:
        x: 输入
        y: 目标答案，也就是 x 的下一个 token
    """

    data_source = train_data if split == "train" else val_data

    # 随机选择 batch_size 个起点
    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    # x 是从 i 开始，长度为 block_size 的片段
    x = torch.stack([
        data_source[i : i + block_size]
        for i in ix
    ])

    # y 是 x 往后移动一个 token
    y = torch.stack([
        data_source[i + 1 : i + block_size + 1]
        for i in ix
    ])

    # 放到 GPU 或 CPU 上
    x = x.to(device)
    y = y.to(device)

    return x, y


if __name__ == "__main__":
    print("训练文本总字符数:", len(text))
    print("词表大小 vocab_size:", tokenizer.vocab_size)
    print("总 token 数:", len(data))
    print("训练集 token 数:", len(train_data))
    print("验证集 token 数:", len(val_data))

    x, y = get_batch("train")

    print("x 的形状:", x.shape)
    print("y 的形状:", y.shape)

    print("第一条 x:", x[0])
    print("第一条 y:", y[0])

    print("第一条 x 解码:")
    print(tokenizer.decode(x[0].tolist()))

    print("第一条 y 解码:")
    print(tokenizer.decode(y[0].tolist()))
