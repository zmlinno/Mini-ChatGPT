# tokenizer.py

import json

UNK_TOKEN = "�"


class CharTokenizer:
    def __init__(self, text=None, chars=None):
        """
        两种创建方式：

        1. 训练时：
           CharTokenizer(text=训练文本)

        2. 推理时：
           CharTokenizer(chars=保存好的词表)
        """

        if chars is None:
            if text is None:
                raise ValueError("必须提供 text 或 chars")

            chars = sorted(list(set(text)))

            # 避免重复加入 UNK
            if UNK_TOKEN in chars:
                chars.remove(UNK_TOKEN)

            # 第 0 个 token 留给未知字符
            chars = [UNK_TOKEN] + chars

        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        """
        文本 -> token id
        如果遇到训练集中没见过的字符，就转成 UNK_TOKEN
        """
        unk_id = self.stoi[UNK_TOKEN]
        return [self.stoi.get(ch, unk_id) for ch in s]

    def decode(self, ids):
        """
        token id -> 文本
        """
        return "".join([self.itos[i] for i in ids])

    def save(self, path):
        """
        保存词表
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chars": self.chars
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):
        """
        加载词表
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return cls(chars=obj["chars"])


if __name__ == "__main__":
    with open("data/chat.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text=text)

    print("训练数据总字符数:", len(text))
    print("词表大小 vocab_size:", tokenizer.vocab_size)

    sample = "<user> 你好"
    ids = tokenizer.encode(sample)

    print("原始文本:", sample)
    print("编码结果:", ids)
    print("解码结果:", tokenizer.decode(ids))

    tokenizer.save("vocab.json")
    print("词表已保存到 vocab.json")