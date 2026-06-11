class CharTokenizer:
    def __init__(self, text):
        """
        text: 训练数据的完整文本
        """

        # 1. 找出训练文本中出现过的所有字符
        chars = sorted(list(set(text)))

        # 2. 建立字符 -> 数字id 的映射
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # 3. 建立数字id -> 字符 的映射
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # 4. 词表大小
        self.vocab_size = len(chars)

    def encode(self, s):
        """
        把字符串转换成 token id 列表
        例如：'你好' -> [12, 35]
        """
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        """
        把 token id 列表转换回字符串
        例如：[12, 35] -> '你好'
        """
        return ''.join([self.itos[i] for i in ids])


if __name__ == "__main__":
    # 读取训练数据
    with open("data/chat.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 创建 tokenizer
    tokenizer = CharTokenizer(text)

    print("训练数据总字符数:", len(text))
    print("词表大小 vocab_size:", tokenizer.vocab_size)

    # 测试编码和解码
    sample = "<user> 你好"
    ids = tokenizer.encode(sample)

    print("原始文本:", sample)
    print("编码结果:", ids)
    print("解码结果:", tokenizer.decode(ids))