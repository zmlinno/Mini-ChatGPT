# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import block_size, n_embd, n_head, n_layer, dropout, device





class Head(nn.Module):
    """
    一个 Attention Head
    """

    def __init__(self, head_size):
        super().__init__()

        # 生成 K、Q、V 的线性层
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角矩阵，用来防止模型看到未来的 token
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [B, T, C]

        B: batch size
        T: token 序列长度
        C: embedding 维度
        """

        B, T, C = x.shape

        # x 生成 Q K V
        k = self.key(x)      # [B, T, head_size]
        q = self.query(x)    # [B, T, head_size]
        v = self.value(x)    # [B, T, head_size]

        # 计算 Q 和 K 的相似度
        # wei shape: [B, T, T]
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # mask：让当前位置只能看自己和前面的 token，不能看未来
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # softmax 得到注意力权重
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 用注意力权重加权 V，得到上下文向量
        out = wei @ v        # [B, T, head_size]

        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        # 多个 head 拼接后，再做一次线性变换
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 把多个 head 的结果拼接起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 输出投影
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    一个 Transformer Block
    """

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Self-Attention + 残差连接
        x = x + self.sa(self.ln1(x))

        # FeedForward + 残差连接
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    """
    MiniGPT 语言模型
    """

    def __init__(self, vocab_size):
        super().__init__()

        # token embedding 表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 位置 embedding 表
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 多层 Transformer Block
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        # 最后的 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # 输出层，也就是你之前问的 W_vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: 输入 token id
        targets: 正确答案，也就是下一个 token id

        idx shape: [B, T]
        targets shape: [B, T]
        """

        B, T = idx.shape

        # 1. token id -> token embedding
        tok_emb = self.token_embedding_table(idx)  # [B, T, C]

        # 2. 位置 id -> position embedding
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)  # [T, C]

        # 3. token embedding + position embedding
        x = tok_emb + pos_emb  # [B, T, C]

        # 4. 经过 Transformer Blocks
        x = self.blocks(x)

        # 5. 最后的归一化
        x = self.ln_f(x)

        # 6. 输出 logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            # CrossEntropy 需要 [B*T, C]
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        根据输入 idx，不断生成新的 token
        """

        for _ in range(max_new_tokens):

            # 如果输入太长，只保留最后 block_size 个 token
            idx_cond = idx[:, -block_size:]

            # 得到预测结果
            logits, loss = self(idx_cond)

            # 只取最后一个位置的预测
            logits = logits[:, -1, :]  # [B, vocab_size]

            # 转成概率
            probs = F.softmax(logits, dim=-1)

            # 根据概率采样一个 token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接到原来的序列后面
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    from data_loader import get_batch, tokenizer

    model = GPTLanguageModel(tokenizer.vocab_size)
    model = model.to(device)

    x, y = get_batch("train")

    logits, loss = model(x, y)

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    # 测试生成
    start_text = "<user> "
    start_ids = tokenizer.encode(start_text)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(idx, max_new_tokens=100)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("生成结果:")
    print(generated_text)