import torch

#一次性训练多少段文本
batch_size = 16

#每一段文本最多看多少个token
#也叫上下文长度 context length
block_size = 64

#训练多少轮
max_iters = 3000

#每隔多少轮打印一次loss
eval_intreval = 300


#学习率
learning_rate = 3e-4

#评估loss时取多少个batch
eval_iters = 100


#模型参数
n_embd = 128  # token向量维度
n_head = 4 #attention head 数量
n_layer = 4 #Transformer 层数
dropout = 0.1 #dropout 概率

#checkpoint配置
save_interval = 600
resume_from_checkpoint = False
checkpoint_path = "checkpoints/latest.pth"

#自动选择设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("使用设备:", device)