# coding=utf-8
import torch
import torch.nn as nn
from llm_test.transformer.resolver import EncoderLayer, DecoderLayer
from llm_test.transformer.PositionalEncoding import PositionalEncoding

"""
Transformer 架构解析
"""
# 参数设置
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
batch_size = 32
seq_length = 50
vocab_size = 10000

# 创建编码器层
encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)

# 创建解码器层
decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

# 创建位置编码
pos_encoding = PositionalEncoding(d_model, dropout)

# 模拟输入数据
src_input = torch.randn(batch_size, seq_length, d_model)  # 源序列
tgt_input = torch.randn(batch_size, seq_length, d_model)  # 目标序列

# 添加位置编码
src_input = pos_encoding(src_input)
tgt_input = pos_encoding(tgt_input)

# 创建mask（示例中使用全1，表示没有mask）
src_mask = torch.ones(batch_size, 1, seq_length, seq_length)
tgt_mask = torch.ones(batch_size, 1, seq_length, seq_length)

# 前向传播
encoder_output = encoder_layer(src_input, src_mask)
decoder_output = decoder_layer(tgt_input, encoder_output, src_mask, tgt_mask)

print(f"输入形状: {src_input.shape}")
print(f"编码器输出形状: {encoder_output.shape}")
print(f"解码器输出形状: {decoder_output.shape}")
