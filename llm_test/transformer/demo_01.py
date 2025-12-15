# coding=utf-8
import torch
import torch.nn as nn
from llm_test.transformer.resolver import EncoderLayer, DecoderLayer
from llm_test.transformer.PositionalEncoding import PositionalEncoding

"""
Transformer 架构解析 实际示例
"""


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 创建多层编码器和解码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 嵌入和位置编码
        src = self.pos_encoding(self.encoder_embedding(src))
        tgt = self.pos_encoding(self.decoder_embedding(tgt))

        # 编码器
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # 解码器
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.output_layer(dec_output)


# 使用示例
transformer = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=8000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# 模拟输入
src_tokens = torch.randint(0, 10000, (32, 50))  # 批次大小32，序列长度50
tgt_tokens = torch.randint(0, 8000, (32, 50))

# 前向传播
output = transformer(src_tokens, tgt_tokens, None, None)
print(f"Transformer输出形状: {output.shape}")  # (32, 50, 8000)
