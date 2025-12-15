# coding=utf-8
import re, collections

"""
子词分词: 将常见的词（如 "agent"）保留为完整的词元，同时将不常见的词（如 "Tokenization"）拆分成多个有意义的子词片段（如 "Token" 和 "ization"）

字节对编码 (Byte-Pair Encoding, BPE) 是最主流的子词分词算法之一[6]，GPT系列模型就采用了这种算法。其核心思想非常简洁，可以理解为一个“贪心”的合并过程：

1.初始化：将词表初始化为所有在语料库中出现过的基本字符。
2.迭代合并：在语料库上，统计所有相邻词元对的出现频率，找到频率最高的一对，将它们合并成一个新的词元，并加入词表。
3.重复：重复第 2 步，直到词表大小达到预设的阈值。
"""


def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


# 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4 # 设置合并次数

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 20)


"""
>>>
第1次合并: ('u', 'g') -> ug
新词表（部分）: ['h ug </w>', 'p ug </w>', 'p u n </w>', 'b u n </w>']
--------------------
第2次合并: ('ug', '</w>') -> ug</w>
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p u n </w>', 'b u n </w>']
--------------------
第3次合并: ('u', 'n') -> un
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un </w>', 'b un </w>']
--------------------
第4次合并: ('un', '</w>') -> un</w>
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un</w>', 'b un</w>']
--------------------
"""