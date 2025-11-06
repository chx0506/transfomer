import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    """层归一化模块"""

    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        """
        初始化层归一化

        参数:
            features: 特征维度大小
            eps: 防止除以零的小常数
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(features))  # 可学习的偏置参数

    def forward(self, x):
        # x: (批次大小, 序列长度, 隐藏层大小)
        # 保持维度用于广播计算
        mean = x.mean(dim=-1, keepdim=True)  # 计算最后一个维度的均值 (批次大小, 序列长度, 1)
        # 保持维度用于广播计算
        std = x.std(dim=-1, keepdim=True)  # 计算最后一个维度的标准差 (批次大小, 序列长度, 1)
        # 添加eps防止除以零或标准差过小的情况
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """前馈神经网络模块（Position-wise Feed Forward Network）"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        初始化前馈网络

        参数:
            d_model: 模型维度
            d_ff: 前馈网络内部维度
            dropout: dropout比率
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # 第一个线性层 (w1和b1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # 第二个线性层 (w2和b2)

    def forward(self, x):
        # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 前馈维度) --> (批次大小, 序列长度, 模型维度)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """输入嵌入层"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        初始化输入嵌入层

        参数:
            d_model: 嵌入维度
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # 嵌入层

    def forward(self, x):
        # (批次大小, 序列长度) --> (批次大小, 序列长度, 模型维度)
        # 根据论文乘以sqrt(d_model)来缩放嵌入向量
        return self.embedding(x) * math.sqrt(self.d_model)


# class PositionalEncoding(nn.Module):
#     """位置编码模块"""
#
#     def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
#         """
#         初始化位置编码
#
#         参数:
#             d_model: 模型维度
#             seq_len: 序列最大长度
#             dropout: dropout比率
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len
#         self.dropout = nn.Dropout(dropout)
#
#         # 创建位置编码矩阵 (序列长度, 模型维度)
#         pe = torch.zeros(seq_len, d_model)
#         # 创建位置向量 (序列长度, 1)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (序列长度, 1)
#         # 计算除数项 (模型维度/iwslt2017)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / iwslt2017)
#
#         # 对偶数位置应用正弦函数
#         pe[:, 0::2] = torch.sin(position * div_term)  # sin(位置 * (10000 ** (2i / 模型维度))
#         # 对奇数位置应用余弦函数
#         pe[:, 1::2] = torch.cos(position * div_term)  # cos(位置 * (10000 ** (2i / 模型维度))
#
#         # 添加批次维度
#         pe = pe.unsqueeze(0)  # (1, 序列长度, 模型维度)
#         # 将位置编码注册为缓冲区（不参与梯度更新）
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         # 将位置编码加到输入上，不计算梯度
#         x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (批次大小, 序列长度, 模型维度)
#         return self.dropout(x)


class ResidualConnection(nn.Module):
    """残差连接模块"""

    def __init__(self, features: int, dropout: float) -> None:
        """
        初始化残差连接

        参数:
            features: 特征维度
            dropout: dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)  # 层归一化

    def forward(self, x, sublayer):
        """残差连接：x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """多头注意力机制模块"""

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        初始化多头注意力

        参数:
            d_model: 模型维度
            h: 注意力头数量
            dropout: dropout比率
        """
        super().__init__()
        self.d_model = d_model  # 嵌入向量大小
        self.h = h  # 注意力头数量
        # 确保模型维度能被头数整除
        assert d_model % h == 0, "模型维度必须能被注意力头数整除"

        self.d_k = d_model // h  # 每个注意力头处理的向量维度
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # 查询矩阵 Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # 键矩阵 Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # 值矩阵 Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出矩阵 Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """计算注意力权重"""
        d_k = query.shape[-1]
        # 应用注意力公式: Q*K^T / sqrt(d_k)
        # (批次大小, 头数, 序列长度, d_k) --> (批次大小, 头数, 序列长度, 序列长度)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # 将掩码位置的值设为负无穷（经过softmax后接近0）
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # 在最后一个维度应用softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (批次大小, 头数, 序列长度, 序列长度) --> (批次大小, 头数, 序列长度, d_k)
        # 返回注意力加权后的值和注意力分数（可用于可视化）
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """前向传播"""
        # 线性变换得到Q、K、V
        query = self.w_q(q)  # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 模型维度)
        key = self.w_k(k)  # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 模型维度)
        value = self.w_v(v)  # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 模型维度)

        # 重塑为多头格式
        # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 头数, d_k) --> (批次大小, 头数, 序列长度, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # 计算注意力
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # 合并所有注意力头
        # (批次大小, 头数, 序列长度, d_k) --> (批次大小, 序列长度, 头数, d_k) --> (批次大小, 序列长度, 模型维度)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 通过输出线性层
        # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 模型维度)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """编码器块（包含自注意力和前馈网络）"""

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        初始化编码器块

        参数:
            features: 特征维度
            self_attention_block: 自注意力模块
            feed_forward_block: 前馈网络模块
            dropout: dropout比率
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # 两个残差连接：一个用于自注意力，一个用于前馈网络
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """前向传播"""
        # 自注意力 + 残差连接
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # 前馈网络 + 残差连接
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """编码器（由多个编码器块组成）"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        初始化编码器

        参数:
            features: 特征维度
            layers: 编码器块列表
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)  # 最终层归一化

    def forward(self, x, mask):
        """前向传播"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 应用层归一化


class DecoderBlock(nn.Module):
    """解码器块（包含掩码自注意力、交叉注意力和前馈网络）"""

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        初始化解码器块

        参数:
            features: 特征维度
            self_attention_block: 掩码自注意力模块
            cross_attention_block: 交叉注意力模块
            feed_forward_block: 前馈网络模块
            dropout: dropout比率
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # 三个残差连接：掩码自注意力、交叉注意力、前馈网络
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """前向传播"""
        # 掩码自注意力 + 残差连接
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 交叉注意力 + 残差连接（查询来自解码器，键值来自编码器）
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        # 前馈网络 + 残差连接
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """解码器（由多个解码器块组成）"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        初始化解码器

        参数:
            features: 特征维度
            layers: 解码器块列表
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)  # 最终层归一化

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """前向传播"""
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # 应用层归一化


class ProjectionLayer(nn.Module):
    """投影层（将解码器输出映射到词汇表概率分布）"""

    def __init__(self, d_model, vocab_size) -> None:
        """
        初始化投影层

        参数:
            d_model: 模型维度
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # 线性投影层

    def forward(self, x):
        # (批次大小, 序列长度, 模型维度) --> (批次大小, 序列长度, 词汇表大小)
        return self.proj(x)


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, projection_layer: ProjectionLayer) -> None:
        """
        初始化Transformer模型

        参数:
            encoder: 编码器
            decoder: 解码器
            src_embed: 源语言嵌入层
            tgt_embed: 目标语言嵌入层
            src_pos: 源语言位置编码
            tgt_pos: 目标语言位置编码
            projection_layer: 投影层
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # self.src_pos = src_pos
        # self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """编码源序列"""
        # (批次大小, 序列长度, 模型维度)
        src = self.src_embed(src)  # 词嵌入
        # src = self.src_pos(src)  # 位置编码
        return self.encoder(src, src_mask)  # 编码器处理

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """解码生成目标序列"""
        # (批次大小, 序列长度, 模型维度)
        tgt = self.tgt_embed(tgt)  # 词嵌入
        # tgt = self.tgt_pos(tgt)  # 位置编码
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # 解码器处理

    def project(self, x):
        """将解码器输出投影到词汇表空间"""
        # (批次大小, 序列长度, 词汇表大小)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    构建Transformer模型

    参数:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        src_seq_len: 源序列最大长度
        tgt_seq_len: 目标序列最大长度
        d_model: 模型维度（默认512）
        N: 编码器/解码器层数（默认6）
        h: 注意力头数（默认8）
        dropout: dropout比率（默认0.1）
        d_ff: 前馈网络内部维度（默认2048）

    返回:
        构建好的Transformer模型
    """
    # 创建嵌入层
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # # 创建位置编码层
    # src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 创建编码器块
    encoder_blocks = []
    for _ in range(N):  # 创建N个编码器块
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 创建解码器块
    decoder_blocks = []
    for _ in range(N):  # 创建N个解码器块
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 掩码自注意力
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 交叉注意力
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # 创建编码器和解码器
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # 创建投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 创建完整的Transformer模型
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)

    # 使用Xavier均匀分布初始化参数
    for p in transformer.parameters():
        if p.dim() > 1:  # 只初始化权重矩阵，不初始化偏置
            nn.init.xavier_uniform_(p)

    return transformer
