import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 源语言特殊标记
        self.sos_token_src = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token_src = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

        # 目标语言特殊标记
        self.sos_token_tgt = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token_tgt = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token_tgt = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
        # token_to_id()：分词器方法，将标记字符串转换为ID
        # torch.Tensor()：创建包含单个ID的张量
        # dtype=torch.int64：指定整数类型（模型输入要求）
        # 文本到ID的转换是将人类可读的自然语言文本转换为机器可处理的数字序列的过程

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]  # 源语言句子 和 目标语言句子对
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # 将句子分为tokens，然后每个tokens映射为对应词汇表中的一个数字
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids  # 返回一个数组

        # 对于源句子和目标句子，有很多种长度，所以要进行填充至一定的序列长度
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 减2是因为有开始和结束标记词
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 减1是因为decode只输入开始标记

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('sentence is too long')  # 检测是否seq_len小于要处理的句子的长度

        # Add SOS and EOS to the Source text
        # ​示例​：[SOS_ID, 34, 56, EOS_ID, PAD_ID, PAD_ID]
        encoder_input = torch.cat(
            [self.sos_token_src,
             torch.tensor(enc_input_tokens, dtype=torch.int64),
             self.eos_token_src,
             torch.tensor([self.pad_token_src] * enc_num_padding_tokens, dtype=torch.int64)

             ]

        )

        # 目标语言句子的ID序列，添加了SOS标记，但不包含EOS标记（用于教师强制训练）
        # [SOS_ID, 78, 92, PAD_ID, PAD_ID]
        decoder_input = torch.cat(
            [
                self.sos_token_tgt,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64)

            ]

        )

        # 创建标签，即跟编码器输出的预测进行对比的真实标签
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token_tgt,
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # 为了方便调试，再次确认是否都满足seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,  # (seq_len)
            'decoder_input': decoder_input,  # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
            # 标识哪些位置是真实标记（1） vs 填充位置（0）
            'decoder_mask': (decoder_input != self.pad_token_tgt).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1,seq_len) & (1,seq_len,seq_len) #进行了广播操作
            # 组合填充掩码（不注意填充）和因果掩码（不注意未来的词），防止解码器看到未来信息——进行逻辑与操作
            # 填充掩码：原始：decoder_input != PAD → [True, True, True, False, False]
            #        扩展：[[[True, True, True, False, False]]]  # (1,1,5)
            # #因果掩码：
            # [[[ True, False, False, False, False],
            #   [ True,  True, False, False, False],
            #   [ True,  True,  True, False, False],
            #   [ True,  True,  True,  True, False],
            #   [ True,  True,  True,  True, True]]]  # (1,5,5)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text

        }


def causal_mask(size):
    # 使用triu这个方法，告诉我一个矩阵对角线上的部分，
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # 返回了对角线上(包含对角线)的元素为1，对角线下的元素为0
    # diagonal = 1即从主对角线位置向上移动一位开始，以下设为0，以上设为1 （故对角线也为0）
    return mask == 0  # 使得矩阵结果可以相反，对角线下的为True即1，对角线上的为0
