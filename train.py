
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


import warnings
from tqdm import tqdm
import os
from pathlib import Path

# 导入Huggingface的数据集和分词器
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    使用贪心算法进行序列解码（推理阶段）

    参数:
        model: 训练好的Transformer模型
        source: 源语言输入序列
        source_mask: 源序列的注意力掩码
        tokenizer_src: 源语言分词器
        tokenizer_tgt: 目标语言分词器
        max_len: 生成序列的最大长度
        device: 使用的设备(CPU/GPU)
    """
    # 获取起始符和结束符的token ID
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # 预计算编码器输出，每个解码步骤重复使用
    encoder_output = model.encode(source, source_mask)
    # 使用起始符初始化解码器输入
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # 循环生成序列直到达到最大长度或遇到结束符
    while True:
        if decoder_input.size(1) == max_len:
            break

        # 为目标序列构建因果注意力掩码（防止看到未来信息）
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # 通过解码器计算输出
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 获取下一个token的概率分布并选择概率最高的token（贪心选择）
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        # 将新生成的token添加到解码器输入中
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # 如果生成结束符则停止
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def calculate_validation_metrics(model, val_dataloader, loss_fn, tokenizer_src,tokenizer_tgt, device, max_examples=20):
    """
    计算验证集上的损失和准确率指标

    参数:
        model: 要验证的模型
        val_dataloader: 验证数据加载器
        loss_fn: 损失函数
        tokenizer_tgt: 目标语言分词器
        device: 计算设备
        max_examples: 最大验证样本数（避免耗时过长）————太大的话计算的很慢
    """
    model.eval()
    total_val_loss = 0
    total_examples = 0

    # 用于计算BLEU等指标
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= max_examples:  # 限制验证样本数量
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # 前向传播
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # 计算损失
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_val_loss += loss.item()
            total_examples += 1

            # 记录预测结果用于计算BLEU
            if batch_idx < 10:  # 只记录前10个样本的预测用于BLEU计算
                # 使用贪心解码生成预测
                model_out = greedy_decode(model, encoder_input, encoder_mask,
                                          tokenizer_src, tokenizer_tgt,
                                          encoder_input.size(1), device)

                # 解码为文本
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                all_predictions.append(model_out_text)
                all_targets.append([target_text])  # BLEU需要参考翻译列表

    # 计算平均验证损失
    avg_val_loss = total_val_loss / total_examples if total_examples > 0 else 0

    # 计算BLEU分数
    bleu_score = 0
    if all_predictions and all_targets:
        try:
            metric = torchmetrics.BLEUScore()
            bleu_score = metric(all_predictions, all_targets)
            bleu_score = bleu_score.item()
        except:
            bleu_score = 0

    return avg_val_loss, bleu_score



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step,
                   writer,
                   num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len,
                                      device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    val_metrics = {}
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    val_metrics['cer'] = cer.item()

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    val_metrics['wer'] = wer.item()

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    val_metrics['bleu'] = bleu.item()

    return val_metrics

def get_all_sentences(ds, lang):
    """生成指定语言的所有句子"""
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    获取或构建指定语言的分词器

    参数:
        config: 配置文件
        ds: 数据集
        lang: 语言代码
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # 如果分词器文件不存在，训练一个新的分词器
        # 代码参考Huggingface官方示例
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # 使用WordLevel模型
        tokenizer.pre_tokenizer = Whitespace()  # 使用空格进行预分词
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # 如果分词器文件已存在，直接加载
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """获取数据加载器和分词器"""

    # 获取本地数据路径
    dataset_dict = load_from_disk("datasets/iwslt2017/iwslt2017-en-it")
    ds_raw = dataset_dict['train'].select(range(3000))
    # 构建源语言和目标语言的分词器
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 将数据集按9:1分割为训练集和验证集
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # 创建双语数据集对象
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # 统计源语言和目标语言句子的最大长度
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'源句子的最大长度: {max_len_src}')
    print(f'目标句子的最大长度: {max_len_tgt}')

    # 创建数据加载器
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """根据配置构建Transformer模型"""
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
                              d_model=config['d_model'],h = config['heads'])
    return model

def train_model(config):
    """主训练函数"""
    # 定义训练设备（优先使用GPU/MPS加速）
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                          "cpu")

    print("使用的设备:", device)
    if (device == 'cuda'):
        print(f"设备名称: {torch.cuda.get_device_name(device.index)}")
        print(f"显存大小: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"设备名称: <mps>")
    else:
        print("注意: 如果您有GPU，建议使用GPU进行训练以加速。")
        print("      Windows系统使用NVIDIA GPU，请参考: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      Mac系统请运行: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # 创建模型权重保存目录
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # 获取数据加载器和分词器
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # 构建模型并移动到相应设备
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # 初始化TensorBoard写入器
    writer = SummaryWriter(config['experiment_name'])

    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 检查是否需要预加载模型权重
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
                                                                                                        preload) if preload else None
    if model_filename:  # 如果找到了模型文件
        print(f'预加载模型: {model_filename}')

        # 加载检查点文件
        state = torch.load(model_filename)

        # 恢复模型权重
        model.load_state_dict(state['model_state_dict'])

        # 设置初始训练轮次（从保存的epoch+1开始）
        initial_epoch = state['epoch'] + 1

        # 恢复优化器状态（包括动量、学习率等）
        optimizer.load_state_dict(state['optimizer_state_dict'])

        # 恢复全局训练步数
        global_step = state['global_step']
    else:
        print('未找到预训练模型，从头开始训练')

    # 定义损失函数（使用标签平滑和忽略填充符）
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # 开始训练循环
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()  # 清理GPU缓存（如果使用GPU）
        model.train()  # 设置模型为训练模式
        batch_iterator = tqdm(train_dataloader, desc=f"训练轮次 {epoch:02d}")
        epoch_train_losses = []
        for batch in batch_iterator:
            # 获取当前批次数据并移动到设备
            encoder_input = batch['encoder_input'].to(device)  # (批次大小, 序列长度)
            decoder_input = batch['decoder_input'].to(device)  # (批次大小, 序列长度)
            encoder_mask = batch['encoder_mask'].to(device)  # (批次大小, 1, 1, 序列长度)
            decoder_mask = batch['decoder_mask'].to(device)  # (批次大小, 1, 序列长度, 序列长度)

            # 前向传播
            encoder_output = model.encode(encoder_input, encoder_mask)  # (批次大小, 序列长度, 模型维度)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (批次大小, 序列长度, 模型维度)
            proj_output = model.project(decoder_output)  # (批次大小, 序列长度, 目标词汇表大小)

            # 获取真实标签
            label = batch['label'].to(device)  # (批次大小, 序列长度)

            # 计算损失
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"损失": f"{loss.item():6.3f}"})
            epoch_train_losses.append(loss.item())

            # 记录损失到TensorBoard
            writer.add_scalar('训练损失', loss.item(), global_step)
            writer.flush()

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

            global_step += 1  # 更新全局步数

        # 计算平均训练损失
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        # 计算验证损失和指标
        print("计算验证指标...")
        avg_val_loss, val_bleu = calculate_validation_metrics(
            model, val_dataloader, loss_fn,tokenizer_src, tokenizer_tgt, device, max_examples=50
        )
        # 每个训练轮次结束后运行验证
        val_metrics = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        #读取最新的模型文件
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # 保存模型检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'train_loss': avg_train_loss,  # 平均训练损失
            'val_loss': avg_val_loss,  # 验证损失
            'val_cer':val_metrics['cer'],
            'val_wer': val_metrics['wer'],
            'val_bleu': val_metrics['bleu'],  # BLEU分数
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 忽略警告信息
    config = get_config()
    config["model_folder"] = "weights2"
    train_model(config)  # 开始训练