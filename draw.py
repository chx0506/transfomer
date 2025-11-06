import matplotlib.pyplot as plt
from pathlib import Path
import torch
import re

def extract_training_data(config):
    """从模型检查点中提取训练数据"""
    model_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    model_files = list(model_dir.glob(f"{config['model_basename']}*.pt"))

    if not model_files:
        print("未找到模型文件")
        return None

    training_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        "val_bleu":[],
        'val_cer':[],
        'val_wer':[]
    }

    for model_file in sorted(model_files):
        try:
            # 提取epoch编号
            match = re.search(rf"{config['model_basename']}(\d+)\.pt", model_file.name)
            if match:
                epoch = int(match.group(1))
                training_data['epochs'].append(epoch)

                # 加载检查点

                checkpoint = torch.load(model_file, map_location='cpu')  # 使用CPU加载避免GPU内存问题

                # 提取训练损失
                if 'train_loss' in checkpoint:
                    training_data['train_loss'].append(float(checkpoint['train_loss']))
                else:
                    print(f"警告: {model_file} 中没有找到训练损失数据")
                    continue

                # 提取验证损失
                if 'val_loss' in checkpoint:
                    training_data['val_loss'].append(float(checkpoint['val_loss']))
                else:
                    print(f"警告: {model_file} 中没有找到val_loss数据")
                    training_data['val_loss'].append(None)

                # # 提取验证BLEU分数
                # if 'val_bleu' in checkpoint:
                #     training_data['val_bleu'].append(float(checkpoint['val_bleu']))
                # else:
                #     print(f"警告: {model_file} 中没有找到val_bleu数据")
                #     training_data['val_bleu'].append(None)

                # # 提取字符错误率(CER)
                # if 'cer' in checkpoint:
                #     training_data['val_cer'].append(float(checkpoint['cer']))
                # else:
                #     print(f"警告: {model_file} 中没有找到cer数据")
                #     training_data['val_cer'].append(None)
                #
                # # 提取词错误率(WER)
                # if 'wer' in checkpoint:
                #     training_data['val_wer'].append(float(checkpoint['wer']))
                # else:
                #     print(f"警告: {model_file} 中没有找到wer数据")
                #     training_data['val_wer'].append(None)

        except Exception as e:
            print(f"处理文件 {model_file} 时出错: {e}")
            continue

        # 过滤掉没有有效数据的epoch
    valid_indices = [i for i, loss in enumerate(training_data['train_loss']) if loss is not None]

    for key in training_data:
        training_data[key] = [training_data[key][i] for i in valid_indices if i < len(training_data[key])]

    # 确保所有数组长度一致
    min_len = min(len(training_data['epochs']),
                  len(training_data['train_loss']),
                  len(training_data['val_loss']))

    for key in training_data:
        training_data[key] = training_data[key][:min_len]

    return training_data if training_data['epochs'] else None


def create_comparison_plot(config, training_data, show_plot=True):
    """创建损失和准确率对比图表"""
    if training_data is None:
        return None, None

    # 创建图表
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    loss_title = 'Training and Validation Loss'
    bleu_title = 'Validation BLEU Score'
    error_title = 'Validation Error Rates'
    xlabel = 'Epoch'
    train_loss_label = 'Training Loss'
    val_loss_label = 'Validation Loss'
    bleu_label = 'BLEU Score'
    cer_label = 'Character Error Rate (CER)'
    wer_label = 'Word Error Rate (WER)'

    # 创建包含三个子图的图表
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 15))

    epochs = training_data['epochs']
    train_losses = training_data['train_loss']
    val_losses = training_data['val_loss']
    val_bleu = training_data['val_bleu']
    # val_cer = training_data['val_cer']
    # val_wer = training_data['val_wer']

    # 第一个子图：训练和验证损失
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label=train_loss_label)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, marker='s', markersize=4, label=val_loss_label)
    ax1.set_title(loss_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # # 第二个子图：BLEU分数
    # ax2.plot(epochs, val_bleu, 'g-', linewidth=2, marker='^', markersize=4, label=bleu_label)
    # ax2.set_title(bleu_title, fontsize=14, fontweight='bold')
    # ax2.set_xlabel(xlabel, fontsize=12)
    # ax2.set_ylabel('BLEU', fontsize=12)
    # ax2.set_ylim(0, 1.0)
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    # # 第三个子图：错误率
    # if any(val_cer) and any(val_wer):
    #     ax3.plot(epochs, val_cer, 'm-', linewidth=2, marker='d', markersize=4, label=cer_label)
    #     ax3.plot(epochs, val_wer, 'c-', linewidth=2, marker='v', markersize=4, label=wer_label)
    #     ax3.set_title(error_title, fontsize=14, fontweight='bold')
    #     ax3.set_xlabel(xlabel, fontsize=12)
    #     ax3.set_ylabel('error', fontsize=12)
    #     ax3.set_ylim(0, 1.0)
    #     ax3.legend()
    #     ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    plot_path = plot_dir / "training_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, plot_path


def draw_training_progress(config):
    """
    主绘图函数 - 根据config绘制训练进度

    参数:
        config: 训练配置字典
        plot_type: 图表类型 ("loss", "comparison", "single")
    """
    print("正在提取训练数据...")

    # 提取训练数据
    training_data = extract_training_data(config)

    if training_data is None:
        print("没有找到训练数据，请先进行训练")
        return

    print(f"找到 {len(training_data['epochs'])} 个训练轮次的数据")

    print("创建对比图表...")
    fig, path = create_comparison_plot(config, training_data, show_plot=True)
    # 打印简要统计
    if training_data['epochs']:
        latest_epoch = max(training_data['epochs'])
        latest_loss = training_data['train_loss'][-1] if training_data['train_loss'] else 0
        latest_acc = training_data['val_loss'][-1] if training_data['val_loss'] else 0
        print(f"\n训练统计:")
        print(f"• 总轮次: {len(training_data['epochs'])}")
        print(f"• 最新轮次: {latest_epoch}")
        print(f"• 最新训练集损失: {latest_loss:.4f}")
        print(f"• 最新验证集损失: {latest_acc:.4f}")


# 使用示例
if __name__ == "__main__":
    # 导入您的config
    from config import get_config

    # 获取配置
    config = get_config()
    config["model_folder"] = "weights2"

    draw_training_progress(config)