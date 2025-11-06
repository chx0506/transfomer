import matplotlib.pyplot as plt
from pathlib import Path
import torch
import re
from config import get_head_configs,get_config

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


def create_comparison_plot( config2,config6,config8,config16,training_data2,training_data6,training_data8,training_data16,show_plot=True):
    """创建损失和准确率对比图表"""

    train_loss_title = 'Training Loss of Different Heads'
    val_loss_title = 'Validation Loss of Different Heads'
    xlabel = 'Epoch'

    print(f"找到 {len(training_data2['epochs'])} 个训练轮次的数据")

    loss_label2 = 'head2'
    loss_label6 = 'head6'
    loss_label8 = 'head8'
    loss_label16 = 'head16'


    # 创建包含三个子图的图表
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 15))

    epochs = training_data2['epochs']

    # 第一个子图：训练损失
    ax1.plot(epochs, training_data2['train_loss'], 'g-', linewidth=2, marker='^', markersize=4, label=loss_label2)
    ax1.plot(epochs, training_data6['train_loss'], 'r-', linewidth=2, marker='s', markersize=4, label=loss_label6)
    ax1.plot(epochs, training_data8['train_loss'][:20], 'b-', linewidth=2, marker='o', markersize=4, label=loss_label8)
    ax1.plot(epochs, training_data16['train_loss'], 'y-', linewidth=2, marker='.', markersize=4, label=loss_label16)
    ax1.set_title(train_loss_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('train_loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 第二个子图：验证损失
    ax2.plot(epochs, training_data2['val_loss'], 'g-', linewidth=2, marker='^', markersize=4, label=loss_label2)
    ax2.plot(epochs, training_data6['val_loss'], 'r-', linewidth=2, marker='s', markersize=4, label=loss_label6)
    ax2.plot(epochs, training_data8['val_loss'][:20], 'b-', linewidth=2, marker='o', markersize=4, label=loss_label8)
    ax2.plot(epochs, training_data16['val_loss'], 'y-', linewidth=2, marker='.', markersize=4, label=loss_label16)
    ax2.set_title(val_loss_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('val_loss', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    print("创建对比图表...")
    # 保存图像
    plot_dir = Path(f"{config2['datasource']}_{config2['model_folder']}")
    plot_path = plot_dir / "training_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, plot_path


# 使用示例
if __name__ == "__main__":

    config_2, config_6, config_16 = get_head_configs()  # 获取配置
    config_8 = get_config(epochs=20)
    config_8["model_folder"] = "weights2"
    training_data2 = extract_training_data(config_2)
    training_data6 = extract_training_data(config_6)
    training_data8 = extract_training_data(config_8)
    training_data16 = extract_training_data(config_16)

    create_comparison_plot(config_2, config_6,config_8,config_16,training_data2,training_data6,training_data8,training_data16)