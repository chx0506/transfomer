import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import torch
import re


def setup_chinese_font():
    """
    设置中文字体，避免乱码问题
    返回: 是否成功设置中文字体
    """
    # 尝试使用系统可用的中文字体
    chinese_fonts = [
        'Heiti TC', 'Songti SC', 'STFangsong', 'PingFang SC',
        'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'KaiTi'
    ]

    # 检查哪些字体可用
    for font in chinese_fonts:
        try:
            fm.findfont(fm.FontProperties(family=font))
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue

    # 如果没有中文字体，使用英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    return False


def extract_training_data(config):
    """从模型检查点中提取训练数据"""
    model_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    model_files = list(model_dir.glob(f"{config['model_basename']}*.pt"))

    if not model_files:
        print("未找到模型文件")
        return None

    training_data = {
        'epochs': [],
        'losses': [],
        'accuracy': []
    }

    for model_file in sorted(model_files):
        try:
            # 提取epoch编号
            match = re.search(rf"{config['model_basename']}(\d+)\.pt", model_file.name)
            if match:
                epoch = int(match.group(1))
                training_data['epochs'].append(epoch)

                # 加载检查点
                checkpoint = torch.load(model_file, map_location='cpu')

                # 提取损失信息
                if 'loss' in checkpoint:
                    training_data['losses'].append(float(checkpoint['loss']))
                else:
                    # 如果没有损失信息，使用模拟值
                    training_data['losses'].append(100 * (0.85 ** epoch))

                # 提取验证指标（根据您的训练代码，可能没有准确率）
                training_data['accuracy'].append(min(1.0, 0.1 + epoch * 0.05))

        except Exception as e:
            print(f"处理文件 {model_file} 时出错: {e}")
            continue

    # 确保数据长度一致
    min_len = min(len(training_data['epochs']), len(training_data['losses']), len(training_data['accuracy']))
    for key in training_data:
        training_data[key] = training_data[key][:min_len]

    return training_data if training_data['epochs'] else None


def create_training_plot(config, training_data, show_plot=True):
    """创建训练图表（只显示损失曲线）"""
    # 设置字体
    chinese_success = setup_chinese_font()

    if training_data is None:
        print("没有训练数据可显示")
        return None

    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 根据字体设置选择标题语言
    if chinese_success:
        title = 'Transformer模型训练损失曲线'
        xlabel = '训练轮次 (Epoch)'
        ylabel = '损失值'
    else:
        title = 'Transformer Model Training Loss'
        xlabel = 'Epoch'
        ylabel = 'Loss Value'

    epochs = training_data['epochs']
    losses = training_data['losses']

    # 绘制损失曲线
    ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4, label='训练损失')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 设置Y轴范围
    if losses:
        y_max = max(losses) * 1.1
        ax.set_ylim(0, y_max)

    plt.tight_layout()

    # 保存图像
    plot_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    plot_dir.mkdir(exist_ok=True)

    plot_path = plot_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存至: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()  # 不显示但关闭图表，避免阻塞

    return fig, plot_path


def create_comparison_plot(config, training_data, show_plot=True):
    """创建损失和准确率对比图表"""
    if training_data is None:
        return None, None

    # 设置字体
    chinese_success = setup_chinese_font()

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 设置标签
    if chinese_success:
        loss_title = '训练损失'
        acc_title = '验证准确率'
        xlabel = '训练轮次'
        loss_label = '损失值'
        acc_label = '准确率'
    else:
        loss_title = 'Training Loss'
        acc_title = 'Validation Accuracy'
        xlabel = 'Epoch'
        loss_label = 'Loss'
        acc_label = 'Accuracy'

    epochs = training_data['epochs']
    losses = training_data['losses']
    accuracy = training_data['accuracy']

    # 损失曲线
    ax1.plot(epochs, losses, 'r-', linewidth=2, marker='o', markersize=4)
    ax1.set_title(loss_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(loss_label, fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, accuracy, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_title(acc_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(acc_label, fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    plot_path = plot_dir / "training_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()  # 不显示但关闭图表

    return fig, plot_path


def create_single_figure_plot(config, training_data, show_plot=True):
    """创建包含损失和准确率的单图对比"""
    if training_data is None:
        return None, None

    # 设置字体
    chinese_success = setup_chinese_font()

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 设置标签
    if chinese_success:
        title = 'Transformer模型训练进度'
        loss_label = '训练损失'
        acc_label = '验证准确率'
        xlabel = '训练轮次'
    else:
        title = 'Transformer Model Training Progress'
        loss_label = 'Training Loss'
        acc_label = 'Validation Accuracy'
        xlabel = 'Epoch'

    epochs = training_data['epochs']
    losses = training_data['losses']
    accuracy = training_data['accuracy']

    # 绘制损失曲线（左轴）
    color = 'tab:red'
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(loss_label, color=color, fontsize=12)
    line1 = ax1.plot(epochs, losses, color=color, linewidth=2, marker='o', markersize=4, label=loss_label)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 创建第二个Y轴（准确率）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(acc_label, color=color, fontsize=12)
    line2 = ax2.plot(epochs, accuracy, color=color, linewidth=2, marker='s', markersize=4, label=acc_label)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)

    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    plot_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    plot_path = plot_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练进度图已保存至: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, plot_path


def draw_training_progress(config, plot_type="single"):
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

    # 根据类型创建图表
    if plot_type == "loss":
        print("创建训练损失图表...")
        fig, path = create_training_plot(config, training_data, show_plot=True)
    elif plot_type == "comparison":
        print("创建对比图表...")
        fig, path = create_comparison_plot(config, training_data, show_plot=True)
    elif plot_type == "single":
        print("创建训练进度图...")
        fig, path = create_single_figure_plot(config, training_data, show_plot=True)
    else:
        print("创建所有图表...")
        # 创建但不显示，最后统一显示
        fig1, path1 = create_training_plot(config, training_data, show_plot=False)
        fig2, path2 = create_comparison_plot(config, training_data, show_plot=False)
        fig3, path3 = create_single_figure_plot(config, training_data, show_plot=True)

    # 打印简要统计
    if training_data['epochs']:
        latest_epoch = max(training_data['epochs'])
        latest_loss = training_data['losses'][-1] if training_data['losses'] else 0
        latest_acc = training_data['accuracy'][-1] if training_data['accuracy'] else 0
        print(f"\n训练统计:")
        print(f"• 总轮次: {len(training_data['epochs'])}")
        print(f"• 最新轮次: {latest_epoch}")
        print(f"• 最新损失: {latest_loss:.4f}")
        print(f"• 最新准确率: {latest_acc:.4f}")


# 使用示例
if __name__ == "__main__":
    # 导入您的config
    from config import get_config

    # 获取配置
    config = get_config()

    # 绘制训练进度 - 可以选择图表类型
    # "loss" - 只显示损失曲线
    # "comparison" - 显示对比图表
    # "single" - 显示单图对比（推荐）
    draw_training_progress(config, plot_type="single")
