from datasets import load_dataset, load_from_disk
import os

# 创建本地目录
local_dir = "./datasets/iwslt20f17"
os.makedirs(local_dir, exist_ok=True)

# 下载到指定目录
dataset = load_dataset(
    "iwslt2017",
    'iwslt2017-en-it',
)

dataset.save_to_disk(os.path.join(local_dir, "iwslt2017-en-it"))
dataset = load_from_disk(os.path.join(local_dir, "iwslt2017-en-it"))


def find_data_files(local_path):
    """查找数据文件"""
    for format in ['parquet', 'json', 'csv']:
        train_file = os.path.join(local_path, f'train.{format}')
        val_file = os.path.join(local_path, f'validation.{format}')

        if os.path.exists(train_file) and os.path.exists(val_file):
            return {'train': train_file, 'validation': val_file, 'format': format}

    return None
