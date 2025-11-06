from pathlib import Path

def get_config(heads=8,epochs= 60):
    return {
        "batch_size": 8,
        "num_epochs": epochs,
        "lr": 10**-4,
        'heads':heads,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'iwslt2017',
        "lang_src": "en",
        "lang_tgt": "it",
        'config_name': 'iwslt2017-en-it',
        "model_folder": f"weights_heads_{heads}",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": f"runs/tmodel_heads_{heads}"
    }

def get_head_configs():
    """返回不同头数的配置列表"""
    config_2 = get_config(heads=2,epochs=20)
    config_6 = get_config(heads=6,epochs=20)
    config_16 = get_config(heads=16,epochs=20)
    return config_2, config_6, config_16

# 获取指定epoch的检查点文件路径
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('') / model_folder / model_filename)

# 获取最新检查点文件路径
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])