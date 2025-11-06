from train import train_model
from config import get_head_configs

config_2, config_6, config_16 = get_head_configs()  # 获取配置

train_model(config_2)  # 开始训练
train_model(config_6)
train_model(config_16)


