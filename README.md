# transformer项目
使用transformer来进行德语和英语的机器翻译

## 1.创建conda环境
```
conda create -n your_conda_name python=3.9
conda activate your_conda_name
```

## 2.安装依赖包
```
pip install -r requirements.txt
```

## 3.使用一键脚本
```
bash scripts/run.sh
```


## 项目代码结构树
```
transfomer/
├── src/                    # 源代码目录
│   ├── train.py           # 主训练脚本
│   ├── model.py           # Transformer模型定义
│   ├── download.py        # 从huggingface上下载数据集到本地
│   ├── Ablation.py        # 不同头数的模型训练
│   ├── no_position.py     # 无位置编码的模型
│   ├── draw.py            # 绘图代码
│   ├── draw2.py           # 不同头数的损失对比图像
│   ├── draw3.py           # 有无位置编码的对比图像
│   ├── dataset.py         # 数据加载和预处理
│   └── config.py          # 配置文件处理
├── scripts/               # 工具脚本目录
│   └── run.sh             # 一键运行脚本
├── results/               # 实验结果目录
│   ├── training_curves/   # 训练曲线图像
│   └── tables/            # 结果表格文件
├── requirements.txt       # Python依赖列表
└── README.md              # 项目说明文档
```
