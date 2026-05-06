# Text-to-SQL Fine-tuning with QLoRA

基于 Qwen2.5-Coder-7B-Instruct 的 Text-to-SQL 微调项目。使用 QLoRA 在单张 16GB 显卡上完成 7B 模型的高效微调，支持中英双语自然语言到 SQL 的转换。

## 特性

- **高效微调**: QLoRA (4-bit 量化 + LoRA)，单卡 16GB 显存即可训练
- **中英双语**: 同时支持中文和英文自然语言问题
- **多数据集**: Spider + BIRD + CSpider 联合训练
- **完整流程**: 数据处理 → 训练 → 评估 → 部署 Demo
- **量化评估**: Execution Accuracy + Exact Match，按难度分级统计

## 环境要求

- Python >= 3.10
- CUDA >= 12.1
- GPU 显存 >= 16GB (推荐 RTX 5070Ti / RTX 4090)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
bash data/download.sh
```

### 3. 数据预处理

```bash
python src/data_process.py --data_dir data --output_dir data
```

### 4. 开始训练

```bash
bash scripts/run_train.sh
```

### 5. 评估

```bash
bash scripts/run_eval.sh
```

### 6. 启动 Demo

```bash
python demo/app.py --model_path outputs/text2sql-qlora/final
```

## 项目结构

```
├── configs/
│   └── train_config.yaml        # 训练超参数配置
├── data/
│   └── download.sh              # 数据集下载脚本
├── src/
│   ├── data_process.py          # 数据预处理
│   ├── data_augment.py          # 数据增强
│   ├── train.py                 # QLoRA 训练
│   ├── evaluate.py              # 模型评估
│   ├── inference.py             # 推理脚本
│   ├── merge_lora.py            # LoRA 权重合并
│   └── utils.py                 # 工具函数
├── scripts/
│   ├── run_train.sh             # 训练启动脚本
│   ├── run_eval.sh              # 评估启动脚本
│   └── run_inference.sh         # 推理示例
├── demo/
│   └── app.py                   # Gradio Web Demo
└── requirements.txt
```
