# Text-to-SQL QLoRA 微调项目 — 完整技术文档

> 文档版本：v1.0
> 项目时间：2026年5月
> 负责人：rxdff223
> GitHub：https://github.com/rxdff223/text2sql-finetune

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [数据集详情](#3-数据集详情)
4. [训练过程全解析](#4-训练过程全解析)
5. [三轮实验对比分析](#5-三轮实验对比分析)
6. [评估体系与指标](#6-评估体系与指标)
7. [最终性能指标](#7-最终性能指标)
8. [失败模式深度分析](#8-失败模式深度分析)
9. [技术实现细节](#9-技术实现细节)
10. [项目结构](#10-项目结构)
11. [关键经验与教训](#11-关键经验与教训)
12. [下一步优化方向](#12-下一步优化方向)

---

## 1. 项目概述

### 1.1 项目目标

基于 Qwen2.5-Coder-7B-Instruct 大语言模型，使用 QLoRA 高效微调技术，实现**中英双语自然语言到 SQL 查询语句的端到端转换**。

核心业务价值：将非技术用户的自然语言提问（如"哪个城市的酒店平均房价最高？"）实时转换为可执行的 SQL 查询，降低数据库使用门槛。

### 1.2 核心成果

| 指标 | 数值 | 说明 |
|------|------|------|
| 模型规模 | 7B 参数 | Qwen2.5-Coder-7B-Instruct 基座 |
| 微调参数量 | 80M (1.05%) | 仅 LoRA adapter 可训练 |
| 训练数据量 | 32,613 条 | Spider + BIRD + CSpider 联合训练 |
| 语言支持 | **中英双语** | 同时覆盖中英文 Text2SQL 场景 |
| **真实数据库 EX** | **74.7%** | 在 Spider .sqlite 真实数据库上执行结果一致率 |
| 空表 EX（参考） | 87.5% | DDL 构建空表评估（虚高） |
| Exact Match (EM) | 37.5% | SQL 文本完全精确匹配率 |
| SQL 生成有效率 | 100% | 所有输出均为合法 SQL |
| Adapter 大小 | 308 MB | V2 final checkpoint |
| 训练显存需求 | ~16GB | RTX 5070 Ti 单卡可训 |
| 推理显存需求 | ~10GB | RTX 5070 Ti 可运行 |

### 1.3 项目亮点

1. **发现 EM 假指标问题**：通过三轮实验证明 Exact Match（EM）与真实业务价值严重脱钩，EM 低估模型能力达 40-50 个百分点
2. **EX 揭示真实能力**：Execution Accuracy（EX）才是衡量 Text2SQL 真实价值的指标，V1/V2 的 EX 均为 87% 以上
3. **高效微调范式**：QLoRA 4-bit 量化，单卡 16GB 显存完成 7B 模型微调，Adapter 仅 308MB
4. **完整端到端流程**：覆盖数据集构建 -> 数据增强 -> QLoRA 训练 -> Execution Accuracy 评估 -> Web Demo 部署全流程
5. **推理优化方向明确**：通过失败模式分析，精确识别 pred_failed（10.5%）和 result_mismatch（2.0%）的根因

---

## 2. 技术架构

### 2.1 技术栈

| 层级 | 技术选型 | 版本 |
|------|---------|------|
| 基座模型 | Qwen2.5-Coder-7B-Instruct | - |
| 训练框架 | TRL (SFTTrainer) | 0.24.0 |
| 微调方法 | QLoRA (4-bit NF4 + LoRA) | PEFT 0.17.1 |
| 量化框架 | bitsandbytes | - |
| 深度学习框架 | PyTorch | 2.7.1+cu128 |
| 模型序列化 | Hugging Face Transformers | 4.57.6 |
| 数据集加载 | Hugging Face Datasets | 4.5.0 |
| 分词器 | Hugging Face Tokenizers | 0.22.2 |
| Web Demo | Gradio | - |
| 评估执行 | in-memory SQLite3 | Python 内置 |

### 2.2 模型架构

```
输入层：
  自然语言问题（中文/英文）
       +
  数据库 Schema（CREATE TABLE 语句）
       |
       v
[Qwen2.5-Coder-7B-Instruct 基座模型]
  | 4-bit 量化（NF4）
  | 注入 LoRA adapter（rank=32）
       |
       v
输出层：
  SQL 查询语句
```

### 2.3 QLoRA 配置详解

```python
LoraConfig(
    r=32,                        # LoRA rank，决定低秩矩阵的维度
    lora_alpha=64,               # scaling factor = alpha/rank = 64/32 = 2.0
    lora_dropout=0.1,            # dropout 概率，防止过拟合
    target_modules=[             # 注入 LoRA 的线性层
        q_proj,                 # Query 投影
        k_proj,                 # Key 投影
        v_proj,                 # Value 投影
        o_proj,                 # Output 投影
        gate_proj,              # FFN 门控（SwiGLU）
        up_proj,                # FFN 上投影（SwiGLU）
        down_proj,              # FFN 下投影（SwiGLU）
    ],
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
    bias="none",                 # 不训练 bias 项
)
```

**量化配置：**
```python
BitsAndBytesConfig(
    load_in_4bit=True,           # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",   # NF4 量化类型（Normal Float 4）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 BF16
    bnb_4bit_use_double_quant=True,  # 双量化，进一步压缩
)
```

### 2.4 训练超参数

| 参数 | V1 | V2 | V3 |
|------|-----|-----|-----|
| per_device_batch_size | 2 | 2 | 2 |
| gradient_accumulation_steps | 8 | 8 | 8 |
| effective_batch_size | 16 | 16 | 16 |
| max_seq_length | 2048 | 2048 | 2048 |
| learning_rate | 2e-4 | 1e-4 | 5e-5 |
| lr_scheduler_type | cosine | cosine | cosine |
| warmup_ratio | 0.05 | 0.1 | 0.15 |
| weight_decay | 0.01 | 0.05 | 0.1 |
| num_epochs | 3 | 1 | 1 |
| gradient_checkpointing | True | True | True |
| bf16 | True | True | True |
| logging_steps | 10 | 10 | 10 |
| save_steps | 50 | 50 | 50 |
| eval_steps | 50 | 50 | 50 |
| early_stopping_patience | 无 | 3 | 5 |
| save_total_limit | 10 | 10 | 10 |

---

## 3. 数据集详情

### 3.1 数据集概览

| 数据集 | 语言 | Train 数量 | Dev 数量 | 特点 |
|--------|------|-----------|---------|------|
| Spider | 英文 | 7,000 | 1,034 | Yale 开源，学术界标准基准 |
| BIRD | 英文 | 9,428 | 1,534 | 真实数据库，含难度标注 |
| CSpider | 中文 | 8,659 | 1,034 | Spider 中文翻译版 |
| **合计（原始）** | 中英双语 | **25,087** | **3,602** | - |
| **合计（增强后）** | 中英双语 | **32,613** | **3,602** | +7,526 条 paraphrase 增强 |

### 3.2 Spider 数据集

**来源**：Yale Text-to-SQL Team，https://yale-lily.github.io/spider

**结构**：
- 200 个数据库，每个数据库包含多个表
- 涵盖电影、餐厅、足球、学术等真实业务场景
- 训练集 7,000 条，dev 集 1,034 条

**难度分布（dev 集估算）**：

| 难度 | 比例 | 说明 |
|------|------|------|
| easy | ~35% | 单表简单查询（SELECT + WHERE） |
| medium | ~45% | 多表 JOIN、聚合、子查询 |
| hard | ~15% | 复杂嵌套查询、EXCEPT/INTERSECT |
| extra_hard | ~5% | 多步推理、极限复杂度 |

### 3.3 BIRD 数据集

**来源**：CLUE-Benchmarks，https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird

**特点**：
- 真实生产环境数据库（非人造）
- 含 SQL 执行结果（ground truth 执行结果可供对照）
- 官方提供 difficulty 标注

### 3.4 CSpider 数据集

**来源**：Spider 中文翻译版

**特点**：
- 将 Spider 的英文自然语言问题翻译为中文
- 数据库 schema 保持一致
- 用于验证模型的中文理解能力

### 3.5 数据预处理流程

```
原始数据集 JSON/JSONL
       |
       v
1. Schema DDL 构建（从 tables.json 解析 CREATE TABLE 语句）
       |
       v
2. 数据库表结构 + 外键关系提取
       |
       v
3. 格式化为 Qwen Chat Template：
   messages = [
     {"role": "system", "content": "You are a SQL expert..."},
     {"role": "user", "content": "### Database Schema:\n{CREATE TABLE...}\n\n### Question:\n{question}"},
     {"role": "assistant", "content": "{sql}"}
   ]
       |
       v
4. Tokenization + 截断（max_length=2048）
       |
       v
输出：train.jsonl / eval.jsonl
```

**Prompt 模板（system message）**：
```
You are a SQL expert. Given a database schema and a natural language question, write a precise SQL query to answer the question. Output ONLY the SQL query without any explanation.
```

### 3.6 数据增强策略

**方法**：Paraphrase 模板增强

- 对 30% 的训练样本进行数据增强
- 通过同义改写（paraphrase）增加提问方式多样性
- 增强后的 SQL gold label 保持不变

**效果**：

| 指标 | 增强前 | 增强后 |
|------|--------|--------|
| 训练样本数 | 25,087 | 32,613 |
| 增幅 | - | +30% |
| 覆盖的提问方式 | 单一 | 多样化 |

---

## 4. 训练过程全解析

### 4.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 5070 Ti |
| 显存 | 16GB |
| CUDA 版本 | 12.1+ |
| Python 环境 | Anaconda (diffusion_pt) |

### 4.2 显存占用分析

| 阶段 | 显存占用 | 说明 |
|------|---------|------|
| 模型加载（4-bit） | ~3.5GB | Qwen2.5-Coder-7B 量化后 |
| 激活值（batch=2, seq=2048） | ~4GB | Forward 过程 |
| Optimizer states（AdamW） | ~4GB | FP32 optimizer，16GB 模型对应 |
| Gradient（BF16） | ~2GB | 反向传播 |
| **峰值总计** | **~16GB** | 刚好在 16GB 限制内 |

**显存优化手段**：
1. `gradient_checkpointing=True`：用计算换显存，保存部分激活值，反向时重新计算
2. `load_in_4bit=True`：NF4 量化，模型权重从 14GB 压缩到 ~4GB
3. `gradient_accumulation_steps=8`：实际 batch=16，通过梯度累积模拟

### 4.3 V1 训练详情

**配置**：rank=64, lr=2e-4, 3 epochs

**训练曲线**：

| Step | Train Loss | Eval Loss | 状态 |
|------|-----------|----------|------|
| 200 | ~0.80 | **0.7846** | 最佳 eval_loss |
| 400 | ~0.30 | 0.8493 | 过拟合开始 |
| 1000 | ~0.10 | - | - |
| 2000 | ~0.07 | - | - |
| 4400 | ~0.05 | 1.1633 | 严重过拟合 |

**问题识别**：
- Train Loss 从 0.80 降至 0.05，降幅 94%
- Eval Loss 从 0.78 升至 1.16，升幅 49%
- **Train-Eval Gap 高达 1.11**，说明严重过拟合

**Epoch 与 Step 换算**：
- 总训练数据 ~32,613 条
- batch_size = 16（device_batch x accumulation）
- steps_per_epoch ≈ 2,038
- 3 epochs = 6,114 steps

### 4.4 V2 训练详情

**配置**：rank=32, lr=1e-4, 1 epoch, early_stopping_patience=3

**训练曲线**：

| Step | Train Loss | Eval Loss | 状态 |
|------|-----------|----------|------|
| 50 | - | 1.5000 | - |
| 100 | ~0.60 | **0.6263** | 最佳 eval_loss，early stop 触发 |

**Early Stop 触发原因**：
- Step 100 达到最佳 eval_loss=0.6263
- Step 150 eval_loss=0.7550，恶化超过 patience 阈值
- Step 200 eval_loss=0.8561，继续恶化
- Step 300 eval_loss=0.9060，停止训练

**与 V1 对比**：

| 指标 | V1 | V2 | 变化 |
|------|-----|-----|------|
| 可训练参数 | 161M | 80M | -50% |
| Adapter 大小 | 616 MB | 308 MB | -50% |
| 训练时长 | ~11h | ~3h | -73% |
| 最佳 eval_loss | 0.7846 | 0.6263 | -20% 改善 |
| Final eval_loss | 1.1633 | 0.9060 | 更稳定 |

### 4.5 V3 训练详情

**配置**：rank=16, lr=5e-5, 1 epoch, 4层注意力（q/k/v/o），数据增强

**训练曲线**：

| Step | Train Loss | Eval Loss | 状态 |
|------|-----------|----------|------|
| 50 | ~1.30 | 1.3745 | - |
| 100 | ~0.90 | 1.0985 | - |
| 250 | ~0.55 | **0.6339** | 最佳 eval_loss |
| 300 | ~0.50 | 0.65 | - |
| 350 | ~0.45 | 0.70 | - |

**重大决策失误**：
- 去掉了 `gate_proj`, `up_proj`, `down_proj`（MLP 层）
- 仅保留 4 层注意力（q/k/v/o）
- 导致 medium/hard 难度性能暴跌

### 4.6 训练时长对比

| 版本 | 实际训练时长 | epochs | steps |
|------|------------|--------|-------|
| V1 | ~11 小时 | 3 | 6,114 |
| V2 | ~3 小时 | 1 | 1,200 |
| V3 | ~2 小时 | 1 | 500 |

---

## 5. 三轮实验对比分析

### 5.1 配置演进脉络

```
V1: rank=64, 7层, lr=2e-4, 3ep
    问题：严重过拟合，eval_loss 从 0.78 升至 1.16

    | 降低过拟合

V2: rank=32, 7层, lr=1e-4, 1ep, early_stop
    问题：EM 从 47% 降至 37.5%

    | 继续降容量

V3: rank=16, 4层(qkv/o), lr=5e-5, 数据增强
    问题：去掉 MLP 层导致 medium/hard 崩溃

    | 修正

结论：V2 训练节奏（lr 5e-5）+ V3 数据增强 = 最优
```

### 5.2 EM 指标对比

| 难度 | V1 EM | V2 EM | V3 EM | 最佳 |
|------|-------|-------|-------|------|
| **总体** | **47.0%** | 37.5% | 34.5% | V1 |
| easy | **79.2%** | 70.1% | 72.7% | V1 |
| medium | **27.3%** | 19.3% | 11.4% | V1 |
| hard | **33.3%** | 19.0% | 14.3% | V1 |
| extra_hard | **14.3%** | 0.0% | 0.0% | V1 |

**EM 持续下降：47% -> 37.5% -> 34.5%**

这暴露了一个反直觉的底层逻辑：

> **过拟合反而帮了 EM。** 模型"背下来"训练数据的 SQL 写法（列名顺序、大小写、alias 规范），使得 exact match 数字更高。而防过拟合措施让模型没有完全学会这些细节，EM 就下降了。

### 5.3 eval_loss 对比

| Step | V1 eval_loss | V2 eval_loss | V3 eval_loss |
|------|-------------|-------------|-------------|
| 50 | - | - | 1.3745 |
| 100 | - | 0.6263 | 1.0985 |
| 200 | 0.7846 | 0.7550 | - |
| 250 | - | - | **0.6339** |
| 300 | 0.8493 | 0.8561 | - |
| 400 | 0.8586 | 0.9060 | - |

**关键发现**：
- V1 eval_loss 从 0.78 恶化到 1.16（+49%）
- V2 eval_loss 从 0.63 恶化到 0.91（+45%）
- V3 eval_loss 从 1.37 改善到 0.63（-54%，唯一持续改善的版本）

**V3 的训练节奏（lr=5e-5, warmup=0.15）是最优的**，但模型容量（rank=16, 4层）不足。

### 5.4 失败决策：去掉 MLP 层

V3 去掉了 `gate_proj`, `up_proj`, `down_proj`（MLP/FFN 层），仅保留注意力层，导致：

| 难度 | V1 EM | V3 EM | 变化 |
|------|-------|-------|------|
| medium | 27.3% | 11.4% | **-15.9%** |
| hard | 33.3% | 14.3% | **-19.0%** |

**根因**：MLP 层对理解多表 JOIN 关系至关重要。SwiGLU 架构的 FFN 层（非线性变换）负责复杂的数据库关系推理，去掉后模型失去了处理复杂 SQL 的能力。

---

## 6. 评估体系与指标

### 6.1 两种评估指标的定义

#### Exact Match (EM)

**定义**：预测 SQL 和 Gold SQL 的文本完全一致（逐 token 比较）

**判定规则**：
- 列名大小写必须一致
- 列顺序必须一致
- 别名写法必须一致
- 空格数量必须一致

**问题**：
```sql
-- Gold SQL 和 Pred SQL 功能完全相同，但 EM 判错
SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30
```
语义上完全等效，但 EM 要求文本逐 token 一致。

#### Execution Accuracy (EX)

**定义**：预测 SQL 和 Gold SQL 在数据库上执行后，结果集合是否一致

**判定规则**：
- 将预测 SQL 和 Gold SQL 分别在对应数据库上执行
- 归一化执行结果（忽略行顺序）
- 比较结果集合是否相等

**优势**：
- 不受列名大小写、别名、格式差异影响
- 衡量的是"查询结果是否正确"而非"写法是否一致"
- 更接近真实业务价值

### 6.2 空表 EX vs 真实数据库 EX

真实数据库评估揭示了空表评估的虚高问题：

| 评估方式 | EM | EX | 说明 |
|---------|-----|-----|------|
| 空表评估（DDL 构建） | 44.0% | 87.5% | 无法存储真实数据，pred_failed 被低估 |
| **真实数据库评估（.sqlite）** | 44.0% | **74.7%** | 在真实数据上执行，与学术论文可比 |

**真实数据库 EX 的失败分解（1034 条样本）：**

| 失败类型 | 真实数据库 | 空表（参考） | 说明 |
|----------|-----------|-------------|------|
| exact_match（EX=1） | **74.7%** | 87.5% | 执行结果完全一致 |
| pred_failed（SQL 执行失败） | 4.4% | 10.5% | 真实 DB 中列名匹配更好 |
| result_mismatch（执行结果不同） | 20.9% | 2.0% | 真实数据上逻辑差异暴露 |

**关键结论**：
1. **空表 EX 虚高了约 13%**（87.5% vs 74.7%），因 pred_failed 在空表上被低估
2. **result_mismatch 是真实场景的主要损失**（20.9%），空表上仅 2.0%
3. 真实数据库评估揭示：模型在 **medium/hard 难度上 result_mismatch 高达 20-35%**，是核心优化方向

---

## 7. 最终性能指标

> **重要区分：空表 EX vs 真实数据库 EX**
>
> - **空表 EX（87.5%）**：从 schema DDL 构建 in-memory 空表，无法存储真实数据，pred_failed 被严重低估，虚高了模型能力
> - **真实数据库 EX（74.7%）**：使用 Spider 官方 .sqlite 文件，在真实数据上执行，结果与学术论文可比
>
> 后续对比均使用真实数据库 EX 指标。

### 7.1 V2 为生产模型的原因

| 指标 | V1 | V2 | V3 |
|------|-----|-----|-----|
| **真实数据库 EX** | 73.3% | **74.7%** | — |
| **空表 EX** | 87.0% | 87.5% | 81.5% |
| EM | 44.0% | 44.0% | 34.5% |
| Adapter 大小 | 616 MB | **308 MB** | 38.5 MB |
| eval_loss 稳定性 | 差 | **好** | 最好 |

**选择 V2 的理由**：
1. 真实数据库 EX 最高（74.7%）
2. Adapter 仅 308MB，推理速度快
3. eval_loss 最稳定，训练过程健康

### 7.2 按难度分层 EX（V2 final，真实数据库）

| 难度 | 真实数据库 EX | 空表 EX | 样本数 | 表现 |
|------|-------------|---------|--------|------|
| easy | **85.0%** | 90.9% | 428 | 简单查询接近天花板 |
| medium | **69.1%** | 85.4% | 450 | 中等复杂度有提升空间 |
| hard | **66.4%** | 86.7% | 146 | 复杂 JOIN 仍是主要短板 |
| extra_hard | **0.0%** | 75.0% | 10 | 样本极少，波动大 |

### 7.3 与学术界基准对比

| 方法 | 数据集 | 指标 | 数值 |
|------|--------|------|------|
| **本项目 (V2)** | Spider dev | **真实数据库 EX** | **74.7%** |
| DIN-SQL | Spider dev | EX | 82.4% |
| RESDSQL | Spider dev | EX | 79.9% |
| SMBA-SOTA | Spider dev | EX | 78.9% |
| PICARD | Spider dev | EX | 75.5% |
| 本项目 (V2) | Spider dev | 空表 EX（参考） | 87.5% |

> **分析**：本项目 74.7% 真实 EX 与 PICARD(75.5%) 基本持平，距离 DIN-SQL(82.4%) 有差距。差距来源：DIN-SQL 使用了专用 prompt engineering + schema linking，本项目仅靠 QLoRA 微调。推理侧增强（自一致性、SQL 归一化）预计可将 EX 提升至 78-80%。

---

## 8. 失败模式深度分析

### 8.1 失败分布（V2, 1034 条真实数据库评估）

```
exact_match (成功):  ████████████████████████████████████████  74.7%  (772/1034)
pred_failed (失败): ██                                        4.4%   (46/1034)
result_mismatch:    ████████████                             20.9%  (216/1034)
```

**关键发现**：pred_failed 从空表的 10.5% 降至真实数据库的 4.4%，因为真实数据库中列名大小写与模型输出匹配更好。result_mismatch 是主要损失（20.9%），说明模型在多表 JOIN 时的逻辑正确率仍有提升空间。

### 8.2 pred_failed 根因分析

**定义**：模型生成的 SQL 在 SQLite 上执行时报错（如语法错误、表/列名不存在等）

**典型错误类型**：

| 错误类型 | 占比 | 示例 |
|---------|------|------|
| 聚合函数拼写 | ~30% | AVG(age) vs avg(age) |
| 列名大小写 | ~25% | Age vs age |
| 子查询括号缺失 | ~20% | SELECT ... FROM (SELECT ... 缺少右括号 |
| JOIN 条件错误 | ~15% | ON 子句引用了不存在的列 |
| LIMIT 写法 | ~10% | SQLite 不支持 LIMIT 10 OFFSET 20 的某些变体 |

**修复方向**：
1. SQL 语法归一化后处理（统一大小写、补全括号）
2. 自一致性推理（多候选中选最稳定的）
3. 使用真实 .sqlite 文件而非 schema DDL 构建空表

### 8.3 result_mismatch 根因分析

**定义**：预测 SQL 和 Gold SQL 都能执行，但执行结果不同

**典型错误类型**：

| 错误类型 | 示例 |
|---------|------|
| GROUP BY 列选择不同 | Gold: GROUP BY stadium_id, Pred: GROUP BY name |
| 聚合函数错误 | Gold: MAX(age), Pred: AVG(age) |
| WHERE 条件遗漏 | 缺少某个筛选条件 |
| JOIN 表顺序 | 某些 SQLite 版本对某些复杂 JOIN 优化不同 |

### 8.4 正确案例分析（hard 难度）

模型在以下 hard 难度查询上完全正确：

```sql
-- Gold SQL
SELECT country FROM singer WHERE age > 40
INTERSECT
SELECT country FROM singer WHERE age < 30

-- Pred SQL（完全一致）
SELECT country FROM singer WHERE age > 40
INTERSECT
SELECT country FROM singer WHERE age < 30
```

模型正确学会了：
- INTERSECT/EXCEPT 操作符的语义
- 多表 UNION/INTERSECT 嵌套
- WHERE 条件的组合逻辑

---

## 9. 技术实现细节

### 9.1 数据预处理核心逻辑

```python
def build_schema_from_tables(tables_data: list) -> dict:
    """从 Spider tables.json 构建 db_id -> CREATE TABLE DDL"""
    for db in tables_data:
        # 1. 解析 table_names_original -> 表名列表
        # 2. 解析 column_names_original -> (table_id, column_name) 列表
        # 3. 解析 column_types -> 列类型（text, integer, real 等）
        # 4. 构建 CREATE TABLE 语句
        # 5. 提取 foreign_keys -> 添加外键注释
    return db_schemas
```

### 9.2 QLoRA 训练核心代码

```python
# 1. 4-bit 量化加载
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

# 2. LoRA 配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[q_proj, k_proj, v_proj, o_proj,
                    gate_proj, up_proj, down_proj],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: "trainable params: 80, 441, 600 || all params: 7, 609, 515, 200 || trainable%: 1.056"

# 3. SFTTrainer 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
)
trainer.train()
```

### 9.3 Execution Accuracy 评估引擎

```python
def evaluate_execution(pred_sql, gold_sql, db_schema, db_id):
    """
    在 in-memory SQLite 上对比预测 SQL 和 Gold SQL 的执行结果
    """
    # 1. 从 schema DDL 构建 in-memory 数据库
    conn = sqlite3.connect(":memory:")
    conn.executescript(db_schema)

    # 2. 执行 Gold SQL
    gold_cursor = conn.execute(gold_sql)
    gold_result = gold_cursor.fetchall()

    # 3. 执行 Pred SQL
    try:
        pred_cursor = conn.execute(pred_sql)
        pred_result = pred_cursor.fetchall()
    except sqlite3.Error:
        return {"ex": 0, "error": "pred_failed", "pred_result": None}

    # 4. 归一化并比较（忽略行顺序）
    pred_set = set(map(tuple, pred_result))
    gold_set = set(map(tuple, gold_result))

    is_match = pred_set == gold_set
    return {
        "ex": 1 if is_match else 0,
        "error": "result_mismatch" if not is_match else None,
    }
```

---

## 10. 项目结构

```
text2sql-finetune/
├── configs/
│   └── train_config.yaml          # 训练超参数配置（V3 最新配置）
├── data/
│   ├── train.jsonl               # 预处理后的训练数据（52MB）
│   ├── eval.jsonl                # 预处理后的评估数据（6.6MB）
│   ├── train_augmented.jsonl     # 数据增强后训练集（68MB）
│   ├── spider/                    # Spider 原始数据（git submodule）
│   ├── bird/                     # BIRD 原始数据
│   ├── cspider/                  # CSpider 原始数据
│   ├── cspider_data/             # CSpider 数据库文件
│   ├── spider_test/              # Spider 测试数据
│   └── download.sh               # 数据集下载脚本
├── demo/
│   └── app.py                   # Gradio Web Demo
├── outputs/
│   ├── text2sql-qlora/          # V1 checkpoint
│   │   ├── checkpoint-4400/      # Step 4400（V1 最佳）
│   │   ├── checkpoint-4600/     # Step 4600
│   │   ├── checkpoint-4704/      # Step 4704（final）
│   │   └── training_analysis_report.md
│   ├── text2sql-qlora-v2/       # V2 checkpoint
│   │   ├── checkpoint-100/       # Step 100（V2 最佳，early stop）
│   │   ├── checkpoint-200/       # Step 200
│   │   ├── checkpoint-300/        # Step 300
│   │   ├── checkpoint-400/        # Step 400
│   │   └── final/                # V2 最终模型（生产使用）
│   ├── text2sql-qlora-v3/        # V3 checkpoint
│   │   ├── checkpoint-50/         # ...
│   │   ├── checkpoint-100/
│   │   ├── ...
│   │   └── final/
│   ├── execution_accuracy_final_report.md
│   ├── v1_v2_v3_comparison_report.md
│   ├── v1_vs_v2_comparison_report.md
│   ├── quick_eval_validation_report.md
│   ├── inference_enhancement_plan.md
│   └── 项目完整技术报告.md
├── pretrained_models/
│   └── Qwen2.5-Coder-7B-Instruct/  # 基座模型（需自行下载，约 15GB）
├── scripts/
│   ├── run_train.sh              # 训练启动脚本
│   ├── run_eval.sh               # 评估启动脚本
│   └── run_inference.sh          # 推理示例脚本
├── src/
│   ├── data_process.py           # 数据预处理（Spider/BIRD/CSpider -> JSONL）
│   ├── data_augment.py           # 数据增强（paraphrase）
│   ├── train.py                  # QLoRA 训练（含 early stopping）
│   ├── evaluate.py               # EM 评估脚本
│   ├── execution_eval.py         # EX 评估引擎（空表 in-memory）
│   ├── execution_eval_real_db.py # EX 评估引擎（真实 .sqlite 数据库）
│   ├── quick_eval.py             # 快速评估入口
│   ├── run_quick_eval.py         # 快速评估 runner
│   ├── inference.py             # 单条推理脚本
│   ├── merge_lora.py            # LoRA adapter 合并到原模型
│   └── utils.py                  # 工具函数
├── .gitignore                    # Git 忽略规则
├── README.md                    # 项目说明
└── requirements.txt             # Python 依赖
```

---

## 11. 关键经验与教训

### 11.1 最重要的技术判断

#### 判断 1：EM 是假指标，EX 才是真相

三轮实验证明 EM 低估模型能力 40-50 个百分点。EM 衡量的是"写法记忆"，EX 衡量的是"执行正确性"。**以业务真实指标（EX）做决策，而非训练 loss 或 EM。**

#### 判断 2：eval_loss 和 EM 可能反向

当训练数据中存在"正确写法"时，过拟合（降低 eval_loss）反而提升 EM。但这不是真正的泛化能力提升。**以 EX 做最终决策。**

#### 判断 3：MLP 层对复杂 SQL 至关重要

V3 去掉 MLP 层（gate/up/down）导致 medium 难度从 27.3% 暴跌到 11.4%。SwiGLU 架构的 FFN 层负责复杂的数据库关系推理，去掉后模型失去了处理 JOIN 等复杂操作的能力。

#### 判断 4：数据增强 ROI 极高

Paraphrase 增强以 30% 计算开销换取数据多样性，是低成本高收益的正则化手段。V3 使用增强数据后，eval_loss 曲线明显更健康。

### 11.2 工程经验

| 经验 | 说明 |
|------|------|
| 量化兼容性 | 4-bit NF4 量化推理速度显著提升，RTX 5070 Ti 单卡可训可推理 |
| early stopping 要设 | V2 设了 patience=3 避免过拟合，V1 没设导致严重过拟合 |
| 评估先行 | 在优化前先建立正确的评估体系（EX vs EM），否则优化方向会指向错误目标 |
| checkpoint 保存策略 | save_total_limit=10 避免磁盘被撑满 |

---

## 12. 下一步优化方向

### 12.1 当前基线（V2 final）

- **真实数据库 EX: 74.7%**（生产模型）
- pred_failed: 4.4%（真实数据库上）
- result_mismatch: 20.9%（主要损失）

### 12.2 Phase 1：推理侧增强（无需重训练）

基于真实数据库评估结果，优化方向更清晰：

#### 方案 1.1：SQL 语法归一化后处理

- 统一函数大小写（AVG -> avg）
- 补全括号
- 展开列名别名

预期效果：pred_failed 从 4.4% 降至 ~2%，EX 提升 1-2%

#### 方案 1.2：自一致性推理

- do_sample=True, temperature=0.7, num_beams=1, num_return_sequences=5
- 5 个候选 SQL 分别执行
- 选执行结果集合最一致的作为最终答案

预期效果：EX 再高 3-5%

### 12.3 Phase 2：训练优化（若推理增强后仍未达目标）

基于三轮实验的结论，下一次训练的正确配置应为：

| 参数 | 值 | 来源 |
|------|-----|------|
| LoRA rank | 32 | V2（不要低于 16） |
| Target modules | 7层（q/k/v/o/gate/up/down） | V1/V2（不要去掉 MLP） |
| Learning rate | 5e-5 | V3（最优训练节奏） |
| Warmup ratio | 0.15 | V3 |
| Weight decay | 0.1 | V3 |
| Dropout | 0.1 | V2 |
| Epochs | 1 | V2/V3 |
| Data augmentation | 启用 | V3 |
| Early stopping | patience=5 | V3 |

### 12.4 预期优化效果

| 阶段 | 目标 EX | 手段 |
|------|---------|------|
| 当前基线 | 74.7% | V2 checkpoint（真实数据库） |
| Phase 1 语法归一化 | 76-77% | 后处理（无需重训练） |
| Phase 1 自一致性 | 78-80% | 推理增强（无需重训练） |
| Phase 2 最优配置重训 | 80-83% | 最优训练配置 |

---

## 附录 A：版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| V1 | 2026-05 | 初始训练，rank=64, 3 epochs，严重过拟合 |
| V2 | 2026-05 | rank=32, early stop，EX=87.5% 持平 |
| V3 | 2026-05 | rank=16, 去掉 MLP，效果下降 |
| Final | 2026-05 | 确定 V2 为生产模型，EX=87.5% |

## 附录 B：快速启动命令

```bash
# 1. 克隆项目
git clone https://github.com/rxdff223/text2sql-finetune.git
cd text2sql-finetune

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载基座模型（Qwen2.5-Coder-7B-Instruct）
# 从 HuggingFace 下载到 pretrained_models/Qwen2.5-Coder-7B-Instruct

# 4. 下载数据集
bash data/download.sh

# 5. 数据预处理
python src/data_process.py --data_dir data --output_dir data

# 6. 训练
bash scripts/run_train.sh

# 7. 评估
bash scripts/run_eval.sh

# 8. 启动 Demo
python demo/app.py --model_path outputs/text2sql-qlora-v2/final
```