# 反洗钱欺诈检测系统 - 项目文档

## 项目结构

```
bitpro-hackathon/
├── data/                           # 数据目录
│   ├── crypto_transfer.csv        # 加密货币交易数据
│   ├── user_info.csv              # 用户信息
│   ├── train_label.csv            # 训练标签
│   └── ...                        # 其他数据文件
│
├── 核心模块
│   ├── data_processor.py          # 数据处理模块
│   ├── graph_builder.py           # 图构建模块
│   ├── resampler.py               # 数据采样模块
│   ├── models.py                  # 模型定义模块
│   └── main.py                    # 主程序（调度所有模块）
│
├── 旧文件（已重构）
│   ├── convertToGraph.py          # ❌ 已整合到 graph_builder.py
│   ├── gcn.py                     # ❌ 已拆分成多个模块
│   └── requirements.txt           # Python依赖
│
└── outputs/                       # 输出目录（自动创建）
    ├── node_embeddings.pt         # 节点嵌入向量
    ├── account_to_id.pt           # 账户ID映射
    └── standardized_transactions.csv  # 标准化交易数据
```

## 各模块详解

### 1. `data_processor.py` - 数据处理模块

**功能**: 读取、清理、标准化交易数据

**主要类**: `DataProcessor`

**关键方法**:
- `load_data()`: 读取CSV数据文件
- `extract_bad_users()`: 提取黑名单用户
- `standardize_transactions()`: 标准化交易并标记洗钱标签
- `create_account_mapping()`: 创建账户到ID的映射
- `map_accounts_to_ids()`: 将账户字符串转换为整数ID

**使用示例**:
```python
from data_processor import DataProcessor

processor = DataProcessor(data_dir='data/')
processor.load_data()
processor.extract_bad_users()
standardized_df = processor.standardize_transactions()
processor.create_account_mapping(standardized_df)
standardized_df = processor.map_accounts_to_ids(standardized_df)
```

---

### 2. `graph_builder.py` - 图构建模块

**功能**: 将交易数据转换成PyTorch Geometric图结构

**主要类**: `GraphBuilder`

**关键方法**:
- `build_graph_from_trading_data()`: 从交易DataFrame构建图
- `load_json_data()`: 读取JSON格式的交易数据
- `build_graph_from_json()`: 从JSON数据构建图（基于IP分组）

**使用示例**:
```python
from graph_builder import GraphBuilder

graph_builder = GraphBuilder()
graph = graph_builder.build_graph_from_trading_data(
    df=standardized_df,
    src_col='src_id',
    dst_col='dst_id',
    num_nodes=processor.num_nodes
)
print(graph)  # 查看图结构
```

---

### 3. `resampler.py` - 数据采样模块

**功能**: 处理类不平衡问题（SMOTE + NearMiss）

**主要类**: `DataResampler`

**关键方法**:
- `split_train_test()`: 分层分割训练/测试集
- `resample_data()`: 使用SMOTE和NearMiss重采样

**使用示例**:
```python
from resampler import DataResampler

resampler = DataResampler()
train_df, test_df = resampler.split_train_test(standardized_df)
train_resampled = resampler.resample_data(train_df)
```

---

### 4. `models.py` - 模型模块

**功能**: 定义和训练图神经网络模型

**主要类**:

#### `Node2VecModel` - Node2Vec模型
- 初始化参数:
  - `embedding_dim`: 嵌入维度（默认128）
  - `walk_length`: 随机游走长度（默认20）
  - `context_size`: skip-gram窗口大小（默认10）
  - `walks_per_node`: 每个节点游走次数（默认10）

- 关键方法:
  - `train()`: 训练模型
  - `get_embeddings()`: 提取节点嵌入向量

#### `GCNModel` - GCN模型
- 图卷积网络实现

#### `GCNTrainer` - GCN训练器
- 训练和评估GCN模型

**使用示例**:
```python
from models import Node2VecModel

n2v_trainer = Node2VecModel(
    edge_index=graph.edge_index,
    embedding_dim=128,
    walk_length=20
)

losses = n2v_trainer.train(epochs=50)
node_embeddings = n2v_trainer.get_embeddings()
```

---

### 5. `main.py` - 主程序

**功能**: 整合所有模块，执行完整的机器学习pipeline

**执行流程**:
1. 数据读取和处理
2. 数据集分割
3. 数据重采样（处理不平衡）
4. 图构建
5. Node2Vec训练
6. 节点嵌入提取
7. 结果保存

**运行方式**:
```bash
python main.py
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

确保以下文件存在于 `data/` 目录下：
- `user_info.csv`
- `crypto_transfer.csv`
- `train_label.csv`

### 3. 运行主程序

```bash
python main.py
```

### 4. 查看输出

输出文件保存在 `outputs/` 目录下：
- `node_embeddings.pt` - PyTorch张量格式的节点嵌入
- `account_to_id.pt` - 账户ID映射
- `standardized_transactions.csv` - 标准化交易数据

---

## 自定义使用 (Advanced)

### 示例1: 自定义参数训练

```python
from data_processor import DataProcessor
from graph_builder import GraphBuilder
from models import Node2VecModel
import torch

# 数据处理
processor = DataProcessor()
processor.load_data()
processor.extract_bad_users()
standardized_df = processor.standardize_transactions()
processor.create_account_mapping(standardized_df)
standardized_df = processor.map_accounts_to_ids(standardized_df)

# 图构建
graph_builder = GraphBuilder()
graph = graph_builder.build_graph_from_trading_data(
    standardized_df,
    num_nodes=processor.num_nodes
)

# Node2Vec训练（自定义参数）
n2v = Node2VecModel(
    edge_index=graph.edge_index,
    embedding_dim=256,      # 增加维度
    walk_length=30,         # 更长的游走
    walks_per_node=20       # 更多游走次数
)

losses = n2v.train(epochs=100)
embeddings = n2v.get_embeddings()

# 保存结果
torch.save(embeddings, 'my_embeddings.pt')
```

### 示例2: 使用JSON数据

```python
from graph_builder import GraphBuilder

graph_builder = GraphBuilder()
graph_builder.load_json_data('./dataset/usdt_twd_trading.json')
graph = graph_builder.build_graph_from_json()
```

---

## 关键概念解释

### 什么是Node2Vec？
Node2Vec是一种图嵌入算法，通过随机游走和skip-gram学习节点的低维表示。
- **参数p**: 返回参数，控制游走回返概率
- **参数q**: 入出参数，控制游走探索倾向

### 什么是SMOTE？
处理类不平衡的过采样技术，通过在少数类样本之间插值生成新样本。

### 什么是NearMiss？
处理类不平衡的欠采样技术，选择最接近多数类样本的样本删除。

---

## 常见问题

### Q: 如何修改Node2Vec的训练轮数？
A: 在 `main.py` 中找到 `n2v_trainer.train(epochs=50)` 行，修改参数值。

### Q: 如何只运行数据处理部分？
A: 只导入并使用 `DataProcessor` 类：
```python
from data_processor import DataProcessor
processor = DataProcessor()
# 使用相应方法...
```

### Q: 输出的嵌入向量如何使用？
A: 可用于下游任务，如分类、聚类、异常检测等：
```python
import torch
embeddings = torch.load('outputs/node_embeddings.pt')
# 用于机器学习模型训练
```

---

## 性能提示

- 如果遇到内存不足，减小 `batch_size` 或 `walks_per_node`
- 在GPU上运行可显著加快速度
- 增加 `epochs` 可能提高模型性能，但训练时间增加

---

## 许可证

MIT License

---

## 作者

Hackathon Team - 反洗钱欺诈检测项目
