"""
配置文件 - 集中管理所有超参数和配置
"""

# ==========================================
# 数据配置
# ==========================================
DATA_DIR = 'data/'

# 数据文件名
USER_INFO_FILE = 'user_info.csv'
CRYPTO_TRANSFER_FILE = 'crypto_transfer.csv'
TRAIN_LABEL_FILE = 'train_label.csv'

# ==========================================
# 数据处理配置
# ==========================================
# 黑名单用户标签值
WHITE_LIST_LABEL = 0
BLACK_LIST_LABEL = 1

# 金额标准化系数
AMOUNT_NORMALIZATION = 1e-8

# ==========================================
# 数据集分割配置
# ==========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==========================================
# 重采样配置
# ==========================================
# SMOTE采样策略 (少数类/多数类比例)
SMOTE_SAMPLING_STRATEGY = 0.1

# ==========================================
# Node2Vec超参数
# ==========================================
NODE2VEC_EMBEDDING_DIM = 128      # 嵌入向量维度
NODE2VEC_WALK_LENGTH = 20         # 随机游走长度
NODE2VEC_CONTEXT_SIZE = 10        # skip-gram窗口大小
NODE2VEC_WALKS_PER_NODE = 10      # 每节点游走次数
NODE2VEC_P = 1.0                  # Return参数
NODE2VEC_Q = 0.5                  # In-out参数
NODE2VEC_NUM_NEGATIVE_SAMPLES = 1 # 负采样数
NODE2VEC_SPARSE = True            # 使用稀疏模式

# Node2Vec训练配置
NODE2VEC_BATCH_SIZE = 128
NODE2VEC_NUM_WORKERS = 0           # Windows上设为0避免多进程错误
NODE2VEC_LR = 0.01                 # 学习率
NODE2VEC_EPOCHS = 50               # 训练轮数

# ==========================================
# GCN超参数
# ==========================================
GCN_HIDDEN_CHANNELS = 64
GCN_OUTPUT_CHANNELS = 2

GCN_LR = 0.01
GCN_WEIGHT_DECAY = 5e-4
GCN_EPOCHS = 100

# ==========================================
# 输出配置
# ==========================================
OUTPUT_DIR = 'outputs/'

# 输出文件名
NODE_EMBEDDINGS_FILE = 'node_embeddings.pt'
ACCOUNT_TO_ID_FILE = 'account_to_id.pt'
STANDARDIZED_TRANSACTIONS_FILE = 'standardized_transactions.csv'
TRAIN_DATA_FILE = 'train_data.csv'
TEST_DATA_FILE = 'test_data.csv'

# ==========================================
# 日志配置
# ==========================================
VERBOSE = True  # 是否输出详细日志

# ==========================================
# 硬件配置
# ==========================================
USE_CUDA = True  # 是否使用GPU（如果可用）
