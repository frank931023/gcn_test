"""
主程序 - 整合所有模块，执行完整的机器学习pipeline
二级缓存机制：
  - 阶段1缓存：标准化数据（standardized_transactions_cached.csv + account_to_id.pt）
  - 阶段7缓存：Node2Vec嵌入（node_embeddings.pt + train_test_split.pt）
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from config import (
    DATA_DIR, TEST_SIZE, RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    NODE2VEC_EMBEDDING_DIM, NODE2VEC_WALK_LENGTH,
    NODE2VEC_CONTEXT_SIZE, NODE2VEC_WALKS_PER_NODE,
    NODE2VEC_P, NODE2VEC_Q, NODE2VEC_EPOCHS,
    OUTPUT_DIR
)
from data_processor import DataProcessor
from graph_builder import GraphBuilder
from resampler import DataResampler
from models import Node2VecModel


# ==========================================
# 缓存检查和加载函数
# ==========================================

def check_stage1_cache_exists():
    """检查阶段1缓存是否存在（标准化数据）"""
    required_files = [
        f'{OUTPUT_DIR}standardized_transactions_cached.csv',
        f'{OUTPUT_DIR}account_to_id.pt'
    ]
    
    exists = all(os.path.exists(f) for f in required_files)
    
    if exists:
        print("\n✓ 检测到【阶段1】缓存（标准化数据）")
        for f in required_files:
            print(f"  - {os.path.basename(f)}")
    
    return exists


def check_stage7_cache_exists():
    """检查阶段7缓存是否存在（Node2Vec嵌入）"""
    required_files = [
        f'{OUTPUT_DIR}node_embeddings.pt',
        f'{OUTPUT_DIR}train_test_split.pt'
    ]
    
    exists = all(os.path.exists(f) for f in required_files)
    
    if exists:
        print("\n✓ 检测到【阶段7】缓存（Node2Vec嵌入）")
        for f in required_files:
            print(f"  - {os.path.basename(f)}")
    
    return exists


def load_stage1_cache():
    """加载阶段1缓存"""
    print("\n【加载阶段1缓存】")
    print("-" * 60)
    
    # 读取数据
    standardized_df = pd.read_csv(f'{OUTPUT_DIR}standardized_transactions_cached.csv')
    account_to_id = torch.load(f'{OUTPUT_DIR}account_to_id.pt')
    
    # 创建processor对象
    processor = DataProcessor(data_dir=DATA_DIR)
    processor.account_to_id = account_to_id
    processor.id_to_account = {i: acc for acc, i in account_to_id.items()}
    processor.num_nodes = len(account_to_id)
    
    print(f"✓ 已加载标准化数据: {len(standardized_df)} 行")
    print(f"✓ 已加载账户映射: {len(account_to_id)} 个账户")
    
    return standardized_df, processor


def load_stage7_cache():
    """加载阶段7缓存"""
    print("\n【加载阶段7缓存】")
    print("-" * 60)
    
    # 读取嵌入
    node_embeddings = torch.load(f'{OUTPUT_DIR}node_embeddings.pt')
    
    # 读取训练/测试集
    split_data = torch.load(f'{OUTPUT_DIR}train_test_split.pt')
    train_df = split_data['train_df']
    test_df = split_data['test_df']
    
    print(f"✓ 已加载节点嵌入: {node_embeddings.shape}")
    print(f"✓ 已加载训练集: {len(train_df)} 行")
    print(f"✓ 已加载测试集: {len(test_df)} 行")
    
    return node_embeddings, train_df, test_df


def main(use_cache=True):
    """主程序流程 - 二级缓存机制
    
    Args:
        use_cache: 是否使用缓存
    """
    
    print("=" * 60)
    print("反洗钱欺诈检测系统 - 完整Pipeline")
    print("=" * 60)
    
    # ==========================================
    # 【第一级缓存】检查阶段1（数据标准化）
    # ==========================================
    if use_cache and check_stage1_cache_exists():
        standardized_df, processor = load_stage1_cache()
        skip_stage1 = True
    else:
        skip_stage1 = False
    
    if not skip_stage1:
        # ==========================================
        # 阶段 1: 数据读取和处理
        # ==========================================
        print("\n【阶段 1】数据读取和处理")
        print("-" * 60)
        
        processor = DataProcessor(data_dir=DATA_DIR)
        processor.load_data()
        processor.extract_bad_users()
        standardized_df = processor.standardize_transactions()
        processor.create_account_mapping(standardized_df)
        standardized_df = processor.map_accounts_to_ids(standardized_df)
        
        # 保存阶段1缓存
        print("\n【保存阶段1缓存】")
        print("-" * 60)
        standardized_df.to_csv(f'{OUTPUT_DIR}standardized_transactions_cached.csv', index=False)
        torch.save(processor.account_to_id, f'{OUTPUT_DIR}account_to_id.pt')
        print("✓ 阶段1缓存已保存")
    
    # ==========================================
    # 阶段 2: 训练集和测试集分割
    # ==========================================
    print("\n【阶段 2】数据集分割")
    print("-" * 60)
    
    resampler = DataResampler()
    train_df, test_df = resampler.split_train_test(
        standardized_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # ==========================================
    # 阶段 2.5: 询问是否进行数据重采样
    # ==========================================
    print("\n" + "="*60)
    print("❓ 数据重采样选项")
    print("="*60)
    print("\n发现类不平衡问题 (洗钱交易 vs 正常交易)")
    print(f"当前训练集中:")
    print(f"  - 总交易数: {len(train_df)}")
    print(f"  - 洗钱交易: {(train_df['is_laundering'] == 1).sum()} ({(train_df['is_laundering'] == 1).sum() / len(train_df) * 100:.2f}%)")
    print(f"  - 正常交易: {(train_df['is_laundering'] == 0).sum()} ({(train_df['is_laundering'] == 0).sum() / len(train_df) * 100:.2f}%)")
    
    print("\n使用 SMOTE + NearMiss 重采样可以:")
    print("  ✓ 平衡正负样本比例")
    print("  ✓ 改善模型性能")
    print("  ✓ 但会增加计算时间")
    
    while True:
        do_resample = input("\n要进行数据重采样吗? (y/n): ").strip().lower()
        if do_resample in ['y', 'n']:
            break
        print("⚠️  请输入 y 或 n")
    
    # ==========================================
    # 阶段 3: 数据重采样处理不平衡 (可选)
    # ==========================================
    if do_resample == 'y':
        print("\n【阶段 3】数据重采样 (处理类不平衡)")
        print("-" * 60)
        
        train_resampled_df = resampler.resample_data(
            train_df,
            smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY
        )
        graph_data = train_resampled_df
        data_source = "重采样后的数据"
    else:
        print("\n【阶段 3】跳过数据重采样")
        print("-" * 60)
        print("✓ 已跳过数据重采样，将使用原始训练数据")
        train_resampled_df = train_df
        graph_data = train_df
        data_source = "原始训练数据"
    
    # ==========================================
    # 【第二级缓存】检查阶段7（Node2Vec嵌入）
    # ==========================================
    if use_cache and check_stage7_cache_exists():
        node_embeddings, train_df, test_df = load_stage7_cache()
        skip_stage4_to_7 = True
    else:
        skip_stage4_to_7 = False
    
    if not skip_stage4_to_7:
        # ==========================================
        # 阶段 4: 构建图结构
        # ==========================================
        print("\n【阶段 4】图构建")
        print("-" * 60)
        print(f"使用 {data_source} 构建图...")
        
        graph_builder = GraphBuilder()
        graph = graph_builder.build_graph_from_trading_data(
            graph_data,
            src_col='src_id',
            dst_col='dst_id',
            num_nodes=processor.num_nodes
        )
        
        # ==========================================
        # 阶段 5: Node2Vec训练
        # ==========================================
        print("\n【阶段 5】Node2Vec模型训练")
        print("-" * 60)
        
        n2v_trainer = Node2VecModel(
            edge_index=graph.edge_index,
            embedding_dim=NODE2VEC_EMBEDDING_DIM,
            walk_length=NODE2VEC_WALK_LENGTH,
            context_size=NODE2VEC_CONTEXT_SIZE,
            walks_per_node=NODE2VEC_WALKS_PER_NODE,
            p=NODE2VEC_P,
            q=NODE2VEC_Q
        )
        
        losses = n2v_trainer.train(epochs=NODE2VEC_EPOCHS)
        
        # ==========================================
        # 阶段 6: 提取节点嵌入
        # ==========================================
        print("\n【阶段 6】节点嵌入提取")
        print("-" * 60)
        
        node_embeddings = n2v_trainer.get_embeddings()
        
        # ==========================================
        # 阶段 7: 保存阶段7缓存
        # ==========================================
        print("\n【阶段 7】保存Node2Vec缓存")
        print("-" * 60)
        
        torch.save(node_embeddings, f'{OUTPUT_DIR}node_embeddings.pt')
        print("✓ 节点嵌入已保存到 outputs/node_embeddings.pt")
        
        torch.save({
            'train_df': train_df,
            'test_df': test_df
        }, f'{OUTPUT_DIR}train_test_split.pt')
        print("✓ 训练/测试集已保存到 outputs/train_test_split.pt")
        
        # 保存标准化数据
        standardized_df.to_csv(f'{OUTPUT_DIR}standardized_transactions.csv', index=False)
        print("✓ 标准化交易数据已保存到 outputs/standardized_transactions.csv")
    
    # ==========================================
    # 阶段 8: GCN 分类模型训练
    # ==========================================
    print("\n【阶段 8】GCN 分类模型训练")
    print("-" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 准备训练/测试样本的索引
    train_indices = train_df['src_id'].values
    test_indices = test_df['src_id'].values
    
    # 准备标签
    train_labels = torch.tensor(train_df['is_laundering'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['is_laundering'].values, dtype=torch.long)
    
    print(f"\n训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"正样本 (洗钱): {(train_labels == 1).sum().item()}")
    print(f"负样本 (正常): {(train_labels == 0).sum().item()}")
    
    # 定义MLP分类器
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # 初始化模型
    mlp_model = MLPClassifier(NODE2VEC_EMBEDDING_DIM, hidden_dim=64, output_dim=2)
    mlp_model = mlp_model.to(device)
    
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练
    print("\n开始训练分类器...")
    epochs = 50
    
    for epoch in tqdm(range(1, epochs + 1), desc="MLP分类器训练"):
        mlp_model.train()
        optimizer.zero_grad()
        
        # 训练集
        train_out = mlp_model(node_embeddings[train_indices].to(device))
        loss = loss_fn(train_out, train_labels.to(device))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            mlp_model.eval()
            with torch.no_grad():
                # 训练集准确率
                train_preds = mlp_model(node_embeddings[train_indices].to(device)).argmax(dim=1)
                train_acc = (train_preds == train_labels.to(device)).float().mean()
                
                # 测试集准确率
                test_preds = mlp_model(node_embeddings[test_indices].to(device)).argmax(dim=1)
                test_acc = (test_preds == test_labels.to(device)).float().mean()
                
                print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
    
    # ==========================================
    # 阶段 9: 模型评估
    # ==========================================
    print("\n【阶段 9】模型评估")
    print("-" * 60)
    
    mlp_model.eval()
    with torch.no_grad():
        # 训练集
        train_logits = mlp_model(node_embeddings[train_indices].to(device))
        train_preds = train_logits.argmax(dim=1).cpu().numpy()
        train_labels_np = train_labels.numpy()
        
        # 测试集
        test_logits = mlp_model(node_embeddings[test_indices].to(device))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_labels_np = test_labels.numpy()
        
        print("\n【训练集性能】")
        train_acc = (train_preds == train_labels_np).mean()
        print(f"  准确率 (Accuracy):  {train_acc:.4f}")
        print(f"  精准率 (Precision): {precision_score(train_labels_np, train_preds, zero_division=0):.4f}")
        print(f"  召回率 (Recall):    {recall_score(train_labels_np, train_preds, zero_division=0):.4f}")
        print(f"  F1 Score:           {f1_score(train_labels_np, train_preds, zero_division=0):.4f}")
        
        train_cm = confusion_matrix(train_labels_np, train_preds)
        print(f"\n  混淆矩阵 (Confusion Matrix):")
        print(f"    正常正确: {train_cm[0, 0]}, 正常错误: {train_cm[0, 1]}")
        print(f"    洗钱错误: {train_cm[1, 0]}, 洗钱正确: {train_cm[1, 1]}")
        
        print("\n【测试集性能】")
        test_acc = (test_preds == test_labels_np).mean()
        print(f"  准确率 (Accuracy):  {test_acc:.4f}")
        print(f"  精准率 (Precision): {precision_score(test_labels_np, test_preds, zero_division=0):.4f}")
        print(f"  召回率 (Recall):    {recall_score(test_labels_np, test_preds, zero_division=0):.4f}")
        print(f"  F1 Score:           {f1_score(test_labels_np, test_preds, zero_division=0):.4f}")
        
        test_cm = confusion_matrix(test_labels_np, test_preds)
        print(f"\n  混淆矩阵 (Confusion Matrix):")
        print(f"    正常正确: {test_cm[0, 0]}, 正常错误: {test_cm[0, 1]}")
        print(f"    洗钱错误: {test_cm[1, 0]}, 洗钱正确: {test_cm[1, 1]}")
    
    # 保存模型
    torch.save(mlp_model.state_dict(), f'{OUTPUT_DIR}mlp_classifier.pt')
    print(f"\n✓ 分类模型已保存到 outputs/mlp_classifier.pt")
    
    # ==========================================
    # 完成
    # ==========================================
    print("\n" + "=" * 60)
    print("✓ Pipeline执行完成！")
    print("=" * 60)
    
    return {
        'processor': processor,
        'node_embeddings': node_embeddings,
        'standardized_df': standardized_df,
        'train_df': train_df,
        'test_df': test_df,
        'mlp_model': mlp_model
    }


if __name__ == '__main__':
    # 创建输出目录
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行主程序
    # 设置 use_cache=True 使用缓存（默认），use_cache=False 强制重新计算
    results = main(use_cache=True)
