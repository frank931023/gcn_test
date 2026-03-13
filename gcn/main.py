"""
主程序 - 整合所有模块，执行完整的机器学习pipeline
"""
import os
import pandas as pd
import torch
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


def check_cache_exists():
    """
    检查是否存在缓存的标准化数据
    
    Returns:
        bool: 缓存文件是否存在
    """
    cache_file = f'{OUTPUT_DIR}standardized_transactions_cached.csv'
    account_mapping_file = f'{OUTPUT_DIR}account_to_id.pt'
    
    exists = os.path.exists(cache_file) and os.path.exists(account_mapping_file)
    
    if exists:
        print("\n✓ 检测到缓存的标准化数据")
        print(f"  - {cache_file}")
        print(f"  - {account_mapping_file}")
    
    return exists


def load_from_cache():
    """
    从缓存加载标准化后的数据和账户映射
    
    Returns:
        tuple: (standardized_df, account_to_id, processor)
    """
    print("\n【加载缓存数据】")
    print("-" * 60)
    print("从缓存文件加载标准化数据...")
    
    cache_file = f'{OUTPUT_DIR}standardized_transactions_cached.csv'
    account_mapping_file = f'{OUTPUT_DIR}account_to_id.pt'
    
    # 读取标准化后的数据
    standardized_df = pd.read_csv(cache_file)
    print(f"✓ 已加载 {len(standardized_df)} 行标准化数据")
    
    # 读取账户映射
    account_to_id = torch.load(account_mapping_file)
    print(f"✓ 已加载 {len(account_to_id)} 个账户映射")
    
    # 创建虚拟的 processor 对象来保存映射信息
    processor = DataProcessor(data_dir=DATA_DIR)
    processor.account_to_id = account_to_id
    processor.id_to_account = {i: acc for acc, i in account_to_id.items()}
    
    return standardized_df, processor


def main(use_cache=True):
    """主程序流程
    
    Args:
        use_cache: 是否使用缓存（如果存在）
    """
    
    print("=" * 60)
    print("反洗钱欺诈检测系统 - 完整Pipeline")
    print("=" * 60)
    
    # 检查是否使用缓存
    if use_cache and check_cache_exists():
        standardized_df, processor = load_from_cache()
    else:
        # ==========================================
        # 阶段 1: 数据读取和处理
        # ==========================================
        print("\n【阶段 1】数据读取和处理")
        print("-" * 60)
        
        # 初始化数据处理器
        processor = DataProcessor(data_dir=DATA_DIR)
        
        # 读取数据
        processor.load_data()
        
        # 提取黑名单用户
        processor.extract_bad_users()
        
        # 标准化交易数据
        standardized_df = processor.standardize_transactions()
        
        # 创建账户映射
        processor.create_account_mapping(standardized_df)
        
        # 将账户映射到ID
        standardized_df = processor.map_accounts_to_ids(standardized_df)
        
        # 保存缓存
        print("\n【保存缓存】")
        print("-" * 60)
        cache_file = f'{OUTPUT_DIR}standardized_transactions_cached.csv'
        account_mapping_file = f'{OUTPUT_DIR}account_to_id.pt'
        
        standardized_df.to_csv(cache_file, index=False)
        print(f"✓ 已保存标准化数据缓存: {cache_file}")
        
        torch.save(processor.account_to_id, account_mapping_file)
        print(f"✓ 已保存账户映射缓存: {account_mapping_file}")
    
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
    # 阶段 4: 构建图结构
    # ==========================================
    print("\n【阶段 4】图构建")
    print("-" * 60)
    print(f"使用 {data_source} 构建图...")
    
    graph_builder = GraphBuilder()
    
    # 从交易数据构建图
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
    
    # 训练Node2Vec
    losses = n2v_trainer.train(epochs=NODE2VEC_EPOCHS)
    
    # ==========================================
    # 阶段 6: 提取节点嵌入
    # ==========================================
    print("\n【阶段 6】节点嵌入提取")
    print("-" * 60)
    
    node_embeddings = n2v_trainer.get_embeddings()
    
    # ==========================================
    # 阶段 7: 保存结果
    # ==========================================
    print("\n【阶段 7】保存结果")
    print("-" * 60)
    
    # 保存节点嵌入
    torch.save(node_embeddings, f'{OUTPUT_DIR}node_embeddings.pt')
    print("✓ 节点嵌入已保存到 outputs/node_embeddings.pt")
    
    # 保存账户映射
    torch.save(processor.account_to_id, f'{OUTPUT_DIR}account_to_id.pt')
    print("✓ 账户映射已保存到 outputs/account_to_id.pt")
    
    # 保存标准化数据
    standardized_df.to_csv(f'{OUTPUT_DIR}standardized_transactions.csv', index=False)
    print("✓ 标准化交易数据已保存到 outputs/standardized_transactions.csv")
    
    # ==========================================
    # 完成
    # ==========================================
    print("\n" + "=" * 60)
    print("✓ Pipeline执行完成！")
    print("=" * 60)
    
    return {
        'processor': processor,
        'resampler': resampler,
        'graph_builder': graph_builder,
        'n2v_trainer': n2v_trainer,
        'graph': graph,
        'node_embeddings': node_embeddings,
        'standardized_df': standardized_df,
        'train_df': train_df,
        'test_df': test_df
    }


if __name__ == '__main__':
    # 创建输出目录
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行主程序
    # 设置 use_cache=True 使用缓存（默认），use_cache=False 强制重新计算
    results = main(use_cache=True)
