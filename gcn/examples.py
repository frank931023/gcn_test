"""
简单示例 - 演示如何单独使用各个模块
"""

# ==========================================
# 示例 1: 演示数据处理模块
# ==========================================
def example_data_processing():
    """演示如何使用数据处理模块"""
    print("\n" + "="*60)
    print("示例 1: 数据处理")
    print("="*60)
    
    from gcn.data_processor import DataProcessor
    
    # 创建处理器
    processor = DataProcessor(data_dir='data/')
    
    # 读取数据
    processor.load_data()
    
    # 提取黑名单用户
    bad_users = processor.extract_bad_users()
    print(f"黑名单用户数: {len(bad_users)}")
    
    # 标准化交易
    standardized_df = processor.standardize_transactions()
    
    print(f"\n标准化后的数据形状: {standardized_df.shape}")
    print(standardized_df.head())


# ==========================================
# 示例 2: 演示图构建模块
# ==========================================
def example_graph_building():
    """演示如何使用图构建模块"""
    print("\n" + "="*60)
    print("示例 2: 图构建")
    print("="*60)
    
    from gcn.data_processor import DataProcessor
    from gcn.graph_builder import GraphBuilder
    
    # 首先处理数据
    processor = DataProcessor(data_dir='data/')
    processor.load_data()
    processor.extract_bad_users()
    standardized_df = processor.standardize_transactions()
    processor.create_account_mapping(standardized_df)
    standardized_df = processor.map_accounts_to_ids(standardized_df)
    
    # 构建图
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph_from_trading_data(
        df=standardized_df,
        src_col='src_id',
        dst_col='dst_id',
        num_nodes=processor.num_nodes
    )
    
    print(f"\n图节点数: {graph.num_nodes}")
    print(f"图边数: {graph.num_edges}")


# ==========================================
# 示例 3: 演示数据采样模块
# ==========================================
def example_resampling():
    """演示如何使用数据采样模块"""
    print("\n" + "="*60)
    print("示例 3: 数据重采样")
    print("="*60)
    
    from gcn.data_processor import DataProcessor
    from gcn.resampler import DataResampler
    
    # 处理数据
    processor = DataProcessor(data_dir='data/')
    processor.load_data()
    processor.extract_bad_users()
    standardized_df = processor.standardize_transactions()
    processor.create_account_mapping(standardized_df)
    standardized_df = processor.map_accounts_to_ids(standardized_df)
    
    # 重采样
    resampler = DataResampler()
    train_df, test_df = resampler.split_train_test(standardized_df)
    train_resampled = resampler.resample_data(train_df)
    
    print(f"\n原始训练集大小: {len(train_df)}")
    print(f"重采样后大小: {len(train_resampled)}")


# ==========================================
# 示例 4: 演示Node2Vec模型
# ==========================================
def example_node2vec_training():
    """演示如何训练Node2Vec模型"""
    print("\n" + "="*60)
    print("示例 4: Node2Vec训练")
    print("="*60)
    
    from gcn.data_processor import DataProcessor
    from gcn.graph_builder import GraphBuilder
    from gcn.resampler import DataResampler
    from gcn.models import Node2VecModel
    
    # 数据处理
    processor = DataProcessor(data_dir='data/')
    processor.load_data()
    processor.extract_bad_users()
    standardized_df = processor.standardize_transactions()
    processor.create_account_mapping(standardized_df)
    standardized_df = processor.map_accounts_to_ids(standardized_df)
    
    # 分割和重采样
    resampler = DataResampler()
    train_df, test_df = resampler.split_train_test(standardized_df)
    train_resampled = resampler.resample_data(train_df)
    
    # 构建图
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph_from_trading_data(
        train_resampled,
        num_nodes=processor.num_nodes
    )
    
    # 训练Node2Vec（注意：epochs设置为5以加快演示）
    n2v = Node2VecModel(
        edge_index=graph.edge_index,
        embedding_dim=64,  # 使用更小的维度以加快速度
        walk_length=10,    # 更短的游走
        walks_per_node=3   # 更少的游走
    )
    
    losses = n2v.train(epochs=5)  # 仅5个epoch用于演示
    embeddings = n2v.get_embeddings()
    
    print(f"\n节点嵌入形状: {embeddings.shape}")
    print(f"最后一个epoch损失: {losses[-1]:.4f}")


# ==========================================
# 示例 5: 自定义参数训练
# ==========================================
def example_custom_training():
    """演示如何自定义参数进行训练"""
    print("\n" + "="*60)
    print("示例 5: 自定义参数训练")
    print("="*60)
    
    from gcn.config import NODE2VEC_EMBEDDING_DIM, NODE2VEC_WALK_LENGTH
    from gcn.models import Node2VecModel
    from gcn.data_processor import DataProcessor
    from gcn.graph_builder import GraphBuilder
    from gcn.resampler import DataResampler
    
    # 从配置文件读取参数
    print(f"从配置文件读取的参数:")
    print(f"  嵌入维度: {NODE2VEC_EMBEDDING_DIM}")
    print(f"  游走长度: {NODE2VEC_WALK_LENGTH}")
    
    # 你可以在这里创建自己的参数并进行训练
    print(f"\n可在 config.py 中修改这些参数，然后运行 main.py 执行完整pipeline")


# ==========================================
# 主函数
# ==========================================
def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("#" * 60)
    print("# 反洗钱欺诈检测系统 - 代码示例")
    print("#" * 60)
    
    # 运行示例（注意：取消注释想要运行的示例）
    
    # example_data_processing()
    # example_graph_building()
    # example_resampling()
    # example_node2vec_training()
    # example_custom_training()
    
    print("\n提示: 取消注释想要运行的示例，然后执行此脚本")
    print("      或者直接运行: python main.py (执行完整pipeline)")


if __name__ == '__main__':
    run_all_examples()
