"""
图构建模块 - 负责将交易数据转换成PyTorch Geometric格式的图
"""
import json
import torch
from torch_geometric.data import Data
from itertools import combinations


class GraphBuilder:
    """构建图结构的类"""
    
    def __init__(self):
        """初始化图构建器"""
        self.node_mapping = {}
        self.node_counter = 0
        self.raw_json_data = None
        
    def build_graph_from_trading_data(self, df, src_col='src_id', dst_col='dst_id', num_nodes=None):
        """
        从交易数据构建图
        
        Args:
            df: 包含交易数据的DataFrame
            src_col: 源账户ID列名
            dst_col: 目标账户ID列名
            num_nodes: 图的节点总数
            
        Returns:
            torch_geometric.data.Data: PyG格式的图数据
        """
        print("从交易数据构建图...")
        
        source_nodes = df[src_col].values
        target_nodes = df[dst_col].values
        
        # 创建边索引张量
        edge_index = torch.tensor(
            [source_nodes, target_nodes],
            dtype=torch.long
        )
        
        if num_nodes is None:
            num_nodes = max(edge_index.max().item() + 1, 1)
        
        graph_data = Data(
            edge_index=edge_index,
            num_nodes=num_nodes
        )
        
        print(f"\n图结构:")
        print(f"  节点数: {graph_data.num_nodes}")
        print(f"  边数: {graph_data.num_edges}")
        print(graph_data)
        
        return graph_data
    
    def load_json_data(self, json_path):
        """
        读取JSON格式的数据
        
        Args:
            json_path: JSON文件路径
        """
        print(f"读取JSON数据: {json_path}")
        with open(json_path, 'r') as f:
            self.raw_json_data = json.load(f)
        print(f"读取了 {len(self.raw_json_data)} 条记录")
        
    def build_node_mapping(self):
        """
        从JSON数据构建节点映射
        
        Returns:
            dict: user_id到node_id的映射
        """
        print("\n构建节点映射...")
        self.node_mapping = {}
        self.node_counter = 0
        
        for row in self.raw_json_data:
            uid = row["user_id"]
            if uid not in self.node_mapping:
                self.node_mapping[uid] = self.node_counter
                self.node_counter += 1
        
        print(f"创建了 {self.node_counter} 个节点")
        return self.node_mapping
    
    def build_graph_from_json(self):
        """
        从JSON数据构建图（基于IP地址分组）
        
        Returns:
            torch_geometric.data.Data: PyG格式的图数据
        """
        if self.raw_json_data is None:
            raise ValueError("请先调用 load_json_data() 来加载数据")
        
        if not self.node_mapping:
            self.build_node_mapping()
        
        print("\n按source_ip_hash分组用户...")
        ip_groups = {}
        
        for row in self.raw_json_data:
            ip = row["source_ip_hash"]
            uid = row["user_id"]
            
            if ip not in ip_groups:
                ip_groups[ip] = []
            
            ip_groups[ip].append(uid)
        
        # 建立边列表：同一IP的用户两两连线
        source_nodes = []
        target_nodes = []
        
        print("构建边...")
        for ip, users in ip_groups.items():
            for u1, u2 in combinations(users, 2):
                src = self.node_mapping[u1]
                dst = self.node_mapping[u2]
                source_nodes.append(src)
                target_nodes.append(dst)
        
        # 转成PyG格式
        edge_index = torch.tensor(
            [source_nodes, target_nodes],
            dtype=torch.long
        )
        
        graph_data = Data(
            edge_index=edge_index,
            num_nodes=self.node_counter
        )
        
        print(f"\n图结构:")
        print(f"  节点数: {graph_data.num_nodes}")
        print(f"  边数: {graph_data.num_edges}")
        print(graph_data)
        
        return graph_data
