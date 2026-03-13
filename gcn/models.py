"""
模型模块 - 定义和训练Node2Vec和其他图神经网络模型
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec, GCNConv
from torch.nn import Linear
from tqdm import tqdm


class Node2VecModel:
    """Node2Vec模型的封装类"""
    
    def __init__(self, edge_index, embedding_dim=128, walk_length=20, 
                 context_size=10, walks_per_node=10, p=1.0, q=0.5):
        """
        初始化Node2Vec模型
        
        Args:
            edge_index: 图的边索引张量
            embedding_dim: 嵌入向量维度
            walk_length: 随机游走长度
            context_size: skip-gram上下文窗口大小
            walks_per_node: 每个节点启动的游走次数
            p: Return参数
            q: In-out参数
        """
        self.edge_index = edge_index
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 初始化Node2Vec模型
        self.model = Node2Vec(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=1,
            p=p,
            q=q,
            sparse=True
        ).to(self.device)
        
        # 创建数据加载器
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        
        # 初始化优化器（稀疏优化器）
        self.optimizer = torch.optim.SparseAdam(
            list(self.model.parameters()), 
            lr=0.01
        )
        
        self.node_embeddings = None
        
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0
        
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            
            # 计算损失
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.loader)
    
    def train(self, epochs=50):
        """
        训练Node2Vec模型
        
        Args:
            epochs: 训练轮数
            
        Returns:
            list: 各epoch的损失值
        """
        print(f"\n开始训练Node2Vec (共 {epochs} epochs)...")
        
        losses = []
        
        for epoch in tqdm(range(1, epochs + 1), desc="训练进度"):
            loss = self.train_epoch()
            losses.append(loss)
            
            # 每5个epoch输出一次进度
            if epoch % 5 == 0 or epoch == 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        
        return losses
    
    def get_embeddings(self):
        """
        提取所有节点的嵌入向量
        
        Returns:
            torch.Tensor: 节点嵌入矩阵 (num_nodes, embedding_dim)
        """
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model()
        
        print(f"\n提取的节点嵌入向量形状: {self.node_embeddings.shape}")
        return self.node_embeddings


class GCNModel(torch.nn.Module):
    """GCN（图卷积网络）模型"""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        初始化GCN模型
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐层特征维度
            out_channels: 输出特征维度（分类数）
        """
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = Linear(out_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征矩阵
            edge_index: 边索引
            
        Returns:
            torch.Tensor: 输出预测
        """
        # 第一个GCN层 + ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # 第二个GCN层 + ReLU激活
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 线性层
        x = self.lin(x)
        
        return x


class GCNTrainer:
    """GCN模型训练器"""
    
    def __init__(self, model, device='cpu'):
        """
        初始化GCN训练器
        
        Args:
            model: GCN模型实例
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=5e-4
        )
    
    def train(self, data, mask, labels, epochs=100):
        """
        训练GCN模型
        
        Args:
            data: PyG格式的图数据
            mask: 训练节点掩码
            labels: 标签
            epochs: 训练轮数
        """
        print(f"\n开始训练GCN (共 {epochs} epochs)...")
        
        self.model.train()
        data = data.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        
        for epoch in tqdm(range(1, epochs + 1), desc="训练进度"):
            self.optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[mask], labels[mask])
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    def evaluate(self, data, mask, labels):
        """
        评估模型性能
        
        Args:
            data: PyG格式的图数据
            mask: 评估节点掩码
            labels: 标签
            
        Returns:
            float: 准确率
        """
        self.model.eval()
        data = data.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            correct = (pred == labels[mask]).sum().item()
            acc = correct / mask.sum().item()
        
        return acc
