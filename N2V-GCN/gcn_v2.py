import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv ,GATv2Conv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 定義圖結構 G 與建立索引映射 ---
def build_graph_data():
    # 讀取邊資料
    shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')
    crypto = pd.read_csv('../data/crypto_transfer.csv')
    
    # 建立 NetworkX 圖
    G = nx.Graph()
    
    # 加入 IP 共用邊
    for _, row in shared_ip.iterrows():
        G.add_edge(int(row['user_id_1']), int(row['user_id_2']))
    
    # 加入內轉關係邊
    crypto_internal = crypto[crypto['relation_user_id'].notnull()]
    for _, row in crypto_internal.iterrows():
        G.add_edge(int(row['user_id']), int(row['relation_user_id']))
        
    # 獲取所有節點並建立 ID 映射 (user_id -> index 0~N)
    all_nodes = sorted(list(G.nodes()))
    id_map = {node_id: i for i, node_id in enumerate(all_nodes)}
    
    # 轉換邊為 PyG 格式 (edge_index)
    edge_list = []
    for u, v in G.edges():
        edge_list.append([id_map[u], id_map[v]])
        edge_list.append([id_map[v], id_map[u]]) # GCN 通常處理無向圖
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return G, id_map, edge_index, all_nodes

# --- 2. 整合特徵 (行為特徵 + Node2Vec Embeddings) ---
def prepare_features(id_map, all_nodes):
    # 讀取行為特徵
    behavior_df = pd.read_csv('extended_features_analysis.csv')
    
    # 讀取 Node2Vec 向量
    n2v_df = pd.read_csv("user_n2v.embeddings", sep=' ', skiprows=1, header=None)
    n2v_df.columns = ['user_id'] + [f'n2v_{i}' for i in range(128)]
    
    # 合併
    final_df = behavior_df.merge(n2v_df, on='user_id', how='inner')
    
    # 轉換可解析的日期字串為 timestamp（秒），嘗試將其他 object 欄位轉數值，無法轉的欄位移除
    for col in list(final_df.columns):
        if col == 'user_id':
            continue
        if final_df[col].dtype == object:
            dt = pd.to_datetime(final_df[col], errors='coerce', utc=True)
            if dt.notna().any():
                final_df[col] = dt.astype('int64') // 10**9
                continue
            num = pd.to_numeric(final_df[col], errors='coerce')
            if num.notna().any():
                final_df[col] = num
            else:
                final_df = final_df.drop(columns=[col])
    
    # 確保特徵矩陣與 id_map 的順序一致
    feature_dim = final_df.drop(columns=['user_id', 'status']).shape[1]
    x_matrix = np.zeros((len(all_nodes), feature_dim))
    y_vector = np.zeros(len(all_nodes))
    
    final_df = final_df.set_index('user_id')
    for i, node_id in enumerate(all_nodes):
        if node_id in final_df.index:
            row = final_df.loc[node_id]
            x_matrix[i] = row.drop('status').values
            y_vector[i] = row['status']
            
    return torch.tensor(x_matrix, dtype=torch.float), torch.tensor(y_vector, dtype=torch.long)

# --- 3. 執行主程式 ---
G, id_map, edge_index, all_nodes = build_graph_data()
x, y = prepare_features(id_map, all_nodes)

# 建立 PyG Data 物件
# 注意：這裡包含全圖資訊，解決了 SMOTE 節點沒連線的問題
data = Data(x=x, edge_index=edge_index, y=y)

# 定義 GCN 模型
class N2V_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(N2V_GCN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=8, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一層卷積
        h1 = self.conv1(x, edge_index)
        h1 = F.elu(h1)
        
        # 第二層卷積
        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        
        # JK 架構：結合初始特徵與二階特徵
        combined = torch.cat([h2, x], dim=1) 
        
        return self.classifier(combined)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = N2V_GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2).to(device)

# --- 4. 針對 FNR=0 的權重設定 ---
# 既然不建議在圖中用 SMOTE，我們把 Class 1 (黑名單) 的權重設得更高
# 100 代表漏掉一個黑名單的懲罰是誤抓一個正常人的 100 倍
weights = torch.tensor([1.0, 100.0]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- 特徵標準化：避免大數值造成梯度/loss 不穩定 ---
scaler = StandardScaler()
# data.x 可能已是 tensor，先轉回 numpy 做 scaling
X_np = data.x.numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
X_scaled = scaler.fit_transform(X_np)
data.x = torch.tensor(X_scaled, dtype=torch.float)

# 把 data 移到 device
data = data.to(device)

# 訓練迴圈：每 20 個 epoch 印出 loss，並評估一次混淆矩陣/FNR
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        # 計算當前預測的 confusion matrix
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
        cm = confusion_matrix(labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            fnr = 0.0

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, FNR: {fnr:.4%}")
        model.train()

# --- 訓練結束後的最終評估與 t-SNE 視覺化 ---
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()

cm = confusion_matrix(labels, preds)
print("\n=== 最終模型評估 ===")
print("Confusion Matrix:\n", cm)
try:
    print(classification_report(labels, preds))
except Exception:
    pass

# 計算 FNR
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
else:
    fnr = 0.0

print(f"Final False Negative Rate (FNR): {fnr:.4%}")

# t-SNE 視覺化 - 使用第一層 conv 的輸出作為 latent features
print("\n執行 t-SNE 降維並產生視覺化...")
with torch.no_grad():
    latent = F.relu(model.conv1(data.x, data.edge_index)).cpu().numpy()

# 若節點過多，隨機抽樣最多 5000 個節點加速 t-SNE
num_nodes = latent.shape[0]
if num_nodes > 5000:
    rng = np.random.RandomState(42)
    idx = rng.choice(num_nodes, size=5000, replace=False)
    latent_sample = latent[idx]
    labels_sample = labels[idx]
else:
    latent_sample = latent
    labels_sample = labels

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(latent_sample)

scatter_df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': ['Blacklist' if l == 1 else 'Normal' for l in labels_sample]
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=scatter_df, x='x', y='y', hue='label', palette={'Blacklist': 'red', 'Normal': 'skyblue'}, alpha=0.6, s=20)
plt.title(f'N2V-GCN 節點嵌入 t-SNE (FNR: {fnr:.2%})')
plt.savefig('gcn_tsne_visualization.png', dpi=150)
print("t-SNE 圖表已儲存為: gcn_tsne_visualization.png")