import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import copy

# --- 1. 定義圖結構 G 與建立索引映射 ---
def build_graph_data():
    # 讀取邊資料
    shared_ip = pd.read_csv('../data/edge/shared_ip_edges.csv')
    crypto = pd.read_csv('../data/crypto_transfer.csv')
    
    # 建立 NetworkX 圖並統計邊的 count/amount
    G = nx.Graph()

    from collections import defaultdict
    edge_stats = defaultdict(lambda: {'count': 0, 'amount': 0.0})

    # IP 共用視為 count
    for _, row in shared_ip.iterrows():
        try:
            u = int(row['user_id_1'])
            v = int(row['user_id_2'])
        except Exception:
            continue
        key = (min(u, v), max(u, v))
        edge_stats[key]['count'] += 1

    # 轉帳邊：累計 count 與 amount (若有)
    crypto_internal = crypto[crypto['relation_user_id'].notnull()]
    for _, row in crypto_internal.iterrows():
        try:
            u = int(row['user_id'])
            v = int(row['relation_user_id'])
        except Exception:
            continue
        amt = 0.0
        if 'amount' in row.index:
            try:
                amt = float(row['amount'])
            except Exception:
                amt = 0.0
        key = (min(u, v), max(u, v))
        edge_stats[key]['count'] += 1
        edge_stats[key]['amount'] += amt

    # 建圖並把屬性放入
    for (u, v), s in edge_stats.items():
        G.add_edge(u, v, count=s['count'], amount=s['amount'])

    # 建立 node id 映射
    all_nodes = sorted(list(G.nodes()))
    id_map = {node_id: i for i, node_id in enumerate(all_nodes)}

    # 建立 edge_index 與 edge_weight（雙向）
    edge_list = []
    counts = []
    amounts = []
    for u, v, attr in G.edges(data=True):
        counts.append(attr.get('count', 1.0))
        amounts.append(attr.get('amount', 0.0))

    counts = np.array(counts, dtype=float)
    amounts = np.array(amounts, dtype=float)

    def minmax(a):
        if a.size == 0:
            return a
        amin = a.min(); amax = a.max()
        if amax - amin == 0:
            return np.zeros_like(a)
        return (a - amin) / (amax - amin)

    ncounts = minmax(counts)
    namounts = minmax(amounts)

    alpha = 0.4
    beta = 0.6

    edge_weight_list = []
    i = 0
    for u, v, attr in G.edges(data=True):
        w = alpha * ncounts[i] + beta * namounts[i]
        iu = id_map[u]; iv = id_map[v]
        edge_list.append([iu, iv]); edge_weight_list.append(w)
        edge_list.append([iv, iu]); edge_weight_list.append(w)
        i += 1

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    return G, id_map, edge_index, all_nodes, edge_weight

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
G, id_map, edge_index, all_nodes, edge_weight = build_graph_data()
x, y = prepare_features(id_map, all_nodes)

# 建立 PyG Data 物件
# 注意：這裡包含全圖資訊，解決了 SMOTE 節點沒連線的問題
data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)

# 定義 GCN 模型
class N2V_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(N2V_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        h1 = F.relu(h1)
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        h2 = F.relu(h2)
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

# --- 特徵標準化 ---
scaler = StandardScaler()
X_np = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
X_scaled = scaler.fit_transform(X_np)
data.x = torch.tensor(X_scaled, dtype=torch.float).to(device)

# --- 特徵標準化 ---
scaler = StandardScaler()
X_np = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
X_scaled = scaler.fit_transform(X_np)
data.x = torch.tensor(X_scaled, dtype=torch.float).to(device)

# --- 產生 stratified train/val/test masks（例如 70/15/15）---
y_np = data.y.cpu().numpy()
idx = np.arange(len(y_np))
tr_idx, temp_idx, y_tr, y_temp = train_test_split(idx, y_np, stratify=y_np, test_size=0.30, random_state=42)
val_idx, te_idx, y_val, y_te = train_test_split(temp_idx, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

train_mask = torch.zeros(len(y_np), dtype=torch.bool)
val_mask = torch.zeros(len(y_np), dtype=torch.bool)
test_mask = torch.zeros(len(y_np), dtype=torch.bool)
train_mask[tr_idx] = True
val_mask[val_idx] = True
test_mask[te_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# --- 訓練 loop（使用 mask 計算 loss，加入 early stopping）---
best_val = 1e9
patience = 20
patience_cnt = 0
best_model_state = None

for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_weight=data.edge_attr.to(device) if data.edge_attr is not None else None)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # val 評估
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_weight=data.edge_attr.to(device) if data.edge_attr is not None else None)
        val_loss = criterion(logits[data.val_mask], data.y[data.val_mask]).item()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        cm = confusion_matrix(labels[data.val_mask.cpu().numpy()], preds[data.val_mask.cpu().numpy()]) 
    # early stop
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_cnt = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        patience_cnt += 1
    if patience_cnt >= patience:
        print("Early stopping at epoch", epoch)
        break

# --- 訓練結束後的最終評估與 t-SNE 視覺化 ---
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index, edge_weight=data.edge_attr.to(device) if data.edge_attr is not None else None)
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
    latent = F.relu(model.conv1(data.x, data.edge_index, edge_weight=data.edge_attr.to(device) if data.edge_attr is not None else None)).cpu().numpy()

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

# --- Precision-Recall 動態閾值分析 ---
with torch.no_grad():
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    idxs = np.where(recalls >= 1.0)[0]
    best_threshold = None
    if len(idxs) > 0 and len(thresholds) > 0:
        idx = idxs[-1]
        thr_idx = max(0, min(len(thresholds) - 1, idx - 1))
        best_threshold = thresholds[thr_idx]
        print(f"達成 FNR=0 的最佳判斷門檻: {best_threshold:.4f}")
    else:
        print("找不到使 Recall==1.0 的閾值，請檢查資料或模型輸出。")

    try:
        plt.figure(figsize=(6, 5))
        plt.plot(recalls, precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig('gcn_pr_curve.png', dpi=150)
        print('Precision-Recall 圖表已儲存為: gcn_pr_curve.png')
    except Exception:
        pass